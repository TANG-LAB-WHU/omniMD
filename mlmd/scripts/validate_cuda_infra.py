#!/usr/bin/env python3
"""Bounded final native CUDA cache acceptance; explicit GPU window required."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def compare(a, b):
    if isinstance(a, list) and isinstance(b, list):
        require(len(a) == len(b), 'comparison shape mismatch')
        return max((compare(x, y) for x, y in zip(a, b)), default=0.)
    require(isinstance(a, (int, float)) and isinstance(b, (int, float))
            and math.isfinite(a) and math.isfinite(b), 'nonfinite/nonnumeric comparison')
    error = abs(a-b)
    require(error <= 1e-9 + 1e-9*abs(a), f'component mismatch: {a} vs {b}')
    return error


def compare_md(a, b):
    require(len(a) == len(b), 'snapshot count mismatch')
    result = {k: 0. for k in ('position', 'velocity', 'energy', 'force', 'virial')}
    for x, y in zip(a, b):
        require(x['step'] == y['step'] and x['time_fs'] == y['time_fs'], 'step mismatch')
        require(x['frame']['species'] == y['frame']['species']
                and x['frame']['cell'] == y['frame']['cell'], 'frame metadata mismatch')
        values = {'position': (x['frame']['positions'], y['frame']['positions']),
                  'velocity': (x['velocities'], y['velocities']),
                  'energy': (x['evaluation']['energy'], y['evaluation']['energy']),
                  'force': (x['evaluation']['forces'], y['evaluation']['forces']),
                  'virial': (x['evaluation']['virial'], y['evaluation']['virial'])}
        for key, pair in values.items():
            result[key] = max(result[key], compare(*pair))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    flags = ('baseline-engine', 'baseline-check', 'engine', 'check', 'public-model',
             'public-frames', 'medium-model', 'medium-frames', 'work')
    for flag in flags:
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--run-gpu', action='store_true')
    args = parser.parse_args()
    require(args.run_gpu, 'Requires --run-gpu and a coordinator-granted exclusive window')
    paths = {name: getattr(args, name.replace('-', '_')).resolve(strict=True)
             for name in flags if name != 'work'}
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    require(not (work/'plan.json').exists(), 'Use a new acceptance output directory')
    hashes = {key: sha(path) for key, path in paths.items()}
    plan = {'source_commit': 'bdf604875434acf99c4ba4d2ca690382d030a53d',
            'native_source_commit': 'b6f36dd2ed5efb78e4184ba61854a06eb27285a0',
            'baseline_commit': 'a74804d8cbba86c1a2563dbbca6444059e740903',
            'sha256': hashes, 'script_sha256': sha(Path(__file__)),
            'started_utc': datetime.now(timezone.utc).isoformat()}
    write(work/'plan.json', plan)

    def execute(name, command):
        print('Running '+name, flush=True)
        start = time.perf_counter()
        process = subprocess.Popen([str(x) for x in command], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf8', start_new_session=os.name != 'nt')
        try:
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            # Stop only this run's process tree; never reset a GPU or retry.
            if os.name == 'nt':
                subprocess.run([str(Path(os.environ['SystemRoot'])/'System32/taskkill.exe'),
                    '/PID', str(process.pid), '/T', '/F'], capture_output=True)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            write(work/'failure.json', {'case': name, 'timeout_seconds': 60, 'action': 'stop; no retry'})
            raise RuntimeError(name+': timeout; owned process tree stopped')
        elapsed = (time.perf_counter()-start)*1000
        (work/(name+'.stdout.txt')).write_text(stdout, encoding='utf8')
        (work/(name+'.stderr.txt')).write_text(stderr, encoding='utf8')
        if process.returncode:
            write(work/'failure.json', {'case': name, 'returncode': process.returncode,
                                      'stderr': stderr, 'action': 'stop; no retry'})
            raise RuntimeError(name+': '+stderr)
        return stdout, elapsed

    report = dict(plan, schema='omnimd-cuda-infra-final-acceptance-v1', comparisons=[])
    for name, model, frames in (('medium216', paths['medium-model'], paths['medium-frames']),
                               ('public192', paths['public-model'], paths['public-frames'])):
        pair = {'case': name, 'model_sha256': sha(model), 'frames_sha256': sha(frames)}
        for label, binary in (('baseline_cuda', paths['baseline-check']), ('final_cuda', paths['check'])):
            stdout, _ = execute(name+'-'+label, [binary, model, frames, '5'])
            result = json.loads(stdout)
            require(result['frames'] == 1 and result['repeats'] == 5, 'timing workload mismatch')
            for key in ('warmed_cpu_total_ms', 'warmed_cuda_total_ms', 'startup_ms'):
                require(math.isfinite(result[key]) and result[key] > 0, 'invalid timing')
            pair[label] = result
        new, old = pair['final_cuda'], pair['baseline_cuda']
        require(new['atoms'] == old['atoms'], 'atom count changed')
        require(new['first_profiles'][0]['neighbor_cache_rebuilt'], 'missing initial candidate build')
        require(new['last_profile']['neighbor_cache_rebuilds'] == 1
                and not new['last_profile']['neighbor_cache_rebuilt']
                and new['last_profile']['neighbor_candidates'] > 0,
                'final guard prevented expected same-frame cache reuse')
        pair['mean_ms_per_eval'] = {'old_cuda': old['warmed_cuda_total_ms']/5,
            'final_cuda': new['warmed_cuda_total_ms']/5,
            'cpu_from_final_build_unchanged_algorithm': new['warmed_cpu_total_ms']/5}
        pair['old_cuda_over_final_cuda_ratio'] = old['warmed_cuda_total_ms']/new['warmed_cuda_total_ms']
        pair['unchanged_cpu_over_final_cuda_ratio'] = new['warmed_cpu_total_ms']/new['warmed_cuda_total_ms']
        report['comparisons'].append(pair)

    input_frames = [json.loads(line) for line in paths['public-frames'].read_text().splitlines() if line.strip()]
    require(len(input_frames) == 1 and len(input_frames[0]['positions']) == 192, 'expected public 192-atom frame')
    frame = input_frames[0]
    request = {'options': {'native_model': str(paths['public-model']), 'device': 'cpu'},
        'dynamics_units': 'EvAngstromFsU', 'frame': frame,
        'masses': [15.999 if kind == 'O' else 1.008 for kind in frame['species']],
        'initialize_velocities': {'temperature_kelvin': 300., 'seed': 20261014,
                                  'rescale_to_temperature': True},
        'remove_center_of_mass': True, 'ensemble': {'kind': 'nve'},
        'timestep_fs': .1, 'steps': 20, 'sample_every': 1, 'checkpoint_every': 10}
    require(set(frame['species']) == {'O', 'H'}, 'unexpected species')

    def md_run(name, binary, r, start, steps):
        ip, op = work/(name+'.request.json'), work/(name+'.jsonl')
        write(ip, r)
        _, elapsed = execute(name, [binary, 'run', ip, op])
        records = [json.loads(line) for line in op.read_text().splitlines()]
        require(records[0]['kind'] == 'header' and records[-1]['kind'] == 'completed', 'incomplete MD')
        backend = 'native-se-e2-a-v1' if r['options']['device'] == 'cpu' else 'native-cuda-se-e2-a-v1'
        require(records[0]['metadata']['backend'] == backend, 'wrong MD backend')
        require(records[0]['model_sha256'] == hashes['public-model'], 'wrong MD model')
        require(records[0]['start_step'] == start and records[-1]['final_step'] == start+steps, 'wrong MD step')
        snapshots = [rec for rec in records if rec['kind'] == 'frame']
        require([rec['step'] for rec in snapshots] == list(range(start,start+steps+1)), 'missing MD snapshot')
        require(all(rec['frame']['units'] == rec['evaluation']['units'] == 'EvAngstrom'
                    for rec in snapshots), 'MD unit mismatch')
        return op, records, snapshots, elapsed

    runs = {}
    for name, binary, device in (('md-old-cuda', paths['baseline-engine'], 'cuda:0'),
                                ('md-final-cpu', paths['engine'], 'cpu'),
                                ('md-final-cuda', paths['engine'], 'cuda:0')):
        r = deepcopy(request)
        r['options']['device'] = device
        runs[name] = md_run(name, binary, r, 0, 20)
    old, cpu, gpu = (runs[key] for key in ('md-old-cuda','md-final-cpu','md-final-cuda'))
    md_report = {'atoms': 192, 'steps': 20, 'timestep_fs': .1, 'ensemble': 'nve',
        'checkpoint_every': 10, 'sample_every': 1,
        'whole_process_ms': {name: value[3] for name, value in runs.items()},
        'final_cpu_vs_final_cuda_max_abs_error': compare_md(cpu[2], gpu[2]),
        'old_cuda_vs_final_cuda_max_abs_error': compare_md(old[2], gpu[2]),
        'max_energy_drift_ev': {name: value[1][-1]['max_absolute_energy_drift_ev'] for name,value in runs.items()},
        'old_cuda_over_final_cuda_whole_process_ratio': old[3]/gpu[3],
        'unchanged_cpu_over_final_cuda_whole_process_ratio': cpu[3]/gpu[3]}
    resumed_request = {'options': {'native_model': str(paths['public-model']), 'device': 'cuda:0'},
        'dynamics_units': 'EvAngstromFsU', 'restart': str(gpu[0])+'.step-10.checkpoint.json',
        'timestep_fs': .1, 'steps': 10, 'sample_every': 1}
    resumed = md_run('md-final-cuda-resumed', paths['engine'], resumed_request, 10, 10)
    require(gpu[2][10:] == resumed[2], 'resumed snapshots differ from uninterrupted final CUDA')
    full_cp = json.loads(Path(str(gpu[0])+'.checkpoint.json').read_text())
    restart_cp = json.loads(Path(str(resumed[0])+'.checkpoint.json').read_text())
    require(full_cp == restart_cp, 'final checkpoint state differs after restart')
    md_report.update(resume_whole_process_ms=resumed[3], restart_step=10,
                     identical_resumed_snapshots=True, identical_final_checkpoint_state=True)
    report['md'] = md_report

    # One persistent potential sees motion, then a >half-skin rebuild and small reuse.
    motion = [deepcopy(gpu[2][index]['frame']) for index in (0,5,10,15,20)]
    for shift in (.25, .26):
        moved = deepcopy(gpu[2][0]['frame'])
        for position in moved['positions']:
            position[0] += shift
        motion.append(moved)
    motion_path = work/'final-motion-rebuild.jsonl'
    with motion_path.open('x', encoding='utf8') as stream:
        for f in motion:
            stream.write(json.dumps(f, allow_nan=False)+'\n')
    stdout, _ = execute('final-motion-rebuild', [paths['check'], paths['public-model'], motion_path, '1'])
    motion_result = json.loads(stdout)
    flags = [p['neighbor_cache_rebuilt'] for p in motion_result['first_profiles']]
    require(flags == [True,False,False,False,False,True,False], f'unexpected cache rebuild sequence: {flags}')
    require(all(p['neighbor_candidates'] > 0 for p in motion_result['first_profiles']), 'normal-scale cache disabled')
    report['final_motion_rebuild'] = motion_result
    for key, path in paths.items():
        require(sha(path) == hashes[key], 'input or executable changed during acceptance: '+key)
    report.update(status='passed', completed_utc=datetime.now(timezone.utc).isoformat(),
        limitations=['Single bounded run; five warmed evaluations per timing case, means only; no distribution estimate.',
          'Baseline checker adds only host timers; final CPU algorithm is unchanged and does not share the CUDA neighbor cache.',
          'CUDA construction and first evaluation are separately timed, not machine-cold measurements.',
          'Whole MD process includes startup, model loading/compilation, JSON output and checkpoint fsync.',
          'Normal desktop graphics were not stopped or controlled; no general speedup or physical-accuracy claim.',
          'No C API GPU, GPU training, driver/system changes or private research inputs.'])
    write(work/'report.json', report)
    print(json.dumps({'status': report['status'], 'comparisons': [
        {key: pair[key] for key in ('case','mean_ms_per_eval','old_cuda_over_final_cuda_ratio',
                                    'unchanged_cpu_over_final_cuda_ratio')} for pair in report['comparisons']],
        'md': md_report, 'motion_rebuild_flags': flags}, indent=2))


if __name__ == '__main__':
    main()
