"""Bounded CUDA cache replay; preparation is file-only, no downloads.
Run --run-gpu only in an explicitly granted exclusive GPU window. Processes
stop on their first error/timeout; no retries, resets or automatic installation.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import time
from replay_cuda import frames, sha, write


def water192():
    result = {'positions': [], 'species': [],
              'cell': [[12.4, 0., 0.], [0., 12.4, 0.], [0., 0., 12.4]],
              'units': 'EvAngstrom'}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                angle = ((i + 2*j + 3*k) % 4) * math.pi / 2
                c, s = math.cos(angle), math.sin(angle)
                origin = [.6 + 3.1*i, .6 + 3.1*j, .6 + 3.1*k]
                for kind, (x, y, z) in zip(('O', 'H', 'H'),
                                         ((0., 0., 0.), (.9572, 0., 0.), (-.239, .927, 0.))):
                    result['positions'].append([origin[0]+c*x-s*y, origin[1]+s*x+c*y, origin[2]+z])
                    result['species'].append(kind)
    return result


def flat(values):
    for value in values:
        if isinstance(value, list):
            yield from flat(value)
        else:
            yield value


def compare_md(cpu, cuda):
    records = [[json.loads(s) for s in p.read_text().splitlines()] for p in (cpu, cuda)]
    for run in records:
        assert run[-1]['kind'] == 'completed' and run[-1]['final_step'] == 20
    headers = [run[0] for run in records]
    for field in ('model_sha256', 'timestep_fs', 'initialize_velocities', 'remove_center_of_mass', 'masses_u'):
        assert headers[0][field] == headers[1][field], field
    initialization = headers[0]['initialize_velocities']
    a, b = [[r for r in run if r['kind'] == 'frame'] for run in records]
    assert len(a) == len(b) == 21
    getters = {'position': lambda r: r['frame']['positions'], 'velocity': lambda r: r['velocities'],
               'energy': lambda r: [r['evaluation']['energy']], 'force': lambda r: r['evaluation']['forces'],
               'virial': lambda r: r['evaluation']['virial']}
    errors = {}
    for field, get in getters.items():
        maximum = 0.
        for x, y in zip(a, b):
            assert x['step'] == y['step']
            left, right = list(flat(get(x))), list(flat(get(y)))
            assert len(left) == len(right)
            for p, q in zip(left, right):
                assert math.isfinite(p) and math.isfinite(q)
                assert abs(p-q) <= 1e-9 + 1e-9*abs(p), (field, p, q)
                maximum = max(maximum, abs(p-q))
        errors[field] = maximum
    return {'snapshots': 21, 'steps': 20, 'timestep_fs': headers[0]['timestep_fs'], 'ensemble': 'NVE',
            'temperature_kelvin': initialization['temperature_kelvin'], 'seed': initialization['seed'],
            'max_absolute_error': errors, 'tolerance': '1e-9 absolute + 1e-9 relative',
            'cpu_max_energy_drift_ev': records[0][-1]['max_absolute_energy_drift_ev'],
            'cuda_max_energy_drift_ev': records[1][-1]['max_absolute_energy_drift_ev']}


def execute(output, name, command):
    started = time.perf_counter()
    try:
        result = subprocess.run([str(x) for x in command], capture_output=True, text=True,
                                encoding='utf-8', timeout=60)
    except subprocess.TimeoutExpired as error:
        write(output/'failure.json', {'case': name, 'error': str(error), 'action': 'stop; no retry/reset'})
        raise
    (output/(name+'.stdout.txt')).write_text(result.stdout, encoding='utf-8')
    (output/(name+'.stderr.txt')).write_text(result.stderr, encoding='utf-8')
    if result.returncode:
        write(output/'failure.json', {'case': name, 'returncode': result.returncode,
                                     'stderr': result.stderr, 'action': 'stop; no retry/reset'})
        raise RuntimeError(name + ': ' + result.stderr)
    return result.stdout, (time.perf_counter()-started)*1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('baseline-check-exe', 'check-exe', 'engine-exe', 'public-model',
                 'medium-model', 'medium-frames', 'output-dir'):
        parser.add_argument('--'+flag, required=True, type=Path)
    parser.add_argument('--run-gpu', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame = water192()
    frames(output/'public-water192.jsonl', [frame])
    request = {'options': {'native_model': str(args.public_model.resolve()), 'device': 'cpu'},
               'frame': frame, 'masses': [15.999 if k == 'O' else 1.008 for k in frame['species']],
               'initialize_velocities': {'temperature_kelvin': 300., 'seed': 20261014,
                                         'rescale_to_temperature': True},
               'remove_center_of_mass': True, 'ensemble': {'kind': 'nve'},
               'timestep_fs': .1, 'steps': 20, 'sample_every': 1}
    for device in ('cpu', 'cuda'):
        r = deepcopy(request)
        r['options']['device'] = 'cpu' if device == 'cpu' else 'cuda:0'
        write(output/f'md-{device}.request.json', r)
    cases = [('medium', args.medium_model.resolve(), args.medium_frames.resolve()),
             ('public192', args.public_model.resolve(), output/'public-water192.jsonl')]
    plan = {'gpu_executed': False, 'cases': [{'name': n, 'model_sha256': sha(m),
             'frames_sha256': sha(f)} for n, m, f in cases], 'repeats': 5,
             'geometry_scope': 'original molecular lattice, not equilibrated water or an accuracy dataset'}
    write(output/'plan.json', plan)
    if not args.run_gpu:
        print(json.dumps(plan, indent=2))
        return
    report = dict(plan, gpu_executed=True, comparisons=[])
    report['executable_sha256'] = {k: sha(getattr(args, k)) for k in
                                 ('baseline_check_exe', 'check_exe', 'engine_exe')}
    for name, model, frame_path in cases:
        pair = {'case': name, 'model_sha256': sha(model), 'frames_sha256': sha(frame_path)}
        for label, executable in (('baseline', args.baseline_check_exe), ('optimized', args.check_exe)):
            stdout, _ = execute(output, label+'-'+name, [executable.resolve(), model, frame_path, '5'])
            pair[label] = json.loads(stdout)
        pair['baseline_over_optimized_cuda_ratio'] = (pair['baseline']['warmed_cuda_total_ms'] /
                                                     pair['optimized']['warmed_cuda_total_ms'])
        report['comparisons'].append(pair)
    report['md_process_elapsed_ms'] = {}
    for label in ('cpu', 'cuda'):
        _, elapsed = execute(output, 'md-'+label, [args.engine_exe.resolve(), 'run',
                             output/f'md-{label}.request.json', output/f'md-{label}.jsonl'])
        report['md_process_elapsed_ms'][label] = elapsed
    report['md'] = compare_md(output/'md-cpu.jsonl', output/'md-cuda.jsonl')
    write(output/'acceptance.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
