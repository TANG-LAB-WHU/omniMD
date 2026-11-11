"""Replicas of one system advanced together through `omnimd-mlmd run-replicas` (host).

    python bench/hap_replicas.py CASE_DIR <target>/release RESULT.csv SYSTEM [SYSTEM ...]
        [--replicas N ...] [--modes MODE ...] [--steps N] [--repeats N] [--other <target>/release]

`CASE_DIR` comes from `hap_prepare.py`; the engine must be built with
`--features native-gpu`. From the first frame of each SYSTEM, NVT dynamics
(CSVR thermostat, 300 K, 0.5 fs, sampled every 50 steps) runs with model 0:
once as a plain `run`, and as `run-replicas` with N replicas that differ in
their velocity and thermostat seeds. The time per step is what the engine
reports for its stepping loop (trajectory output included, loading the model
left out), the least of `--repeats` runs because the card is shared with a
desktop. `--other` times the plain run of a second build as well (for example
the one before a change of the integrator); a build that does not report its
rate is timed by the difference of two runs of different length.
"""
import argparse
import csv
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

MASS = {'Ca': 40.078, 'P': 30.9738, 'O': 15.999, 'H': 1.008, 'C': 12.011}
TIMESTEP_FS = 0.5
MODES = {'float64': {}, 'float32': {'precision': 'float32'},
         'float64-compressed': {'compress_min_distance': 0.5},
         'float32-compressed': {'precision': 'float32', 'compress_min_distance': 0.5}}


def engine(release):
    exe = release / 'omnimd-mlmd'
    return exe if exe.exists() else exe.with_suffix('.exe')


RATE = re.compile(r'steps of \d+ system\(s\) in [0-9.]+ s, ([0-9.]+) ms per step')


def timed(exe, command, request, work, name):
    """Whole-process seconds, and the milliseconds per step the engine reports (None if it does not)."""
    path, output = work / f'{name}.json', work / name
    path.write_text(json.dumps(request))
    start = time.perf_counter()
    proc = subprocess.run([str(exe), command, str(path), str(output)], capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise SystemExit((proc.stderr.strip().splitlines() or ['engine failed'])[-1][:400])
    reported = RATE.search(proc.stderr)
    return elapsed, float(reported.group(1)) if reported else None


def system_input(frame, seed):
    return {'frame': frame, 'masses': [MASS[s] for s in frame['species']],
            'initialize_velocities': {'temperature_kelvin': 300.0, 'seed': 20261329 + seed,
                                      'rescale_to_temperature': True},
            'remove_center_of_mass': True,
            'ensemble': {'kind': 'nvt_csvr', 'temperature_kelvin': 300.0, 'tau_fs': 50.0, 'seed': 1 + seed}}


def ms_per_step(exe, command, request, work, tag, steps, repeats):
    """Milliseconds per step, the least of `repeats` runs."""
    short = max(steps // 10, 50)
    best = float('inf')
    for attempt in range(repeats):
        full, reported = timed(exe, command, dict(request, steps=steps), work, f'{tag}-{attempt}-full')
        if reported is None:
            # A build that does not say: two runs of different length.
            brief, _ = timed(exe, command, dict(request, steps=short), work, f'{tag}-{attempt}-short')
            reported = (full - brief) * 1e3 / (steps - short)
        best = min(best, reported)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('result', type=Path)
    parser.add_argument('systems', nargs='+')
    parser.add_argument('--replicas', nargs='*', type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument('--modes', nargs='*', default=['float32-compressed', 'float64'], choices=list(MODES))
    parser.add_argument('--steps', type=int, default=20000, help='steps of the plain run; replica runs take fewer')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--other', type=Path, help='a second build whose plain run is timed too')
    args = parser.parse_args()
    manifest = json.loads((args.case_dir / 'manifest.json').read_text())
    frames = [json.loads(line) for line in (args.case_dir / 'frames.jsonl').read_text().splitlines()]
    model = str((args.case_dir / f"model-{manifest['models'][0]}.native.json").resolve())
    exe = engine(args.release)
    with args.result.open('w', newline='') as out, tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        writer = csv.writer(out)
        writer.writerow(['system', 'natoms', 'mode', 'route', 'replicas', 'ms_per_step', 'us_per_replica_step',
                         'ns_per_day_per_replica', 'ns_per_day_in_all', 'replica_steps_vs_plain_run'])

        def record(system, natoms, mode, route, replicas, ms, plain):
            per_replica = ms / replicas
            ns_per_day = TIMESTEP_FS * 1e-6 * 86400 / (per_replica * 1e-3)
            writer.writerow([system, natoms, mode, route, replicas, f'{ms:.4f}', f'{per_replica * 1e3:.1f}',
                             f'{ns_per_day:.1f}', f'{ns_per_day * replicas:.1f}',
                             f'{plain / per_replica:.2f}' if plain else ''])
            out.flush()
            print(f"{system.split('__')[0]:<18} {natoms:>4} atoms  {mode:<19} {route:<12} {replicas:>3} replicas  "
                  f"{ms:8.3f} ms per step  {per_replica * 1e3:7.1f} us per replica  "
                  f"{ns_per_day * replicas:8.1f} ns/day in all" + (f"  x{plain / per_replica:.2f}" if plain else ''),
                  flush=True)

        for system in args.systems:
            index = next(i for i, f in enumerate(manifest['frames']) if f['system'] == system)
            frame, natoms = frames[index], manifest['frames'][index]['natoms']
            for mode in args.modes:
                options = {'native_model': model, 'device': 'cuda:0', 'gpu_runtime': MODES[mode]}
                shared = {'options': options, 'timestep_fs': TIMESTEP_FS, 'sample_every': 50}
                tag = f'{system}-{mode}'
                if args.other:
                    ms = ms_per_step(engine(args.other), 'run', dict(shared, **system_input(frame, 0)), work,
                                     tag + '-other', args.steps, args.repeats)
                    record(system, natoms, mode, 'run (other)', 1, ms, None)
                plain = ms_per_step(exe, 'run', dict(shared, **system_input(frame, 0)), work, tag + '-run',
                                    args.steps, args.repeats)
                record(system, natoms, mode, 'run', 1, plain, None)
                for count in args.replicas:
                    replicas = [dict(system_input(frame, seed), name=f'r{seed}') for seed in range(count)]
                    request = dict(shared, schema='omnimd-replicas-v1', replicas=replicas)
                    steps = max(args.steps // count, 2000)
                    ms = ms_per_step(exe, 'run-replicas', request, work, f'{tag}-{count}', steps, args.repeats)
                    record(system, natoms, mode, 'run-replicas', count, ms, plain)


if __name__ == '__main__':
    main()
