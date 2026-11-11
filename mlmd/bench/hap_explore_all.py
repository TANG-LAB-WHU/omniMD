"""An exploration of the HAP systems as one replica run, and its candidates (host).

    python bench/hap_explore_all.py CASE_DIR <target>/release OUTPUT_DIR [--result RESULT.json]
        [--systems SYSTEM ...] [--temperatures 300 500] [--npt-temperatures 300]
        [--steps 40000] [--stride 50] [--window 0.10 0.30] [--max-per-run 30]
        [--mode float32-compressed]

`CASE_DIR` comes from `hap_prepare.py`; the engine must be built with
`--features native-gpu`. Every system of the case (or those named) starts
from its first frame at every temperature: NVT with the CSVR thermostat
(tau 50 fs), 0.5 fs steps, model 0 driving and the other models of the case
watching every `--stride`-th frame. At `--npt-temperatures` the systems that
are not slabs (`hap001*`) run at 1.01325 bar with a fully flexible cell
instead (stochastic cell rescaling, tau 500 fs). All runs are replicas of one
`omnimd-mlmd run-replicas`; `collect-candidates` then writes the candidates,
at most `--max-per-run` of a run, and the summary.

This is the protocol of the pipeline's `dp_explore.py` (LAMMPS with
`pair_style deepmd`, Nose-Hoover) as far as the engines share it; thermostat
and barostat differ. (`hap_explore.py` repeats single runs of the pipeline
from their LAMMPS inputs and compares the statistics with its record.) Printed and written to `--result`: the seconds of the
stepping loop as the engine reports them, of the whole run-replicas process
(models loaded, tables built, trajectories written) and of the collection,
and what the summary says of every run. OUTPUT_DIR must not exist.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

MASS = {'Ca': 40.078, 'P': 30.9738, 'O': 15.999, 'H': 1.008, 'C': 12.011}
TIMESTEP_FS = 0.5
MODES = {'float64': {}, 'float32': {'precision': 'float32'},
         'float64-compressed': {'compress_min_distance': 0.5},
         'float32-compressed': {'precision': 'float32', 'compress_min_distance': 0.5}}
RATE = re.compile(r'(\d+) steps of (\d+) system\(s\) in ([0-9.]+) s, ([0-9.]+) ms per step')


def engine(release):
    exe = release / 'omnimd-mlmd'
    return exe if exe.exists() else exe.with_suffix('.exe')


def run(exe, command, request, path, output):
    """Runs an engine command; its whole-process seconds and what it printed."""
    path.write_text(json.dumps(request))
    start = time.perf_counter()
    proc = subprocess.run([str(exe), command, str(path), str(output)], capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise SystemExit((proc.stderr.strip().splitlines() or ['engine failed'])[-1][:600])
    return elapsed, proc.stdout + proc.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--result', type=Path, help='timings and the digest of the summary as JSON')
    parser.add_argument('--systems', nargs='*', help='systems to explore (default: all of the case)')
    parser.add_argument('--temperatures', nargs='+', type=float, default=[300.0, 500.0])
    parser.add_argument('--npt-temperatures', nargs='*', type=float, default=[],
                        help='temperatures at which systems that are not slabs run at constant pressure')
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--stride', type=int, default=50)
    parser.add_argument('--window', nargs=2, type=float, default=[0.10, 0.30], metavar=('LOW', 'HIGH'))
    parser.add_argument('--max-per-run', type=int, default=30)
    parser.add_argument('--mode', default='float32-compressed', choices=list(MODES))
    parser.add_argument('--seed', type=int, default=20261029)
    args = parser.parse_args()
    manifest = json.loads((args.case_dir / 'manifest.json').read_text())
    frames = [json.loads(line) for line in (args.case_dir / 'frames.jsonl').read_text().splitlines()]
    systems = args.systems or sorted({f['system'] for f in manifest['frames']})
    options = [{'native_model': str((args.case_dir / f'model-{k}.native.json').resolve()), 'device': 'cuda:0',
                'gpu_runtime': MODES[args.mode]} for k in manifest['models']]
    window = {'min_ev_per_angstrom': args.window[0], 'max_ev_per_angstrom': args.window[1]}
    replicas, atoms = [], {}
    for system in systems:
        index = next(i for i, f in enumerate(manifest['frames']) if f['system'] == system)
        frame, short = frames[index], system.split('__')[0]
        for temperature in args.temperatures:
            npt = temperature in args.npt_temperatures and not short.startswith('hap001')
            name = f"{short}_T{int(temperature)}{'_npt' if npt else ''}"
            seed = args.seed + int(temperature) + 1000 * len(replicas)
            ensemble = {'kind': 'nvt_csvr', 'temperature_kelvin': temperature, 'tau_fs': 50.0, 'seed': seed}
            if npt:
                ensemble = dict(ensemble, kind='npt_csvr_crescale', pressure_bar=1.01325, barostat_tau_fs=500.0,
                                compressibility_per_bar=1e-6, coupling='triclinic')
            replicas.append({'name': name, 'frame': frame, 'masses': [MASS[s] for s in frame['species']],
                             'initialize_velocities': {'temperature_kelvin': temperature, 'seed': seed + 1,
                                                       'rescale_to_temperature': True},
                             'remove_center_of_mass': True, 'ensemble': ensemble})
            atoms[name] = len(frame['species'])
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    exe = engine(args.release)
    request = {'schema': 'omnimd-replicas-v1', 'options': options[0], 'replicas': replicas,
               'timestep_fs': TIMESTEP_FS, 'steps': args.steps, 'sample_every': args.stride,
               'deviation': {'models': options[1:], 'force_window': window}}
    whole, said = run(exe, 'run-replicas', request, output / 'explore.json', output / 'explored')
    rate = RATE.search(said)
    stepping = float(rate.group(3)) if rate else None
    collect = {'schema': 'omnimd-candidates-v1', 'replica_runs': [str(output / 'explored')],
               'max_per_trajectory': args.max_per_run}
    collecting, said = run(exe, 'collect-candidates', collect, output / 'collect.json', output / 'candidates')
    summary = json.loads((output / 'candidates' / 'summary.json').read_text())
    job_steps = args.steps * len(replicas)
    print(f"{len(replicas)} runs of {args.steps} steps, {sum(atoms.values())} atoms a step, {args.mode}, "
          f"{len(options)} models every {args.stride} steps")
    print(f"run-replicas: {whole:.1f} s in all" + (f", {stepping:.1f} s stepping "
          f"({1e3 * stepping / args.steps:.3f} ms per step, {job_steps / stepping:.0f} run-steps per second)"
          if stepping else ''))
    print(f"collect-candidates: {collecting:.1f} s; {said.strip()}")
    print(f"{'run':<28} {'atoms':>5} {'frames':>6} {'accurate':>8} {'candidate':>9} {'failed':>6} {'picked':>6} "
          f"{'largest':>8}")
    runs = []
    for of in summary['trajectories']:
        name = of['id_prefix']
        runs.append({'run': name, 'natoms': atoms[name], 'ensemble': of['ensemble']['kind'],
                     'frames': of['frames'], 'accurate_frames': of['accurate_frames'],
                     'candidate_frames': of['candidate_frames'], 'failed_frames': of['failed_frames'],
                     'picked': len(of['picked']),
                     'max_force_deviation_ev_per_angstrom': of['max_force_deviation_ev_per_angstrom']})
        print(f"{name:<28} {atoms[name]:>5} {of['frames']:>6} {of['accurate_frames']:>8} "
              f"{of['candidate_frames']:>9} {of['failed_frames']:>6} {len(of['picked']):>6} "
              f"{of['max_force_deviation_ev_per_angstrom']:>8.3f}")
    print(f"accurate fraction {summary['accurate_fraction']:.3f} of all frames, "
          f"{summary['mean_accurate_fraction']:.3f} as the mean of the runs; {summary['picked_frames']} picked")
    if args.result:
        args.result.write_text(json.dumps({
            'schema': 'omnimd-hap-explore-v1', 'mode': args.mode, 'models': len(options),
            'steps': args.steps, 'timestep_fs': TIMESTEP_FS, 'sample_every': args.stride,
            'force_window': window, 'max_per_run': args.max_per_run, 'runs': runs,
            'atoms_per_step': sum(atoms.values()), 'run_steps': job_steps,
            'run_replicas_seconds': round(whole, 2), 'stepping_seconds': stepping,
            'ms_per_step': round(1e3 * stepping / args.steps, 4) if stepping else None,
            'run_steps_per_second': round(job_steps / stepping, 1) if stepping else None,
            'collect_candidates_seconds': round(collecting, 2),
            'frames': summary['frames'], 'accurate_frames': summary['accurate_frames'],
            'candidate_frames': summary['candidate_frames'], 'failed_frames': summary['failed_frames'],
            'accurate_fraction': summary['accurate_fraction'],
            'mean_accurate_fraction': summary['mean_accurate_fraction'],
            'picked_frames': summary['picked_frames']}, indent=1) + '\n')


if __name__ == '__main__':
    main()
