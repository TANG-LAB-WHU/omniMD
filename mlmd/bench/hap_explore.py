"""Repeat exploration runs of a pipeline with an ensemble of native models and
compare the model-deviation classification with the pipeline's record (host).

    python bench/hap_explore.py EXPLORE_DIR CASE_DIR <target>/release OUT.json RUN [RUN ...]
        [--mode float32-compressed] [--window 0.1 0.3]

`EXPLORE_DIR/RUN` holds `conf.lmp`, `in.lmp` and `model_devi.out` of a finished
LAMMPS run with `fix nvt` or `fix npt ... tri`; it is only read. `CASE_DIR`
holds `manifest.json` (type map and model numbers) and `model-K.native.json`.
For every run: `omnimd-mlmd run` with the first model and the temperature,
pressure, time constants, length and sampling interval of `in.lmp`, then
`import-trajectory` and `model-deviation` with every model (float64 on the GPU
runtime) and the force window. Reported next to the recorded values: the
fractions of frames below, inside and above the window and the mean and the
largest of the per-frame largest force deviation. The ensembles and the
trajectories differ, so only the statistics are comparable, and for runs in
which the models fail not even those closely.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hap_npt  # noqa: E402  (read_start, read_deviation, MASS)


def protocol(run):
    text = (run / 'in.lmp').read_text(encoding='utf-8', errors='replace')
    number = r'(\S+)'
    npt = re.search(rf'fix\s+\S+\s+all\s+npt\s+temp\s+{number}\s+{number}\s+{number}\s+tri\s+{number}\s+{number}\s+{number}', text)
    nvt = re.search(rf'fix\s+\S+\s+all\s+nvt\s+temp\s+{number}\s+{number}\s+{number}', text)
    fix = npt or nvt
    if not fix or fix.group(1) != fix.group(2):
        raise ValueError('in.lmp needs one fix nvt or fix npt ... tri with a constant temperature')
    out = {'temperature_kelvin': float(fix.group(1)), 'thermostat_fs': float(fix.group(3)) * 1e3,
           'timestep_fs': float(re.search(r'^timestep\s+(\S+)', text, re.M).group(1)) * 1e3,
           'steps': int(re.findall(r'^run\s+(\d+)', text, re.M)[-1]),
           'sample_every': int(re.search(r'^thermo\s+(\d+)', text, re.M).group(1))}
    if npt:
        if npt.group(4) != npt.group(5):
            raise ValueError('fix npt needs a constant pressure')
        out.update(pressure_bar=float(npt.group(4)), barostat_fs=float(npt.group(6)) * 1e3)
    return out


def classify(largest, low, high):
    n = len(largest)
    return {'frames': n, 'accurate': sum(x < low for x in largest) / n,
            'candidate': sum(low <= x < high for x in largest) / n, 'failed': sum(x >= high for x in largest) / n,
            'mean_max_devi_f': sum(largest) / n, 'max_devi_f_max': max(largest)}


def engine(exe, command, request, work, name):
    (work / f'{name}.json').write_text(json.dumps(request))
    proc = subprocess.run([str(exe), command, str(work / f'{name}.json'), str(work / f'{name}.jsonl')],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr.strip().splitlines() or [command + ' failed'])[-1][:300])
    return work / f'{name}.jsonl'


def explore(exe, run, name, manifest, native, runtime, window):
    """One run through the engine: classification of its frames and time per step."""
    low, high = window
    settings = protocol(run)
    frame = hap_npt.read_start(run, manifest['type_map'])
    ensemble = {'kind': 'nvt_csvr', 'temperature_kelvin': settings['temperature_kelvin'],
                'tau_fs': settings['thermostat_fs'], 'seed': 1}
    if 'pressure_bar' in settings:
        ensemble.update(kind='npt_csvr_crescale', pressure_bar=settings['pressure_bar'],
                        barostat_tau_fs=settings['barostat_fs'], compressibility_per_bar=1e-6,
                        coupling='triclinic')
    request = {
        'options': {'native_model': native(manifest['models'][0]), 'device': 'cuda:0', 'gpu_runtime': runtime},
        'frame': frame, 'masses': [hap_npt.MASS[s] for s in frame['species']],
        'initialize_velocities': {'temperature_kelvin': settings['temperature_kelvin'], 'seed': 20261329,
                                  'rescale_to_temperature': True},
        'remove_center_of_mass': True, 'ensemble': ensemble, 'timestep_fs': settings['timestep_fs'],
        'steps': settings['steps'], 'sample_every': settings['sample_every']}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        start = time.perf_counter()
        trajectory = engine(exe, 'run', request, work, 'run')
        seconds = time.perf_counter() - start
        dataset = engine(exe, 'import-trajectory', {'schema': 'omnimd-trajectory-import-v1',
                                                    'trajectory': str(trajectory), 'id_prefix': name},
                         work, 'explored')
        members = [{'id': f'model-{k}', 'options': {'native_model': native(k), 'device': 'cuda:0',
                                                    'gpu_runtime': {}}} for k in manifest['models']]
        deviations = engine(exe, 'model-deviation', {
            'schema': 'omnimd-model-deviation-v1', 'dataset': str(dataset), 'models': members,
            'force_window': {'min_ev_per_angstrom': low, 'max_ev_per_angstrom': high}}, work, 'deviation')
        lines = [json.loads(line) for line in deviations.read_text(encoding='utf-8').splitlines()]
    largest = [r['statistics']['force_ev_per_angstrom']['max'] for r in lines if r['kind'] == 'deviation']
    return dict(classify(largest, low, high), ms_per_step=seconds * 1e3 / settings['steps'], protocol=settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('explore_dir', type=Path)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('output_json', type=Path)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--mode', default='float32-compressed',
                        choices=['float64', 'float32', 'float64-compressed', 'float32-compressed'])
    parser.add_argument('--window', type=float, nargs=2, default=[0.1, 0.3])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    manifest = json.loads((args.case_dir / 'manifest.json').read_text(encoding='utf-8'))
    exe = args.release / 'omnimd-mlmd'
    if not exe.exists():
        exe = exe.with_suffix('.exe')
    runtime = {}
    if args.mode.startswith('float32'):
        runtime['precision'] = 'float32'
    if args.mode.endswith('compressed'):
        runtime['compress_min_distance'] = 0.5

    def native(k):
        return str((args.case_dir / f'model-{k}.native.json').resolve())

    results = {}
    print(f"{'run':30} {'':9} {'accurate':>9} {'candidate':>9} {'failed':>9} {'mean max':>9} {'largest':>9}")
    for name in args.runs:
        run = args.explore_dir / name
        entry = {'recorded': classify(hap_npt.read_deviation(run)[0], *args.window)}
        try:
            entry['omnimd'] = explore(exe, run, name, manifest, native, runtime, args.window)
        except (RuntimeError, ValueError) as error:
            entry['omnimd'] = {'error': str(error)}
        results[name] = entry
        for label in ('omnimd', 'recorded'):
            row = entry[label]
            if 'error' in row:
                print(f"{name[:30]:30} {label:9} {row['error'][:70]}")
            else:
                print(f"{name[:30]:30} {label:9} {row['accurate']:9.3f} {row['candidate']:9.3f} {row['failed']:9.3f} "
                      f"{row['mean_max_devi_f']:9.4f} {row['max_devi_f_max']:9.4f}", flush=True)
    with args.output_json.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump({'mode': args.mode, 'window_ev_per_angstrom': args.window, 'models': manifest['models'],
                   'runs': results}, handle, indent=1)
        handle.write('\n')


if __name__ == '__main__':
    main()
