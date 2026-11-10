"""Constant-pressure run of a prepared model through `omnimd-mlmd`, compared
with a recorded LAMMPS `fix npt` run of the same model (host).

    python bench/hap_npt.py LAMMPS_RUN_DIR CASE_DIR <target>/release OUT.json
        [--steps 100000] [--mode float64] [--compressibility 1e-6] [--every 10] [--deviation] [--seed 1]

`LAMMPS_RUN_DIR` holds `conf.lmp` (the starting structure), `log.lammps`
(thermo columns Step Temp PotEng TotEng Press Volume) and `traj.dump` (box per
frame) of a finished `fix npt ... tri` run; it is only read. `CASE_DIR` comes
from `hap_prepare.py` and provides `model-0.native.json`, the model that
drives the LAMMPS run. The engine must be built with `--features native-gpu`.

The engine runs the same protocol with its own algorithms (CSVR thermostat,
stochastic cell rescaling with a fully flexible cell): the temperature,
pressure and time constants are read from the `fix npt` line of `in.lmp`.
Both runs sample the same ensemble, so the averages after equilibration must
agree within their statistical errors; the trajectories themselves differ.
Reported for each: mean and standard error (block averages over the last 80
percent) of volume, potential energy, pressure, temperature, cell lengths and
angles, and the standard deviation of the volume.

With `--deviation` the sampled frames then go through `import-trajectory` and
`model-deviation` with every prepared model (float64 on the GPU runtime) and
the force window 0.1 to 0.3 eV/A of the pipeline, and the per-frame largest and
mean force deviation are compared with `model_devi.out` of the LAMMPS run.
"""
import argparse
import json
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path

MASS = {'Ca': 40.078, 'P': 30.9738, 'O': 15.999, 'H': 1.008, 'C': 12.011}
BLOCKS = 8


def statistics(series):
    """Mean, standard error from block averages, and standard deviation of the last 80 percent."""
    kept = series[len(series) // 5:]
    size = len(kept) // BLOCKS
    blocks = [sum(kept[b * size:(b + 1) * size]) / size for b in range(BLOCKS)]
    mean = sum(blocks) / BLOCKS
    error = math.sqrt(sum((b - mean) ** 2 for b in blocks) / (BLOCKS - 1) / BLOCKS)
    used = kept[:size * BLOCKS]
    spread = math.sqrt(sum((x - mean) ** 2 for x in used) / len(used))
    return {'mean': mean, 'error': error, 'std': spread, 'samples': len(used)}


def shape(cell):
    """Lengths and angles (degrees) of the rows of a cell."""
    norm = [math.sqrt(sum(x * x for x in row)) for row in cell]

    def angle(i, j):
        return math.degrees(math.acos(sum(a * b for a, b in zip(cell[i], cell[j])) / (norm[i] * norm[j])))
    return {'a': norm[0], 'b': norm[1], 'c': norm[2], 'alpha': angle(1, 2), 'beta': angle(0, 2),
            'gamma': angle(0, 1)}


def summarize(columns, cells):
    out = {name: statistics(values) for name, values in columns.items()}
    shapes = [shape(cell) for cell in cells]
    for key in ('a', 'b', 'c', 'alpha', 'beta', 'gamma'):
        out[key] = statistics([s[key] for s in shapes])
    return out


def lammps_cell(bounds):
    """Row lattice vectors from the three `BOX BOUNDS xy xz yz` lines of a dump."""
    (xlo, xhi, xy), (ylo, yhi, xz), (zlo, zhi, yz) = bounds
    xlo -= min(0.0, xy, xz, xy + xz)
    xhi -= max(0.0, xy, xz, xy + xz)
    ylo -= min(0.0, yz)
    yhi -= max(0.0, yz)
    return [[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]]


def read_lammps(run):
    """Thermo series and per-frame cells of the recorded run."""
    lines = (run / 'log.lammps').read_text(encoding='utf-8', errors='replace').splitlines()
    start = next(i for i, line in enumerate(lines) if line.split()[:2] == ['Step', 'Temp'])
    names = lines[start].split()
    rows = []
    for line in lines[start + 1:]:
        if line.startswith('Loop time'):
            break
        fields = line.split()
        if len(fields) == len(names):
            rows.append([float(x) for x in fields])
    column = {name: [row[i] for row in rows] for i, name in enumerate(names)}
    loop = re.search(r'Loop time of ([\d.eE+-]+) on .* for (\d+) steps', '\n'.join(lines))
    cells, dump = [], (run / 'traj.dump').read_text(encoding='utf-8', errors='replace').splitlines()
    for i, line in enumerate(dump):
        if line.startswith('ITEM: BOX BOUNDS'):
            cells.append(lammps_cell([[float(x) for x in dump[i + k].split()] for k in (1, 2, 3)]))
    summary = summarize({'volume_angstrom3': column['Volume'], 'potential_energy_ev': column['PotEng'],
                         'pressure_bar': column['Press'], 'temperature_kelvin': column['Temp']}, cells)
    summary['ms_per_step'] = float(loop.group(1)) / int(loop.group(2)) * 1e3 if loop else None
    return summary


def read_start(run, type_map):
    """Frame of `conf.lmp` (ASE `atomic` style, positive-volume triclinic box)."""
    lines = (run / 'conf.lmp').read_text(encoding='utf-8', errors='replace').splitlines()
    find = lambda tag: next(line.split() for line in lines if line.strip().endswith(tag))
    xlo, xhi = map(float, find('xlo xhi')[:2])
    ylo, yhi = map(float, find('ylo yhi')[:2])
    zlo, zhi = map(float, find('zlo zhi')[:2])
    tilt = next((line.split() for line in lines if line.strip().endswith('xy xz yz')), ['0', '0', '0'])
    xy, xz, yz = map(float, tilt[:3])
    natoms = int(find('atoms')[0])
    first = next(i for i, line in enumerate(lines) if line.startswith('Atoms')) + 2
    atoms = sorted((int(f[0]), int(f[1]), [float(x) for x in f[2:5]])
                   for f in (line.split() for line in lines[first:first + natoms]))
    return {'positions': [[p[0] - xlo, p[1] - ylo, p[2] - zlo] for _, _, p in atoms],
            'species': [type_map[t - 1] for _, t, _ in atoms],
            'cell': [[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]],
            'units': 'EvAngstrom'}


def read_deviation(run):
    """Per-frame (max_devi_f, avg_devi_f) of the recorded run."""
    rows = [line.split() for line in (run / 'model_devi.out').read_text(encoding='utf-8', errors='replace').splitlines()
            if line.strip() and not line.startswith('#')]
    return [float(r[4]) for r in rows], [float(r[6]) for r in rows]


def deviation_summary(largest, mean, low, high):
    return {'max_devi_f': statistics(largest), 'avg_devi_f': statistics(mean), 'max_devi_f_max': max(largest),
            'frames': len(largest), 'accurate': sum(x < low for x in largest),
            'candidate': sum(low <= x < high for x in largest), 'failed': sum(x >= high for x in largest)}


def protocol(run):
    """Temperature (K), thermostat time (fs), pressure (bar), barostat time (fs), timestep (fs), thermo interval."""
    text = (run / 'in.lmp').read_text(encoding='utf-8', errors='replace')
    fix = re.search(r'fix\s+\S+\s+all\s+npt\s+temp\s+(\S+)\s+(\S+)\s+(\S+)\s+tri\s+(\S+)\s+(\S+)\s+(\S+)', text)
    if not fix or fix.group(1) != fix.group(2) or fix.group(4) != fix.group(5):
        raise SystemExit('in.lmp needs one `fix npt temp T T Tdamp tri P P Pdamp` line with constant targets')
    timestep = float(re.search(r'^timestep\s+(\S+)', text, re.M).group(1)) * 1e3
    return {'temperature_kelvin': float(fix.group(1)), 'thermostat_fs': float(fix.group(3)) * 1e3,
            'pressure_bar': float(fix.group(4)), 'barostat_fs': float(fix.group(6)) * 1e3,
            'timestep_fs': timestep, 'sample_every': int(re.search(r'^thermo\s+(\d+)', text, re.M).group(1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lammps_run', type=Path)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('output_json', type=Path)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--mode', default='float64',
                        choices=['float64', 'float32', 'float64-compressed', 'float32-compressed'])
    parser.add_argument('--compressibility', type=float, default=1e-6, help='assumed, per bar')
    parser.add_argument('--every', type=int, default=10, help='steps between cell moves')
    parser.add_argument('--deviation', action='store_true', help='also compare the model deviation of the frames')
    parser.add_argument('--seed', type=int, default=1, help='seed of thermostat and barostat')
    args = parser.parse_args()
    manifest = json.loads((args.case_dir / 'manifest.json').read_text(encoding='utf-8', errors='replace'))
    exe = args.release / 'omnimd-mlmd'
    if not exe.exists():
        exe = exe.with_suffix('.exe')
    settings = protocol(args.lammps_run)
    frame = read_start(args.lammps_run, manifest['type_map'])
    runtime = {}
    if args.mode.startswith('float32'):
        runtime['precision'] = 'float32'
    if args.mode.endswith('compressed'):
        runtime['compress_min_distance'] = 0.5
    request = {
        'options': {'native_model': str((args.case_dir / f"model-{manifest['models'][0]}.native.json").resolve()),
                    'device': 'cuda:0', 'gpu_runtime': runtime},
        'frame': frame, 'masses': [MASS[s] for s in frame['species']],
        'initialize_velocities': {'temperature_kelvin': settings['temperature_kelvin'], 'seed': 20261329,
                                  'rescale_to_temperature': True},
        'remove_center_of_mass': True,
        'ensemble': {'kind': 'npt_csvr_crescale', 'temperature_kelvin': settings['temperature_kelvin'],
                     'tau_fs': settings['thermostat_fs'], 'seed': args.seed,
                     'pressure_bar': settings['pressure_bar'],
                     'barostat_tau_fs': settings['barostat_fs'], 'compressibility_per_bar': args.compressibility,
                     'coupling': 'triclinic', 'barostat_every': args.every},
        'timestep_fs': settings['timestep_fs'], 'steps': args.steps, 'sample_every': settings['sample_every']}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / 'request.json').write_text(json.dumps(request))
        start = time.perf_counter()
        proc = subprocess.run([str(exe), 'run', str(work / 'request.json'), str(work / 'out.jsonl')],
                              capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            raise SystemExit((proc.stderr.strip().splitlines() or ['engine failed'])[-1][:400])
        records = [json.loads(line) for line in (work / 'out.jsonl').read_text(encoding='utf-8', errors='replace').splitlines()]
        deviation = None
        if args.deviation:
            low, high = 0.1, 0.3
            (work / 'import.json').write_text(json.dumps({
                'schema': 'omnimd-trajectory-import-v1', 'trajectory': str(work / 'out.jsonl'),
                'id_prefix': args.lammps_run.name}))
            members = [{'id': f'model-{k}', 'options': {
                'native_model': str((args.case_dir / f'model-{k}.native.json').resolve()), 'device': 'cuda:0',
                'gpu_runtime': {}}} for k in manifest['models']]
            (work / 'deviation.json').write_text(json.dumps({
                'schema': 'omnimd-model-deviation-v1', 'dataset': str(work / 'explored.jsonl'), 'models': members,
                'force_window': {'min_ev_per_angstrom': low, 'max_ev_per_angstrom': high}}))
            start = time.perf_counter()
            for command, source, target in (('import-trajectory', 'import.json', 'explored.jsonl'),
                                            ('model-deviation', 'deviation.json', 'deviation.jsonl')):
                step = subprocess.run([str(exe), command, str(work / source), str(work / target)],
                                      capture_output=True, text=True)
                if step.returncode != 0:
                    raise SystemExit((step.stderr.strip().splitlines() or [command + ' failed'])[-1][:400])
            seconds = time.perf_counter() - start
            lines = [json.loads(line) for line in (work / 'deviation.jsonl').read_text(encoding='utf-8').splitlines()]
            forces = [r['statistics']['force_ev_per_angstrom'] for r in lines if r['kind'] == 'deviation']
            ours_deviation = deviation_summary([f['max'] for f in forces], [f['mean'] for f in forces], low, high)
            done = lines[-1]
            assert (done['accurate_frames'], done['candidate_frames'], done['failed_frames']) == (
                ours_deviation['accurate'], ours_deviation['candidate'], ours_deviation['failed'])
            ours_deviation.update(models=len(members), import_and_deviation_seconds=seconds)
            deviation = {'force_window_ev_per_angstrom': [low, high], 'omnimd': ours_deviation,
                         'lammps_recorded': deviation_summary(*read_deviation(args.lammps_run), low, high)}
    frames = [r for r in records if r['kind'] == 'frame']
    completed = records[-1]
    ours = summarize({'volume_angstrom3': [f['volume_angstrom3'] for f in frames],
                      'potential_energy_ev': [f['evaluation']['energy'] for f in frames],
                      'pressure_bar': [f['pressure_bar'] for f in frames],
                      'temperature_kelvin': [f['temperature_kelvin'] for f in frames]},
                     [f['frame']['cell'] for f in frames])
    ours.update(ms_per_step=elapsed * 1e3 / args.steps, whole_process_seconds=elapsed,
                barostat_moves=completed['barostat_moves'], barostat_refused=completed['barostat_refused'],
                max_effective_energy_drift_ev=completed['max_absolute_effective_energy_drift_ev'])
    theirs = read_lammps(args.lammps_run)
    report = {'system': args.lammps_run.name, 'natoms': len(frame['species']), 'protocol': settings,
              'steps': args.steps, 'mode': args.mode, 'ensemble': request['ensemble'],
              'omnimd': ours, 'lammps_recorded': theirs}
    if deviation:
        report['model_deviation'] = deviation
    with args.output_json.open('x') as handle:
        json.dump(report, handle, indent=1)
    print(f"{'quantity':22} {'omniMD':>24} {'LAMMPS (recorded)':>24} {'difference / error':>20}")
    for key in ('volume_angstrom3', 'potential_energy_ev', 'pressure_bar', 'temperature_kelvin', 'a', 'b', 'c',
                'alpha', 'beta', 'gamma'):
        a, b = ours[key], theirs[key]
        sigma = math.hypot(a['error'], b['error'])
        print(f"{key:22} {a['mean']:14.4f} +- {a['error']:7.4f} {b['mean']:14.4f} +- {b['error']:7.4f} "
              f"{(a['mean'] - b['mean']) / sigma:14.2f} sigma")
    print(f"volume std: {ours['volume_angstrom3']['std']:.3f} vs {theirs['volume_angstrom3']['std']:.3f}; "
          f"{ours['ms_per_step']:.3f} ms/step vs {theirs['ms_per_step']:.1f} recorded; refused "
          f"{ours['barostat_refused']}/{ours['barostat_moves']}")
    if deviation:
        for key in ('max_devi_f', 'avg_devi_f'):
            a, b = deviation['omnimd'][key], deviation['lammps_recorded'][key]
            print(f"{key:22} {a['mean']:14.5f} +- {a['error']:7.5f} {b['mean']:14.5f} +- {b['error']:7.5f} "
                  f"{(a['mean'] - b['mean']) / math.hypot(a['error'], b['error']):14.2f} sigma")
        for name in ('omnimd', 'lammps_recorded'):
            d = deviation[name]
            print(f"{name}: {d['frames']} frames, accurate {d['accurate']}, candidate {d['candidate']}, "
                  f"failed {d['failed']}, largest {d['max_devi_f_max']:.4f}")
        print(f"import and deviation: {deviation['omnimd']['import_and_deviation_seconds']:.1f} s")


if __name__ == '__main__':
    main()
