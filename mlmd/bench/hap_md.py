"""Exploration-style runs of a prepared ensemble through `omnimd-mlmd` (host).

    python bench/hap_md.py CASE_DIR <target>/release results/hap-omnimd-<tag>.csv SYSTEM [SYSTEM ...]

`CASE_DIR` comes from `hap_prepare.py`; the engine must be built with
`--features native-gpu`. Three things are measured or checked:

1. Model deviation: `model-deviation` with every model on the GPU runtime over
   all prepared frames, compared with the deviation computed directly from
   DeePMD's reference forces (`deepmd-ref-model-K.jsonl`): per frame, the
   largest, smallest and mean over atoms of the standard deviation of the
   force vector over models, as in DeePMD's `model_devi.out`.
2. NVT dynamics (CSVR thermostat, 300 K, 0.5 fs, sampled every 50 steps) from
   the first frame of each SYSTEM with model 0, in four modes. The time per
   step is the difference between two whole-process runs of different length.
3. The cost of the 4-model deviation per sampled frame.
"""
import argparse
import csv
import json
import subprocess
import tempfile
import time
from pathlib import Path

FIELDS = ['engine', 'route', 'device', 'precision', 'natoms', 'ms_per_eval',
          'us_per_atom_step', 'ns_per_day', 'peak_mem_mb', 'status', 'note']
MASS = {'Ca': 40.078, 'P': 30.9738, 'O': 15.999, 'H': 1.008, 'C': 12.011}
TIMESTEP_FS = 0.5
MODES = [('float64', {}), ('float32', {'precision': 'float32'}),
         ('float64-compressed', {'compress_min_distance': 0.5}),
         ('float32-compressed', {'precision': 'float32', 'compress_min_distance': 0.5})]


def engine(exe, command, request, work, name):
    path = work / f'{name}.json'
    output = work / f'{name}.jsonl'
    path.write_text(json.dumps(request))
    start = time.perf_counter()
    proc = subprocess.run([str(exe), command, str(path), str(output)], capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr.strip().splitlines() or ['engine failed'])[-1][:300])
    return elapsed, [json.loads(line) for line in output.read_text().splitlines()]


def options(case_dir, model, runtime):
    return {'native_model': str((case_dir / f'model-{model}.native.json').resolve()), 'device': 'cuda:0',
            'gpu_runtime': runtime}


def reference_deviation(case_dir, models):
    """Per frame (max, min, mean) over atoms of the force std over models."""
    refs = [[json.loads(line) for line in (case_dir / f'deepmd-ref-model-{k}.jsonl').read_text().splitlines()]
            for k in models]
    out = []
    for frame in zip(*refs):
        per_atom = []
        for atom in zip(*[f['forces'] for f in frame]):
            mean = [sum(f[a] for f in atom) / len(atom) for a in range(3)]
            per_atom.append((sum(sum((f[a] - mean[a]) ** 2 for a in range(3)) for f in atom) / len(atom)) ** 0.5)
        out.append((max(per_atom), min(per_atom), sum(per_atom) / len(per_atom)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('output_csv', type=Path)
    parser.add_argument('systems', nargs='+')
    parser.add_argument('--steps', type=int, default=2000)
    args = parser.parse_args()
    manifest = json.loads((args.case_dir / 'manifest.json').read_text())
    exe = args.release / 'omnimd-mlmd'
    if not exe.exists():
        exe = exe.with_suffix('.exe')
    frames = [json.loads(line) for line in (args.case_dir / 'frames.jsonl').read_text().splitlines()]
    models = manifest['models']
    with args.output_csv.open('x', newline='') as handle, tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        def record(row):
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)

        # 1. Deviation of the whole ensemble on every prepared frame.
        request = {'schema': 'omnimd-model-deviation-v1', 'dataset': str((args.case_dir / 'dataset.jsonl').resolve()),
                   'models': [{'id': f'model-{k}', 'options': options(args.case_dir, k, {})} for k in models]}
        elapsed, records = engine(exe, 'model-deviation', request, work, 'deviation')
        ours = [r['statistics']['force_ev_per_angstrom'] for r in records if r.get('kind') == 'deviation']
        expected = reference_deviation(args.case_dir, models)
        worst = max(max(abs(o['max'] - e[0]), abs(o['min'] - e[1]), abs(o['mean'] - e[2]))
                    for o, e in zip(ours, expected))
        assert len(ours) == len(expected) == len(frames)
        atoms = sum(f['natoms'] for f in manifest['frames'])
        record({'engine': 'omnimd', 'route': f'engine-model-deviation-{len(models)}-models', 'device': 'gpu',
                'precision': 'float64', 'natoms': atoms // len(frames), 'status': 'ok',
                'ms_per_eval': elapsed * 1e3 / len(frames),
                'note': f'{len(frames)} frames, whole process {elapsed:.2f} s including loading {len(models)} '
                        f'models; max_devi_f/min_devi_f/avg_devi_f differ from DeePMD reference forces by at '
                        f'most {worst:.1e} eV/A'})

        # 2. NVT from the first frame of each system.
        for system in args.systems:
            index = next(i for i, f in enumerate(manifest['frames']) if f['system'] == system)
            frame, natoms = frames[index], manifest['frames'][index]['natoms']
            base = {'frame': frame, 'masses': [MASS[s] for s in frame['species']],
                    'initialize_velocities': {'temperature_kelvin': 300.0, 'seed': 20261329,
                                              'rescale_to_temperature': True},
                    'remove_center_of_mass': True,
                    'ensemble': {'kind': 'nvt_csvr', 'temperature_kelvin': 300.0, 'tau_fs': 50.0, 'seed': 1},
                    'timestep_fs': TIMESTEP_FS, 'sample_every': 50}
            short = args.steps // 10
            for mode, runtime in MODES:
                row = {'engine': 'omnimd', 'route': f'engine-nvt-{mode}', 'device': 'gpu',
                       'precision': mode.split('-')[0], 'natoms': natoms}
                try:
                    tag = f'{system}-{mode}'
                    request = dict(base, options=options(args.case_dir, models[0], runtime))
                    brief, _ = engine(exe, 'run', dict(request, steps=short), work, tag + '-short')
                    full, trajectory = engine(exe, 'run', dict(request, steps=args.steps), work, tag + '-full')
                    seconds = (full - brief) / (args.steps - short)
                    temperatures = [r['temperature_kelvin'] for r in trajectory if 'step' in r]
                    row.update(status='ok', ms_per_eval=seconds * 1e3, us_per_atom_step=seconds * 1e6 / natoms,
                               ns_per_day=TIMESTEP_FS * 1e-6 * 86400 / seconds,
                               note=f'{system}; NVT CSVR 300 K 0.5 fs, sampled every 50 steps; ({args.steps} - '
                                    f'{short}) steps from two whole-process runs ({full:.2f} s, {brief:.2f} s); '
                                    f'mean sampled T {sum(temperatures) / len(temperatures):.0f} K')
                except RuntimeError as error:
                    row.update(status='error', note=f'{system}: {error}')
                record(row)


if __name__ == '__main__':
    main()
