"""Score models trained by `omnimd-native-gpu-train` on a prepared test set and
compare them with a pipeline's recorded DeePMD test errors (host).

    python bench/hap_train_score.py TEST_DATASET.jsonl RECORDED_sealed_test.json <target>/release OUT.json MODEL.native.json [MODEL ...]

Runs `omnimd-native-gpu-test` on every model. Per system and overall it
prints the smallest and largest force RMSE (eV/A) and energy RMSE (meV/atom)
over the given models next to the smallest and largest over the recorded
DeePMD models. The recorded file has per model and system the keys
`energy_rmse_eV_atom`, `force_rmse_eV_A` and `frames`; overall values are
aggregated over force components and frames with the atom counts of the test
set. `OUT.json` keeps every number.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def spread(values, scale=1.0):
    return f'{min(values) * scale:.4f} - {max(values) * scale:.4f}' if len(values) > 1 else f'{values[0] * scale:.4f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('recorded', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('output_json', type=Path)
    parser.add_argument('models', nargs='+', type=Path)
    parser.add_argument('--precision', default='float64', choices=['float64', 'float32'])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    exe = args.release / 'omnimd-native-gpu-test'
    if not exe.exists():
        exe = exe.with_suffix('.exe')
    recorded = json.loads(args.recorded.read_text(encoding='utf-8'))
    reports = {}
    for model in args.models:
        proc = subprocess.run([str(exe), str(model), str(args.dataset), '--precision', args.precision],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f'{model}: {proc.stderr.strip()[-300:]}')
        reports[model.name] = json.loads(proc.stdout)
    first = next(iter(reports.values()))
    systems = sorted(first['groups'])
    # Recorded overall errors, weighted like the test command weights its own.
    recorded_all = {}
    for name, per_system in recorded.items():
        force = sum(per_system[s]['force_rmse_eV_A'] ** 2 * first['groups'][s]['atoms'] for s in systems)
        energy = sum(per_system[s]['energy_rmse_eV_atom'] ** 2 * first['groups'][s]['frames'] for s in systems)
        recorded_all[name] = {'force_rmse': math.sqrt(force / first['all']['atoms']),
                              'energy_rmse_per_atom': math.sqrt(energy / first['all']['frames'])}
    print(f"{'system':36} {'force ours':>17} {'force recorded':>17} {'energy ours':>17} {'energy recorded':>17}")
    for system in systems + ['all']:
        ours = [r['groups'][system] if system != 'all' else r['all'] for r in reports.values()]
        theirs = ([{'force_rmse': m[system]['force_rmse_eV_A'], 'energy_rmse_per_atom': m[system]['energy_rmse_eV_atom']}
                   for m in recorded.values()] if system != 'all' else list(recorded_all.values()))
        print(f"{system[:36]:36} {spread([o['force_rmse'] for o in ours]):>17} "
              f"{spread([t['force_rmse'] for t in theirs]):>17} "
              f"{spread([o['energy_rmse_per_atom'] for o in ours], 1e3):>17} "
              f"{spread([t['energy_rmse_per_atom'] for t in theirs], 1e3):>17}")
    with args.output_json.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump({'dataset': str(args.dataset), 'precision': args.precision, 'models': reports,
                   'recorded': recorded, 'recorded_overall': recorded_all}, handle, indent=1)
        handle.write('\n')


if __name__ == '__main__':
    main()
