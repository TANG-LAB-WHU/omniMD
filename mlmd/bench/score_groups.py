"""Score groups of native models on a labelled dataset, side by side (host).

    python bench/score_groups.py TEST_DATASET.jsonl <target>/release OUT.json NAME=MODEL.native.json[,MODEL...] [NAME=...]

Runs `omnimd-native-gpu-test` on every model. For every group it prints, per
system and overall, the smallest and largest force RMSE (eV/A) and energy
RMSE (meV/atom) over the models of the group: for example the seeds of one
trainer next to those of another, both scored by the same command. `OUT.json`
(a new file) keeps every report under its group and model file.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def spread(values, scale=1.0):
    low, high = min(values) * scale, max(values) * scale
    return f'{low:.4f}' if len(values) == 1 else f'{low:.4f} - {high:.4f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('output_json', type=Path)
    parser.add_argument('groups', nargs='+', help='NAME=MODEL[,MODEL...]')
    parser.add_argument('--precision', default='float64', choices=['float64', 'float32'])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    exe = args.release / 'omnimd-native-gpu-test'
    if not exe.exists():
        exe = exe.with_suffix('.exe')
    groups = {}
    for item in args.groups:
        name, _, models = item.partition('=')
        if not name or not models or name in groups:
            raise SystemExit(f'{item!r}: expected NAME=MODEL[,MODEL...] with a name of its own')
        groups[name] = {}
        for model in models.split(','):
            proc = subprocess.run([str(exe), model, str(args.dataset), '--precision', args.precision],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                raise SystemExit(f'{model}: {proc.stderr.strip()[-300:]}')
            groups[name][Path(model).as_posix()] = json.loads(proc.stdout)
    first = next(iter(next(iter(groups.values())).values()))
    systems = sorted(first['groups'])
    width = max(17, *(len(name) + 7 for name in groups))
    print(f"{'system':36} " + ' '.join(f"{'force ' + name:>{width}}" for name in groups) + ' '
          + ' '.join(f"{'energy ' + name:>{width}}" for name in groups))
    for system in systems + ['all']:
        reports = {name: [r['groups'][system] if system != 'all' else r['all'] for r in models.values()]
                   for name, models in groups.items()}
        print(f'{system[:36]:36} '
              + ' '.join(f"{spread([r['force_rmse'] for r in reports[name]]):>{width}}" for name in groups) + ' '
              + ' '.join(f"{spread([r['energy_rmse_per_atom'] for r in reports[name]], 1e3):>{width}}"
                         for name in groups))
    with args.output_json.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump({'dataset': args.dataset.as_posix(), 'precision': args.precision, 'groups': groups}, handle,
                  indent=1)
        handle.write('\n')


if __name__ == '__main__':
    main()
