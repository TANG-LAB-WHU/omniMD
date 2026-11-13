"""Train DeePMD-kit on an omniMD training request (run in the DeePMD-kit image).

    python bench/deepmd_train.py REQUEST.json SYSTEMS_DIR OUT_DIR

`REQUEST.json` is a request of `omnimd-native-gpu-train` whose `model` is the
`model` section of a DeePMD input script, and `SYSTEMS_DIR` holds the
request's training frames as DeePMD npy systems (`dataset_to_deepmd.py`
writes them from its dataset). The `model`, `learning_rate` and `loss`
sections and the number of steps, the batch size, the seed and the display
interval go into a DeePMD input script unchanged, so both trainers run the
same settings. The request's `precision` becomes the precision of DeePMD's
networks; note that omniMD's `float32` keeps float64 parameters under
float32 networks, which DeePMD-kit has no counterpart of.

A new `OUT_DIR` receives `input.json`, what `dp --pt train` and `dp --pt
freeze` leave (`train.log`, `lcurve.out`, checkpoints, `graph.pth`),
`model.native.json` (the trained model converted for omniMD, to be scored
by `omnimd-native-gpu-test` like omniMD's own) and `summary.json`: the
trainer's own `average training time` per batch and the wall time of
`dp --pt train` from start to exit.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('systems', type=Path)
    parser.add_argument('out_dir', type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding='utf-8'))
    model = request['model']
    if not isinstance(model, dict):
        raise SystemExit('the request must carry a model section, not the path of a model to continue from')
    systems = sorted(p.resolve().as_posix() for p in args.systems.iterdir() if (p / 'type.raw').is_file())
    if not systems:
        raise SystemExit(f'no DeePMD systems under {args.systems}')
    precision = request.get('precision', 'float64')
    model = json.loads(json.dumps(model))
    model['descriptor']['precision'] = precision
    model['fitting_net']['precision'] = precision
    training = request['training']
    steps = training['numb_steps']
    config = {
        'model': model,
        'learning_rate': {'type': 'exp', **request['learning_rate']},
        'loss': {'type': 'ener', **request['loss']},
        'training': {
            'training_data': {'systems': systems, 'batch_size': training.get('batch_size', 1)},
            'numb_steps': steps, 'seed': training.get('seed', 0), 'disp_file': 'lcurve.out',
            'disp_freq': training.get('disp_freq', 100), 'save_freq': training.get('save_freq', steps),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    out = args.out_dir.resolve()
    (out / 'input.json').write_text(json.dumps(config, indent=1) + '\n', encoding='utf-8')

    started = time.time()
    with (out / 'train.log').open('w', encoding='utf-8') as log:
        code = subprocess.run(['dp', '--pt', 'train', 'input.json'], cwd=out, stdout=log,
                              stderr=subprocess.STDOUT).returncode
    wall = time.time() - started
    text = (out / 'train.log').read_text(encoding='utf-8', errors='replace')
    if code != 0:
        raise SystemExit('dp --pt train failed:\n' + text[-2000:])
    timing = re.search(r'average training time: ([0-9.eE+-]+) s/batch(?: \((\d+) batches excluded\))?', text)
    subprocess.run(['dp', '--pt', 'freeze', '-o', 'graph.pth'], cwd=out, check=True, capture_output=True)
    subprocess.run(['dp', 'convert-backend', 'graph.pth', 'model.dp'], cwd=out, check=True, capture_output=True)
    converter = Path(__file__).resolve().parent.parent / 'native' / 'tools' / 'convert_deepmd.py'
    subprocess.run([sys.executable, str(converter), str(out / 'model.dp'), str(out / 'model.native.json'),
                    '--units', 'EvAngstrom', '--geometry', 'dpmodel'], check=True)
    (out / 'model.dp').unlink()

    import deepmd
    curve = [line for line in (out / 'lcurve.out').read_text().splitlines() if line and not line.startswith('#')]
    summary = {
        'deepmd_version': deepmd.__version__, 'steps': steps, 'systems': len(systems), 'precision': precision,
        'average_training_time_s_per_batch': float(timing.group(1)) if timing else None,
        'batches_excluded_from_that_average': int(timing.group(2) or 0) if timing else None,
        'steps_per_second': 1.0 / float(timing.group(1)) if timing else None,
        'train_wall_seconds': wall, 'steps_per_wall_second': steps / wall,
        'last_curve_line': curve[-1].split() if curve else None,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
