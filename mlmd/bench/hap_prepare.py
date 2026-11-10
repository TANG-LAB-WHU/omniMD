"""Prepare a DeePMD se_e2_a ensemble and its test frames for omniMD (run in the
DeePMD-kit image).

    python bench/hap_prepare.py TRAIN_DIR DATA_DIR OUT_DIR [--models 0 1 2 3]

`TRAIN_DIR/model-K/graph.pth` are frozen PyTorch-backend models and `DATA_DIR`
holds DeePMD npy systems (`type_map.raw`, `type.raw`, `set.*/coord.npy`, ...).
Nothing is written outside `OUT_DIR`:

- `model-K.native.json`: the model converted for omniMD (`dp convert-backend`
  to the dpmodel format, then `native/tools/convert_deepmd.py`);
- `frames.jsonl`: every frame of every system, one omniMD `Frame` per line;
- `dataset.jsonl`: the same frames with their energy, force and (where the
  system has them) virial labels in the omnimd-dataset-jsonl-v1 format;
- `deepmd-ref-model-K.jsonl`: DeePMD's own energy, forces and virial for each
  frame (DeepPot on CPU);
- `manifest.json`: systems, frame ids and atom counts.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np  # noqa: E402
import deepmd  # noqa: E402
from deepmd.infer import DeepPot  # noqa: E402


def load_systems(data_dir):
    frames = []
    for system in sorted(p for p in data_dir.iterdir() if (p / 'type.raw').is_file()):
        type_map = (system / 'type_map.raw').read_text().split()
        types = np.loadtxt(system / 'type.raw', dtype=int, ndmin=1)
        for data_set in sorted(system.glob('set.*')):
            coords = np.load(data_set / 'coord.npy').reshape(-1, len(types), 3)
            boxes = np.load(data_set / 'box.npy').reshape(-1, 3, 3)
            energies = np.load(data_set / 'energy.npy').reshape(-1)
            forces = np.load(data_set / 'force.npy').reshape(-1, len(types), 3)
            virial_file = data_set / 'virial.npy'
            virials = np.load(virial_file).reshape(-1, 3, 3) if virial_file.is_file() else None
            for k in range(len(coords)):
                frames.append({'id': f'{system.name}/{data_set.name}/{k}', 'system': system.name,
                               'type_map': type_map, 'types': types, 'coords': coords[k], 'cell': boxes[k],
                               'energy': float(energies[k]), 'forces': forces[k],
                               'virial': None if virials is None else virials[k].tolist()})
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_dir', type=Path)
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--models', type=int, nargs='*', default=[0, 1, 2, 3],
                        help='models to convert and evaluate; none writes only the frames and dataset')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    converter = Path(__file__).resolve().parent.parent / 'native' / 'tools' / 'convert_deepmd.py'

    frames = load_systems(args.data_dir)
    type_map = frames[0]['type_map']
    assert all(f['type_map'] == type_map for f in frames), 'systems must share one type map'
    with (args.out_dir / 'frames.jsonl').open('w', newline='\n') as plain, \
            (args.out_dir / 'dataset.jsonl').open('w', newline='\n') as dataset:
        header = {'schema': 'omnimd-dataset-jsonl-v1', 'type_map': type_map, 'units': 'EvAngstrom',
                  'virial_convention': 'MinusDEpsilonRow', 'source': str(args.data_dir), 'selected_sets': []}
        dataset.write(json.dumps({'kind': 'header', 'header': header}) + '\n')
        for index, f in enumerate(frames):
            frame = {'positions': f['coords'].tolist(), 'species': [type_map[t] for t in f['types']],
                     'cell': f['cell'].tolist(), 'units': 'EvAngstrom'}
            plain.write(json.dumps(frame) + '\n')
            labels = {'energy': f['energy'], 'forces': f['forces'].tolist(), 'virial': f['virial']}
            dataset.write(json.dumps({'kind': 'frame', 'data': {'index': index, 'id': f['id'], 'frame': frame,
                                                                'labels': labels}}) + '\n')
        dataset.write(json.dumps({'kind': 'completed', 'frames': len(frames), 'source_sha256': {}}) + '\n')

    for k in args.models:
        frozen = args.train_dir / f'model-{k}' / 'graph.pth'
        portable = args.out_dir / f'model-{k}.dp'
        subprocess.run(['dp', 'convert-backend', str(frozen), str(portable)], check=True)
        subprocess.run([sys.executable, str(converter), str(portable), str(args.out_dir / f'model-{k}.native.json'),
                        '--units', 'EvAngstrom', '--geometry', 'dpmodel'], check=True)
        portable.unlink()
        potential = DeepPot(str(frozen))
        assert potential.get_type_map() == type_map, (potential.get_type_map(), type_map)
        with (args.out_dir / f'deepmd-ref-model-{k}.jsonl').open('w', newline='\n') as handle:
            for f in frames:
                energy, forces, virial = potential.eval(f['coords'].reshape(1, -1), f['cell'].reshape(1, 9),
                                                        f['types'].tolist())
                handle.write(json.dumps({'id': f['id'], 'energy': float(energy.ravel()[0]),
                                         'forces': forces.reshape(-1, 3).tolist(),
                                         'virial': virial.reshape(3, 3).tolist()}) + '\n')
        print('model', k, 'converted and evaluated on', len(frames), 'frames', flush=True)

    manifest = {'deepmd_version': deepmd.__version__, 'type_map': type_map, 'models': args.models,
                'train_dir': str(args.train_dir), 'data_dir': str(args.data_dir),
                'frames': [{'id': f['id'], 'system': f['system'], 'natoms': int(len(f['types']))} for f in frames]}
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')


if __name__ == '__main__':
    main()
