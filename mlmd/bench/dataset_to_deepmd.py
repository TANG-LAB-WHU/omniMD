"""Write an omnimd-dataset-jsonl-v1 file as DeePMD npy systems (needs numpy).

    python bench/dataset_to_deepmd.py DATASET.jsonl OUT_DIR

The reverse of `hap_prepare.py`, for training DeePMD-kit on the very frames
omniMD trains on. One system per group of frames, where a frame's group is
its id up to the first `/` (the system name for datasets made from DeePMD
systems): `OUT_DIR/<group>/` with `type_map.raw`, `type.raw` and
`set.000/{coord,box,energy,force}.npy`, and `virial.npy` when every frame of
the group has that label. Frames keep their order in the file, and the
numbers are written as they are read (eV and A, the virial as DeePMD stores
it).

The frames of a group must have the same atoms in the same order, a cell,
an energy and forces; anything else stops the conversion, and an existing
`OUT_DIR` is never written into.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('out_dir', type=Path)
    args = parser.parse_args()

    type_map, groups, completed = None, {}, None
    with args.dataset.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            record = json.loads(line)
            kind = record.get('kind')
            if kind == 'header':
                header = record['header']
                if header.get('schema') != 'omnimd-dataset-jsonl-v1' or header.get('units') != 'EvAngstrom':
                    raise SystemExit('expected an omnimd-dataset-jsonl-v1 file in eV and A')
                type_map = header['type_map']
            elif kind == 'frame':
                if type_map is None:
                    raise SystemExit(f'line {number}: a frame before the header')
                data = record['data']
                groups.setdefault(data['id'].split('/')[0], []).append(data)
            elif kind == 'completed':
                completed = record['frames']
            else:
                raise SystemExit(f'line {number}: unknown record kind {kind!r}')
    total = sum(len(frames) for frames in groups.values())
    if completed != total:
        raise SystemExit(f'the file ends without its closing record, or with another count ({completed}) than its {total} frames')

    systems = {}
    for name, frames in groups.items():
        species = frames[0]['frame']['species']
        for data in frames:
            frame, labels = data['frame'], data['labels']
            if frame['species'] != species:
                raise SystemExit(f"{data['id']}: other atoms, or another order, than the first frame of {name}")
            if frame.get('cell') is None:
                raise SystemExit(f"{data['id']}: no cell")
            if labels.get('energy') is None or labels.get('forces') is None:
                raise SystemExit(f"{data['id']}: needs an energy and forces")
        count = len(frames)
        arrays = {
            'coord': np.array([d['frame']['positions'] for d in frames], dtype=np.float64).reshape(count, -1),
            'box': np.array([d['frame']['cell'] for d in frames], dtype=np.float64).reshape(count, 9),
            'energy': np.array([d['labels']['energy'] for d in frames], dtype=np.float64),
            'force': np.array([d['labels']['forces'] for d in frames], dtype=np.float64).reshape(count, -1),
        }
        if arrays['coord'].shape[1] != 3 * len(species) or arrays['force'].shape != arrays['coord'].shape:
            raise SystemExit(f'{name}: positions and forces need three numbers for each of the {len(species)} atoms')
        if all(d['labels'].get('virial') is not None for d in frames):
            arrays['virial'] = np.array([d['labels']['virial'] for d in frames], dtype=np.float64).reshape(count, 9)
        systems[name] = ([type_map.index(s) for s in species], arrays)

    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, (types, arrays) in systems.items():
        system = args.out_dir / name
        (system / 'set.000').mkdir(parents=True)
        (system / 'type_map.raw').write_text(''.join(f'{t}\n' for t in type_map), encoding='utf-8', newline='\n')
        (system / 'type.raw').write_text(''.join(f'{t}\n' for t in types), encoding='utf-8', newline='\n')
        for key, values in arrays.items():
            np.save(system / 'set.000' / f'{key}.npy', values)
        print(f"{name}: {len(arrays['energy'])} frames of {len(types)} atoms"
              + ('' if 'virial' in arrays else ', no virial'))
    print(f'{total} frames in {len(systems)} systems under {args.out_dir}')


if __name__ == '__main__':
    main()
