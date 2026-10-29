"""Create the benchmark model and water boxes (run inside DeepMD-kit 3.1.3).

The model uses the official water se_e2_a example architecture with untrained,
seeded weights: throughput does not depend on weight values, and every engine
evaluates the same parameters. Boxes are 64-molecule cells at 0.997 g/cm^3,
randomly oriented and replicated to 192, 1536, 12288 and 98304 atoms.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

TYPES = ['O', 'H']
REPLICAS = [1, 2, 4, 8]


def build_model(output, precision, seed):
    from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeA
    from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet
    from deepmd.dpmodel.model.ener_model import EnergyModel
    from deepmd.dpmodel.utils.serialization import save_dp_model
    from deepmd.pt.utils.serialization import deserialize_to_file

    descriptor_config = dict(rcut=6.0, rcut_smth=0.5, sel=[46, 92], neuron=[25, 50, 100],
                             axis_neuron=16, resnet_dt=False, type_one_side=False,
                             precision=precision, type_map=TYPES, seed=seed)
    fitting_config = dict(ntypes=2, dim_descrpt=100 * 16, neuron=[240, 240, 240], resnet_dt=True,
                          precision=precision, type_map=TYPES, seed=seed + 1)
    descriptor = DescrptSeA(**descriptor_config)
    fitting = EnergyFittingNet(**fitting_config)
    # Plausible normalization for liquid water; angular mean zero and isotropic std
    # as required by the native converter.
    descriptor.davg[...] = 0.0
    descriptor.davg[:, :, 0] = 0.25
    descriptor.dstd[...] = 0.2
    descriptor.dstd[:, :, 0] = 0.3
    fitting.bias_atom_e[:] = np.array([[-93.5], [-187.0]]) / 3.0
    model = EnergyModel(descriptor=descriptor, fitting=fitting, type_map=TYPES)
    definition = {'type': 'standard', 'type_map': TYPES,
                  'descriptor': {'type': 'se_e2_a',
                                 **{k: v for k, v in descriptor_config.items() if k != 'type_map'}},
                  'fitting_net': {'type': 'ener',
                                  **{k: v for k, v in fitting_config.items()
                                     if k not in ('ntypes', 'dim_descrpt', 'type_map')}}}
    data = {'model': model.serialize(), 'model_def_script': definition}
    stem = output / f'water-se_e2_a-{precision}'
    # Both writers mutate their input, so each gets its own copy.
    save_dp_model(str(stem.with_suffix('.dp')), deepcopy(data))
    deserialize_to_file(str(stem.with_suffix('.pth')), deepcopy(data))
    return stem


def water_box(rng):
    molecules, spacing = 4, (30.0 * 64) ** (1 / 3) / 4
    bond, half_angle = 0.9572, np.deg2rad(104.52) / 2
    local = np.array([[0.0, 0.0, 0.0],
                      [bond * np.sin(half_angle), bond * np.cos(half_angle), 0.0],
                      [-bond * np.sin(half_angle), bond * np.cos(half_angle), 0.0]])
    coords, types = [], []
    for index in np.ndindex(molecules, molecules, molecules):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        coords.append(local @ rotation.T + (np.array(index) + 0.5) * spacing)
        types += [0, 1, 1]
    return np.concatenate(coords), np.array(types), spacing * molecules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--seed', type=int, default=20261029)
    args = parser.parse_args()
    import deepmd
    if deepmd.__version__ != '3.1.3':
        raise RuntimeError('benchmark recipe requires DeepMD-kit 3.1.3')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    models = {p: build_model(args.output_dir, p, args.seed).name for p in ('float64', 'float32')}

    base_coords, base_types, length = water_box(np.random.default_rng(args.seed))
    cases = []
    for n in REPLICAS:
        shifts = np.array(list(np.ndindex(n, n, n)), dtype=float) * length
        coords = (base_coords[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        types = np.tile(base_types, n ** 3)
        cell = np.eye(3) * length * n
        name = f'water-{len(types)}'
        np.savez(args.output_dir / f'{name}.npz', coords=coords, cell=cell, atype=types)
        frame = {'positions': coords.tolist(), 'species': [TYPES[t] for t in types],
                 'cell': cell.tolist(), 'units': 'EvAngstrom'}
        (args.output_dir / f'{name}.jsonl').write_text(json.dumps(frame) + '\n')
        cases.append({'name': name, 'natoms': int(len(types)), 'box_angstrom': float(length * n)})
    manifest = {'deepmd_version': deepmd.__version__, 'seed': args.seed, 'models': models,
                'cases': cases}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
