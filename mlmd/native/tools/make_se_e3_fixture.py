"""Fixtures for the se_e3 descriptor against DeePMD-kit itself.

Run in the DeePMD-kit image from `mlmd`:

    python native/tools/make_se_e3_fixture.py native/fixtures

Builds two small se_e3 (se_at, se_a_3be) energy models of three types, each
type with a normalization of its own: `se-e3-model.native.json`, whose `sel`
holds every neighbor of the frames, and `se-e3-tight-model.native.json`,
whose `sel` ([3, 4, 2]) is far below the neighbor counts, so that the result
depends on which neighbors DeePMD keeps. Both are converted with
`convert_deepmd.py`. Writes `se-e3-deepmd-reference.json` with DeePMD's
energy, forces and virial (PyTorch backend, CPU, float64) for a periodic
triclinic frame and for the same atoms without a cell under the first model,
and for the periodic frame under the second.

The radial mean and width of the models are set to numbers the descriptor
must not use: it reads the direction entries of the environment only.
"""
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np  # noqa: E402
import deepmd  # noqa: E402
from deepmd.dpmodel.descriptor.se_t import DescrptSeT  # noqa: E402
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet  # noqa: E402
from deepmd.dpmodel.model.ener_model import EnergyModel  # noqa: E402
from deepmd.dpmodel.utils.serialization import save_dp_model  # noqa: E402
from deepmd.infer import DeepPot  # noqa: E402
from deepmd.pt.utils.serialization import deserialize_to_file  # noqa: E402

TYPES = ['O', 'H', 'C']
SEED = 20261115
RCUT = 4.0


def build(sel, seed):
    descriptor_config = dict(rcut=RCUT, rcut_smth=0.5, sel=sel, neuron=[6, 12], resnet_dt=False,
                             precision='float64', type_map=TYPES, seed=seed)
    fitting_config = dict(ntypes=3, dim_descrpt=12, neuron=[8, 8], resnet_dt=True,
                          precision='float64', type_map=TYPES, seed=seed + 1)
    descriptor = DescrptSeT(**descriptor_config)
    fitting = EnergyFittingNet(**fitting_config)
    # Per center type, the same for every slot. The descriptor reads the three
    # direction entries (zero mean, one width); the radial entry it must ignore.
    for center, (radial_mean, radial_width, width) in enumerate(
            [(0.10, 0.30, 0.22), (0.16, 0.25, 0.31), (0.07, 0.41, 0.27)]):
        descriptor.davg[center, :, 0] = radial_mean
        descriptor.dstd[center, :, 0] = radial_width
        descriptor.dstd[center, :, 1:] = width
    fitting.bias_atom_e[:] = np.array([[-3.0], [-1.5], [-2.25]])
    model = EnergyModel(descriptor=descriptor, fitting=fitting, type_map=TYPES)
    definition = {'type': 'standard', 'type_map': TYPES,
                  'descriptor': {'type': 'se_e3',
                                 **{k: v for k, v in descriptor_config.items() if k != 'type_map'}},
                  'fitting_net': {'type': 'ener',
                                  **{k: v for k, v in fitting_config.items()
                                     if k not in ('ntypes', 'dim_descrpt', 'type_map')}}}
    return {'model': model.serialize(), 'model_def_script': definition}


def neighbor_counts(coords, lattice, atype):
    """The largest number of neighbors of each type within the cutoff, over the atoms (periodic)."""
    shifts = np.array([[i, j, k] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)]) @ lattice
    worst = [0] * len(TYPES)
    for center in coords:
        delta = coords[None, :, :] + shifts[:, None, :] - center
        dist = np.linalg.norm(delta, axis=-1)
        near = (dist < RCUT) & (dist > 1e-9)
        for t in range(len(TYPES)):
            worst[t] = max(worst[t], int(near[:, atype == t].sum()))
    return worst


def main():
    out = Path(sys.argv[1])
    rng = np.random.default_rng(SEED)
    lattice = np.array([[7.0, 0.0, 0.0], [0.9, 6.6, 0.0], [-0.5, 0.4, 6.8]])
    grid = (3, 3, 3)
    points = np.stack(np.meshgrid(*[np.arange(n) / n for n in grid], indexing='ij'), -1).reshape(-1, 3)
    points += rng.uniform(-0.1, 0.1, points.shape) / np.array(grid)
    coords = points @ lattice
    # Nine O, twelve H, six C, mixed.
    atype = rng.permutation(np.array([0] * 9 + [1] * 12 + [2] * 6))
    worst = neighbor_counts(coords, lattice, atype)
    roomy = [count + 3 for count in worst]
    species = [TYPES[t] for t in atype]
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, sel, seed in (('se-e3', roomy, SEED), ('se-e3-tight', [3, 4, 2], SEED + 10)):
            data = build(sel, seed)
            stem = Path(tmp) / name
            save_dp_model(str(stem.with_suffix('.dp')), deepcopy(data))
            deserialize_to_file(str(stem.with_suffix('.pth')), deepcopy(data))
            target = out / f'{name}-model.native.json'
            target.unlink(missing_ok=True)
            subprocess.run([sys.executable, str(Path(__file__).with_name('convert_deepmd.py')),
                            str(stem.with_suffix('.dp')), str(target),
                            '--units', 'EvAngstrom', '--geometry', 'dpmodel'], check=True)
            pot = DeepPot(str(stem.with_suffix('.pth')))
            cells = [lattice] if name == 'se-e3-tight' else [lattice, None]
            for cell in cells:
                energy, forces, virial = pot.eval(coords.reshape(1, -1),
                                                  None if cell is None else cell.reshape(1, 9),
                                                  atype.tolist())
                cases.append({'model': target.name, 'sel': sel,
                              'frame': {'positions': coords.tolist(), 'species': species,
                                        'cell': None if cell is None else cell.tolist(),
                                        'units': 'EvAngstrom'},
                              'energy': float(energy.ravel()[0]),
                              'forces': forces.reshape(-1, 3).tolist(),
                              'virial': virial.reshape(3, 3).tolist()})
                print(target.name, 'periodic' if cell is not None else 'open', 'energy', cases[-1]['energy'],
                      'largest force', float(np.abs(forces).max()))
    reference = {'deepmd_version': deepmd.__version__, 'rcut': RCUT,
                 'largest_neighbor_counts_periodic': worst, 'cases': cases}
    (out / 'se-e3-deepmd-reference.json').write_text(json.dumps(reference, indent=1) + '\n')
    print('atoms', len(coords), 'largest neighbor counts', worst, 'sel', roomy)


if __name__ == '__main__':
    main()
