"""Fixtures for the se_e2_r descriptor against DeePMD-kit itself.

Run in the DeePMD-kit image from `mlmd`:

    python native/tools/make_se_r_fixture.py native/fixtures

Builds two small se_e2_r (se_r) energy models with different normalization
for the two types: `se-r-model.native.json`, whose `sel` holds every neighbor
of the frames, and `se-r-tight-model.native.json`, whose `sel` ([4, 6]) is far
below the neighbor counts, so that the result depends on which neighbors
DeePMD keeps. Both are converted with `convert_deepmd.py`. Writes
`se-r-deepmd-reference.json` with DeePMD's energy, forces and virial (PyTorch
backend, CPU, float64) for a periodic triclinic frame and for the same atoms
without a cell under the first model, and for the periodic frame under the
second.
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
from deepmd.dpmodel.descriptor.se_r import DescrptSeR  # noqa: E402
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet  # noqa: E402
from deepmd.dpmodel.model.ener_model import EnergyModel  # noqa: E402
from deepmd.dpmodel.utils.serialization import save_dp_model  # noqa: E402
from deepmd.infer import DeepPot  # noqa: E402
from deepmd.pt.utils.serialization import deserialize_to_file  # noqa: E402

TYPES = ['O', 'H']
SEED = 20261112
RCUT = 4.0


def build(sel, seed):
    descriptor_config = dict(rcut=RCUT, rcut_smth=0.5, sel=sel, neuron=[6, 12], resnet_dt=False,
                             type_one_side=True, precision='float64', type_map=TYPES, seed=seed)
    fitting_config = dict(ntypes=2, dim_descrpt=12, neuron=[8, 8], resnet_dt=True,
                          precision='float64', type_map=TYPES, seed=seed + 1)
    descriptor = DescrptSeR(**descriptor_config)
    fitting = EnergyFittingNet(**fitting_config)
    # Per center type, the same for every slot: a mean and a width that differ between the types.
    descriptor.davg[0, :, 0], descriptor.davg[1, :, 0] = 0.10, 0.16
    descriptor.dstd[0, :, 0], descriptor.dstd[1, :, 0] = 0.30, 0.25
    fitting.bias_atom_e[:] = np.array([[-3.0], [-1.5]])
    model = EnergyModel(descriptor=descriptor, fitting=fitting, type_map=TYPES)
    definition = {'type': 'standard', 'type_map': TYPES,
                  'descriptor': {'type': 'se_e2_r',
                                 **{k: v for k, v in descriptor_config.items() if k != 'type_map'}},
                  'fitting_net': {'type': 'ener',
                                  **{k: v for k, v in fitting_config.items()
                                     if k not in ('ntypes', 'dim_descrpt', 'type_map')}}}
    return {'model': model.serialize(), 'model_def_script': definition}


def neighbor_counts(coords, lattice, atype):
    """The largest number of neighbors of each type within the cutoff, over the atoms (periodic)."""
    shifts = np.array([[i, j, k] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)]) @ lattice
    worst = [0, 0]
    for i, center in enumerate(coords):
        delta = coords[None, :, :] + shifts[:, None, :] - center
        dist = np.linalg.norm(delta, axis=-1)
        near = (dist < RCUT) & (dist > 1e-9)
        for t in (0, 1):
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
    atype = rng.permutation(np.arange(len(coords)) % 3 != 0).astype(int)  # one O per two H
    worst = neighbor_counts(coords, lattice, atype)
    roomy = [worst[0] + 3, worst[1] + 3]
    species = [TYPES[t] for t in atype]
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, sel, seed in (('se-r', roomy, SEED), ('se-r-tight', [4, 6], SEED + 10)):
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
            cells = [lattice] if name == 'se-r-tight' else [lattice, None]
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
                print(target.name, 'periodic' if cell is not None else 'open', 'energy', cases[-1]['energy'])
    reference = {'deepmd_version': deepmd.__version__, 'rcut': RCUT,
                 'largest_neighbor_counts_periodic': worst, 'cases': cases}
    (out / 'se-r-deepmd-reference.json').write_text(json.dumps(reference, indent=1) + '\n')
    print('atoms', len(coords), 'largest neighbor counts', worst, 'sel', roomy)


if __name__ == '__main__':
    main()
