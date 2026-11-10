"""Fixture for the GPU parity test with `sel` smaller than the neighbor counts.

Run in the DeePMD-kit image from `mlmd`:

    python native/tools/make_truncation_fixture.py native/fixtures

Builds a small se_e2_a model whose `sel` ([4, 6]) is far below the ~27
neighbors each atom has within the cutoff, so the result depends on which
neighbors DeePMD keeps (the nearest `sum(sel)` of any type, then at most
`sel[t]` per type). Writes `truncation-model.native.json` (converted with
`convert_deepmd.py`) and `truncation-deepmd-reference.json` with the frame and
DeePMD's energy, forces and virial (PyTorch backend, CPU, float64).
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
from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeA  # noqa: E402
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet  # noqa: E402
from deepmd.dpmodel.model.ener_model import EnergyModel  # noqa: E402
from deepmd.dpmodel.utils.serialization import save_dp_model  # noqa: E402
from deepmd.infer import DeepPot  # noqa: E402
from deepmd.pt.utils.serialization import deserialize_to_file  # noqa: E402

TYPES = ['O', 'H']
SEED = 20261110


def main():
    out = Path(sys.argv[1])
    descriptor_config = dict(rcut=4.0, rcut_smth=0.5, sel=[4, 6], neuron=[4, 8], axis_neuron=2,
                             resnet_dt=False, type_one_side=False, precision='float64',
                             type_map=TYPES, seed=SEED)
    fitting_config = dict(ntypes=2, dim_descrpt=8 * 2, neuron=[8, 8], resnet_dt=True,
                          precision='float64', type_map=TYPES, seed=SEED + 1)
    descriptor = DescrptSeA(**descriptor_config)
    fitting = EnergyFittingNet(**fitting_config)
    descriptor.davg[...] = 0.0
    descriptor.davg[:, :, 0] = 0.1
    descriptor.dstd[...] = 0.2
    descriptor.dstd[:, :, 0] = 0.3
    fitting.bias_atom_e[:] = np.array([[-3.0], [-1.5]])
    model = EnergyModel(descriptor=descriptor, fitting=fitting, type_map=TYPES)
    definition = {'type': 'standard', 'type_map': TYPES,
                  'descriptor': {'type': 'se_e2_a',
                                 **{k: v for k, v in descriptor_config.items() if k != 'type_map'}},
                  'fitting_net': {'type': 'ener',
                                  **{k: v for k, v in fitting_config.items()
                                     if k not in ('ntypes', 'dim_descrpt', 'type_map')}}}
    data = {'model': model.serialize(), 'model_def_script': definition}

    rng = np.random.default_rng(SEED)
    lattice = np.array([[7.0, 0.0, 0.0], [0.9, 6.6, 0.0], [-0.5, 0.4, 6.8]])
    grid = (3, 3, 3)
    points = np.stack(np.meshgrid(*[np.arange(n) / n for n in grid], indexing='ij'), -1).reshape(-1, 3)
    points += rng.uniform(-0.1, 0.1, points.shape) / np.array(grid)
    coords = points @ lattice
    atype = rng.permutation(np.arange(len(coords)) % 3 != 0).astype(int)  # one O per two H

    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / 'truncation'
        save_dp_model(str(stem.with_suffix('.dp')), deepcopy(data))
        deserialize_to_file(str(stem.with_suffix('.pth')), deepcopy(data))
        subprocess.run([sys.executable, str(Path(__file__).with_name('convert_deepmd.py')),
                        str(stem.with_suffix('.dp')), str(out / 'truncation-model.native.json'),
                        '--units', 'EvAngstrom', '--geometry', 'dpmodel'], check=True)
        energy, forces, virial = DeepPot(str(stem.with_suffix('.pth'))).eval(
            coords.reshape(1, -1), lattice.reshape(1, 9), atype.tolist())

    reference = {
        'deepmd_version': deepmd.__version__,
        'frame': {'positions': coords.tolist(), 'species': [TYPES[t] for t in atype],
                  'cell': lattice.tolist(), 'units': 'EvAngstrom'},
        'energy': float(energy.ravel()[0]),
        'forces': forces.reshape(-1, 3).tolist(),
        'virial': virial.reshape(3, 3).tolist(),
    }
    (out / 'truncation-deepmd-reference.json').write_text(json.dumps(reference, indent=1) + '\n')
    print('energy', reference['energy'], 'atoms', len(coords))


if __name__ == '__main__':
    main()
