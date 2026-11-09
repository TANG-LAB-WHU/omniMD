"""Reference energy/forces/virial from DeepMD-kit for the bench cases (run in
the DeepMD image). CPU float64 DeepPot, so results are deterministic and do
not depend on GPU memory. One JSON line per case: {natoms, energy, forces,
virial} with virial as a 3x3 row-major list.

    python deepmd_reference.py cases water-se_e2_a-float64.pth 192 1536 > ref.jsonl
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np  # noqa: E402
from deepmd.infer import DeepPot  # noqa: E402

case_dir, model = Path(sys.argv[1]), sys.argv[2]
pot = DeepPot(str(case_dir / model))
for natoms in sys.argv[3:]:
    data = np.load(case_dir / f'water-{natoms}.npz')
    energy, forces, virial = pot.eval(data['coords'].reshape(1, -1), data['cell'].reshape(1, 9),
                                      data['atype'].tolist())
    print(json.dumps({'natoms': int(natoms), 'energy': float(energy.ravel()[0]),
                      'forces': forces.reshape(-1, 3).tolist(),
                      'virial': virial.reshape(3, 3).tolist()}), flush=True)
