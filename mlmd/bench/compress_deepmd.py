"""Compress the benchmark models with DeePMD-kit (run in the DeePMD image).

    python compress_deepmd.py cases [MIN_DISTANCE]

Same steps as `dp --pt compress` (stride 0.01, extrapolate 5). The benchmark
models are untrained, so they carry no minimum neighbor distance; it is set
here (default 0.5 Å, below the shortest O-H distance of the water frames)
instead of being computed from training data. Writes
`<model>-compressed.pth` next to each model and adds them to the manifest
under `compressed_models`.
"""
import json
import sys
from pathlib import Path

import torch
from deepmd.pt.model.model import get_model
from deepmd.pt.utils import env


def main():
    case_dir = Path(sys.argv[1])
    min_distance = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    manifest_path = case_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    compressed = {}
    for precision, stem in manifest['models'].items():
        saved = torch.jit.load(str(case_dir / f'{stem}.pth'), map_location='cpu')
        model = get_model(json.loads(saved.model_def_script))
        model.load_state_dict(saved.state_dict())
        model.min_nbor_dist = torch.tensor(min_distance, dtype=env.GLOBAL_PT_FLOAT_PRECISION,
                                           device=env.DEVICE)
        model.enable_compression(5, 0.01, 0.1, -1)
        torch.jit.save(torch.jit.script(model), str(case_dir / f'{stem}-compressed.pth'))
        compressed[precision] = f'{stem}-compressed'
        print('compressed', stem, flush=True)
    manifest['compressed_models'] = compressed
    manifest['compressed_min_distance'] = min_distance
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
