"""Fixture for `tests/stats.rs`: a small dataset and DeePMD-kit's statistics on it.

Run in the DeePMD-kit image from `mlmd/native`:

    python tools/make_stats_fixture.py fixtures

Writes `stats-dataset.jsonl` (omnimd-dataset-jsonl-v1) and
`stats-deepmd-reference.json` with, from DeePMD-kit's own code:
neighbor-stat (`NeighborStatOP`), se_e2_a input statistics
(`DescrptSeA.compute_input_stats`) and the energy bias
(`compute_stats_from_redu`). Frames differ in size and composition, one has no
periodic cell, and `sel` is smaller than the neighbor counts of some atoms so
the nearest-neighbor truncation matters.
"""
import json
import sys
from pathlib import Path

import numpy as np
import deepmd
from deepmd.dpmodel.descriptor import DescrptSeA
from deepmd.dpmodel.utils.neighbor_stat import NeighborStatOP
from deepmd.utils.out_stat import compute_stats_from_redu

TYPE_MAP = ['O', 'H', 'C']
RCUT, RCUT_SMTH, SEL = 4.0, 0.5, [6, 12, 4]


def make_frame(rng, grid, lattice, weights, periodic):
    """Jittered lattice (no close contacts) with random types."""
    points = np.stack(np.meshgrid(*[np.arange(n) / n for n in grid], indexing='ij'), -1).reshape(-1, 3)
    points += rng.uniform(-0.12, 0.12, points.shape) / np.array(grid)
    coords = points @ lattice
    atype = rng.choice(len(TYPE_MAP), size=len(coords), p=weights)
    return coords, atype, lattice if periodic else None


def main():
    out = Path(sys.argv[1])
    rng = np.random.default_rng(20261110)
    specs = [
        ((3, 3, 3), np.diag([7.5, 7.5, 7.5]), [0.34, 0.66, 0.0], True),
        ((3, 4, 3), np.array([[8.0, 0.0, 0.0], [1.5, 9.0, 0.0], [-1.0, 0.7, 8.2]]), [0.3, 0.5, 0.2], True),
        ((4, 3, 3), np.array([[9.5, 0.0, 0.0], [0.0, 7.0, 0.0], [0.8, 0.0, 7.4]]), [0.5, 0.3, 0.2], True),
        ((2, 2, 2), np.diag([3.6, 3.6, 3.6]), [0.4, 0.6, 0.0], True),  # box below rcut: several images
        ((3, 3, 2), np.diag([6.0, 6.0, 4.0]), [0.2, 0.6, 0.2], False),  # open boundary
        ((4, 4, 3), np.array([[10.0, 0.0, 0.0], [2.0, 9.5, 0.0], [0.0, 1.0, 7.8]]), [0.25, 0.55, 0.2], True),
    ]
    frames = [make_frame(rng, *spec) for spec in specs]
    true_bias = np.array([-431.7, -13.9, -1029.3])
    counts = np.array([[np.sum(atype == t) for t in range(len(TYPE_MAP))] for _, atype, _ in frames], dtype=float)
    energies = counts @ true_bias + rng.normal(0.0, 0.8, len(frames))

    with (out / 'stats-dataset.jsonl').open('w', newline='\n') as handle:
        header = {'schema': 'omnimd-dataset-jsonl-v1', 'type_map': TYPE_MAP, 'units': 'EvAngstrom',
                  'virial_convention': 'MinusDEpsilonRow', 'source': 'tools/make_stats_fixture.py',
                  'selected_sets': []}
        handle.write(json.dumps({'kind': 'header', 'header': header}) + '\n')
        for index, ((coords, atype, cell), energy) in enumerate(zip(frames, energies)):
            frame = {'positions': coords.tolist(), 'species': [TYPE_MAP[t] for t in atype],
                     'cell': None if cell is None else cell.tolist(), 'units': 'EvAngstrom'}
            labels = {'energy': float(energy), 'forces': np.zeros_like(coords).tolist(), 'virial': None}
            data = {'index': index, 'id': f'stats-{index}', 'frame': frame, 'labels': labels}
            handle.write(json.dumps({'kind': 'frame', 'data': data}) + '\n')
        handle.write(json.dumps({'kind': 'completed', 'frames': len(frames), 'source_sha256': {}}) + '\n')

    # One DeePMD "system" per frame: the statistics pool sums over systems.
    systems = []
    min_rr2, max_nnei = np.inf, np.zeros(len(TYPE_MAP), dtype=int)
    neighbor_op = NeighborStatOP(len(TYPE_MAP), RCUT, mixed_types=False)
    for coords, atype, cell in frames:
        nloc = len(atype)
        box = None if cell is None else cell.reshape(1, 9)
        systems.append({'coord': coords.reshape(1, nloc, 3), 'atype': atype.reshape(1, nloc).astype(np.int64),
                        'box': box, 'natoms': np.array([[nloc, nloc, *counts[len(systems)].astype(int)]])})
        rr2, nnei = neighbor_op.call(coords.reshape(1, nloc * 3), atype.reshape(1, nloc), box)
        min_rr2 = min(min_rr2, float(np.min(rr2)))
        max_nnei = np.maximum(max_nnei, np.max(np.asarray(nnei).reshape(-1, len(TYPE_MAP)), axis=0))

    descriptor = DescrptSeA(RCUT, RCUT_SMTH, SEL, precision='float64')
    descriptor.compute_input_stats(systems)
    davg, dstd = np.asarray(descriptor.davg), np.asarray(descriptor.dstd)
    # DeePMD tiles the per-type values over the neighbor slots.
    assert np.all(davg == davg[:, :1, :]) and np.all(dstd == dstd[:, :1, :])

    bias, std = compute_stats_from_redu(energies.reshape(-1, 1), counts, rcond=None)
    reference = {
        'deepmd_version': deepmd.__version__,
        'rcut': RCUT, 'rcut_smth': RCUT_SMTH, 'sel': SEL,
        'min_nbor_dist': float(np.sqrt(min_rr2)),
        'max_nbor_size': [int(v) for v in max_nnei],
        'davg': davg[:, 0, :].tolist(), 'dstd': dstd[:, 0, :].tolist(),
        'energy_bias': np.asarray(bias).ravel().tolist(), 'energy_std': float(np.asarray(std).ravel()[0]),
    }
    (out / 'stats-deepmd-reference.json').write_text(json.dumps(reference, indent=1) + '\n')
    print(json.dumps(reference, indent=1))


if __name__ == '__main__':
    main()
