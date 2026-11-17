"""Fixture for `tests/stats.rs`: DeePMD-kit's input statistics of the
descriptors of a hybrid, on the dataset of `make_stats_fixture.py`.

Run in the DeePMD-kit image from `mlmd/native`:

    python tools/make_hybrid_stats_fixture.py fixtures

Reads `stats-dataset.jsonl` and writes `hybrid-stats-deepmd-reference.json`.
A hybrid of an se_e2_a and an se_e2_r descriptor gets its statistics from
DeePMD-kit's own code (`DescrptHybrid.compute_input_stats`), which hands the
frames to every descriptor of the list: each takes its mean and width over
the neighbors it selects as the descriptor of a model, within its own cutoff
and with its own slots (`EnvMatStatSe` builds its list from the descriptor).

In the forward pass of a hybrid the descriptors read something else: of one
list made with the largest cutoff and of every type the most slots, the first
slots of each type. With slots short of the neighbor counts, as here, that
gives a descriptor other neighbors than its own selection. The file also has
the statistics over those (`as_in_the_forward_pass`, from DeePMD-kit's
neighbor list and environment matrix with the cut of `DescrptHybrid.call`),
to show that the two differ and which one DeePMD-kit trains with.
"""
import json
import sys
from pathlib import Path

import numpy as np
import deepmd
from deepmd.dpmodel.descriptor import DescrptHybrid, DescrptSeA, DescrptSeR
from deepmd.dpmodel.utils.env_mat import EnvMat
from deepmd.dpmodel.utils.nlist import extend_input_and_build_neighbor_list

# The second has the shorter cutoff; each has fewer slots of one type than
# the list of both, [6, 14, 4] within 4.0.
PARTS = [
    {'type': 'se_e2_a', 'rcut': 4.0, 'rcut_smth': 0.5, 'sel': [6, 12, 4]},
    {'type': 'se_e2_r', 'rcut': 3.0, 'rcut_smth': 1.0, 'sel': [3, 14, 2]},
]


def load(path):
    """The frames of an omnimd-dataset-jsonl-v1 file as DeePMD systems of one frame."""
    type_map, systems = None, []
    for line in path.read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        if record['kind'] == 'header':
            type_map = record['header']['type_map']
        elif record['kind'] == 'frame':
            frame = record['data']['frame']
            atype = np.array([type_map.index(name) for name in frame['species']], dtype=np.int64)
            nloc = len(atype)
            box = None if frame['cell'] is None else np.array(frame['cell'], dtype=float).reshape(1, 9)
            counts = [int(np.sum(atype == t)) for t in range(len(type_map))]
            systems.append({'coord': np.array(frame['positions'], dtype=float).reshape(1, nloc, 3),
                            'atype': atype.reshape(1, nloc), 'box': box,
                            'natoms': np.array([[nloc, nloc, *counts]])})
    return type_map, systems


def moments(values):
    """Mean and width of DeePMD-kit's `StatItem` over all entries of `values`."""
    number, total, squared = values.size, float(values.sum()), float(np.square(values).sum())
    if number == 0:
        return 0.0, 0.1
    mean = total / number
    width = np.sqrt(squared / number - mean * mean)
    return mean, (1e-2 if abs(width) < 1e-2 else float(width))


def forward_statistics(systems, ntypes, part, list_rcut, list_sel):
    """Statistics of a descriptor over what it reads in the forward pass of the hybrid."""
    sel = part['sel']
    # `DescrptHybrid`: of every type the first slots of the list, as many as the descriptor has.
    starts = np.cumsum([0] + list_sel[:-1])
    cut = np.concatenate([np.arange(start, start + count) for start, count in zip(starts, sel)])
    radial_only = part['type'] == 'se_e2_r'
    last = 1 if radial_only else 4
    rows = [[] for _ in range(ntypes)]
    for system in systems:
        nloc = system['atype'].shape[1]
        coord, atype, _, nlist = extend_input_and_build_neighbor_list(
            system['coord'], system['atype'], list_rcut, list_sel, mixed_types=False, box=system['box'])
        zero = np.zeros((ntypes, sum(sel), last))
        one = np.ones((ntypes, sum(sel), last))
        # The switch of the descriptor is zero from its own cutoff on: such a neighbor is an empty slot.
        env, _, _ = EnvMat(part['rcut'], part['rcut_smth']).call(coord, atype, nlist[:, :, cut], zero, one,
                                                                 radial_only)
        env = np.asarray(env).reshape(nloc, sum(sel), last)
        for t in range(ntypes):
            rows[t].append(env[system['atype'][0] == t].reshape(-1, last))
    davg, dstd = [], []
    for t in range(ntypes):
        entries = np.concatenate(rows[t], axis=0)
        mean, width = moments(entries[:, :1])
        if radial_only:
            davg.append([mean])
            dstd.append([width])
        else:
            _, angular = moments(entries[:, 1:])
            davg.append([mean, 0.0, 0.0, 0.0])
            dstd.append([width, angular, angular, angular])
    return davg, dstd


def main():
    out = Path(sys.argv[1])
    type_map, systems = load(out / 'stats-dataset.jsonl')
    ntypes = len(type_map)
    descriptors = []
    for part in PARTS:
        kind = DescrptSeA if part['type'] == 'se_e2_a' else DescrptSeR
        descriptors.append(kind(part['rcut'], part['rcut_smth'], part['sel'], precision='float64'))
    hybrid = DescrptHybrid(list=descriptors)
    list_rcut, list_sel = float(hybrid.get_rcut()), [int(s) for s in hybrid.get_sel()]
    assert list_rcut == max(p['rcut'] for p in PARTS)
    assert list_sel == [max(p['sel'][t] for p in PARTS) for t in range(ntypes)]
    hybrid.compute_input_stats(systems)

    parts = []
    for part, descriptor in zip(PARTS, hybrid.descrpt_list):
        davg, dstd = np.asarray(descriptor.davg), np.asarray(descriptor.dstd)
        # DeePMD tiles the per-type values over the neighbor slots.
        assert np.all(davg == davg[:, :1, :]) and np.all(dstd == dstd[:, :1, :])
        # The same descriptor as the descriptor of a model has the same statistics.
        alone = type(descriptor)(part['rcut'], part['rcut_smth'], part['sel'], precision='float64')
        alone.compute_input_stats(systems)
        assert np.array_equal(np.asarray(alone.davg), davg) and np.array_equal(np.asarray(alone.dstd), dstd)
        forward_davg, forward_dstd = forward_statistics(systems, ntypes, part, list_rcut, list_sel)
        parts.append({**part, 'davg': davg[:, 0, :].tolist(), 'dstd': dstd[:, 0, :].tolist(),
                      'as_in_the_forward_pass': {'davg': forward_davg, 'dstd': forward_dstd}})
    # A descriptor that reads the list as it is has the same statistics either way: the
    # check of `forward_statistics` against DeePMD-kit's own.
    for part in PARTS:
        whole = {**part, 'rcut': list_rcut, 'sel': list_sel}
        kind = DescrptSeA if part['type'] == 'se_e2_a' else DescrptSeR
        alone = kind(list_rcut, part['rcut_smth'], list_sel, precision='float64')
        alone.compute_input_stats(systems)
        davg, dstd = forward_statistics(systems, ntypes, whole, list_rcut, list_sel)
        assert np.allclose(davg, np.asarray(alone.davg)[:, 0, :], rtol=0, atol=1e-12), part['type']
        assert np.allclose(dstd, np.asarray(alone.dstd)[:, 0, :], rtol=0, atol=1e-12), part['type']

    reference = {'deepmd_version': deepmd.__version__, 'dataset': 'stats-dataset.jsonl',
                 'list': {'rcut': list_rcut, 'sel': list_sel}, 'parts': parts}
    (out / 'hybrid-stats-deepmd-reference.json').write_text(json.dumps(reference, indent=1) + '\n')
    print(json.dumps(reference, indent=1))


if __name__ == '__main__':
    main()
