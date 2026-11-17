"""Fixtures for the hybrid descriptor against DeePMD-kit itself.

Run in the DeePMD-kit image from `mlmd`:

    python native/tools/make_hybrid_fixture.py native/fixtures

Builds two small energy models of three types whose descriptor is a hybrid of
an se_e2_a, an se_e2_r and an se_e3 descriptor, each with a cutoff, a
smoothing radius, slots, a width, nets and a normalization of its own:

- `hybrid-model.native.json`: every descriptor has room for all neighbors
  within its own cutoff, and for a few more (a short run of dynamics moves
  atoms across the cutoffs), but each reads the shared neighbor list
  differently.
  The se_e2_a part has the shortest list for two types and the longest for
  the third, where its slots hold neighbors beyond its cutoff that must count
  as empty; the se_e2_r part has the largest cutoff; the se_e3 part the
  smallest cutoff and the fewest slots.
- `hybrid-tight-model.native.json`: the slots are far below the neighbor
  counts, in the shared list and in every part, so that the result depends
  on which neighbors DeePMD-kit keeps: the nearest `sum(sel)` of the list (of
  any type), of those at most `sel[t]` of each type, and of those the first
  `sel[t]` of each part. The script counts the atoms for which that is
  another choice than each part would make alone, and stops if there are
  none.

Both are converted with `convert_deepmd.py`. Writes
`hybrid-deepmd-reference.json` with DeePMD's energy, forces and virial
(PyTorch backend, CPU, float64) for a periodic triclinic frame and for the
same atoms without a cell under the first model, and for the periodic frame
under the second.
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
from deepmd.dpmodel.descriptor.hybrid import DescrptHybrid  # noqa: E402
from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeA  # noqa: E402
from deepmd.dpmodel.descriptor.se_r import DescrptSeR  # noqa: E402
from deepmd.dpmodel.descriptor.se_t import DescrptSeT  # noqa: E402
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet  # noqa: E402
from deepmd.dpmodel.model.ener_model import EnergyModel  # noqa: E402
from deepmd.dpmodel.utils.serialization import save_dp_model  # noqa: E402
from deepmd.infer import DeepPot  # noqa: E402
from deepmd.pt.utils.serialization import deserialize_to_file  # noqa: E402

TYPES = ['O', 'H', 'C']
SEED = 20261116
# Per part: DeePMD-kit's name, class, cutoff, smoothing radius, embedding neurons,
# extra arguments, and per center type the radial mean, the radial width and the
# width of the directions (se_e2_r has the radial entry only; se_e3 must not read it).
PARTS = {
    'se_e2_a': (DescrptSeA, [4, 8], dict(axis_neuron=3, type_one_side=False),
                [(0.11, 0.33, 0.21), (0.05, 0.27, 0.29), (0.08, 0.38, 0.24)]),
    'se_e2_r': (DescrptSeR, [5, 10], dict(type_one_side=True),
                [(0.10, 0.30, None), (0.16, 0.25, None), (0.07, 0.41, None)]),
    'se_e3': (DescrptSeT, [3, 6], dict(),
              [(0.13, 0.35, 0.22), (0.09, 0.28, 0.31), (0.06, 0.44, 0.27)]),
}


def build(parts, seed):
    """`parts`: (name, rcut, rcut_smth, sel) of each descriptor of the hybrid, in order."""
    descriptors, configs = [], []
    for index, (name, rcut, rcut_smth, sel) in enumerate(parts):
        cls, neuron, extra, stats = PARTS[name]
        config = dict(rcut=rcut, rcut_smth=rcut_smth, sel=sel, neuron=neuron, resnet_dt=False,
                      precision='float64', type_map=TYPES, seed=seed + 10 * index, **extra)
        descriptor = cls(**config)
        for center, (radial_mean, radial_width, width) in enumerate(stats):
            descriptor.davg[center, :, 0] = radial_mean
            descriptor.dstd[center, :, 0] = radial_width
            if width is not None:
                descriptor.dstd[center, :, 1:] = width
        descriptors.append(descriptor)
        configs.append({'type': name, **{k: v for k, v in config.items() if k != 'type_map'}})
    hybrid = DescrptHybrid(list=descriptors)
    fitting_config = dict(ntypes=len(TYPES), dim_descrpt=hybrid.get_dim_out(), neuron=[8, 8],
                          resnet_dt=True, precision='float64', type_map=TYPES, seed=seed + 1)
    fitting = EnergyFittingNet(**fitting_config)
    fitting.bias_atom_e[:] = np.array([[-3.0], [-1.5], [-2.25]])
    model = EnergyModel(descriptor=hybrid, fitting=fitting, type_map=TYPES)
    definition = {'type': 'standard', 'type_map': TYPES,
                  'descriptor': {'type': 'hybrid', 'list': configs},
                  'fitting_net': {'type': 'ener',
                                  **{k: v for k, v in fitting_config.items()
                                     if k not in ('ntypes', 'dim_descrpt', 'type_map')}}}
    return {'model': model.serialize(), 'model_def_script': definition}, hybrid


def neighbors(coords, lattice, atype, rcut):
    """Of every atom its neighbors within `rcut` (periodic), nearest first: (distance, type, image)."""
    shifts = np.array([[i, j, k] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)]) @ lattice
    lists = []
    for center in coords:
        delta = coords[None, :, :] + shifts[:, None, :] - center
        dist = np.linalg.norm(delta, axis=-1)
        found = [(dist[s, j], int(atype[j]), (s, j))
                 for s, j in zip(*np.nonzero((dist < rcut) & (dist > 1e-9)))]
        lists.append(sorted(found))
    return lists


def worst_counts(lists):
    return [max(sum(1 for _, t, _ in near if t == kind) for near in lists) for kind in range(len(TYPES))]


def by_type(near, sel):
    """At most `sel[t]` of each type of a list that is nearest first."""
    kept, used = [], [0] * len(sel)
    for entry in near:
        if used[entry[1]] < sel[entry[1]]:
            used[entry[1]] += 1
            kept.append(entry)
    return kept


def select(near, sel):
    """DeePMD-kit's choice for a descriptor alone: the nearest sum(sel) of any type, then by type."""
    return by_type(near[:sum(sel)], sel)


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
    species = [TYPES[t] for t in atype]
    worst = {rcut: worst_counts(neighbors(coords, lattice, atype, rcut)) for rcut in (2.9, 3.4, 4.4)}
    roomy = [('se_e2_a', 3.4, 0.6, [worst[3.4][0] + 4, worst[3.4][1] + 4, worst[4.4][2] + 6]),
             ('se_e2_r', 4.4, 2.2, [worst[4.4][0] + 5, worst[4.4][1] + 4, worst[4.4][2] + 4]),
             ('se_e3', 2.9, 0.4, [worst[2.9][0] + 4, worst[2.9][1] + 5, worst[2.9][2] + 4])]
    tight = [('se_e2_a', 4.0, 0.5, [6, 7, 4]),
             ('se_e3', 3.6, 0.9, [3, 4, 2]),
             ('se_e2_r', 4.4, 1.0, [5, 9, 3])]
    cases, models = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, parts, seed in (('hybrid', roomy, SEED), ('hybrid-tight', tight, SEED + 100)):
            data, hybrid = build(parts, seed)
            stem = Path(tmp) / name
            save_dp_model(str(stem.with_suffix('.dp')), deepcopy(data))
            deserialize_to_file(str(stem.with_suffix('.pth')), deepcopy(data))
            target = out / f'{name}-model.native.json'
            target.unlink(missing_ok=True)
            subprocess.run([sys.executable, str(Path(__file__).with_name('convert_deepmd.py')),
                            str(stem.with_suffix('.dp')), str(target),
                            '--units', 'EvAngstrom', '--geometry', 'dpmodel'], check=True)
            # The list all parts read, and what each part has of it against what it
            # would select alone.
            rcut, sel = hybrid.get_rcut(), hybrid.get_sel()
            shared = neighbors(coords, lattice, atype, rcut)
            listed = [select(near, sel) for near in shared]
            cut = sum(1 for near, kept in zip(shared, listed) if len(kept) < len(near))
            differ = []
            for _, part_rcut, _, part_sel in parts:
                mine = [[e for e in by_type(kept, part_sel) if e[0] < part_rcut] for kept in listed]
                alone = [select([e for e in near if e[0] < part_rcut], part_sel) for near in shared]
                differ.append(sum(1 for a, b in zip(mine, alone) if a != b))
            models[target.name] = {
                'rcut': rcut, 'sel': sel,
                'parts': [{'kind': kind, 'rcut': r, 'rcut_smth': s, 'sel': part_sel}
                          for kind, r, s, part_sel in parts],
                'atoms_whose_list_is_cut': cut,
                'atoms_where_a_part_differs_from_its_own_selection': differ}
            print(target.name, 'list: rcut', rcut, 'sel', sel, 'cut for', cut, 'atoms; parts differ from'
                  ' their own selection for', differ, 'atoms')
            if name == 'hybrid':
                assert cut == 0 and differ == [0, 0, 0], 'the roomy model must hold every neighbor'
            else:
                assert cut > 0 and all(count > 0 for count in differ), \
                    'the tight model must tell the selection of a hybrid from that of its parts'
            pot = DeepPot(str(stem.with_suffix('.pth')))
            for cell in ([lattice] if name == 'hybrid-tight' else [lattice, None]):
                energy, forces, virial = pot.eval(coords.reshape(1, -1),
                                                  None if cell is None else cell.reshape(1, 9),
                                                  atype.tolist())
                cases.append({'model': target.name,
                              'frame': {'positions': coords.tolist(), 'species': species,
                                        'cell': None if cell is None else cell.tolist(),
                                        'units': 'EvAngstrom'},
                              'energy': float(energy.ravel()[0]),
                              'forces': forces.reshape(-1, 3).tolist(),
                              'virial': virial.reshape(3, 3).tolist()})
                print(target.name, 'periodic' if cell is not None else 'open', 'energy', cases[-1]['energy'],
                      'largest force', float(np.abs(forces).max()))
    reference = {'deepmd_version': deepmd.__version__,
                 'largest_neighbor_counts_periodic': {str(rcut): counts for rcut, counts in worst.items()},
                 'models': models, 'cases': cases}
    (out / 'hybrid-deepmd-reference.json').write_text(json.dumps(reference, indent=1) + '\n')
    print('atoms', len(coords), 'largest neighbor counts', worst)


if __name__ == '__main__':
    main()
