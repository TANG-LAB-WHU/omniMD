"""Focused conversion checks for the generated DeepMD fixture, no inference."""
from copy import deepcopy
import json
import sys
from convert_deepmd import ConversionError, convert, load_data

source = load_data(sys.argv[1])
converted = convert(source, 'tensorflow')
assert converted == convert(load_data(sys.argv[2]), 'tensorflow')
assert converted['atom_energy_bias'] == [-0.6599999999999999, 0.17]
assert converted['descriptor']['rcut_smth'] == 2.0999999046325684
assert convert(source, 'dpmodel')['descriptor']['rcut_smth'] == 2.1
# Match a known central/neighbor pair to its source index; swapped ordering
# must not accidentally pass because this fixture uses distinct pair networks.
emb = source['model']['descriptor']['embeddings']['networks']
assert converted['descriptor']['embedding'][1][0]['layers'][0]['weights'] == emb[1]['layers'][0]['@variables']['w']['value']
assert converted['descriptor']['embedding'][0][1]['layers'][0]['weights'] == emb[2]['layers'][0]['@variables']['w']['value']
rejections = []
mutations = [
 ('producer version', lambda d: d.update(version='3.2.0')),
 ('unknown descriptor field', lambda d: d['model']['descriptor'].update(compression={})),
 ('hybrid descriptor', lambda d: d['model']['descriptor'].update(type='hybrid')),
 ('non-unit output scale', lambda d: d['model']['@variables']['out_std']['value'][0][0].__setitem__(0, 2.0)),
 ('per-slot normalization', lambda d: d['model']['descriptor']['@variables']['davg']['value'][0][1].__setitem__(0, 0.0)),
 ('float32 weights', lambda d: d['model']['descriptor']['embeddings']['networks'][0]['layers'][0]['@variables']['w'].update(dtype='float32')),
 ('malformed weights', lambda d: d['model']['fitting']['nets']['networks'][0]['layers'][0]['@variables']['w']['value'].pop()),
 ('vacuum subtraction', lambda d: d['model']['fitting'].update(atom_ener=[0.0,None])),
]
for name, mutate in mutations:
    data = deepcopy(source)
    mutate(data)
    try:
        convert(data, 'tensorflow')
    except ConversionError:
        rejections.append(name)
    else:
        raise AssertionError('did not reject '+name)
print(json.dumps({'dp_yaml_equal':True,'geometry_modes_checked':True,'biases_and_pair_order_checked':True,
                  'rejected':rejections}))
