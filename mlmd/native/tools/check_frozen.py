"""Focused frozen-export checks: offsets/geometry plus topology/normalization failures."""
from copy import deepcopy
import json
import sys
from pathlib import Path
from tensorflow.python.framework.tensor_util import make_tensor_proto
from export_frozen import FrozenGraph, ConversionError


def main():
    original = FrozenGraph(sys.argv[1])
    model, profile = original.export()
    expected = json.loads(Path(sys.argv[2]).read_text())
    assert model == expected, 'frozen export differs from original complete interchange conversion'
    assert model['atom_energy_bias'] == [-0.7+0.04,0.2-0.03]

    def fresh():
        g = deepcopy(original)
        g.values = {}
        return g

    changed = fresh()
    name = 'model_attr/model_attr/t_out_bias'
    values = changed.tensor(name).copy()
    values[0,0,0] += 0.125
    changed.nodes[name].attr['value'].tensor.CopyFrom(make_tensor_proto(values))
    changed.values = {}
    result, _ = changed.export()
    assert result['atom_energy_bias'][0] == model['atom_energy_bias'][0]+0.125
    assert result['fitting'] == model['fitting']
    changed = fresh()
    changed.nodes['ProdEnvMatA'].attr['rcut_r_smth'].f = 2.2
    result, _ = changed.export()
    assert result['descriptor']['rcut_smth'] == changed.nodes['ProdEnvMatA'].attr['rcut_r_smth'].f
    assert result['descriptor']['rcut_smth'] != 2.2

    rejected = []
    def reject(label, mutate):
        g = fresh()
        mutate(g)
        g.values = {}
        try: g.export()
        except ConversionError: rejected.append(label)
        else: raise AssertionError('accepted '+label)
    def constant(g, name, values):
        g.nodes[name].attr['value'].tensor.CopyFrom(make_tensor_proto(values, dtype=g.nodes[name].attr['value'].tensor.dtype))
    reject('inconsistent C API cutoff',lambda g:constant(g,'descrpt_attr/rcut',3.5))
    reject('changed activation', lambda g:setattr(g.nodes['filter_type_0/Tanh'],'op','Sigmoid'))
    reject('changed descriptor divisor', lambda g:constant(g,'filter_type_0/truediv/y',65.0))
    reject('unapplied residual timestep',lambda g:g.nodes['filter_type_0/add'].input.__setitem__(1,'filter_type_0/Reshape_2'))
    reject('changed force provenance',lambda g:g.nodes['o_force'].input.__setitem__(0,'o_atom_energy'))
    reject('non-unit output scale',lambda g:constant(g,'model_attr/model_attr/t_out_std',[[[2.0],[1.0]]]))
    def invalid_normalization(g):
        values = g.tensor('descrpt_attr/t_std').copy()
        values[0,0] = 0.0
        constant(g,'descrpt_attr/t_std',values)
    reject('nonuniform or nonpositive normalization',invalid_normalization)
    print(json.dumps({'profile':profile,'matches_complete_interchange':True,
                      'both_bias_sites_preserved':True,'changed_bias_and_effective_geometry_checked':True,
                      'rejected':rejected}))

if __name__ == '__main__': main()
