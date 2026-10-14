"""Data-only frozen GraphDef -> native v1 for two audited exact graph profiles.

TensorFlow is used only for protobuf definitions and tensor decoding; no supplied
GraphDef is imported, compiled, deserialized as a Model, or executed. See FROZEN.md.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from google.protobuf.message import DecodeError
from convert_deepmd import ConversionError, need

ROOTS = ('o_energy', 'o_force', 'o_virial', 'model_attr/tmap',
         'model_attr/model_type', 'model_attr/model_version')
PROFILE_DIR = Path(__file__).with_name('frozen_profiles')


def profile_specs():
    """Reviewed architecture mappings; canonical graph equality gates their use."""
    result = []
    for modern in (False, True):
        p = {'id': 'synthetic-tf-3.1.3-v1' if modern else 'public-deeppot-1.1-v1',
             'sel': [32, 32] if modern else [46, 92],
             'embedding_widths': [1, 2, 4, 4] if modern else [1, 2, 4, 8],
             'axis': 2 if modern else 4, 'fitting_widths': [8, 5, 5, 1] if modern else [32, 6, 6, 6, 1],
             'modern': modern, 'parameters': {}}
        params = p['parameters']
        params['model_attr/tmap'] = {'dtype': 7, 'shape': []}
        params['descrpt_attr/rcut'] = {'dtype': 2, 'shape': []}
        for name in ('descrpt_attr/t_avg', 'descrpt_attr/t_std'):
            params[name] = {'dtype': 2, 'shape': [2, sum(p['sel'])*4]}
        for center in range(2):
            for neighbor in range(2):
                widths = p['embedding_widths']
                for layer, (ni, no) in enumerate(zip(widths, widths[1:]), 1):
                    prefix = f'filter_type_{center}'
                    for key, shape in [('matrix', [ni,no]), ('bias', [no] if modern else [1,no])]:
                        params[f'{prefix}/{key}_{layer}_{neighbor}'] = {'dtype': 2, 'shape': shape}
                    if modern:
                        params[f'{prefix}/idt_{layer}_{neighbor}'] = {'dtype': 2, 'shape': [1,no]}
            widths = p['fitting_widths']
            for layer, (ni, no) in enumerate(zip(widths, widths[1:])):
                final = layer == len(widths)-2
                prefix = f'final_layer_type_{center}' if final else f'layer_{layer}_type_{center}'
                params[prefix+'/matrix'] = {'dtype': 2, 'shape': [ni,no]}
                params[prefix+'/bias'] = {'dtype': 2, 'shape': [no]}
                if not final and layer > 0:
                    params[prefix+'/idt'] = {'dtype': 2, 'shape': [no]}
        if modern:
            params['fitting_attr/t_bias_atom_e'] = {'dtype': 2, 'shape': [2]}
            params['model_attr/model_attr/t_out_bias'] = {'dtype': 2, 'shape': [1,2,1]}
        result.append(p)
    return result


class FrozenGraph:
    def __init__(self, path):
        from tensorflow.core.framework.graph_pb2 import GraphDef
        path = Path(path)
        need(path.suffix == '.pb' and path.stat().st_size <= 256*1024*1024,
             'expected a .pb GraphDef of at most 256 MiB')
        self.raw = path.read_bytes()
        self.graph = GraphDef()
        self.graph.ParseFromString(self.raw)
        need(not self.graph.library.function and not self.graph.library.gradient,
             'function libraries are unsupported')
        need(len(self.graph.node) <= 100_000, 'graph exceeds node limit')
        self.nodes = {n.name:n for n in self.graph.node}
        need(len(self.nodes) == len(self.graph.node), 'duplicate node names')
        self.reachable = set()
        pending = list(ROOTS) + [n for n in self.nodes if n.startswith(
            ('model_attr/', 'descrpt_attr/', 'fitting_attr/'))]
        while pending:
            name = pending.pop().lstrip('^').split(':')[0]
            need(name in self.nodes, f'missing connected node: {name}')
            if name in self.reachable: continue
            self.reachable.add(name)
            pending.extend(self.nodes[name].input)
        self.values = {}

    def tensor(self, name):
        from tensorflow.python.framework.tensor_util import MakeNdarray
        import numpy as np
        need(name in self.nodes and self.nodes[name].op == 'Const', f'{name}: expected Const')
        if name not in self.values:
            tensor = self.nodes[name].attr['value'].tensor
            dims = [d.size for d in tensor.tensor_shape.dim]
            need(not tensor.tensor_shape.unknown_rank and all(d >= 0 for d in dims)
                 and len(dims) <= 8 and math.prod(dims) <= 8_000_000, f'{name}: invalid tensor shape')
            value = MakeNdarray(tensor)
            need(value.dtype.kind in 'fiubOS', f'{name}: unsupported tensor dtype')
            if value.dtype.kind == 'f': need(np.isfinite(value).all(), f'{name}: nonfinite tensor')
            self.values[name] = value
        return self.values[name]

    def canonical(self, profile):
        """All E/F/V ancestors: pin ops, ports, attributes, shapes and structural data.

        Only listed Const payloads and two effective float cutoff attrs vary.
        Names are part of the profile, but names alone never establish semantics.
        """
        from google.protobuf.json_format import MessageToDict
        params = profile['parameters']
        need(set(params) <= self.reachable, 'profile parameter missing from output ancestry')
        rows = []
        for name in sorted(self.reachable):
            node = self.nodes[name]
            attrs = {k:MessageToDict(v, preserving_proto_field_name=True,
                                    use_integers_for_enums=True) for k,v in node.attr.items()}
            if node.op == 'Const':
                value = self.tensor(name)
                tensor = node.attr['value'].tensor
                info = {'dtype':tensor.dtype, 'shape':list(value.shape)}
                if name in params:
                    need(info == params[name], f'{name}: unsupported parameter dtype/shape')
                    info['value'] = '<parameter>'
                else:
                    # Decode string bytes explicitly; do not depend on protobuf's tensor encoding.
                    if value.dtype.kind in 'OS':
                        info['value'] = [bytes(v).hex() for v in value.flat]
                    else:
                        info['value'] = value.tolist()
                attrs['value'] = info
            if name == 'ProdEnvMatA':
                for key in ('rcut_r', 'rcut_r_smth'):
                    need(node.attr[key].WhichOneof('value') == 'f', f'{key}: expected float attribute')
                    attrs[key] = {'f':'<effective-cutoff>'}
            rows.append({'name':name, 'op':node.op, 'inputs':list(node.input), 'attrs':attrs})
        return rows

    def select_profile(self):
        failures = []
        for profile in profile_specs():
            try:
                actual = self.canonical(profile)
                expected = [json.loads(line) for line in
                            (PROFILE_DIR/(profile['id']+'.jsonl')).read_text(encoding='utf-8').splitlines()]
                need(actual == expected, 'connected graph differs from audited profile')
                return profile
            except ConversionError as exc:
                failures.append(f"{profile['id']}: {exc}")
        raise ConversionError('unsupported frozen graph; ' + '; '.join(failures))

    def export(self):
        p = self.select_profile()  # No extraction based on names before structural equality.
        types = self.tensor('model_attr/tmap').item().decode('utf-8').split()
        need(len(types) == 2 and len(set(types)) == 2 and
             all(re.fullmatch(r'[A-Z][a-z]?', t) for t in types), 'invalid ordered two-species type map')
        env = self.nodes['ProdEnvMatA']
        rc, rs = env.attr['rcut_r'].f, env.attr['rcut_r_smth'].f
        need(math.isfinite(rc) and math.isfinite(rs) and 0 <= rs < rc, 'invalid effective cutoffs')
        need(self.tensor('descrpt_attr/rcut').item() == rc,
             'C API cutoff metadata must equal the effective environment-op cutoff')
        avg = self.tensor('descrpt_attr/t_avg').reshape(2,-1,4)
        std = self.tensor('descrpt_attr/t_std').reshape(2,-1,4)
        normalization = []
        for i in range(2):
            mean, scale = avg[i,0], std[i,0]
            need((avg[i] == mean).all() and (std[i] == scale).all(), 'nonuniform slot normalization')
            need((mean[1:] == 0).all() and (scale > 0).all() and (scale[1:] == scale[1]).all(),
                 'normalization must have zero angular mean and positive isotropic angular scale')
            normalization.append({'mean':mean.tolist(), 'std':scale.tolist()})
        def layer(w, b, dt, residual, act):
            return {'weights':self.tensor(w).tolist(), 'bias':self.tensor(b).reshape(-1).tolist(),
                    'activation':act, 'residual':residual,
                    'timestep':None if dt is None else self.tensor(dt).reshape(-1).tolist()}
        embedding = []
        for center in range(2):
            neighbors = []
            for neighbor in range(2):
                layers = []
                for i in range(1,4):
                    prefix, suffix = f'filter_type_{center}/', f'_{i}_{neighbor}'
                    residual = 'identity' if p['modern'] and i == 3 else 'concat2'
                    layers.append(layer(prefix+'matrix'+suffix, prefix+'bias'+suffix,
                                        prefix+'idt'+suffix if p['modern'] else None, residual, 'tanh'))
                neighbors.append({'layers':layers})
            embedding.append(neighbors)
        fitting = []
        for center in range(2):
            layers = []
            for i in range(len(p['fitting_widths'])-1):
                final = i == len(p['fitting_widths'])-2
                prefix = f'final_layer_type_{center}' if final else f'layer_{i}_type_{center}'
                dt = None if final or i == 0 else prefix+'/idt'
                layers.append(layer(prefix+'/matrix', prefix+'/bias', dt,
                                    'none' if dt is None else 'identity', 'linear' if final else 'tanh'))
            fitting.append({'layers':layers})
        bias = [0.0,0.0]
        if p['modern']:
            # The audited output path adds fitting bias, multiplies by unit out_std,
            # then adds model out_bias. Final affine biases remain inside fitting.
            bias = (self.tensor('fitting_attr/t_bias_atom_e') +
                    self.tensor('model_attr/model_attr/t_out_bias').reshape(2)).tolist()
        result = {'format':'omnimd-native-se-e2-a', 'version':1, 'precision':'float64',
                  'units':'EvAngstrom', 'type_map':types,
                  'descriptor':{'kind':'se_e2_a','rcut':rc,'rcut_smth':rs,'sel':p['sel'],
                                'axis_neuron':p['axis'],'normalization':normalization,'embedding':embedding},
                  'fitting':fitting, 'atom_energy_bias':bias}
        json.dumps(result,allow_nan=False)
        return result, p['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--units', choices=['EvAngstrom'], required=True,
                        help='explicit user assertion; a GraphDef does not attest physical units')
    args = parser.parse_args()
    try:
        graph = FrozenGraph(args.input)
        result, profile = graph.export()
        encoded = (json.dumps(result,indent=2,allow_nan=False)+'\n').encode('utf-8')
        with args.output.open('xb') as handle: handle.write(encoded)
        print(json.dumps({'profile':profile,'input_sha256':hashlib.sha256(graph.raw).hexdigest(),
                          'output_sha256':hashlib.sha256(encoded).hexdigest(),
                          'geometry':'effective TensorFlow op attributes',
                          'effective_rcut':result['descriptor']['rcut'],
                          'effective_rcut_smth':result['descriptor']['rcut_smth'],
                          'verified_output_ancestor_nodes':len(graph.reachable)}))
        return 0
    except (ValueError, TypeError, KeyError, OSError, ImportError, OverflowError, DecodeError) as exc:
        print(f'frozen graph export failed: {exc}',file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
