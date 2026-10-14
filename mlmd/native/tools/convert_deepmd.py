"""Strict DeepMD-kit 3.1.3 .dp/.yaml -> omnimd-native-se-e2-a v1.

Reads data only. Does not import DeepMD/TensorFlow, load pickle/checkpoints,
execute model graphs, infer missing architecture, or install dependencies.
.dp requires h5py; .yaml/.yml requires PyYAML's SafeLoader.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import struct
from pathlib import Path


class ConversionError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise ConversionError(message)


def fields(obj, names, where, optional=()):
    need(isinstance(obj, dict), f'{where}: expected object')
    required, allowed = set(names.split()), set(names.split()) | set(optional)
    need(not (set(obj) - allowed), f'{where}: unsupported fields {sorted(set(obj)-allowed)}')
    need(not (required - set(obj)), f'{where}: missing fields {sorted(required-set(obj))}')
    return obj


def schema(obj, name, version, names, where, optional=()):
    fields(obj, '@class @version ' + names, where, optional)
    need(obj['@class'] == name and type(obj['@version']) is int and obj['@version'] == version,
         f'{where}: expected {name} serialization version {version}')


def boolean(value, where):
    need(type(value) is bool, f'{where}: expected boolean')
    return value


def number(value, where):
    need(type(value) in (int, float) and math.isfinite(value), f'{where}: expected finite number')
    return value


def integer(value, where, minimum=0):
    need(type(value) is int and value >= minimum, f'{where}: expected integer >= {minimum}')
    return value


def unique(pairs):
    obj = {}
    for key, value in pairs:
        need(key not in obj, f'duplicate key {key!r}')
        obj[key] = value
    return obj


def array(obj, shape, where):
    fields(obj, '@class @is_variable @version dtype value', where)
    need(obj['@class'] == 'np.ndarray' and obj['@is_variable'] is True
         and type(obj['@version']) is int and obj['@version'] == 1 and obj['dtype'] == 'float64',
         f'{where}: expected typed float64 ndarray v1')
    def check(value, dims):
        if not dims:
            return float(number(value, where))
        need(isinstance(value, list) and len(value) == dims[0], f'{where}: expected shape {shape}')
        return [check(v, dims[1:]) for v in value]
    return check(obj['value'], shape)


def load_data(path):
    path = Path(path)
    need(path.stat().st_size <= 256 * 1024 * 1024, 'input exceeds 256 MiB limit')
    if path.suffix == '.dp':
        import h5py
        with h5py.File(path, 'r') as handle:
            data = json.loads(handle.attrs['json'], object_pairs_hook=unique)
            def decode(value, variable=False):
                if isinstance(value, dict):
                    return {k:decode(v, variable or k == '@variables') for k,v in value.items()}
                if isinstance(value, list): return [decode(v, variable) for v in value]
                if value is None or not variable: return value
                need(isinstance(value, str) and re.fullmatch(r'/variable_[0-9]+', value),
                     'HDF5 variable must be an internal root dataset reference')
                need(isinstance(handle.get(value, getlink=True), h5py.HardLink),
                     f'{value}: external and soft links are unsupported')
                ds = handle[value]
                need(isinstance(ds, h5py.Dataset) and not ds.is_virtual and not ds.external
                     and ds.compression is None and ds.dtype.kind == 'f' and ds.dtype.itemsize == 8,
                     f'{value}: expected ordinary, uncompressed float64 dataset')
                need(ds.size <= 8_000_000, f'{value}: dataset exceeds 8000000 values')
                return {'@class':'np.ndarray', '@is_variable':True, '@version':1,
                        'dtype':'float64', 'value':ds[()].tolist()}
            return decode(data)
    if path.suffix in ('.yaml', '.yml'):
        import yaml
        class UniqueSafeLoader(yaml.SafeLoader):
            pass
        def mapping(loader, node):
            return unique((loader.construct_object(k, deep=True), loader.construct_object(v, deep=True))
                          for k,v in node.value)
        UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        return yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueSafeLoader)
    raise ConversionError('supported inputs are DeepMD 3.1.3 .dp, .yaml, or .yml; no .pb/.pth/pickle')


def precision(obj, where):
    need(obj['precision'] == 'float64', f'{where}: only explicit float64 precision is supported')


def activation(value, where):
    supported = {'none':'linear', 'linear':'linear', 'tanh':'tanh', 'sigmoid':'sigmoid', 'softplus':'softplus'}
    need(value in supported, f'{where}: unsupported activation {value!r}')
    return supported[value]


def layer(obj, ni, no, where):
    schema(obj, 'Layer', 2, 'bias use_timestep activation_function resnet precision trainable @variables', where)
    precision(obj, where)
    for key in ('bias','use_timestep','resnet','trainable'): boolean(obj[key], f'{where}.{key}')
    variables = fields(obj['@variables'], 'w b idt', f'{where}.variables')
    w = array(variables['w'], [ni,no], f'{where}.w')
    if obj['bias']:
        b = array(variables['b'], [no], f'{where}.b')
    else:
        need(variables['b'] is None, f'{where}: bias=false requires null b')
        b = [0.0] * no
    if obj['use_timestep']:
        dt = array(variables['idt'], [no], f'{where}.idt')
    else:
        need(variables['idt'] is None, f'{where}: use_timestep=false requires null idt')
        dt = None
    skip = 'none'
    if obj['resnet']:
        if ni == no: skip = 'identity'
        elif 2 * ni == no: skip = 'concat2'
        # In DeepMD resnet=true on incompatible widths is an ordinary layer.
    return {'weights':w, 'bias':b, 'activation':activation(obj['activation_function'],where),
            'residual':skip, 'timestep':dt}


def network(obj, kind, ni, hidden, act, where):
    fitting = kind == 'fitting'
    keys = 'in_dim neuron activation_function resnet_dt precision layers '
    keys += 'out_dim bias_out' if fitting else 'bias'
    schema(obj, 'FittingNetwork' if fitting else 'EmbeddingNetwork', 1 if fitting else 2, keys, where)
    precision(obj, where)
    need(obj['in_dim'] == ni and obj['neuron'] == hidden, f'{where}: conflicting network dimensions')
    need(obj['activation_function'] == act, f'{where}: conflicting activation metadata')
    boolean(obj['resnet_dt'], f'{where}.resnet_dt')
    boolean(obj['bias_out' if fitting else 'bias'], f'{where}.bias')
    if fitting: need(obj['out_dim'] == 1, f'{where}: fitting output must be scalar')
    widths = [ni, *hidden, *([1] if fitting else [])]
    need(isinstance(obj['layers'],list) and len(obj['layers']) == len(widths)-1,
         f'{where}: incorrect layer count')
    layers = [layer(v, widths[i], widths[i+1], f'{where}.layer[{i}]') for i,v in enumerate(obj['layers'])]
    for i,l in enumerate(layers):
        need(l['activation'] == ('linear' if fitting and i == len(layers)-1 else activation(act,where)),
             f'{where}: layer activation conflicts with network metadata')
    if fitting:
        need(layers[-1]['residual'] == 'none' and layers[-1]['timestep'] is None,
             f'{where}: final fitting layer cannot have residual or timestep')
    return {'layers':layers}


def collection(obj, ndim, nt, kind, ni, hidden, act, where):
    schema(obj,'NetworkCollection',1,'ndim ntypes network_type networks',where)
    need(obj['ndim'] == ndim and obj['ntypes'] == nt and obj['network_type'] == kind+'_network',
         f'{where}: incorrect collection layout')
    need(isinstance(obj['networks'],list) and len(obj['networks']) == nt**ndim,
         f'{where}: network count mismatch')
    return [network(v,kind,ni,hidden,act,f'{where}[{i}]') for i,v in enumerate(obj['networks'])]


def widths(value, where):
    need(isinstance(value,list) and 0 < len(value) <= 127, f'{where}: invalid neuron list')
    for x in value: integer(x,where,1); need(x <= 65536, f'{where}: width exceeds native limit')
    return value


def convert(data, geometry):
    need(geometry in ("dpmodel", "tensorflow"), "geometry must explicitly select dpmodel or tensorflow")
    fields(data, 'software version time model', 'envelope',
           ('model_def_script','backend','tf_version','pt_version','@variables'))
    need(data['software'] == 'deepmd-kit' and data['version'] == '3.1.3',
         'only producer deepmd-kit 3.1.3 is supported')
    if '@variables' in data:
        fields(data['@variables'], '', 'envelope.variables', ('min_nbor_dist',))
        if 'min_nbor_dist' in data['@variables']:
            array(data['@variables']['min_nbor_dist'], [], 'min_nbor_dist')
    m = data['model']
    schema(m,'Model',2,'type type_map atom_exclude_types pair_exclude_types rcond preset_out_bias @variables descriptor fitting','model')
    need(m['type'] == 'standard', 'only the standard model is supported')
    need(m['atom_exclude_types'] == [] and m['pair_exclude_types'] == [], 'model exclusions are unsupported')
    types = m['type_map']
    elements = 'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split()
    need(isinstance(types,list) and 0 < len(types) <= 256 and all(t in elements for t in types)
         and len(set(types)) == len(types), 'invalid ordered chemical type_map')
    nt = len(types)
    mv = fields(m['@variables'],'out_bias out_std','model.variables')
    offset = array(mv['out_bias'],[1,nt,1],'out_bias')[0]
    need(array(mv['out_std'],[1,nt,1],'out_std') == [[[1.0] for _ in types]],
         'non-unit out_std is unsupported (DeepMD backends differ in its semantics)')
    d = m['descriptor']
    schema(d,'Descriptor',2,'type rcut rcut_smth sel neuron axis_neuron resnet_dt trainable type_one_side exclude_types env_protection set_davg_zero activation_function precision spin env_mat embeddings @variables type_map','descriptor')
    need(d['type'] == 'se_e2_a', 'only se_e2_a is supported (no hybrid/type embedding/spin)')
    precision(d,'descriptor')
    need(d['type_map'] is None or d['type_map'] == types, 'descriptor type_map mismatch')
    need(d['exclude_types'] == [] and d['spin'] is None and d['env_protection'] == 0,
         'descriptor exclusions, spin, and environment protection are unsupported')
    for k in ('resnet_dt','trainable','type_one_side','set_davg_zero'): boolean(d[k], f'descriptor.{k}')
    rc, rs = number(d['rcut'],'rcut'), number(d['rcut_smth'],'rcut_smth')
    need(0 <= rs < rc and math.isfinite(rc*rc), 'invalid cutoff/smoothing radii')
    env = fields(d['env_mat'],'rcut rcut_smth protection use_exp_switch','env_mat')
    need(env == {'rcut':rc,'rcut_smth':rs,'protection':0.0,'use_exp_switch':False},
         'unsupported or inconsistent environment matrix (including exponential switch)')
    sel = d['sel']
    need(isinstance(sel,list) and len(sel) == nt,'sel/type_map size mismatch')
    for x in sel: integer(x,'sel')
    slots = sum(sel)
    need(0 < slots <= 65536,'sum(sel) outside native capacity range')
    neurons = widths(d['neuron'],'embedding neurons')
    axis = integer(d['axis_neuron'],'axis_neuron',1)
    need(axis <= neurons[-1], 'axis_neuron exceeds embedding width')
    dv = fields(d['@variables'],'davg dstd','descriptor.variables')
    avg = array(dv['davg'],[nt,slots,4],'davg')
    std = array(dv['dstd'],[nt,slots,4],'dstd')
    normalization = []
    for i in range(nt):
        mean, scale = avg[i][0], std[i][0]
        need(all(row == mean for row in avg[i]) and all(row == scale for row in std[i]),
             f'type {i}: nonuniform per-slot normalization is unsupported')
        need(mean[1:] == [0.0]*3 and all(s>0 for s in scale) and scale[1] == scale[2] == scale[3],
             f'type {i}: angular normalization must be isotropic with zero mean')
        normalization.append({'mean':mean,'std':scale})
    ndim = 1 if d['type_one_side'] else 2
    embeddings = collection(d['embeddings'],ndim,nt,'embedding',1,neurons,d['activation_function'],'embedding')
    # DeepMD NetworkCollection uses the FIRST tuple component as least significant.
    embedding = [[embeddings[j if ndim == 1 else i+nt*j] for j in range(nt)] for i in range(nt)]
    f = m['fitting']
    schema(f,'Fitting',4,'type var_name ntypes dim_descrpt neuron resnet_dt numb_fparam numb_aparam dim_case_embd default_fparam rcond activation_function precision mixed_types exclude_types @variables type_map tot_ener_zero trainable layer_name use_aparam_as_mask spin dim_out atom_ener nets','fitting')
    precision(f,'fitting')
    need(f['type'] == 'ener' and f['var_name'] == 'energy' and f['dim_out'] == 1 and f['ntypes'] == nt,
         'only scalar energy fitting with matching ntypes is supported')
    need(f['dim_descrpt'] == neurons[-1]*axis, 'fitting descriptor size mismatch')
    need(f['type_map'] is None or f['type_map'] == types,'fitting type_map mismatch')
    for k in ('numb_fparam','numb_aparam','dim_case_embd'):
        need(type(f[k]) is int and f[k] == 0, f'{k} is unsupported')
    for k in ('mixed_types','tot_ener_zero','use_aparam_as_mask'):
        need(f[k] is False, f'{k}=true is unsupported')
    need(f['default_fparam'] is None and f['spin'] is None and f['exclude_types'] == [],
         'fitting defaults/spin/exclusions are unsupported')
    need(f['atom_ener'] is None or f['atom_ener'] == [] or f['atom_ener'] == [None]*nt, 'atom_ener vacuum subtraction is unsupported')
    fv = fields(f['@variables'],'bias_atom_e case_embd fparam_avg fparam_inv_std aparam_avg aparam_inv_std','fitting.variables')
    for k,v in fv.items():
        if k != 'bias_atom_e': need(v is None, f'{k} must be null')
    bias = array(fv['bias_atom_e'],[nt,1],'bias_atom_e')
    fitting = collection(f['nets'],1,nt,'fitting',neurons[-1]*axis,widths(f['neuron'],'fitting neurons'),f['activation_function'],'fitting networks')
    if geometry == 'tensorflow':
        # TensorFlow AttrValue.float stores environment-op radii as IEEE f32,
        # even when model weights, coordinates and environment math are f64.
        rc, rs = (struct.unpack('<f', struct.pack('<f', value))[0] for value in (rc,rs))
        need(0 <= rs < rc and math.isfinite(rc), 'TensorFlow cutoff rounding collapses/overflows radii')
    result = {'format':'omnimd-native-se-e2-a','version':1,'precision':'float64','units':'EvAngstrom',
              'type_map':types,'descriptor':{'kind':'se_e2_a','rcut':rc,'rcut_smth':rs,'sel':sel,
                  'axis_neuron':axis,'normalization':normalization,'embedding':embedding},
              'fitting':fitting,'atom_energy_bias':[bias[i][0]+offset[i][0] for i in range(nt)]}
    # Validate the entire result is JSON-finite before exposing any output file.
    json.dumps(result,allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('output', help='new native JSON file; existing files are not overwritten')
    parser.add_argument('--units', required=True, choices=['EvAngstrom'],
                        help='explicit attestation of the source training units; DeepMD does not store reliable units')
    parser.add_argument('--geometry', required=True, choices=['dpmodel','tensorflow'],
                        help='dpmodel preserves saved radii; tensorflow matches the f32 op attributes used by a frozen TF model')
    args = parser.parse_args()
    try:
        result = convert(load_data(args.input), args.geometry)
        encoded = json.dumps(result,indent=2,allow_nan=False)+'\n'
        with Path(args.output).open('x',encoding='utf-8',newline='\n') as out: out.write(encoded)
        print(json.dumps({'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                          'output_sha256':hashlib.sha256(encoded.encode()).hexdigest(),
                          'geometry':args.geometry,'producer':'deepmd-kit 3.1.3','format':result['format'],'type_map':result['type_map']}))
    except (ConversionError, ValueError, TypeError, KeyError, OSError, ImportError, RecursionError, OverflowError) as exc:
        print(f'conversion failed: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
