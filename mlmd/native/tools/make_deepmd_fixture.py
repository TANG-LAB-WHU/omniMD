"""Create a small untrained DeepMD 3.1.3 model using official APIs.

Run only in an already provisioned DeepMD 3.1.3 + TensorFlow environment.
This does not train, load a checkpoint, use pickle, or fetch any data.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    args = parser.parse_args()
    import deepmd
    import numpy as np
    from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeA
    from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet
    from deepmd.dpmodel.model.ener_model import EnergyModel
    from deepmd.dpmodel.utils.serialization import save_dp_model
    from deepmd.tf.utils.serialization import deserialize_to_file
    if deepmd.__version__ != '3.1.3':
        raise RuntimeError('fixture recipe requires DeepMD-kit 3.1.3')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    types = ['O', 'H']
    descriptor_config = dict(rcut=3.0, rcut_smth=2.1, sel=[32,32], neuron=[2,4,4],
                             axis_neuron=2, resnet_dt=True, type_one_side=False,
                             precision='float64', type_map=types, seed=20261014)
    fitting_config = dict(ntypes=2, dim_descrpt=8, neuron=[5,5], resnet_dt=True,
                          precision='float64', type_map=types, seed=20261015)
    descriptor = DescrptSeA(**descriptor_config)
    fitting = EnergyFittingNet(**fitting_config)
    descriptor.davg[0,:,:] = np.array([0.13,0,0,0])
    descriptor.davg[1,:,:] = np.array([-0.08,0,0,0])
    descriptor.dstd[0,:,:] = np.array([0.7,0.9,0.9,0.9])
    descriptor.dstd[1,:,:] = np.array([1.1,0.8,0.8,0.8])
    fitting.bias_atom_e[:] = np.array([[-0.7],[0.2]])
    model = EnergyModel(descriptor=descriptor, fitting=fitting, type_map=types)
    serialized = model.serialize()
    # Keep both offset locations nonzero so the converter must preserve both.
    serialized['@variables']['out_bias'][:] = np.array([[[0.04],[-0.03]]])
    definition = {'type':'standard', 'type_map':types,
                  'descriptor':{'type':'se_e2_a', **{k:v for k,v in descriptor_config.items() if k != 'type_map'}},
                  'fitting_net':{'type':'ener', **{k:v for k,v in fitting_config.items()
                                                 if k not in ('ntypes','dim_descrpt','type_map')}}}
    data = {'model':serialized, 'model_def_script':definition}
    save_dp_model(str(output/'synthetic.dp'), deepcopy(data))
    save_dp_model(str(output/'synthetic.yaml'), deepcopy(data))
    deserialize_to_file(str(output/'synthetic.pb'), data)
    # Diagnostic tree contains values/shapes only, never pickled Python objects.
    def plain(value):
        if isinstance(value, np.ndarray):
            return {'dtype':str(value.dtype), 'shape':list(value.shape)}
        if isinstance(value, dict): return {k:plain(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)): return [plain(v) for v in value]
        return value
    (output/'schema.json').write_text(json.dumps(plain(data),indent=2)+'\n')
    print(json.dumps({'deepmd_version':deepmd.__version__, 'output_dir':str(output),
                      'model_kind':'synthetic untrained se_e2_a', 'seeds':[20261014,20261015]}))


if __name__ == '__main__':
    main()
