"""Replay tiny native final-readout training using original synthetic labels.

No training framework, downloads, installed packages, or private inputs. Python
stdlib only; the actual optimizer and analytic derivatives run in native Rust.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def run(command, **kwargs):
    return subprocess.run([str(s) for s in command],check=True,text=True,encoding='utf-8',
                          stdout=subprocess.PIPE,**kwargs).stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-exe',type=Path,required=True)
    parser.add_argument('--train-exe',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output_dir.resolve()
    evaluator,trainer=args.eval_exe.resolve(),args.train_exe.resolve()
    for exe in (evaluator,trainer):
        if not exe.is_file(): raise ValueError('native binaries must be built first')
    output.mkdir(parents=True,exist_ok=False)
    teacher=json.loads((root/'fixtures/synthetic-v1.json').read_text())
    initial=deepcopy(teacher)
    for kind,net in enumerate(teacher['fitting']):
        for i,weight in enumerate(net['layers'][-1]['weights']):
            weight[0]=0.8*math.sin((kind*7+i+1)*0.7)
    for net in initial['fitting']:
        for weight in net['layers'][-1]['weights']: weight[0]=0.0
    write(output/'teacher.json',teacher)
    write(output/'initial.json',initial)
    generator=random.Random(20261014)
    base=json.loads((root/'fixtures/frame.jsonl').read_text())
    frames=[]
    for i in range(48):
        frame=deepcopy(base)
        if i%2==0: frame['cell']=None
        for r in frame['positions']:
            for axis in range(3): r[axis]+=generator.uniform(-0.24,0.24)
        if i%3==0:
            frame['positions'].pop()
            frame['species'].pop()
        frames.append(frame)
    lines=''.join(json.dumps(f,allow_nan=False)+'\n' for f in frames)
    (output/'frames.jsonl').write_text(lines,encoding='utf-8')
    targets=[json.loads(line) for line in run([evaluator,output/'teacher.json'],input=lines).splitlines()]
    (output/'teacher.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in targets),encoding='utf-8')
    header={'kind':'header','header':{'schema':'omnimd-dataset-jsonl-v1','type_map':teacher['type_map'],
            'units':'EvAngstrom','virial_convention':'MinusDEpsilonRow',
            'source':'original synthetic native teacher; data seed 20261014; no physical accuracy claim',
            'selected_sets':['set.000']}}
    records=[header]
    for i,(frame,label) in enumerate(zip(frames,targets)):
        records.append({'kind':'frame','data':{'index':i,'id':f'set.000/{i}','frame':frame,
                        'labels':{k:label[k] for k in ('energy','forces','virial')}}})
    records.append({'kind':'completed','frames':len(frames),'source_sha256':
                    {'teacher.json':digest(output/'teacher.json'),'replay_readout_training.py':digest(__file__)}})
    (output/'dataset.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    request={'schema':'omnimd-native-readout-train-v1','initial_model':'initial.json',
             'dataset':'dataset.jsonl','output_directory':'trained',
             'config':{'split':{'mode':'seeded','seed':20261015,'validation_fraction':0.25},
                       'fit_bias':False,'energy_weight':1.0,'force_weight':1.0,
                       'energy_scale_ev_per_atom':1.0,'force_scale_ev_per_angstrom':0.01,
                       'ridge_inverse_ev2':0.0,'rank_rtol':1e-12}}
    write(output/'train.request.json',request)
    summary=json.loads(run([trainer,output/'train.request.json']))
    report=json.loads((output/'trained/report.json').read_text())
    predictions=[json.loads(line) for line in run([evaluator,output/'trained/model.json'],input=lines).splitlines()]
    (output/'trained.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in predictions),encoding='utf-8')
    errors={'energy':0.0,'forces':0.0,'virial':0.0}
    for ref,actual in zip(targets,predictions):
        errors['energy']=max(errors['energy'],abs(ref['energy']-actual['energy']))
        for field in ('forces','virial'):
            errors[field]=max(errors[field],max(abs(a-b) for r,s in zip(ref[field],actual[field]) for a,b in zip(r,s)))
    if report['final_training']['normalized_ef_objective']>=report['initial_training']['normalized_ef_objective']*1e-10:
        raise AssertionError('training did not reduce the actual normalized objective')
    if errors['energy']>1e-8 or errors['forces']>1e-8 or errors['virial']>1e-8:
        raise AssertionError(f'exported-model predictions do not reproduce teacher: {errors}')
    trained_model=json.loads((output/'trained/model.json').read_text())
    frozen=deepcopy(trained_model)
    for net,old in zip(frozen['fitting'],initial['fitting']):
        net['layers'][-1]['weights']=old['layers'][-1]['weights']
    if frozen!=initial: raise AssertionError('parameters outside selected final weights changed')
    receipt={'scope':'actual native final-readout training only; not full-network DeepMD training',
             'data_seed':20261014,'split_seed':20261015,'frames':len(frames),
             'train_frames':len(report['training_frames']),'validation_frames':len(report['validation_frames']),
             'data_rank':report['data_rank'],'parameters':len(report['parameters']),
             'initial_training':report['initial_training'],'final_training':report['final_training'],
             'initial_validation':report['initial_validation'],'final_validation':report['final_validation'],
             'exported_model_max_abs_error_to_teacher':errors,
             'native_v1_reload_max_abs_energy_force_virial':report['native_v1_reload_max_abs_energy_force_virial'],
             'frozen_parameters_unchanged':True,
             'sha256':{name:digest(path) for name,path in [('dataset.jsonl',output/'dataset.jsonl'),
                        ('teacher.json',output/'teacher.json'),('initial.json',output/'initial.json'),
                        ('model.json',output/'trained/model.json'),('report.json',output/'trained/report.json'),
                        ('eval_executable',evaluator),('train_executable',trainer)]}}
    write(output/'acceptance.json',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
