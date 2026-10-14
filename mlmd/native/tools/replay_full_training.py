"""Original tiny synthetic E/F training and exact checkpoint-resume replay.
Python stdlib only. Teacher labels, mixed AD and Adam all run through Rust binaries.
This is an implementation check, not evidence of physical accuracy/generalization.
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
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run(command, **kwargs):
    return subprocess.run([str(s) for s in command], check=True, text=True, encoding='utf-8',
                          stdout=subprocess.PIPE, **kwargs).stdout


def perturb(model):
    index = 0
    networks = [n for row in model['descriptor']['embedding'] for n in row] + model['fitting']
    for net in networks:
        for layer in net['layers']:
            for vector in [*layer['weights'], layer['bias'], layer['timestep'] or []]:
                for i in range(len(vector)):
                    index += 1
                    vector[i] += 0.12*math.sin(index*0.63)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-exe', type=Path, required=True)
    parser.add_argument('--train-exe', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    evaluator, trainer = args.eval_exe.resolve(), args.train_exe.resolve()
    for exe in (evaluator, trainer):
        if not exe.is_file():
            raise ValueError('native binaries must be built first')
    output.mkdir(parents=True, exist_ok=False)
    initial = json.loads((root/'fixtures/full-tanh-v1.json').read_text())
    teacher = deepcopy(initial)
    parameters = perturb(teacher)
    write(output/'initial.json', initial)
    write(output/'teacher.json', teacher)
    generator = random.Random(20261014)
    base = json.loads((root/'fixtures/frame.jsonl').read_text())
    frames = []
    for i in range(24):
        frame = deepcopy(base)
        frame['cell'] = None if i % 2 == 0 else [[3.8,0,0],[.3,3.9,0],[.1,.2,4.]]
        for r in frame['positions']:
            for axis in range(3):
                r[axis] += generator.uniform(-0.2,0.2)
        if i % 3 == 0:
            frame['positions'].pop()
            frame['species'].pop()
        frames.append(frame)
    lines = ''.join(json.dumps(f, allow_nan=False)+'\n' for f in frames)
    (output/'frames.jsonl').write_text(lines, encoding='utf-8')
    targets = [json.loads(s) for s in run([evaluator,output/'teacher.json'], input=lines).splitlines()]
    records = [{'kind':'header','header':{'schema':'omnimd-dataset-jsonl-v1','type_map':initial['type_map'],
                'units':'EvAngstrom','virial_convention':'MinusDEpsilonRow',
                'source':'original synthetic full-network teacher; data seed 20261014; no physical accuracy claim',
                'selected_sets':['set.000']}}]
    for i,(frame,label) in enumerate(zip(frames,targets)):
        records.append({'kind':'frame','data':{'index':i,'id':f'set.000/{i}','frame':frame,
                        'labels':{k:label[k] for k in ('energy','forces','virial')}}})
    records.append({'kind':'completed','frames':len(frames),'source_sha256':
                    {'teacher.json':digest(output/'teacher.json'),'replay_full_training.py':digest(__file__)}})
    (output/'dataset.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    request = {'schema':'omnimd-native-full-train-v1','initial_model':'initial.json',
        'dataset':'dataset.jsonl','output_directory':'continuous','config':{
            'split':{'mode':'seeded','seed':20261015,'validation_fraction':.25},'steps':300,
            'energy_weight':1.,'force_weight':1.,'energy_scale_ev_per_atom':.1,'force_scale_ev_per_angstrom':.1,
            'learning_rate':.005,'beta1':.9,'beta2':.999,'epsilon':1e-8,
            'gradient_clip_norm':10.,'checkpoint_every':150}}
    write(output/'continuous.request.json', request)
    run([trainer,output/'continuous.request.json'])
    resumed = deepcopy(request)
    resumed.pop('initial_model')
    resumed.update(checkpoint='continuous/checkpoint-step-150.json', output_directory='resumed')
    resumed['config']['steps'] = 150
    write(output/'resumed.request.json',resumed)
    run([trainer,output/'resumed.request.json'])
    a = json.loads((output/'continuous/checkpoint.json').read_text())
    b = json.loads((output/'resumed/checkpoint.json').read_text())
    for key in ('model','initial_model','first_moment','second_moment','step','training_indices',
                'validation_indices','dataset_sha256','parameter_layout_sha256'):
        if a[key] != b[key]:
            raise AssertionError(f'serialized resume changed {key}')
    report = json.loads((output/'continuous/report.json').read_text())
    for split in ('training','validation'):
        old,new = report['initial_'+split],report['final_'+split]
        for key in ('normalized_ef_objective','energy_per_atom_rmse_ev','force_component_rmse_ev_per_angstrom'):
            if not new[key] < old[key]*.5:
                raise AssertionError(f'{split} {key} did not improve at least twofold: {old[key]} -> {new[key]}')
    for group,change in report['parameter_changes_from_initial'].items():
        if change['changed'] != change['parameters'] or change['delta_l2'] <= 1e-3:
            raise AssertionError(f'group not actually trained: {group}: {change}')
    predictions = [json.loads(s) for s in run([evaluator,output/'continuous/model.json'],input=lines).splitlines()]
    (output/'trained.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in predictions),encoding='utf-8')
    errors = dict(energy=0.,forces=0.,virial=0.)
    for ref,actual in zip(targets,predictions):
        errors['energy'] = max(errors['energy'],abs(ref['energy']-actual['energy']))
        for field in ('forces','virial'):
            errors[field] = max(errors[field],max(abs(a-b) for r,s in zip(ref[field],actual[field]) for a,b in zip(r,s)))
    receipt = {'scope':'full supported tanh network; original synthetic teacher; no physical accuracy claim',
        'data_seed':20261014,'split_seed':20261015,'frames':len(frames),'train_frames':len(report['training_frames']),
        'validation_frames':len(report['validation_frames']),'parameters':parameters,'steps':300,
        'initial_training':report['initial_training'],'final_training':report['final_training'],
        'initial_validation':report['initial_validation'],'final_validation':report['final_validation'],
        'parameter_changes_from_initial':report['parameter_changes_from_initial'],
        'resume_150_plus_150_equals_continuous_300':True,
        'exported_model_max_abs_error_to_teacher':errors,
        'native_v1_reload_max_abs_energy_force_virial':report['native_v1_reload_max_abs_energy_force_virial'],
        'sha256':{name:digest(p) for name,p in [('dataset.jsonl',output/'dataset.jsonl'),
            ('initial.json',output/'initial.json'),('teacher.json',output/'teacher.json'),
            ('model.json',output/'continuous/model.json'),('report.json',output/'continuous/report.json'),
            ('checkpoint.json',output/'continuous/checkpoint.json'),('eval_executable',evaluator),('train_executable',trainer)]}}
    write(output/'acceptance.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
