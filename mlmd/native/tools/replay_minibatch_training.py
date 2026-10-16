"""Original synthetic deterministic minibatch E/F/V and checkpoint replay.
Python stdlib only; generated datasets, weights and raw outputs belong in ignored storage.
This verifies implementation, not physical accuracy or full-batch convergence equivalence.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import time
from replay_full_training import digest, perturb, run, write


def values(model):
    return [x for net in [n for row in model['descriptor']['embedding'] for n in row]+model['fitting']
            for layer in net['layers'] for row in [*layer['weights'],layer['bias'],layer['timestep'] or []] for x in row]


def comparable(state):
    result = deepcopy(state)
    result['config'].pop('steps', None)
    result['config'].pop('fixed_geometry_cache', None)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-exe', type=Path, required=True)
    p.add_argument('--train-exe', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--cuda-gradient', choices=['hyperdual','reverse'], default='hyperdual')
    p.add_argument('--fixed-geometry-cache', action='store_true')
    p.add_argument('--steps', type=int, default=31)
    a = p.parse_args()
    if a.steps < 5 or a.steps > 100000:
        raise ValueError('steps must be 5..100000')
    if a.device != 'cpu' and not (a.device.startswith('cuda:') and a.device[5:].isascii() and a.device[5:].isdecimal()):
        raise ValueError('device must be cpu or explicit cuda:N')
    if a.device == 'cpu' and (a.fixed_geometry_cache or a.cuda_gradient == 'reverse'):
        raise ValueError('CUDA options require explicit CUDA device')
    root = Path(__file__).resolve().parents[1]
    out = a.output_dir.resolve()
    evaluator, trainer = a.eval_exe.resolve(), a.train_exe.resolve()
    out.mkdir(parents=True, exist_ok=False)
    initial = json.loads((root/'fixtures/full-tanh-v1.json').read_text())
    teacher = deepcopy(initial)
    parameters = perturb(teacher)
    write(out/'initial.json', initial)
    write(out/'teacher.json', teacher)
    base = json.loads((root/'fixtures/frame.jsonl').read_text())
    frames = []
    for i in range(8):
        f = deepcopy(base)
        f['cell'] = None if i % 2 == 0 else [[3.8,0,0],[.3,3.9,0],[.1,.2,4.]]
        for atom,r in enumerate(f['positions']):
            for axis in range(3):
                r[axis] += .19*math.sin((i+1)*(atom*3+axis+1)*.618)
        if i % 3 == 0:
            f['positions'].pop()
            f['species'].pop()
        frames.append(f)
    lines = ''.join(json.dumps(f,allow_nan=False)+'\n' for f in frames)
    labels = [json.loads(line) for line in run([evaluator,out/'teacher.json'],input=lines).splitlines()]
    records = [{'kind':'header','header':{'schema':'omnimd-dataset-jsonl-v1','type_map':initial['type_map'],
        'units':'EvAngstrom','virial_convention':'MinusDEpsilonRow','source':'original deterministic synthetic minibatch teacher; no physical accuracy claim','selected_sets':['set.000']}}]
    records += [{'kind':'frame','data':{'index':i,'id':f'set.000/{i}','frame':f,'labels':{k:label[k] for k in ('energy','forces','virial')}}}
                for i,(f,label) in enumerate(zip(frames,labels))]
    records += [{'kind':'completed','frames':len(frames),'source_sha256':{'teacher.json':digest(out/'teacher.json'),'replay_minibatch_training.py':digest(__file__)}}]
    (out/'dataset.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    req = {'schema':'omnimd-native-full-train-v1','device':a.device,'initial_model':'initial.json','dataset':'dataset.jsonl','output_directory':'continuous',
           'config':{'split':{'mode':'explicit','validation_ids':['set.000/2','set.000/6','set.000/7']},'steps':a.steps,
            'minibatch':{'batch_size':2,'seed':20261016,'order':'shuffle_each_epoch'},
            'energy_weight':1.,'force_weight':1.,'virial_loss':{'weight':.7,'scale_ev_per_atom':.2},
            'energy_scale_ev_per_atom':.1,'force_scale_ev_per_angstrom':.1,
            'learning_rate':.005,'beta1':.9,'beta2':.999,'epsilon':1e-8,'gradient_clip_norm':10.,'checkpoint_every':1}}
    if a.cuda_gradient == 'reverse': req['config']['cuda_gradient'] = 'reverse'
    if a.fixed_geometry_cache: req['config']['fixed_geometry_cache'] = True
    write(out/'continuous.request.json',req)
    start = time.perf_counter()
    run([trainer,out/'continuous.request.json'])
    process_ms = (time.perf_counter()-start)*1000
    state = json.loads((out/'continuous/checkpoint.json').read_text())
    report = json.loads((out/'continuous/report.json').read_text())
    expected_frames = 5*(a.steps//3)+2*(a.steps%3)
    mb = report['minibatch']
    assert mb['frames_processed'] == expected_frames
    assert mb['partial_batches'] == a.steps//3
    assert mb['end']['epoch'] == a.steps//3 and mb['end']['cursor'] == 2*(a.steps%3)
    assert state['schema'] == 'omnimd-native-full-checkpoint-v5'
    assert [f['index'] for f in report['training_frames']] == [0,1,3,4,5]
    assert report['dataset_sha256'] == digest(out/'dataset.jsonl')
    for split in (2,3,4):
        resumed = deepcopy(req)
        resumed.pop('initial_model')
        resumed.update(checkpoint=f'continuous/checkpoint-step-{split}.json',output_directory=f'resume-{split}')
        resumed['config']['steps'] = a.steps-split
        if a.device != 'cpu': resumed['config']['fixed_geometry_cache'] = not a.fixed_geometry_cache
        write(out/f'resume-{split}.request.json',resumed)
        run([trainer,out/f'resume-{split}.request.json'])
        actual = json.loads((out/f'resume-{split}/checkpoint.json').read_text())
        assert comparable(actual) == comparable(state), f'resume at {split} changed state'
    rejected = []
    for corruption in ('missing_sampler','changed_order','changed_seed'):
        broken = deepcopy(state)
        if corruption == 'missing_sampler': broken.pop('sampler')
        if corruption == 'changed_order': broken['sampler']['order'][0] = 2
        if corruption == 'changed_seed': broken['sampler']['seed'] += 1
        write(out/f'{corruption}.json',broken)
        bad = deepcopy(req)
        bad.pop('initial_model')
        bad.update(checkpoint=f'{corruption}.json',output_directory=f'rejected-{corruption}')
        bad['config']['steps'] = 1
        write(out/f'{corruption}.request.json',bad)
        result = subprocess.run([str(trainer),str(out/f'{corruption}.request.json')],capture_output=True,text=True)
        assert result.returncode != 0 and not (out/f'rejected-{corruption}').exists()
        rejected.append(corruption)
    comparison = None
    if a.device != 'cpu':
        cpu = deepcopy(req)
        cpu.update(device='cpu',output_directory='cpu')
        cpu['config'].pop('cuda_gradient',None)
        cpu['config'].pop('fixed_geometry_cache',None)
        write(out/'cpu.request.json',cpu)
        run([trainer,out/'cpu.request.json'])
        reference = json.loads((out/'cpu/checkpoint.json').read_text())
        maximum = max(abs(x-y) for x,y in zip(values(state['model'])+state['first_moment']+state['second_moment'],values(reference['model'])+reference['first_moment']+reference['second_moment']))
        assert maximum < 1e-8
        comparison = {'parameter_moment_max_abs':maximum}
    receipt = {'scope':'bounded deterministic minibatch implementation replay; no physical accuracy, scaling or convergence-equivalence claim',
        'device':a.device,'execution':report['execution'],'parameters':parameters,'steps':a.steps,'training_indices':[0,1,3,4,5],
        'checkpoint_schema':state['schema'],'minibatch':mb,'timing':report['timing'],'complete_training_process_ms':process_ms,
        'timing_scope':'one process including IO, initialization, 4 assessments, each-step checkpoint writes and export; excludes teacher, resumes and CPU comparison',
        'processed_training_frames_per_gradient_second':expected_frames/(report['timing']['gradient_ms']/1000),
        'fixed_geometry_cache':report.get('fixed_geometry_cache'),'exact_serialized_resume_steps':[2,3,4],
        'cache_toggled_on_resume':a.device!='cpu','corrupt_checkpoint_rejected_before_output':rejected,'cpu_comparison':comparison,
        'initial_training':report['initial_training'],'final_training':report['final_training'],'initial_validation':report['initial_validation'],'final_validation':report['final_validation'],
        'native_v1_reload_max_abs_energy_force_virial':report['native_v1_reload_max_abs_energy_force_virial'],
        'sha256':{name:digest(path) for name,path in [('dataset',out/'dataset.jsonl'),('model',out/'continuous/model.json'),('checkpoint',out/'continuous/checkpoint.json'),('trainer',trainer),('evaluator',evaluator),('script',__file__)]}}
    write(out/'acceptance.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
