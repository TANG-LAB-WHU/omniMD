"""Focused integrated engine -> CUDA trainer acceptance; requires an exclusive GPU window.
Reuses original synthetic replay inputs, never retries a failed command, and
keeps generated data/weights/raw logs outside the repository.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def frozen(model):
    value = deepcopy(model)
    for net in [n for row in value['descriptor']['embedding'] for n in row] + value['fitting']:
        for layer in net['layers']:
            layer['weights'] = [[None for _ in row] for row in layer['weights']]
            layer['bias'] = [None for _ in layer['bias']]
            if layer['timestep'] is not None:
                layer['timestep'] = [None for _ in layer['timestep']]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    engine, evaluator, fixtures, work = [p.resolve() for p in
        (args.engine, args.evaluator, args.fixtures, args.work)]
    trainer = engine.with_name('omnimd-native-train'+engine.suffix)
    for p in (engine, evaluator, trainer):
        if not p.is_file():
            raise ValueError(f'missing binary: {p}')
    started = datetime.now(timezone.utc).isoformat()
    work.mkdir(parents=True, exist_ok=False)
    for name in ('initial.json','dataset.jsonl','frames.jsonl'):
        shutil.copyfile(fixtures/name, work/name)
    commands = []

    def run(label, command, input_text=None, expected_error=None):
        p = subprocess.run([str(s) for s in command], input=input_text,
                           text=True, capture_output=True, timeout=120)
        (work/(label+'.stdout')).write_text(p.stdout, encoding='utf-8')
        (work/(label+'.stderr')).write_text(p.stderr, encoding='utf-8')
        commands.append({'label':label, 'command':[str(s) for s in command], 'exit_code':p.returncode})
        write(work/'commands.json', commands)
        if expected_error is not None:
            if p.returncode == 0 or expected_error not in p.stderr:
                raise AssertionError(f'{label}: unexpected rejection: {p.stderr}')
        elif p.returncode != 0:
            raise RuntimeError(f'{label}: command failed; no retry: {p.stderr}')
        return p.stdout

    capabilities = json.loads(run('capabilities', [engine,'capabilities']))
    assert capabilities['training']['cuda']['full_network'] is True
    req = read(fixtures/'continuous.request.json')
    assert req['schema'] == 'omnimd-native-full-train-v1'
    req.update(device='cuda:0', initial_model='initial.json', dataset='dataset.jsonl', output_directory='continuous')
    req['config'].update(steps=6, checkpoint_every=3)
    write(work/'continuous.request.json',req)
    run('continuous', [engine,'train-native',work/'continuous.request.json'])
    resume = deepcopy(req)
    resume.pop('initial_model')
    resume.update(checkpoint='continuous/checkpoint-step-3.json', output_directory='resumed')
    resume['config']['steps']=3
    write(work/'resumed.request.json',resume)
    run('resumed',[engine,'train-native',work/'resumed.request.json'])
    a = read(work/'continuous/checkpoint.json')
    b = read(work/'resumed/checkpoint.json')
    mid = read(work/'continuous/checkpoint-step-3.json')
    assert a['step'] == b['step'] == 6 and mid['step'] == 3
    for key in ('model','initial_model','first_moment','second_moment','step',
                'training_indices','validation_indices','dataset_sha256',
                'parameter_layout_sha256','schema','execution'):
        assert a[key] == b[key], f'exact resume changed {key}'
    for checkpoint, expected in ((a,req['config']),(b,resume['config'])):
        assert checkpoint['config'] == expected
    assert a['schema'] == 'omnimd-native-full-checkpoint-v2'
    assert a['execution']['backend'] == 'cuda_hyperdual_f64_v1'
    assert a['execution']['visible_device'] == 0
    assert len(a['execution']['program_sha256']) == 64 and a['execution']['driver_version'] > 0
    assert a['dataset_sha256'] == sha(work/'dataset.jsonl')
    assert any(x != 0 for x in a['first_moment']) and any(x != 0 for x in a['second_moment'])
    assert frozen(a['model']) == frozen(read(work/'initial.json'))
    reports = [read(work/name/'report.json') for name in ('continuous','resumed')]
    for report, start in zip(reports,(0,3)):
        assert report['step_start'] == start and report['step_end'] == 6
        assert report['execution'] == a['execution']
        assert report['timing']['gradient_ms'] > 0 and report['timing']['adam_ms'] > 0
        assert report['native_v1_reload_max_abs_energy_force_virial'] == [0,0,0]
        assert report['source_sha256'] == reports[0]['source_sha256']
        for group,change in report['parameter_changes_from_initial'].items():
            assert change['changed'] == change['parameters'] and change['delta_l2'] > 0, group
    assert (work/'continuous/model.json').read_bytes() == (work/'resumed/model.json').read_bytes()
    lines = (work/'frames.jsonl').read_text(encoding='utf-8')
    predictions = [json.loads(s) for s in run('export-eval',[evaluator,work/'continuous/model.json'],lines).splitlines()]
    frames = [json.loads(s) for s in lines.splitlines()]
    assert len(predictions) == len(frames) == 24
    for value,frame in zip(predictions,frames):
        assert len(value['forces']) == len(frame['positions'])
        assert len(value['virial']) == 3 and all(len(r)==3 for r in value['virial'])
        assert all(math.isfinite(x) for x in [value['energy'],*[x for r in value['forces'] for x in r],*[x for r in value['virial'] for x in r]])
    wrong = deepcopy(resume)
    wrong.update(device='cpu',output_directory='rejected-backend')
    write(work/'wrong-backend.request.json',wrong)
    run('reject-backend',[engine,'train-native',work/'wrong-backend.request.json'],expected_error='checkpoint training backend/device changed')
    assert not (work/'rejected-backend').exists()
    receipt = {
        'schema':'omnimd-cuda-full-training-integration-v1', 'status':'passed',
        'scope':'Six-step engine CLI integration only; original synthetic 24-frame fixture. Independent gradient and 300-step acceptance remain in B receipt. No performance or chemical-accuracy claim.',
        'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'source_native_commit':'f4dbf62bb4acb97e8f7e12e62de625e08af07a05',
        'integrated_native_commit':'02b5457',
        'execution':a['execution'],'execution_scope':reports[0]['execution_scope'],
        'frames':24,'training_frames':len(a['training_indices']),'validation_frames':len(a['validation_indices']),
        'continuous_steps':6,'resume_from_step':3,'resume_additional_steps':3,
        'model_and_moments_exact_resume':True,'frozen_model_fields_unchanged':True,
        'backend_change_rejected_without_output':True,
        'parameter_changes_from_initial':reports[0]['parameter_changes_from_initial'],
        'initial_training':reports[0]['initial_training'],'final_training':reports[0]['final_training'],
        'initial_validation':reports[0]['initial_validation'],'final_validation':reports[0]['final_validation'],
        'native_v1_reload_max_abs_energy_force_virial':reports[0]['native_v1_reload_max_abs_energy_force_virial'],
        'sha256':{name:sha(path) for name,path in [('engine',engine),('trainer',trainer),('evaluator',evaluator),('script',Path(__file__)),('dataset',work/'dataset.jsonl'),('initial_model',work/'initial.json'),('export_model',work/'continuous/model.json'),('checkpoint',work/'continuous/checkpoint.json')]},
        'checks': [{'label':c['label'],'exit_code':c['exit_code']} for c in commands],
        'source_sha256':reports[0]['source_sha256']}
    write(work/'acceptance.json',receipt)
    print(json.dumps(receipt,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
