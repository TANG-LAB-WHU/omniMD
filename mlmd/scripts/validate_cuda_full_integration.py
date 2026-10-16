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
    parser.add_argument('--cuda-gradient', choices=('hyperdual','reverse'), default='hyperdual')
    parser.add_argument('--fixed-geometry-cache', action='store_true', help='Check cached training plus both cache-toggle restart directions')
    parser.add_argument('--legacy-uncached-work', type=Path, help='Required with cache: previous same-executor uncached six-step acceptance')
    parser.add_argument('--virial-loss', action='store_true', help='Enable weight 1, scale 0.1 eV/atom and a short CPU comparison')
    parser.add_argument('--legacy-ef-work', type=Path, help='Optional previous six-step E/F integration output for CUDA-v2 continuation check')
    args = parser.parse_args()
    if args.fixed_geometry_cache and not args.legacy_uncached_work:
        parser.error('--fixed-geometry-cache requires --legacy-uncached-work')
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
    if args.cuda_gradient=='reverse':
        assert capabilities['training']['cuda']['gradient_executors']['reverse']=='explicit opt-in'
        req['config']['cuda_gradient']='reverse'
    else:
        req['config'].pop('cuda_gradient',None)
    if args.virial_loss:
        assert capabilities['training']['virial_loss']['full_network'] is True
        req['config']['virial_loss']={'weight':1.0,'scale_ev_per_atom':0.1}
    else:
        req['config'].pop('virial_loss',None)
    if args.fixed_geometry_cache:
        assert capabilities['training']['cuda']['fixed_geometry_cache']['cpu_only'] is False
        req['config']['fixed_geometry_cache']=True
    else:
        req['config'].pop('fixed_geometry_cache',None)
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
    assert a['schema'] == ('omnimd-native-full-checkpoint-v4' if args.cuda_gradient=='reverse' else 'omnimd-native-full-checkpoint-v3' if args.virial_loss else 'omnimd-native-full-checkpoint-v2')
    if args.virial_loss:
        assert a['loss_contract'] == b['loss_contract'] == 'energy-force-row-virial-per-atom-nine-v1'
    assert a['execution']['backend'] == ('cuda_reverse_f64_v1' if args.cuda_gradient=='reverse' else 'cuda_hyperdual_f64_v1')
    if args.cuda_gradient=='reverse' and not args.virial_loss:
        assert a['loss_contract']==b['loss_contract']=='energy-force-per-atom-v1'
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
    cache_stats=None
    if args.fixed_geometry_cache:
        cache_stats=[report['fixed_geometry_cache'] for report in reports]
        for stats,steps in zip(cache_stats,(6,3)):
            assert stats['frames']==len(a['training_indices'])
            assert stats['hits']==steps*len(a['training_indices'])
            assert 0<stats['retained_payload_bytes']<=stats['policy_limit_bytes']==64*1024*1024
            assert len(stats['geometry_sha256'])==64
        assert cache_stats[0]['geometry_sha256']==cache_stats[1]['geometry_sha256']
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
    run('reject-backend',[engine,'train-native',work/'wrong-backend.request.json'],expected_error='explicit CUDA' if args.cuda_gradient=='reverse' else 'checkpoint training backend/device changed')
    assert not (work/'rejected-backend').exists()
    cpu_comparison=None
    loss_change_rejected=False
    legacy_exact=None
    if args.virial_loss:
        cpu=deepcopy(req)
        cpu.update(device='cpu',output_directory='cpu-reference')
        cpu['config'].pop('cuda_gradient',None)
        cpu['config'].pop('fixed_geometry_cache',None)
        write(work/'cpu.request.json',cpu)
        run('cpu-reference',[engine,'train-native',work/'cpu.request.json'])
        state=read(work/'cpu-reference/checkpoint.json')
        assert state['schema']=='omnimd-native-full-checkpoint-v3'
        assert state['loss_contract']==a['loss_contract']
        assert state['training_indices']==a['training_indices'] and state['validation_indices']==a['validation_indices']
        def values(m):
            return [v for net in [n for row in m['descriptor']['embedding'] for n in row]+m['fitting']
                    for layer in net['layers'] for vector in [*layer['weights'],layer['bias'],layer['timestep'] or []] for v in vector]
        parameter_delta=max(abs(x-y) for x,y in zip(values(state['model']),values(a['model'])))
        moment_delta=max(abs(x-y) for key in ('first_moment','second_moment') for x,y in zip(state[key],a[key]))
        cpu_pred=[json.loads(s) for s in run('cpu-export-eval',[evaluator,work/'cpu-reference/model.json'],lines).splitlines()]
        delta=[0.,0.,0.]
        for c,g in zip(cpu_pred,predictions):
            delta[0]=max(delta[0],abs(c['energy']-g['energy']))
            for k,key in enumerate(('forces','virial'),1):
                delta[k]=max(delta[k],max(abs(x-y) for r,s in zip(c[key],g[key]) for x,y in zip(r,s)))
        assert max(parameter_delta,moment_delta,*delta)<1e-10
        cpu_comparison={'parameter_max_abs':parameter_delta,'moment_max_abs':moment_delta,'efv_max_abs':delta}
        wrong_loss=deepcopy(resume)
        wrong_loss['output_directory']='rejected-loss'
        wrong_loss['config']['virial_loss']['weight']=2.0
        write(work/'wrong-loss.request.json',wrong_loss)
        run('reject-loss',[engine,'train-native',work/'wrong-loss.request.json'],expected_error='checkpoint')
        assert not (work/'rejected-loss').exists()
        loss_change_rejected=True
    executor_change_rejected=False
    if args.cuda_gradient=='reverse':
        wrong_executor=deepcopy(resume)
        wrong_executor['output_directory']='rejected-executor'
        wrong_executor['config'].pop('cuda_gradient')
        write(work/'wrong-executor.request.json',wrong_executor)
        run('reject-executor',[engine,'train-native',work/'wrong-executor.request.json'],expected_error='checkpoint')
        assert not (work/'rejected-executor').exists()
        executor_change_rejected=True
    cache_toggle_exact=None
    historical_uncached_exact=None
    if args.fixed_geometry_cache:
        def numerical_state(checkpoint):
            state=deepcopy(checkpoint)
            state['config'].pop('steps')
            state['config'].pop('fixed_geometry_cache',None)
            return state
        cache_off=deepcopy(resume)
        cache_off['config'].pop('fixed_geometry_cache')
        cache_off['output_directory']='resumed-cache-off'
        write(work/'resumed-cache-off.request.json',cache_off)
        run('resume-cache-off',[engine,'train-native',work/'resumed-cache-off.request.json'])
        assert numerical_state(read(work/'resumed-cache-off/checkpoint.json'))==numerical_state(a)
        assert 'fixed_geometry_cache' not in read(work/'resumed-cache-off/report.json')
        legacy=args.legacy_uncached_work.resolve()
        old=read(legacy/'continuous/checkpoint.json')
        assert old['step']==6 and 'fixed_geometry_cache' not in old['config']
        assert old['dataset_sha256']==sha(work/'dataset.jsonl')
        assert old['execution']==a['execution']
        assert numerical_state(old)==numerical_state(a)
        legacy_on=deepcopy(resume)
        legacy_on.update(checkpoint=str(legacy/'continuous/checkpoint-step-3.json'),output_directory='legacy-off-to-on')
        write(work/'legacy-off-to-on.request.json',legacy_on)
        run('legacy-off-to-on',[engine,'train-native',work/'legacy-off-to-on.request.json'])
        assert numerical_state(read(work/'legacy-off-to-on/checkpoint.json'))==numerical_state(old)
        assert read(work/'legacy-off-to-on/report.json')['fixed_geometry_cache']['hits']==3*len(a['training_indices'])
        cache_toggle_exact=True
        historical_uncached_exact=True
    if args.legacy_ef_work:
        legacy=args.legacy_ef_work.resolve()
        legacy_request=read(legacy/'continuous.request.json')
        legacy_request.pop('initial_model')
        legacy_request.update(checkpoint=str(legacy/'continuous/checkpoint-step-3.json'),
            dataset=str(legacy/'dataset.jsonl'),output_directory=str(work/'legacy-ef-resumed'))
        legacy_request['config']['steps']=3
        assert legacy_request['device']=='cuda:0' and 'virial_loss' not in legacy_request['config']
        write(work/'legacy-ef.request.json',legacy_request)
        run('legacy-ef-resume',[engine,'train-native',work/'legacy-ef.request.json'])
        expected=read(legacy/'continuous/checkpoint.json')
        expected['config']['steps']=3
        assert expected['schema']=='omnimd-native-full-checkpoint-v2'
        assert read(work/'legacy-ef-resumed/checkpoint.json')==expected
        legacy_exact=True
    receipt = {
        'schema':'omnimd-cuda-full-training-integration-v1', 'status':'passed',
        'scope':'Six-step engine CLI integration only; original synthetic 24-frame fixture. Independent gradient and 300-step acceptance remain in B receipt. No performance or chemical-accuracy claim.',
        'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'source_native_commit':'bbfecc639fc045de778344e3ebf3ec97ae07a41c' if args.fixed_geometry_cache else 'f33d332adcafcc2a360b22e9605d6da43870eaf9' if args.cuda_gradient=='reverse' else 'd12fdb5c36a4fa74a10fb14bbb489a664cae0a2c' if args.virial_loss else 'f4dbf62bb4acb97e8f7e12e62de625e08af07a05',
        'integrated_native_commit':'a6a841d' if args.fixed_geometry_cache else 'f4e794a' if args.cuda_gradient=='reverse' else '7d3d472' if args.virial_loss else '02b5457',
        'fixed_geometry_cache':args.fixed_geometry_cache,'cache_statistics':cache_stats,
        'cache_toggle_both_directions_exact':cache_toggle_exact,'historical_uncached_checkpoint_exact':historical_uncached_exact,
        'cuda_gradient':args.cuda_gradient,'executor_change_rejected_without_output':executor_change_rejected,
        'virial_loss':req['config'].get('virial_loss'),'checkpoint_schema':a['schema'],
        'loss_contract':a.get('loss_contract'),'cpu_comparison':cpu_comparison,
        'changed_loss_rejected_without_output':loss_change_rejected,'legacy_cuda_v2_resume_exact':legacy_exact,
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
