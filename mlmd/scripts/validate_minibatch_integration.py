"""Short integrated minibatch CLI acceptance. Exclusive GPU window required.
Uses saved original synthetic project inputs; no retries or performance benchmark.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparable(state):
    state=deepcopy(state)
    state['config'].pop('steps')
    state['config'].pop('fixed_geometry_cache',None)
    return json.dumps(state,sort_keys=True,allow_nan=False)


def values(model):
    return [v for net in [n for row in model['descriptor']['embedding'] for n in row]+model['fitting']
            for layer in net['layers'] for row in [*layer['weights'],layer['bias'],layer['timestep'] or []] for v in row]


def sampler(config, indices, step):
    mask=(1<<64)-1
    gamma=0x9e3779b97f4a7c15
    def next_value(state):
        state=(state+gamma)&mask
        z=state
        z=((z^(z>>30))*0xbf58476d1ce4e5b9)&mask
        z=((z^(z>>27))*0x94d049bb133111eb)&mask
        return state,z^(z>>31)
    count=(len(indices)+config['batch_size']-1)//config['batch_size']
    epoch=step//count
    _,rng=next_value((config['seed']+gamma*epoch)&mask)
    order=list(indices)
    if config['order']=='shuffle_each_epoch':
        for i in range(len(order)-1,0,-1):
            bound=i+1
            threshold=((1<<64)-bound)%bound
            while True:
                rng,v=next_value(rng)
                if v>=threshold:
                    break
            j=v%bound
            order[i],order[j]=order[j],order[i]
    return {'algorithm':'omnimd-epoch-splitmix64-v1','seed':config['seed'],'order_mode':config['order'],
            'epoch':epoch,'cursor':(step%count)*config['batch_size'],'order':order,'rng_state':rng}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--evaluator',type=Path,required=True)
    p.add_argument('--baseline-work',type=Path,required=True,help='Prior accepted uncached or cached full-batch six-step reverse E/F/V integration')
    p.add_argument('--work',type=Path,required=True)
    args=p.parse_args()
    engine,evaluator,baseline,work=[v.resolve() for v in (args.engine,args.evaluator,args.baseline_work,args.work)]
    trainer=engine.with_name('omnimd-native-train'+engine.suffix)
    for path in (engine,evaluator,trainer):
        assert path.is_file(),path
    golden=sampler({'batch_size':2,'seed':20261016,'order':'shuffle_each_epoch'},[0,2,4,5,7],0)
    assert golden['order']==[4,2,7,5,0] and golden['rng_state']==13274578833983480863
    started=datetime.now(timezone.utc).isoformat()
    work.mkdir(parents=True,exist_ok=False)
    for name in ('initial.json','dataset.jsonl','frames.jsonl'):
        shutil.copyfile(baseline/name,work/name)
    commands=[]
    def run(label,command,input_text=None,error=None):
        result=subprocess.run([str(v) for v in command],input=input_text,text=True,capture_output=True,timeout=120)
        (work/(label+'.stdout')).write_text(result.stdout,encoding='utf-8')
        (work/(label+'.stderr')).write_text(result.stderr,encoding='utf-8')
        commands.append({'label':label,'command':[str(v) for v in command],'exit_code':result.returncode})
        write(work/'commands.json',commands)
        if error is None:
            if result.returncode:
                raise RuntimeError(f'{label} failed; no retry: {result.stderr}')
        else:
            assert result.returncode!=0 and error in result.stderr,(label,result.stderr)
        return result.stdout
    def train(label,request,error=None):
        path=work/(label+'.request.json')
        write(path,request)
        run(label,[engine,'train-native',path],error=error)
        if error:
            assert not (work/request['output_directory']).exists()
            return None
        return read(work/request['output_directory']/'checkpoint.json')
    caps=json.loads(run('capabilities',[engine,'capabilities']))
    assert caps['training']['minibatch']['streaming'] is False
    req=read(baseline/'continuous.request.json')
    req.update(device='cuda:0',initial_model='initial.json',dataset='dataset.jsonl',output_directory='continuous')
    req['config'].update(steps=7,checkpoint_every=1,cuda_gradient='reverse',fixed_geometry_cache=True,
                         minibatch={'batch_size':7,'seed':20261016,'order':'shuffle_each_epoch'})
    state=train('continuous',req)
    assert state['schema']=='omnimd-native-full-checkpoint-v5' and state['step']==7
    assert state['execution']['backend']=='cuda_reverse_f64_v1'
    assert state['loss_contract']=='energy-force-row-virial-per-atom-nine-v1'
    assert state['dataset_sha256']==sha(work/'dataset.jsonl')
    indices=state['training_indices']
    assert len(indices)==18 and len(state['validation_indices'])==6
    sizes=[]
    for step in range(1,8):
        current=read(work/f'continuous/checkpoint-step-{step}.json')
        expected=sampler(req['config']['minibatch'],indices,step)
        assert current['sampler']==expected
        assert set(expected['order'])==set(indices)
        assert not set(expected['order']).intersection(state['validation_indices'])
        sizes.append(min(7,18-((step-1)%3)*7))
    assert sizes==[7,7,4,7,7,4,7]
    report=read(work/'continuous/report.json')
    assert report['minibatch']['frames_processed']==43
    assert report['minibatch']['partial_batches']==2
    assert report['minibatch']['batch_size_counts']=={'4':2,'7':5}
    assert report['minibatch']['start']==sampler(req['config']['minibatch'],indices,0)
    assert report['minibatch']['end']==state['sampler']
    assert report['fixed_geometry_cache']['frames']==18 and report['fixed_geometry_cache']['hits']==43
    assert report['native_v1_reload_max_abs_energy_force_virial']==[0,0,0]
    def resume(label,checkpoint,steps,cache):
        r=deepcopy(req)
        r.pop('initial_model')
        r.update(checkpoint=checkpoint,output_directory=label)
        r['config']['steps']=steps
        if cache:
            r['config']['fixed_geometry_cache']=True
        else:
            r['config'].pop('fixed_geometry_cache',None)
        actual=train(label,r)
        assert comparable(actual)==comparable(state),label
    resume('resume-before-tail-cache-off','continuous/checkpoint-step-2.json',5,False)
    resume('resume-after-tail','continuous/checkpoint-step-3.json',4,True)
    resume('resume-mid-epoch-cache-on','resume-before-tail-cache-off/checkpoint-step-4.json',3,True)
    cpu=deepcopy(req)
    cpu.update(device='cpu',output_directory='cpu-reference')
    cpu['config'].pop('cuda_gradient')
    cpu['config'].pop('fixed_geometry_cache')
    reference=train('cpu-reference',cpu)
    assert reference['sampler']==state['sampler']
    delta=max(abs(a-b) for a,b in zip(values(reference['model'])+reference['first_moment']+reference['second_moment'],
                                     values(state['model'])+state['first_moment']+state['second_moment']))
    assert delta<1e-10
    lines=(work/'frames.jsonl').read_text(encoding='utf-8')
    predictions=[]
    for folder in ('continuous','cpu-reference'):
        predictions.append([json.loads(line) for line in run(folder+'-export',[evaluator,work/folder/'model.json'],lines).splitlines()])
    assert len(predictions[0])==len(predictions[1])==24
    efv=[0.,0.,0.]
    for gpu,cpu_value in zip(*predictions):
        efv[0]=max(efv[0],abs(gpu['energy']-cpu_value['energy']))
        for i,key in enumerate(('forces','virial'),1):
            efv[i]=max(efv[i],max(abs(a-b) for r,s in zip(gpu[key],cpu_value[key]) for a,b in zip(r,s)))
    assert max(efv)<1e-10
    rejected=[]
    for mode in ('batch_size','seed','order','missing_sampler','cursor'):
        bad=deepcopy(req)
        bad.pop('initial_model')
        bad.update(checkpoint='continuous/checkpoint.json',output_directory='rejected-'+mode)
        bad['config']['steps']=1
        if mode in ('batch_size','seed'):
            bad['config']['minibatch'][mode]+=1
        elif mode=='order':
            bad['config']['minibatch']['order']='sequential'
        else:
            broken=deepcopy(state)
            if mode=='missing_sampler':
                broken.pop('sampler')
            else:
                broken['sampler']['cursor']+=1
            write(work/(mode+'.checkpoint.json'),broken)
            bad['checkpoint']=mode+'.checkpoint.json'
        train('reject-'+mode,bad,error='checkpoint')
        rejected.append(mode)
    legacy=read(baseline/'continuous.request.json')
    legacy.pop('initial_model')
    legacy.update(checkpoint=str(baseline/'continuous/checkpoint-step-3.json'),dataset='dataset.jsonl',output_directory='legacy-fullbatch')
    legacy['config']['steps']=3
    assert 'minibatch' not in legacy['config']
    old=train('legacy-fullbatch',legacy)
    assert old['schema']=='omnimd-native-full-checkpoint-v4' and 'sampler' not in old
    assert comparable(old)==comparable(read(baseline/'continuous/checkpoint.json'))
    receipt={'schema':'omnimd-minibatch-integration-v1','status':'passed','started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
             'scope':'Seven-update reverse E/F/V engine CLI integration on existing original synthetic data; no new benchmark, chemical accuracy or full-batch convergence claim.',
             'source_native_commit':'3e7091c143ffa424c935d86adbd9ecb7f1e97d5c','integrated_native_commit':'5be7e49','source_base':'b8ce8fa589e673136902ed162639e012547532b7',
             'execution':state['execution'],'checkpoint_schema':state['schema'],'updates':7,'training_frames':18,'validation_frames':6,
             'batch_sizes':sizes,'sampler_matches_independent_integer_reference_all_steps':True,'minibatch':report['minibatch'],
             'cache_statistics':report['fixed_geometry_cache'],'exact_resume_points':[2,3,4],'both_cache_toggle_directions_exact':True,
             'cpu_parameter_moment_max_abs':delta,'cpu_export_efv_max_abs':efv,'native_v1_reload_max_abs_energy_force_virial':report['native_v1_reload_max_abs_energy_force_virial'],
             'rejected_before_output':rejected,'historical_fullbatch_v4_exact':True,
             'historical_checkpoint_sha256':{name:sha(baseline/'continuous'/name) for name in ('checkpoint-step-3.json','checkpoint.json')},
             'initial_training':report['initial_training'],'final_training':report['final_training'],'initial_validation':report['initial_validation'],'final_validation':report['final_validation'],
             'sha256':{name:sha(path) for name,path in [('engine',engine),('trainer',trainer),('evaluator',evaluator),('script',Path(__file__)),('dataset',work/'dataset.jsonl'),('initial_model',work/'initial.json'),('export_model',work/'continuous/model.json'),('checkpoint',work/'continuous/checkpoint.json')]},
             'checks':[{'label':c['label'],'exit_code':c['exit_code']} for c in commands], 'source_sha256':report['source_sha256']}
    write(work/'acceptance.json',receipt)
    print(json.dumps(receipt,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
