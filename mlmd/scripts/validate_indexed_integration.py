"""Short integrated indexed-dataset acceptance on saved original synthetic inputs.
Requires an exclusive GPU window; stops on failures without retry or benchmarking.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from validate_minibatch_integration import read, write, sha, comparable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--baseline-work',type=Path,required=True,help='Accepted prior seven-step memory minibatch integration with CPU reference')
    p.add_argument('--work',type=Path,required=True)
    a=p.parse_args()
    engine,baseline,work=[v.resolve() for v in (a.engine,a.baseline_work,a.work)]
    trainer=engine.with_name('omnimd-native-train'+engine.suffix)
    assert engine.is_file() and trainer.is_file()
    started=datetime.now(timezone.utc).isoformat()
    work.mkdir(parents=True,exist_ok=False)
    for name in ('initial.json','dataset.jsonl','frames.jsonl'):
        shutil.copyfile(baseline/name,work/name)
    commands=[]
    def run(label,command,error=None):
        result=subprocess.run([str(v) for v in command],text=True,capture_output=True,timeout=120)
        (work/(label+'.stdout')).write_text(result.stdout,encoding='utf-8')
        (work/(label+'.stderr')).write_text(result.stderr,encoding='utf-8')
        commands.append({'label':label,'command':[str(v) for v in command],'exit_code':result.returncode})
        write(work/'commands.json',commands)
        if error is None:
            if result.returncode:
                raise RuntimeError(f'{label} failed; no retry: {result.stderr}')
        else:
            assert result.returncode and error in result.stderr,(label,result.stderr)
        return result.stdout
    def train(label,req,error=None):
        write(work/(label+'.request.json'),req)
        run(label,[engine,'train-native',work/(label+'.request.json')],error)
        if error:
            assert not (work/req['output_directory']).exists()
            return None,None
        folder=work/req['output_directory']
        state,report=read(folder/'checkpoint.json'),read(folder/'report.json')
        assert report['native_v1_reload_max_abs_energy_force_virial']==[0,0,0]
        if req.get('dataset_access','memory')=='indexed':
            access=report['dataset_access']
            assert access['mode']=='indexed_jsonl_v1' and access['frames']==24
            assert access['file_bytes']==(work/req['dataset']).stat().st_size
            assert access['frames_decoded_per_read']==1
            assert access['index_payload_bytes']<=access['index_payload_limit_bytes']==64*1024*1024
            assert access['parser_record_limit_bytes']==256*1024
            assert access['integrity_verifications']==req['config']['steps']+2
        else:
            assert 'dataset_access' not in report
        return state,report
    capabilities=json.loads(run('capabilities',[engine,'capabilities']))
    access=capabilities['training']['dataset_access']
    assert access['default']=='memory' and access['indexed']['frame_limit']==100000
    assert capabilities['training']['minibatch']['default_memory_frame_limit']==1024
    req=read(baseline/'continuous.request.json')
    assert req['config']['steps']==7 and req['config']['minibatch']['batch_size']==7
    req.update(dataset_access='indexed',initial_model='initial.json',dataset='dataset.jsonl',output_directory='indexed')
    state,report=train('indexed',req)
    old=read(baseline/'continuous/checkpoint.json')
    old_report=read(baseline/'continuous/report.json')
    assert old['dataset_sha256']==sha(work/'dataset.jsonl')
    assert state['schema']=='omnimd-native-full-checkpoint-v5' and comparable(state)==comparable(old)
    assert (work/'indexed/model.json').read_bytes()==(baseline/'continuous/model.json').read_bytes()
    metrics=('initial_training','final_training','initial_validation','final_validation')
    for key in metrics:
        assert report[key]==old_report[key],key
    assert report['minibatch']==old_report['minibatch']
    assert report['fixed_geometry_cache']['hits']==43
    for field in ('frames','retained_payload_bytes','geometry_sha256'):
        assert report['fixed_geometry_cache'][field]==old_report['fixed_geometry_cache'][field]
    back=deepcopy(req)
    back.pop('initial_model')
    back.update(dataset_access='memory',checkpoint='indexed/checkpoint-step-2.json',output_directory='to-memory-cache-off')
    back['config']['steps']=5
    back['config'].pop('fixed_geometry_cache')
    memory,memory_report=train('to-memory-cache-off',back)
    assert comparable(memory)==comparable(state)
    for key in ('final_training','final_validation'):
        assert memory_report[key]==report[key]
    forward=deepcopy(req)
    forward.pop('initial_model')
    forward.update(checkpoint=str(baseline/'continuous/checkpoint-step-3.json'),output_directory='old-memory-to-indexed')
    forward['config']['steps']=4
    resumed,resumed_report=train('old-memory-to-indexed',forward)
    assert comparable(resumed)==comparable(state)
    for key in ('final_training','final_validation'):
        assert resumed_report[key]==report[key]
    cpu=read(baseline/'cpu-reference.request.json')
    cpu.update(dataset_access='indexed',initial_model='initial.json',dataset='dataset.jsonl',output_directory='cpu-indexed')
    cpu_state,cpu_report=train('cpu-indexed',cpu)
    assert comparable(cpu_state)==comparable(read(baseline/'cpu-reference/checkpoint.json'))
    for key in metrics:
        assert cpu_report[key]==read(baseline/'cpu-reference/report.json')[key]
    assert (work/'cpu-indexed/model.json').read_bytes()==(baseline/'cpu-reference/model.json').read_bytes()
    no_batch=deepcopy(cpu)
    no_batch['config'].pop('minibatch')
    no_batch['output_directory']='rejected-no-minibatch'
    train('reject-no-minibatch',no_batch,'requires explicit minibatch')
    unknown=deepcopy(cpu)
    unknown.update(dataset_access='streamed',output_directory='rejected-access')
    train('reject-access',unknown,'unknown variant')
    original=(work/'dataset.jsonl').read_bytes()
    assert b'\n' in original
    changed=original.replace(b'\n',b' \n',1)
    (work/'changed-bytes.jsonl').write_bytes(changed)
    changed_req=deepcopy(forward)
    changed_req.update(dataset='changed-bytes.jsonl',output_directory='rejected-hash')
    train('reject-hash',changed_req,'checkpoint')
    receipt={'schema':'omnimd-indexed-training-integration-v1','status':'passed','started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
             'source_native_commit':'be573fe0f91d323932c85e47448b0aace5950f32','integrated_native_commit':'dc24b26','source_base':'8f80ee542b643386c753f0f23d0e4c29d9270f81',
             'scope':'Existing 24-frame synthetic fixture, seven reverse E/F/V minibatch updates; storage and cache-switch integration only. No repeated capacity, memory or throughput benchmark.',
             'execution':state['execution'],'checkpoint_schema':state['schema'],'frames':24,'training_frames':18,'validation_frames':6,'updates':7,
             'state_and_all_four_metrics_exact_vs_prior_memory':True,'cpu_state_and_all_four_metrics_exact_vs_prior_memory':True,
             'model_export_bytes_exact_vs_prior_memory':True,'indexed_to_memory_cache_off_exact_step_2':True,'old_memory_to_indexed_cache_on_exact_step_3':True,
             'minibatch':report['minibatch'],'indexed_access':report['dataset_access'],'cpu_indexed_access':cpu_report['dataset_access'],
             'cache_statistics':report['fixed_geometry_cache'],'native_v1_reload_max_abs_energy_force_virial':report['native_v1_reload_max_abs_energy_force_virial'],
             'rejected_before_output':['indexed without minibatch','unknown dataset access mode','changed raw dataset bytes on resume'],
             'historical_checkpoint_sha256':{name:sha(baseline/'continuous'/name) for name in ('checkpoint-step-3.json','checkpoint.json')},
             'sha256':{name:sha(path) for name,path in [('engine',engine),('trainer',trainer),('script',Path(__file__)),('helper',Path(__file__).with_name('validate_minibatch_integration.py')),('dataset',work/'dataset.jsonl'),('initial_model',work/'initial.json'),('model',work/'indexed/model.json'),('checkpoint',work/'indexed/checkpoint.json')]},
             'checks':[{'label':c['label'],'exit_code':c['exit_code']} for c in commands],'source_sha256':report['source_sha256']}
    write(work/'acceptance.json',receipt)
    print(json.dumps(receipt,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
