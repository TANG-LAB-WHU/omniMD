"""Original synthetic bounded indexed-dataset acceptance. Python stdlib only.
Large records are whitespace padded intentionally to cross both the historical
1024-frame and 128-MiB file limits without claiming a physical benchmark.
"""
import argparse
from copy import deepcopy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from replay_full_training import perturb, write, run


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(65536),b''): h.update(part)
    return h.hexdigest()


def launch(trainer, request):
    start=time.perf_counter()
    proc=subprocess.Popen([str(trainer),str(request)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout,stderr=proc.communicate()
    if proc.returncode: raise RuntimeError(stderr.decode('utf-8'))
    peak=None
    if os.name=='nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        c=Counters();c.cb=ctypes.sizeof(c)
        get=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
        get.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        get.restype=wintypes.BOOL
        if not get(wintypes.HANDLE(int(proc._handle)),ctypes.byref(c),ctypes.sizeof(c)): raise ctypes.WinError(ctypes.get_last_error())
        peak=int(c.PeakWorkingSetSize)
    return {'process_ms':(time.perf_counter()-start)*1000,'peak_working_set_bytes':peak}


def comparable(checkpoint):
    s=deepcopy(checkpoint);s['config'].pop('steps',None);return s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-exe',type=Path,required=True)
    p.add_argument('--train-exe',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();evaluator=a.eval_exe.resolve();trainer=a.train_exe.resolve();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    initial=json.loads((root/'fixtures/full-tanh-v1.json').read_text());teacher=deepcopy(initial);parameters=perturb(teacher)
    write(out/'initial.json',initial);write(out/'teacher.json',teacher)
    base=json.loads((root/'fixtures/frame.jsonl').read_text());frames=[]
    for i in range(8):
        f=deepcopy(base);f['cell']=None if i%2==0 else [[3.8,0.,0.],[.3,3.9,0.],[.1,.2,4.]]
        for atom,r in enumerate(f['positions']):
            for axis in range(3):r[axis]+=.19*math.sin((i+1)*(atom*3+axis+1)*.618)
        if i%3==0:f['positions'].pop();f['species'].pop()
        frames.append(f)
    labels=[json.loads(line) for line in run([evaluator,out/'teacher.json'],input=''.join(json.dumps(f)+'\n' for f in frames)).splitlines()]
    header={'kind':'header','header':{'schema':'omnimd-dataset-jsonl-v1','type_map':initial['type_map'],'units':'EvAngstrom','virial_convention':'MinusDEpsilonRow','source':'original synthetic indexed reader stress; repeated geometries, optional whitespace padding; no physical accuracy claim','selected_sets':['set.000']}}
    def emit(name,count,pad):
        with (out/name).open('wb') as f:
            f.write((json.dumps(header)+'\n').encode())
            for i in range(count):
                line=json.dumps({'kind':'frame','data':{'index':i,'id':f'set.000/{i}','frame':frames[i%8],'labels':{k:labels[i%8][k] for k in ('energy','forces','virial')}}},allow_nan=False).encode()
                assert len(line)<65536
                if pad:line+=b' '*(65535-len(line))
                f.write(line+b'\n')
            f.write((json.dumps({'kind':'completed','frames':count,'source_sha256':{'teacher.json':digest(out/'teacher.json'),'replay_indexed_training.py':digest(__file__)}})+'\n').encode())
    emit('small.jsonl',8,False);emit('large.jsonl',2051,True)
    config={'split':{'mode':'explicit','validation_ids':['set.000/2','set.000/6','set.000/7']},'steps':7,'minibatch':{'batch_size':2,'seed':20261016,'order':'shuffle_each_epoch'},'energy_weight':1.,'force_weight':1.,'virial_loss':{'weight':.7,'scale_ev_per_atom':.2},'energy_scale_ev_per_atom':.1,'force_scale_ev_per_angstrom':.1,'learning_rate':.005,'beta1':.9,'beta2':.999,'epsilon':1e-8,'gradient_clip_norm':10.,'checkpoint_every':1}
    def execute(name,c,dataset,access,checkpoint=None):
        r={'schema':'omnimd-native-full-train-v1','dataset_access':access,'device':'cpu','dataset':dataset,'output_directory':name,'config':c}
        r.update(checkpoint=checkpoint) if checkpoint else r.update(initial_model='initial.json')
        path=out/(name+'.request.json');write(path,r);timing=launch(trainer,path)
        state=json.loads((out/name/'checkpoint.json').read_text());report=json.loads((out/name/'report.json').read_text())
        return state,report,timing
    memory,mr,mt=execute('memory',config,'small.jsonl','memory')
    indexed,ir,it=execute('indexed',config,'small.jsonl','indexed')
    assert comparable(memory)==comparable(indexed)
    for field in ('initial_training','final_training','initial_validation','final_validation'):assert mr[field]==ir[field]
    resumes=[]
    for step,mode in [(2,'indexed'),(3,'memory')]:
        c=deepcopy(config);c['steps']=7-step
        source='memory' if mode=='indexed' else 'indexed'
        result,_,_=execute(f'switch-{step}',c,'small.jsonl',mode,f'{source}/checkpoint-step-{step}.json')
        assert comparable(result)==comparable(memory)
        resumes.append({'step':step,'from':source,'to':mode,'exact':True})
    large=deepcopy(config);large['split']['validation_ids']=['set.000/1025'];large['steps']=19;large['checkpoint_every']=16;large['minibatch']['batch_size']=128
    ls,lr,lt=execute('large',large,'large.jsonl','indexed')
    assert lr['dataset_access']['frames']==2051 and len(ls['training_indices'])==2050
    assert lr['minibatch']['frames_processed']==2306 and lr['minibatch']['partial_batches']==1
    assert ls['sampler']['epoch']==1 and ls['sampler']['cursor']==256
    assert lr['dataset_access']['file_bytes']>128*1024*1024
    assert lr['dataset_access']['index_payload_bytes']<1024*1024
    assert lr['dataset_access']['peak_decoded_frame_payload_bytes']<4096
    if lt['peak_working_set_bytes'] is not None: assert lt['peak_working_set_bytes']<96*1024*1024
    resume=deepcopy(large);resume['steps']=3
    rs,rr,rt=execute('large-resume',resume,'large.jsonl','indexed','large/checkpoint-step-16.json')
    assert comparable(ls)==comparable(rs)
    assert rr['minibatch']['frames_processed']==258
    bad={'schema':'omnimd-native-full-train-v1','dataset_access':'memory','device':'cpu','initial_model':'initial.json','dataset':'large.jsonl','output_directory':'legacy-rejected','config':large}
    write(out/'legacy.request.json',bad)
    rejected=subprocess.run([str(trainer),str(out/'legacy.request.json')],capture_output=True,text=True)
    assert rejected.returncode and not (out/'legacy-rejected').exists()
    assert '128 MiB' in rejected.stderr
    receipt={'scope':__doc__,'parameters':parameters,'small':{'frames':8,'memory_process':mt,'indexed_process':it,'access':ir['dataset_access'],'state_and_all_metrics_exact':True,'bidirectional_resume':resumes},
        'large':{'frames':2051,'training_frames':2050,'validation_frames':1,'record_padding_bytes':65536,'process':lt,'access':lr['dataset_access'],'steps':19,'processed_frames':2306,'partial_batches':1,'end_epoch':1,'end_cursor':256,'resume_step_16_exact':True,'resume_processed_frames':258,'resume_process':rt,'legacy_128mib_guard_preserved':True,'dataset_sha256':digest(out/'large.jsonl'),'checkpoint_sha256':digest(out/'large/checkpoint.json'),'model_sha256':digest(out/'large/model.json'),'native_v1_reload_max_abs_energy_force_virial':lr['native_v1_reload_max_abs_energy_force_virial']},
        'memory_scope':'PeakWorkingSetSize of completed Windows native training subprocess (includes runtime/model/AD/index/report/checkpoint buffers); not a portable hard RSS cap. Input is whitespace padded; payload counters exclude allocator, serde temporary buffers, partition/sampler/report and neural working memory.',
        'sha256':{'script':digest(__file__),'trainer':digest(trainer),'evaluator':digest(evaluator)}}
    assert receipt['large']['dataset_sha256']==lr['dataset_sha256']
    write(out/'acceptance.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
