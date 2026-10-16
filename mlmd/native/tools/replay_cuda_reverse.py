"""Original synthetic complete-process comparison for bounded reverse CUDA training.
Run only during an authorized GPU window. Python stdlib; no downloads.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import time
from replay_full_training import digest, perturb, run, write


def medium_model(root):
    m=json.loads((root/'fixtures/full-tanh-v1.json').read_text())
    m['descriptor']['axis_neuron']=2
    def layer(ni,no,residual='none',linear=False):
        return {'weights':[[.12*math.sin((i*no+j+1)*.7) for j in range(no)] for i in range(ni)],
                'bias':[.03]*no,'activation':'linear' if linear else 'tanh','residual':residual,
                'timestep':None if linear else [.7]*no}
    for row in m['descriptor']['embedding']:
        for net in row:
            net['layers']=[layer(1,4),layer(4,4,'identity')]
    for net in m['fitting']:
        first=layer(8,8);first['timestep']=None
        net['layers']=[first,layer(8,8,'identity'),layer(8,1,linear=True)]
    return m


def values(m):
    return [v for net in [n for row in m['descriptor']['embedding'] for n in row]+m['fitting']
            for layer in net['layers'] for vector in [*layer['weights'],layer['bias'],layer['timestep'] or []] for v in vector]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-exe',type=Path,required=True)
    parser.add_argument('--train-exe',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); evaluator=args.eval_exe.resolve(); trainer=args.train_exe.resolve()
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];initial=medium_model(root);teacher=deepcopy(initial)
    parameters=perturb(teacher);assert parameters==466
    write(output/'initial.json',initial);write(output/'teacher.json',teacher)
    frames=[]
    for k in range(8):
        positions=[]
        for i in range(64):
            positions.append([(i//2)*8.+(i%2)*(1.1+.04*math.sin(k+i)),(i%2)*.2,.1])
        frames.append({'positions':positions,'species':['O' if i%2==0 else 'H' for i in range(64)],'cell':None,'units':'EvAngstrom'})
    lines=''.join(json.dumps(f)+'\n' for f in frames)
    labels=[json.loads(s) for s in run([evaluator,output/'teacher.json'],input=lines).splitlines()]
    header={'schema':'omnimd-dataset-jsonl-v1','type_map':['O','H'],'units':'EvAngstrom','virial_convention':'MinusDEpsilonRow','source':'original synthetic separated-pair teacher; no physical accuracy claim','selected_sets':[]}
    records=[{'kind':'header','header':header}]
    records += [{'kind':'frame','data':{'index':i,'id':f'pairs/{i}','frame':f,'labels':{k:l[k] for k in ('energy','forces','virial')}}} for i,(f,l) in enumerate(zip(frames,labels))]
    records += [{'kind':'completed','frames':8,'source_sha256':{'teacher.json':digest(output/'teacher.json'),'replay_cuda_reverse.py':digest(__file__)}}]
    (output/'dataset.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    config={'split':{'mode':'explicit','validation_ids':['pairs/6','pairs/7']},'steps':100,
            'energy_weight':1.,'force_weight':1.,'virial_loss':{'weight':1.,'scale_ev_per_atom':.1},
            'energy_scale_ev_per_atom':.1,'force_scale_ev_per_angstrom':.1,'learning_rate':.005,
            'beta1':.9,'beta2':.999,'epsilon':1e-8,'gradient_clip_norm':10.,'checkpoint_every':50}
    rows=[];states={}
    # Rotate execution order to reduce systematic warm-up/order bias.
    for repeat in range(3):
        order=['cpu','hyperdual','reverse'];order=order[repeat:]+order[:repeat]
        for backend in order:
            name=f'{backend}-{repeat}';cfg=deepcopy(config)
            if backend=='reverse':cfg['cuda_gradient']='reverse'
            request={'schema':'omnimd-native-full-train-v1','device':'cpu' if backend=='cpu' else 'cuda:0',
                     'initial_model':'initial.json','dataset':'dataset.jsonl','output_directory':name,'config':cfg}
            write(output/(name+'.json'),request);start=time.perf_counter();run([trainer,output/(name+'.json')]);wall=(time.perf_counter()-start)*1000
            report=json.loads((output/name/'report.json').read_text());state=json.loads((output/name/'checkpoint.json').read_text())
            if backend in states:
                assert state==states[backend],f'nonrepeatable {backend} checkpoint'
            states[backend]=state
            rows.append({'backend':backend,'repeat':repeat,'complete_training_process_ms':wall,'timing':report['timing'],
                         'gradient_profile':report.get('cuda_gradient_profile'),'final_training':report['final_training'],'final_validation':report['final_validation']})
    errors={}
    for backend in ('hyperdual','reverse'):
        errors[backend]=max(abs(a-b) for a,b in zip(values(states['cpu']['model']),values(states[backend]['model'])))
        assert errors[backend]<1e-8
    receipt={'scope':'bounded synthetic 466-parameter 64-atom E/F/V training; not a production scaling benchmark',
             'frames':8,'train_frames':6,'validation_frames':2,'steps':100,'parameters':466,'runs':rows,
             'median_complete_process_ms':{b:statistics.median(r['complete_training_process_ms'] for r in rows if r['backend']==b) for b in states},
             'three_repeats_checkpoint_exact':True,'parameter_max_abs_vs_cpu':errors,
             'timing_scope':'complete fresh training subprocess, including input, driver/NVRTC initialization, all updates, assessment, two intermediate and final checkpoints and model reload verification; excludes fixture generation',
             'sha256':{n:digest(output/n) for n in ('dataset.jsonl','initial.json','teacher.json')},
             'executable_sha256':{'eval':digest(evaluator),'train':digest(trainer)}}
    write(output/'acceptance.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
