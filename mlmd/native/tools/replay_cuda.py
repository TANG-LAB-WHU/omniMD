"""Prepare or run bounded native CUDA EFV parity/timing. No downloads.
Preparation is CPU/file-only. --run-gpu must be explicitly selected only after
an exclusive healthy GPU window is granted. Stop on the first failed/timed-out
process; never reset a device or retry a CUDA failure.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def frames(path,values):
    Path(path).write_text(''.join(json.dumps(f,allow_nan=False)+'\n' for f in values),encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def medium_model(base):
    m=deepcopy(base)
    counter=0
    def layer(ni,no,residual,linear=False):
        nonlocal counter
        counter+=1
        return {'weights':[[.35/math.sqrt(ni)*math.sin((i*no+j+counter)*.31) for j in range(no)] for i in range(ni)],
                'bias':[.03*math.cos((j+counter)*.67) for j in range(no)],
                'activation':'linear' if linear else 'tanh','residual':residual,
                'timestep':None if linear else [.9+.03*math.sin(j+counter) for j in range(no)]}
    m['descriptor'].update(sel=[16,16],axis_neuron=4,embedding=[[
        {'layers':[layer(1,8,'none'),layer(8,16,'concat2'),layer(16,32,'concat2')]}
        for _ in range(2)] for _ in range(2)])
    m['fitting']=[{'layers':[layer(128,64,'none'),layer(64,64,'identity'),layer(64,1,'none',True)]} for _ in range(2)]
    return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-exe',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--public-model',type=Path)
    parser.add_argument('--public-frames',type=Path)
    parser.add_argument('--run-gpu',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    model=json.loads((root/'fixtures/full-tanh-v1.json').read_text())
    write(output/'tiny-model.json',model)
    base=json.loads((root/'fixtures/frame.jsonl').read_text())
    nonperiodic=deepcopy(base);nonperiodic['cell']=None
    wrapped=deepcopy(base)
    wrapped['positions'][1]=[a+b for a,b in zip(wrapped['positions'][1],wrapped['cell'][1])]
    self_image={'positions':[[.3,.4,.5]],'species':['O'],'cell':[[2.5,0.,0.],[0.,2.5,0.],[0.,0.,2.5]],'units':'EvAngstrom'}
    frames(output/'tiny-frames.jsonl',[base,nonperiodic,wrapped,self_image])
    write(output/'medium-model.json',medium_model(model))
    medium={'positions':[],'species':[],'cell':[[12.,0.,0.],[0.,12.,0.],[0.,0.,12.]],'units':'EvAngstrom'}
    for i in range(6):
        for j in range(6):
            for k in range(6):
                medium['positions'].append([2*i+.03*math.sin(j+k),2*j+.02*math.cos(i+k),2*k+.01*math.sin(i+j)])
                medium['species'].append('O' if (i+j+k)%2==0 else 'H')
    frames(output/'medium-frames.jsonl',[medium])
    cases=[('tiny',output/'tiny-model.json',output/'tiny-frames.jsonl',3),
           ('medium',output/'medium-model.json',output/'medium-frames.jsonl',5)]
    if bool(args.public_model)!=bool(args.public_frames):
        raise ValueError('supply both public model and public frames')
    if args.public_model:
        cases.insert(1,('public',args.public_model.resolve(),args.public_frames.resolve(),3))
    plan={'gpu_executed':False,'cases':[{'name':name,'model':str(m),'frames':str(f),'repeats':n,
           'model_sha256':sha(m),'frames_sha256':sha(f)} for name,m,f,n in cases]}
    write(output/'plan.json',plan)
    if not args.run_gpu:
        print(json.dumps(plan,indent=2));return
    # Explicit operator-controlled GPU window. No automatic retries.
    receipts=[]
    for name,m,f,n in cases:
        command=[str(args.check_exe.resolve()),str(m),str(f),str(n)]
        try:
            result=subprocess.run(command,check=True,text=True,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            failure={'case':name,'gpu_attempted':True,'success':False,'error':str(error),
                     'stderr':str(getattr(error,'stderr','')),'action':'stopped; no retry/reset; return to explicit CPU backend'}
            write(output/'failure.json',failure)
            raise
        (output/(name+'.stdout.json')).write_text(result.stdout,encoding='utf-8')
        (output/(name+'.stderr.txt')).write_text(result.stderr,encoding='utf-8')
        receipt=json.loads(result.stdout);receipt['case']=name
        receipt['model_sha256']=sha(m);receipt['frames_sha256']=sha(f);receipts.append(receipt)
    report={'gpu_executed':True,'scope':'bounded same-model native CPU/CUDA FP64 EFV inference; no GPU training claim',
            'check_executable_sha256':sha(args.check_exe),'cases':receipts}
    write(output/'acceptance.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
