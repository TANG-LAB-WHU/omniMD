"""CPU-only candidate -> external label return -> readout retraining acceptance.
Uses saved original synthetic model-deviation fixtures. No external scientific
calculator, GPU, downloads or overwrite; saved synthetic predictions emulate a
returned label file, explicitly marked as software fixtures.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def write_rows(path,values):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        for value in values:
            stream.write(json.dumps(value,allow_nan=False)+'\n')


def identity_sha(context,source,identity):
    """Independent Python implementation of the documented v1 binary encoding."""
    h=hashlib.sha256()
    def n(x): h.update(struct.pack('<Q',x))
    def s(x):
        raw=x.encode('utf-8'); n(len(raw)); h.update(raw)
    def x(value): h.update(struct.pack('<d',value))
    def vector(values):
        n(len(values))
        for value in values: x(value)
    s('omnimd-label-identity-v1');s(source.lower());n(identity['index']);s(identity['id'])
    s(context['units']);s(context['virial_convention']);n(len(context['type_map']))
    for value in context['type_map']: s(value)
    frame=identity['frame'];n(len(frame['positions']))
    for species,r in zip(frame['species'],frame['positions']):
        s(species)
        for value in r: x(value)
    n(frame['cell'] is not None)
    if frame['cell'] is not None:
        for r in frame['cell']:
            for value in r: x(value)
    layout=context.get('parameter_layout');n(layout is not None)
    if layout is not None:
        for key in ('fparam','aparam'):
            n(len(layout[key]))
            for field in layout[key]:s(field['name']);s(field['unit'])
    p=identity.get('parameters');n(p is not None)
    if p is not None:
        s(p['frame_parameter_policy']);vector(p['fparam']);n(len(p['aparam']))
        for r in p['aparam']:vector(r)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--deviation-work',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    args=parser.parse_args()
    engine,previous,work=[x.resolve() for x in (args.engine,args.deviation_work,args.work)]
    work.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1')
    commands=[]
    def run(label,command,success=True):
        p=subprocess.run([str(x) for x in command],capture_output=True,text=True,timeout=120,env=env)
        commands.append({'label':label,'exit_code':p.returncode})
        (work/f'{len(commands):02d}.stdout').write_text(p.stdout,encoding='utf-8')
        (work/f'{len(commands):02d}.stderr').write_text(p.stderr,encoding='utf-8')
        print(f'{len(commands)}. {label}: {p.returncode}',flush=True)
        if (p.returncode==0)!=success:raise AssertionError(p.stderr)
        return p
    def request(name,value):
        p=work/(name+'.request.json');write(p,value);return p
    source=previous/'dataset.jsonl'
    ensemble=read(previous/'ensemble.request.json')
    ensemble['dataset']=str(source)
    for i,member in enumerate(ensemble['models']):
        member['options']={'native_model':str(previous/f'model-{i}.json'),'device':'cpu'}
    run('CPU model-deviation',[engine,'model-deviation',request('deviation',ensemble),work/'deviations.jsonl'])
    deviations=rows(work/'deviations.jsonl')
    assert deviations[-1]['dataset_sha256']==digest(source)
    ids=[r['id'] for r in deviations[1:-1] if r['candidate']]
    assert len(ids)==6
    select=request('select',{'schema':'omnimd-select-dataset-v1','dataset':str(source),'expected_source_sha256':digest(source),'ids':list(reversed(ids))})
    run('Select reviewed candidates',[engine,'select-dataset',select,work/'selected'])
    selected=work/'selected/dataset.jsonl'
    originals=rows(selected)
    assert [r['data']['id'] for r in originals[1:-1]]==ids
    saved=rows(previous/'01.stdout.txt')
    source_records=rows(source)
    targets={r['data']['id']:e for r,e in zip(source_records[1:-1],saved)}
    assert len(targets)==12
    all_bindings=[]
    def roundtrip(name,data):
        prepare=request(name+'-prepare',{'schema':'omnimd-prepare-labels-v1','dataset':str(data),'expected_source_sha256':digest(data)})
        job=work/(name+'-job')
        run(name+' prepare label job',[engine,'prepare-labels',prepare,job])
        payload=rows(job/'job.jsonl')
        for item in payload[1:-1]:
            identity=item['identity']
            assert identity['identity_sha256']==identity_sha(payload[0]['context'],digest(data),identity)
            all_bindings.append(identity['identity_sha256'])
        header=deepcopy(payload[0]);header['schema']='omnimd-label-return-v1'
        header['provenance']={'provider':'original synthetic replay','method':'saved native CPU software-fixture predictions; no external calculation or physical label claim',
                              'source_sha256':{'saved_predictions':digest(previous/'01.stdout.txt'),'teacher_model':digest(previous/'model-0.json')}}
        response=[header]
        for item in reversed(payload[1:-1]):
            identity=item['identity']
            evaluation=targets[identity['id']]
            response.append({'kind':'frame','identity':identity,'labels':{k:evaluation[k] for k in ('energy','forces','virial')}})
        response.append({'kind':'completed','frames':len(response)-1})
        returned=work/(name+'-returned.jsonl');write_rows(returned,response)
        req=request(name+'-import',{'schema':'omnimd-import-labels-v1','dataset':str(data),'expected_source_sha256':digest(data),'labels':str(returned),'expected_labels_sha256':digest(returned)})
        output=work/(name+'-labeled')
        run(name+' import returned labels',[engine,'import-labels',req,output])
        receipt=read(output/'receipt.json')
        assert receipt['output_sha256']==digest(output/'dataset.jsonl')
        assert receipt['label_return_sha256']==digest(output/'labels.jsonl')==digest(returned)
        assert receipt['request_sha256']==digest(output/'request.json')==digest(req)
        before,after=rows(data),rows(output/'dataset.jsonl')
        assert before[0]==after[0]
        for old,new in zip(before[1:-1],after[1:-1]):
            for key in ('index','id','frame','parameters'):assert old['data'].get(key)==new['data'].get(key)
            assert new['data']['labels']=={k:targets[old['data']['id']][k] for k in ('energy','forces','virial')}
        return output,receipt,response,req
    labeled,receipt,response,import_req=roundtrip('v1',selected)
    run('Inspect completed labeled dataset',[engine,'inspect-dataset',labeled/'dataset.jsonl',work/'inspection.json'])
    assert read(work/'inspection.json')['labels']['energy_and_force_frames']==6
    config={'split':{'mode':'explicit','validation_ids':[ids[-1]]},'fit_bias':False,'energy_weight':1.,'force_weight':1.,'energy_scale_ev_per_atom':.1,'force_scale_ev_per_angstrom':.01,'ridge_inverse_ev2':0.,'rank_rtol':1e-12}
    train={'schema':'omnimd-native-readout-train-v1','device':'cpu','initial_model':str(previous/'model-1.json'),'dataset':str(labeled/'dataset.jsonl'),'output_directory':str(work/'trained'),'config':config}
    run('CPU retrain on returned labels',[engine,'train-native',request('train',train)])
    trained=read(work/'trained/report.json')
    assert len(trained['training_frames'])==5 and len(trained['validation_frames'])==1
    assert trained['dataset_sha256']==digest(labeled/'dataset.jsonl')
    assert trained['final_validation']['energy_per_atom_rmse_ev']<1e-10
    assert trained['final_validation']['force_component_rmse_ev_per_angstrom']<1e-10
    # v2 fixture tests identity and IO only, not a parameter-dependent physical calculator.
    v2=deepcopy(originals);v2[0]['header']['schema']='omnimd-dataset-jsonl-v2'
    v2[0]['header']['parameter_layout']={'fparam':[{'name':'condition','unit':'fixture'}],'aparam':[{'name':'atomic_condition','unit':'fixture'}]}
    for i,item in enumerate(v2[1:-1]):
        item['data']['parameters']={'frame_parameter_policy':'model_default' if i==5 else 'explicit','fparam':[] if i==5 else [300.+i],
                                   'aparam':[[.125*(j+1)] for j in range(len(item['data']['frame']['species']))]}
    v2[0]['header']['source']='original synthetic parameter identity fixture only; no parameter-dependent calculator validation'
    v2file=work/'v2.jsonl';write_rows(v2file,v2)
    v2out,v2receipt,_,_=roundtrip('v2',v2file)
    run('Inspect preserved v2 dataset',[engine,'inspect-dataset',v2out/'dataset.jsonl',work/'v2-inspection.json'])
    # One end-to-end stale geometry rejection plus duplicate/conflict/refusal routing.
    rejected=[]
    for case in ('coordinate','duplicate','conflict','wrong-source-hash','wrong-return-hash'):
        bad=deepcopy(response);req=read(import_req)
        if case=='coordinate':bad[1]['identity']['frame']['positions'][0][0]+=.01
        if case=='duplicate':bad.insert(2,deepcopy(bad[1]))
        if case=='conflict':
            known=next(r['data']['id'] for r in originals[1:-1] if r['data']['labels']['energy'] is not None)
            next(r for r in bad[1:-1] if r['identity']['id']==known)['labels']['energy']+=1.
        path=work/(case+'.jsonl');write_rows(path,bad)
        req.update(labels=str(path),expected_labels_sha256=digest(path))
        if case=='wrong-source-hash':req['expected_source_sha256']='00'*32
        if case=='wrong-return-hash':req['expected_labels_sha256']='00'*32
        output=work/(case+'-output')
        run('Reject '+case,[engine,'import-labels',request(case,req),output],False)
        assert not output.exists();rejected.append(case)
    result={'schema':'omnimd-external-label-replay-v1','status':'passed','device':'cpu','CUDA_VISIBLE_DEVICES':'-1',
            'scope':'Original saved synthetic software fixtures; no external calculation or scientific accuracy claim. v2 is an identity/IO check only.',
            'source_frames':12,'selected_frames':6,'initial_labeled_frames':3,'final_labeled_frames':6,'training_frames':5,'validation_frames':1,
            'source_order_preserved':True,'v2_parameters_preserved':True,'independent_binary_identity_checks':len(all_bindings),'selected_ids':ids,
            'training_data_rank':trained['data_rank'],'initial_validation':trained['initial_validation'],'final_validation':trained['final_validation'],
            'native_v1_reload_max_abs_energy_force_virial':trained['native_v1_reload_max_abs_energy_force_virial'],
            'sha256':{'engine':digest(engine),'trainer':digest(engine.with_name('omnimd-native-train'+engine.suffix)),'script':digest(__file__),
                      'source':digest(source),'selected':digest(selected),'returned':receipt['label_return_sha256'],'labeled':receipt['output_sha256'],
                      'v2_labeled':v2receipt['output_sha256'],'trained_model':digest(work/'trained/model.json')},
            'rejections':rejected,'commands':commands}
    write(work/'report.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
