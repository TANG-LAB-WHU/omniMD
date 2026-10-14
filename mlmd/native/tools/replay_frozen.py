"""Replay an existing supported frozen graph against A's C API and native Rust.

Requires existing executables, input model and JSONL frames; no download, training,
package installation, or network. Every generated artifact goes in a new directory.
"""
import argparse
import json
from pathlib import Path
import sys
from replay_deepmd import run, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pb','frames','capi-exe','native-exe','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--image',required=True)
    parser.add_argument('--units',choices=['EvAngstrom'],required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    capi, native = args.capi_exe.resolve(), args.native_exe.resolve()
    pb, frames, output = args.pb.resolve(), args.frames.resolve(), args.output_dir.resolve()
    for p in (capi,native,pb,frames):
        if not p.is_file(): raise ValueError(f'input does not exist: {p}')
    parsed_frames = [json.loads(line) for line in frames.read_text().splitlines() if line.strip()]
    if not parsed_frames: raise ValueError('empty frame input')
    image_id = run(['docker','image','inspect','--format','{{.Id}}',args.image]).strip()
    output.mkdir(parents=True,exist_ok=False)
    docker = ['docker','run','--rm','--pull','never','--network','none',
              '--mount',f'type=bind,source={root},target=/native,readonly',
              '--mount',f'type=bind,source={pb},target=/input.pb,readonly',
              '--mount',f'type=bind,source={capi},target=/capi-eval,readonly',
              '--mount',f'type=bind,source={output},target=/out']
    for var in ['PYTHONDONTWRITEBYTECODE=1','TF_CPP_MIN_LOG_LEVEL=3','OMP_NUM_THREADS=1',
                'TF_INTRA_OP_PARALLELISM_THREADS=1','TF_INTER_OP_PARALLELISM_THREADS=1']:
        docker.extend(['-e',var])
    docker.append(image_id)
    conversion = json.loads(run([*docker,'python','/native/tools/export_frozen.py',
                                '/input.pb','/out/native.json','--units',args.units]))
    frame_lines = ''.join(json.dumps(f)+'\n' for f in parsed_frames)
    (output/'frames.jsonl').write_text(frame_lines,encoding='utf-8')
    model = json.loads((output/'native.json').read_text())
    request = {'options':{'library':'/opt/deepmd-kit/lib/libdeepmd_c.so','model':'/input.pb',
                         'device':'Cpu','type_map':model['type_map'],'units':args.units},'frames':parsed_frames}
    (output/'capi.request.json').write_text(json.dumps(request),encoding='utf-8')
    run([*docker,'/capi-eval','/out/capi.request.json','/out/capi.json'])
    raw_native = run([native,output/'native.json'],input=frame_lines)
    (output/'native.jsonl').write_text(raw_native,encoding='utf-8')
    capi_result = json.loads((output/'capi.json').read_text())
    (output/'capi.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in capi_result['evaluations']),encoding='utf-8')
    comparison = json.loads(run([sys.executable,root/'tools/compare_evaluations.py',
        output/'capi.jsonl',output/'native.jsonl','--energy-atol','1e-12',
        '--force-atol','1e-12','--virial-atol','1e-12','--rtol','1e-12']))
    report = {'scope':'supported frozen graph numerical compatibility; no chemical accuracy or speed claim',
              'reference_deepmd_version':'3.1.3','docker_image_id':image_id,
              'conversion':conversion,'comparison':comparison,
              'sha256':{name:sha(path) for name,path in [('input.pb',pb),('native.json',output/'native.json'),
                       ('frames.jsonl',output/'frames.jsonl'),('capi_executable',capi),('native_executable',native)]}}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
