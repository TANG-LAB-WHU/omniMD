"""Replay synthetic DeepMD .dp/.yaml conversion and C API/native comparison.

Needs an existing trusted local Docker image with DeepMD 3.1.3 + TF + h5py +
PyYAML, the already-built A Linux C API evaluator, and the host native evaluator.
No images or packages are downloaded. All generated data go in a new directory.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, text=True, encoding='utf-8',
                          stdout=subprocess.PIPE, **kwargs).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='already present trusted DeepMD 3.1.3 image')
    parser.add_argument('--capi-exe', required=True, type=Path, help='A Linux omnimd-deepmd executable on host')
    parser.add_argument('--native-exe', required=True, type=Path, help='host omnimd-native-eval executable')
    parser.add_argument('--output-dir', required=True, type=Path, help='new output directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    capi, native, output = args.capi_exe.resolve(), args.native_exe.resolve(), args.output_dir.resolve()
    if not capi.is_file() or not native.is_file(): raise ValueError('both evaluators must be built already')
    image_id = run(['docker','image','inspect','--format','{{.Id}}',args.image]).strip()
    output.mkdir(parents=True, exist_ok=False)
    docker = ['docker','run','--rm','--pull','never','--network','none',
              '--mount',f'type=bind,source={root},target=/native,readonly',
              '--mount',f'type=bind,source={output},target=/out',
              '--mount',f'type=bind,source={capi},target=/capi-eval,readonly']
    for var in ['PYTHONDONTWRITEBYTECODE=1','TF_CPP_MIN_LOG_LEVEL=3','OMP_NUM_THREADS=1',
                'TF_INTRA_OP_PARALLELISM_THREADS=1','TF_INTER_OP_PARALLELISM_THREADS=1']:
        docker.extend(['-e',var])
    docker.append(image_id)
    generation = run([*docker,'python','/native/tools/make_deepmd_fixture.py','/out'])
    reports = []
    for suffix in ('dp','yaml'):
        reports.append(json.loads(run([*docker,'python','/native/tools/convert_deepmd.py',
            '/out/synthetic.'+suffix, '/out/native-from-'+suffix+'.json',
            '--geometry','tensorflow','--units','EvAngstrom'])))
    conversion_checks = json.loads(run([*docker,'python','/native/tools/check_conversion.py',
                                       '/out/synthetic.dp','/out/synthetic.yaml']))
    base = json.loads((root/'fixtures/frame.jsonl').read_text())
    cluster = deepcopy(base); cluster['cell'] = None
    wrapped = deepcopy(base)
    for k in range(3): wrapped['positions'][1][k] += base['cell'][0][k] - base['cell'][1][k]
    frames = [cluster,base,wrapped]
    frame_lines = ''.join(json.dumps(f)+'\n' for f in frames)
    (output/'frames.jsonl').write_text(frame_lines,encoding='utf-8')
    model = json.loads((output/'native-from-dp.json').read_text())
    request = {'options':{'library':'/opt/deepmd-kit/lib/libdeepmd_c.so','model':'/out/synthetic.pb',
                          'device':'Cpu','type_map':model['type_map'],'units':'EvAngstrom'},'frames':frames}
    (output/'capi.request.json').write_text(json.dumps(request),encoding='utf-8')
    run([*docker,'/capi-eval','/out/capi.request.json','/out/capi.json'])
    raw_native = run([native,output/'native-from-dp.json'],input=frame_lines)
    (output/'native.jsonl').write_text(raw_native,encoding='utf-8')
    capi_result = json.loads((output/'capi.json').read_text())
    (output/'capi.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in capi_result['evaluations']),encoding='utf-8')
    comparison = json.loads(run([sys.executable,root/'tools/compare_evaluations.py',
        output/'capi.jsonl',output/'native.jsonl','--energy-atol','1e-12',
        '--force-atol','1e-12','--virial-atol','1e-12','--rtol','1e-12']))
    report = {'fixture':'original synthetic, untrained; numerical compatibility only',
              'deepmd_version':'3.1.3','seeds':[20261014,20261015],
              'docker_image_id':image_id,'coordinate_and_weight_precision':'float64',
              'geometry':'tensorflow','source_rcut_smth':2.1,
              'effective_rcut_smth':model['descriptor']['rcut_smth'],
              'frame_cases':['nonperiodic','triclinic periodic','periodic wrapping'],
              'conversion_checks':conversion_checks,'comparison':comparison,
              'sha256':{p.name:sha(p) for p in [output/'synthetic.dp',output/'synthetic.pb',
                         output/'native-from-dp.json',output/'frames.jsonl',capi,native]},
              'converter_reports':reports}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError,OSError,subprocess.CalledProcessError) as exc:
        print(f'cross-backend replay failed: {exc}',file=sys.stderr)
        raise SystemExit(1)
