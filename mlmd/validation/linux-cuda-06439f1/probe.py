"""One opt-in native CUDA Linux probe; existing attempt markers prevent retries."""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/work')
BIN = ROOT / 'target/release'
MODEL = Path('/src/mlmd/native/fixtures/full-tanh-v1.json')
FRAMES = Path('/src/mlmd/native/fixtures/frame.jsonl')

def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def execute(name, command, data=None):
    save(ROOT / (name + '.attempt.json'), {'command': command, 'started_unix': time.time()})
    started = time.perf_counter()
    try:
        result = subprocess.run(command, input=data, text=True, capture_output=True, timeout=90)
        elapsed = time.perf_counter() - started
    except subprocess.TimeoutExpired as error:
        save(ROOT / (name + '.failure.json'), {'reason': 'timeout', 'timeout_seconds': 90,
             'stdout': str(error.stdout), 'stderr': str(error.stderr)})
        raise
    (ROOT / (name + '.stdout.txt')).write_text(result.stdout)
    (ROOT / (name + '.stderr.txt')).write_text(result.stderr)
    if result.returncode != 0:
        save(ROOT / (name + '.failure.json'), {'exit_code': result.returncode, 'stderr': result.stderr,
                                               'seconds': elapsed, 'retry_permitted': False})
        raise RuntimeError(f'{name} failed: {result.returncode}: {result.stderr}')
    return result, elapsed

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['compile', 'probe'])
    args = parser.parse_args()
    library = '/opt/deepmd-kit/lib/libnvrtc.so.12'
    if os.environ.get('OMNIMD_NVRTC_LIBRARY') != library:
        raise RuntimeError('set the observed NVRTC path explicitly')
    if args.command == 'compile':
        result, elapsed = execute('nvrtc-compile', [str(BIN / 'omnimd-native-cuda-check'), '--compile-only', str(MODEL)])
        receipt = json.loads(result.stdout)
        if receipt['gpu_executed'] is not False or receipt['ptx_bytes'] <= 0:
            raise RuntimeError('invalid compile-only result')
        receipt['process_seconds'] = elapsed
        save(ROOT / 'compile.report.json', receipt)
        print(json.dumps(receipt))
        return
    if not (ROOT / 'compile.report.json').exists():
        raise RuntimeError('compile-only must pass before the one GPU probe')
    frame_text = FRAMES.read_text()
    frames = [json.loads(line) for line in frame_text.splitlines() if line.strip()]
    if len(frames) != 1 or len(frames[0]['positions']) != 4:
        raise RuntimeError('expected one original four-atom frame')
    cpu, cpu_seconds = execute('probe-cpu', [str(BIN / 'omnimd-native-eval'), str(MODEL)], frame_text)
    gpu, gpu_seconds = execute('probe-gpu', [str(BIN / 'omnimd-native-cuda-eval'), str(MODEL)], frame_text)
    a, b = json.loads(cpu.stdout), json.loads(gpu.stdout)
    errors = {}
    for name in ('energy', 'forces', 'virial'):
        aa = [a[name]] if name == 'energy' else [x for row in a[name] for x in row]
        bb = [b[name]] if name == 'energy' else [x for row in b[name] for x in row]
        expected = {'energy': 1, 'forces': 12, 'virial': 9}[name]
        if len(aa) != expected or len(bb) != expected or a['units'] != b['units'] or a['units'] != 'EvAngstrom':
            raise RuntimeError('output shape or unit mismatch')
        for x, y in zip(aa, bb):
            if not math.isfinite(x) or not math.isfinite(y) or abs(x-y) > 1e-9 + 1e-9*abs(x):
                raise RuntimeError(f'CPU/GPU {name} mismatch: {x} vs {y}')
        errors[name] = max(abs(x-y) for x,y in zip(aa,bb))
    receipt = {'status': 'passed', 'candidate': '06439f1cf91306b4d19c570e20abaeb7e140adf6',
               'model_sha256': digest(MODEL), 'frames_sha256': digest(FRAMES),
               'frames': 1, 'atoms': 4, 'max_absolute_error': errors,
               'tolerance': '1e-9 + 1e-9*abs(CPU) per component',
               'device': gpu.stderr.strip(), 'gpu_evaluations': 1,
               'gpu_execution_evidence': 'Explicit native CUDA evaluator completed its three driver kernels and context synchronization; no CPU fallback exists in this executable.',
               'process_seconds': {'cpu': cpu_seconds, 'cuda': gpu_seconds},
               'timing_scope': 'One cold process per backend; includes startup/JIT, IO and evaluation; no steady-state speed comparison.',
               'binary_sha256': {name: digest(BIN / name) for name in ['omnimd-native-eval','omnimd-native-cuda-eval']}}
    save(ROOT / 'probe.report.json', receipt)
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
