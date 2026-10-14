#!/usr/bin/env python3
"""Compare existing CPU/CUDA evaluator outputs. Does not invoke a GPU or model."""
import argparse
import json
import math
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument('cpu', type=Path)
p.add_argument('cuda', type=Path)
p.add_argument('--atol', type=float, default=1e-8)
p.add_argument('--rtol', type=float, default=1e-8)
a = p.parse_args()
cpu, gpu = (json.loads(path.read_text()) for path in (a.cpu, a.cuda))
assert gpu['metadata']['backend'] == 'deepmd-c-api-cuda-kernels-verified'
assert gpu['cuda_kernel_launches'] > 0
assert cpu['metadata']['backend'] == 'deepmd-c-api'
assert cpu['metadata']['type_map'] == gpu['metadata']['type_map']
assert cpu['metadata']['precision'] == gpu['metadata']['precision']
assert len(cpu['evaluations']) == len(gpu['evaluations']) > 0

def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield value

errors = {name: 0.0 for name in ('energy', 'forces', 'virial')}
for c, g in zip(cpu['evaluations'], gpu['evaluations']):
    assert c['units'] == g['units'] == 'EvAngstrom'
    for name in errors:
        cv, gv = list(flatten(c[name])), list(flatten(g[name]))
        assert len(cv) == len(gv)
        for x, y in zip(cv, gv):
            assert math.isfinite(x) and math.isfinite(y)
            errors[name] = max(errors[name], abs(x-y))
            assert math.isclose(x, y, abs_tol=a.atol, rel_tol=a.rtol), (name, x, y)
report = {'frames': len(cpu['evaluations']), 'max_absolute_error': errors,
          'cuda_kernel_launches': gpu['cuda_kernel_launches'], 'atol': a.atol, 'rtol': a.rtol,
          'cpu_timing': cpu.get('timing'), 'gpu_timing': gpu.get('timing')}
if cpu.get('timing') and gpu.get('timing'):
    report['cpu_over_gpu_speed_ratio'] = cpu['timing']['seconds_per_evaluation'] / gpu['timing']['seconds_per_evaluation']
print(json.dumps(report, indent=2))
