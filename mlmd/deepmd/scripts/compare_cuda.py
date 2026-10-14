#!/usr/bin/env python3
"""Compare existing CPU/CUDA outputs from the same model and input frames.

Does not invoke a GPU or model. The caller must establish model/input identity;
the evaluator output does not contain those hashes. Tolerance is
max(atol, rtol * max(abs(cpu), abs(cuda))), including near-zero values.
"""
import argparse
import json
import math
from pathlib import Path


def require(condition, message):
    # These checks must also run under python -O / PYTHONOPTIMIZE.
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def matrix(value, rows, label):
    require(isinstance(value, list) and len(value) > 0, f'{label}: expected nonempty matrix')
    require(rows is None or len(value) == rows, f'{label}: invalid row count')
    require(all(isinstance(row, list) and len(row) == 3 for row in value),
            f'{label}: every row must have three entries')
    return [item for row in value for item in row]


def validate_timing(timing, frames, label):
    require(isinstance(timing, dict), f'{label}: timing must be an object')
    for key in ('steady_evaluations', 'steady_passes', 'warmup_passes'):
        require(positive_integer(timing.get(key)), f'{label}: {key} must be a positive integer')
    require(timing['steady_evaluations'] == timing['steady_passes'] * frames,
            f'{label}: steady_evaluations disagrees with passes * frames')
    for key in ('load_seconds', 'first_pass_seconds', 'warmup_seconds',
                'steady_seconds', 'seconds_per_evaluation'):
        value = timing.get(key)
        require(finite_number(value) and value > 0, f'{label}: {key} must be finite and positive')
    expected = timing['steady_seconds'] / timing['steady_evaluations']
    require(math.isclose(timing['seconds_per_evaluation'], expected, rel_tol=1e-12, abs_tol=1e-15),
            f'{label}: seconds_per_evaluation disagrees with steady total / evaluations')
    require(isinstance(timing.get('scope'), str) and timing['scope'], f'{label}: timing scope is required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cpu', type=Path)
    parser.add_argument('cuda', type=Path)
    parser.add_argument('--atol', type=float, default=1e-8)
    parser.add_argument('--rtol', type=float, default=1e-8)
    args = parser.parse_args()
    require(finite_number(args.atol) and args.atol >= 0, 'atol must be finite and nonnegative')
    require(finite_number(args.rtol) and args.rtol >= 0, 'rtol must be finite and nonnegative')
    cpu, gpu = (json.loads(path.read_text(encoding='utf-8')) for path in (args.cpu, args.cuda))
    require(gpu['metadata']['backend'] == 'deepmd-c-api-cuda-kernels-verified',
            'CUDA output lacks verified kernel execution metadata')
    require(positive_integer(gpu.get('cuda_kernel_launches')), 'CUDA output lacks successful kernel launches')
    require(cpu['metadata']['backend'] == 'deepmd-c-api', 'CPU output must use the CPU C API backend')
    for key in ('type_map', 'precision', 'cutoff', 'units'):
        require(cpu['metadata'][key] == gpu['metadata'][key], f'CPU/CUDA metadata differs: {key}')
    require(isinstance(cpu['evaluations'], list) and isinstance(gpu['evaluations'], list),
            'evaluations must be arrays')
    frames = len(cpu['evaluations'])
    require(frames == len(gpu['evaluations']) and frames > 0, 'CPU/CUDA frame count differs or is empty')
    errors = {name: 0.0 for name in ('energy', 'forces', 'virial')}
    for index, (c, g) in enumerate(zip(cpu['evaluations'], gpu['evaluations'])):
        require(c['units'] == g['units'] == 'EvAngstrom', f'frame {index}: invalid units')
        for name in errors:
            cv, gv = ([c[name]], [g[name]]) if name == 'energy' else (
                matrix(c[name], 3 if name == 'virial' else None, f'CPU frame {index} {name}'),
                matrix(g[name], 3 if name == 'virial' else None, f'CUDA frame {index} {name}'))
            require(len(cv) == len(gv), f'frame {index}: {name} shape differs')
            for component, (x, y) in enumerate(zip(cv, gv)):
                require(finite_number(x) and finite_number(y),
                        f'frame {index} {name}[{component}]: nonfinite or nonnumeric value')
                errors[name] = max(errors[name], abs(x - y))
                require(math.isclose(x, y, abs_tol=args.atol, rel_tol=args.rtol),
                        f'frame {index} {name}[{component}]: CPU={x}, CUDA={y}, tolerance exceeded')
    report = {'frames': frames, 'max_absolute_error': errors,
              'cuda_kernel_launches': gpu['cuda_kernel_launches'], 'atol': args.atol, 'rtol': args.rtol,
              'cpu_timing': cpu.get('timing'), 'gpu_timing': gpu.get('timing')}
    for label, result in (('CPU', cpu), ('CUDA', gpu)):
        if result.get('timing') is not None:
            validate_timing(result['timing'], frames, label)
    if cpu.get('timing') is not None and gpu.get('timing') is not None:
        ct, gt = cpu['timing'], gpu['timing']
        for key in ('steady_evaluations', 'steady_passes', 'warmup_passes', 'scope'):
            require(ct[key] == gt[key], f'CPU/CUDA timing conditions differ: {key}')
        # Kernel count covers first pass, warmup and steady evaluations together.
        total_evaluations = frames * (1 + gt['warmup_passes'] + gt['steady_passes'])
        require(gpu['cuda_kernel_launches'] >= total_evaluations,
                'kernel count is inconsistent with at least one verified kernel per evaluation')
        report['cuda_kernel_counted_evaluations'] = total_evaluations
        report['cpu_over_gpu_speed_ratio'] = ct['seconds_per_evaluation'] / gt['seconds_per_evaluation']
        report['gpu_over_cpu_time_ratio'] = gt['seconds_per_evaluation'] / ct['seconds_per_evaluation']
        report['gpu_timing_result'] = ('faster' if report['gpu_over_cpu_time_ratio'] < 1 else
                                       'slower' if report['gpu_over_cpu_time_ratio'] > 1 else 'equal')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
