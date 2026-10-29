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


def validate_timing(timing, frames, label, batch_size, new_output):
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
    if new_output:
        calls_per_pass = (frames + batch_size - 1) // batch_size
        expected_calls = {
            'batch_size': batch_size,
            'first_pass_capi_calls': calls_per_pass,
            'warmup_capi_calls': calls_per_pass * timing['warmup_passes'],
            'steady_capi_calls': calls_per_pass * timing['steady_passes'],
            'total_capi_calls': calls_per_pass * (1 + timing['warmup_passes'] + timing['steady_passes']),
        }
        for key, expected in expected_calls.items():
            require(positive_integer(timing.get(key)) and timing[key] == expected,
                    f'{label}: timing {key} disagrees with batch size and pass counts')


def validate_execution(result, frames, label):
    new_fields = ('batch_size', 'capi_calls', 'model_details')
    timing = result.get('timing')
    timing_fields = ('batch_size', 'first_pass_capi_calls', 'warmup_capi_calls',
                     'steady_capi_calls', 'total_capi_calls')
    new_output = any(key in result for key in new_fields) or (
        isinstance(timing, dict) and any(key in timing for key in timing_fields))
    if new_output:
        require(all(key in result for key in new_fields),
                f'{label}: new output requires batch_size, capi_calls and model_details')
        details = result['model_details']
        require(isinstance(details, dict), f'{label}: model_details must be an object')
        for key in ('frame_parameter_dim', 'atom_parameter_dim'):
            value = details.get(key)
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                    f'{label}: model_details {key} must be a nonnegative integer')
        default_fparam = details.get('has_default_fparam')
        require(default_fparam is None or isinstance(default_fparam, bool),
                f'{label}: has_default_fparam must be a boolean or null')
    batch_size = result.get('batch_size', 1)
    require(positive_integer(batch_size), f'{label}: batch_size must be a positive integer')
    policy = result.get('frame_parameter_policy', 'explicit')
    require(policy in ('explicit', 'model_default'), f'{label}: unknown frame_parameter_policy')
    if policy == 'model_default':
        details = result.get('model_details', {})
        require(batch_size == 1, f'{label}: model_default requires batch_size=1')
        require(details.get('has_default_fparam') is True
                and positive_integer(details.get('frame_parameter_dim')),
                f'{label}: model_default requires a positive frame_parameter_dim and has_default_fparam=true')
    passes = 1
    if timing is not None:
        validate_timing(timing, frames, label, batch_size, new_output)
        if 'frame_parameter_policy' in timing:
            require(timing['frame_parameter_policy'] == policy,
                    f'{label}: timing frame_parameter_policy disagrees with output policy')
        passes += timing['warmup_passes'] + timing['steady_passes']
    expected_calls = ((frames + batch_size - 1) // batch_size) * passes
    if new_output:
        require(positive_integer(result['capi_calls']) and result['capi_calls'] == expected_calls,
                f'{label}: capi_calls disagrees with batch size and pass counts')
    return batch_size, expected_calls, frames * passes


def compare_values(cv, gv, name, index, errors, atol, rtol):
    require(len(cv) == len(gv), f'frame {index}: {name} shape differs')
    for component, (x, y) in enumerate(zip(cv, gv)):
        require(finite_number(x) and finite_number(y),
                f'frame {index} {name}[{component}]: nonfinite or nonnumeric value')
        errors[name] = max(errors[name], abs(x - y))
        require(math.isclose(x, y, abs_tol=atol, rel_tol=rtol),
                f'frame {index} {name}[{component}]: CPU={x}, CUDA={y}, tolerance exceeded')


def atomic_values(value, atoms, label):
    require(isinstance(value, dict), f'{label}: atomic evaluation must be an object')
    energies = value.get('energies')
    virials = value.get('virials')
    require(isinstance(energies, list) and len(energies) == atoms,
            f'{label}: atomic energies must have one entry per atom')
    require(isinstance(virials, list) and len(virials) == atoms,
            f'{label}: atomic virials must have one matrix per atom')
    return energies, [component for i, virial in enumerate(virials)
                      for component in matrix(virial, 3, f'{label} atom {i} virial')]


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
    cpu_batch, cpu_calls, _ = validate_execution(cpu, frames, 'CPU')
    gpu_batch, gpu_calls, gpu_evaluations = validate_execution(gpu, frames, 'CUDA')
    require(cpu_batch == gpu_batch, 'CPU/CUDA batch_size differs')
    policy = gpu.get('frame_parameter_policy', 'explicit')
    require(cpu.get('frame_parameter_policy', 'explicit') == policy,
            'CPU/CUDA frame_parameter_policy differs')
    require(cpu.get('model_details') == gpu.get('model_details'), 'CPU/CUDA model_details differs')
    require(gpu['cuda_kernel_launches'] >= gpu_calls,
            'kernel count is inconsistent with at least one verified kernel per combined C API call')
    atomic = 'atomic_evaluations' in cpu
    require(atomic == ('atomic_evaluations' in gpu), 'CPU/CUDA atomic output presence differs')
    if atomic:
        for label, result in (('CPU', cpu), ('CUDA', gpu)):
            require(isinstance(result['atomic_evaluations'], list)
                    and len(result['atomic_evaluations']) == frames,
                    f'{label}: atomic_evaluations must contain one entry per frame')
    errors = {name: 0.0 for name in ('energy', 'forces', 'virial')}
    if atomic:
        errors.update(atomic_energy=0.0, atomic_virial=0.0)
    for index, (c, g) in enumerate(zip(cpu['evaluations'], gpu['evaluations'])):
        require(c['units'] == g['units'] == 'EvAngstrom', f'frame {index}: invalid units')
        for name in ('energy', 'forces', 'virial'):
            cv, gv = ([c[name]], [g[name]]) if name == 'energy' else (
                matrix(c[name], 3 if name == 'virial' else None, f'CPU frame {index} {name}'),
                matrix(g[name], 3 if name == 'virial' else None, f'CUDA frame {index} {name}'))
            compare_values(cv, gv, name, index, errors, args.atol, args.rtol)
        if atomic:
            ca = atomic_values(cpu['atomic_evaluations'][index], len(c['forces']), f'CPU frame {index}')
            ga = atomic_values(gpu['atomic_evaluations'][index], len(g['forces']), f'CUDA frame {index}')
            for name, cv, gv in zip(('atomic_energy', 'atomic_virial'), ca, ga):
                compare_values(cv, gv, name, index, errors, args.atol, args.rtol)
    report = {'frames': frames, 'max_absolute_error': errors,
              'cuda_kernel_launches': gpu['cuda_kernel_launches'], 'atol': args.atol, 'rtol': args.rtol,
              'batch_size': gpu_batch, 'cpu_capi_calls': cpu_calls, 'gpu_capi_calls': gpu_calls,
              'frame_parameter_policy': policy,
              'cuda_kernel_counted_capi_calls': gpu_calls,
              'cpu_timing': cpu.get('timing'), 'gpu_timing': gpu.get('timing')}
    if cpu.get('timing') is not None and gpu.get('timing') is not None:
        ct, gt = cpu['timing'], gpu['timing']
        for key in ('steady_evaluations', 'steady_passes', 'warmup_passes', 'scope'):
            require(ct[key] == gt[key], f'CPU/CUDA timing conditions differ: {key}')
        # Retain the legacy frame count; kernel verification is per C API call.
        report['cuda_kernel_counted_evaluations'] = gpu_evaluations
        report['cpu_over_gpu_speed_ratio'] = ct['seconds_per_evaluation'] / gt['seconds_per_evaluation']
        report['gpu_over_cpu_time_ratio'] = gt['seconds_per_evaluation'] / ct['seconds_per_evaluation']
        report['gpu_timing_result'] = ('faster' if report['gpu_over_cpu_time_ratio'] < 1 else
                                       'slower' if report['gpu_over_cpu_time_ratio'] > 1 else 'equal')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
