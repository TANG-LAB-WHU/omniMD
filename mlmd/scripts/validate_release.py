#!/usr/bin/env python3
"""One bounded release replay: full training/restart -> data -> prediction -> min -> MD.
Uses existing public regression models plus original synthetic teacher data.
Requires the already-used NumPy for comparisons; performs no downloads.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from validate_structure_workflow import read, records, write_xyz


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(result):
    return {k: v for k, v in result.items() if k not in ('final_extxyz', 'final_dataset')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--native-eval', type=Path, required=True)
    parser.add_argument('--native-train', type=Path, required=True)
    parser.add_argument('--public-request', type=Path, required=True)
    parser.add_argument('--public-native-model', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]

    def invoke(command, name, allowed_failure=False):
        proc = subprocess.run([str(x) for x in command], capture_output=True, text=True)
        (args.work / (name + '.log')).write_text(proc.stdout + proc.stderr, encoding='utf8')
        if proc.returncode and not allowed_failure:
            raise RuntimeError(f'{name} failed; inspect its local log')
        return proc

    def call(command, request, name, allow_nonconverged=False):
        path, out = args.work / (name + '.request.json'), args.work / (name + '.jsonl')
        path.write_text(json.dumps(request, allow_nan=False), encoding='utf8')
        cmd = [args.binary, command, path]
        if command != 'train-native':
            cmd.append(out)
        proc = invoke(cmd, name, allow_nonconverged)
        if allow_nonconverged:
            end = records(out)[-1]
            assert end['kind'] == 'completed'
            assert (proc.returncode == 0) == end['converged']
        return out

    capabilities = json.loads(invoke([args.binary, 'capabilities'], 'capabilities').stdout)
    assert capabilities['training']['full_network']
    assert 'omnimd-native-full-train-v1' in capabilities['training']['request_schemas']
    training = args.work / 'training'
    invoke([sys.executable, root / 'native/tools/replay_full_training.py',
            '--eval-exe', args.native_eval, '--train-exe', args.native_train,
            '--output-dir', training], 'full-training')
    training_receipt = read(training / 'acceptance.json')

    # Exercise the unified CLI using the exact same dataset and restart settings.
    # This one additional Adam update is separate from the 300-step acceptance.
    request = deepcopy(read(training / 'continuous.request.json'))
    request.pop('initial_model')
    request.update(dataset=str(training / 'dataset.jsonl'),
                   checkpoint=str(training / 'continuous/checkpoint.json'),
                   output_directory=str(args.work / 'unified-step-301'))
    request['config']['steps'] = 1
    call('train-native', request, 'unified-training')
    checkpoint = read(args.work / 'unified-step-301/checkpoint.json')
    assert checkpoint['step'] == 301
    options = {'native_model': str(args.work / 'unified-step-301/model.json')}

    original = records(training / 'dataset.jsonl')
    selected = [r['data'] for r in original if r['kind'] == 'frame' and r['data']['frame']['cell']]
    source = args.work / 'synthetic.extxyz'
    write_xyz(source, selected)
    dataset = call('import-xyz', {'input': str(source), 'format': 'Extxyz',
        'units': 'EvAngstrom', 'pbc': 'Periodic',
        'type_map': original[0]['header']['type_map']}, 'imported')
    imported = records(dataset)[1:-1]
    assert len(imported) == len(selected) == 12
    for actual, expected in zip(imported, selected):
        assert actual['data']['frame'] == expected['frame']
        assert actual['data']['labels'] == expected['labels']
    predicted = records(call('batch-eval', {'options': options, 'dataset': str(dataset)}, 'predictions'))
    assert predicted[-1]['kind'] == 'completed'

    settings = {'force_tolerance_ev_per_angstrom': .005, 'energy_tolerance_ev': 1e-6,
        'initial_step_angstrom2_per_ev': .02, 'max_displacement_angstrom': .02,
        'max_backtracks': 32, 'max_iterations': 50, 'armijo': 1e-4}
    minimum = records(call('minimize', {'options': options,
        'dataset_frame': {'dataset': str(dataset), 'index': 0},
        'minimization': settings}, 'trained-minimum', True))
    assert minimum[-1]['final_energy_ev'] <= minimum[-1]['initial_energy_ev']
    geometry_dataset = Path(minimum[-1]['final_dataset'])
    geometry = records(geometry_dataset)[1]['data']
    assert geometry['frame']['cell'] == selected[0]['frame']['cell']
    assert all(v is None for v in geometry['labels'].values())
    reimport = records(call('import-xyz', {'input': minimum[-1]['final_extxyz'], 'format': 'Extxyz',
        'units': 'EvAngstrom', 'pbc': 'Periodic',
        'type_map': original[0]['header']['type_map']}, 'reimported'))
    assert reimport[1]['data']['frame'] == geometry['frame']
    masses = [{'O': 15.999, 'H': 1.008}[s] for s in geometry['frame']['species']]
    md = records(call('run', {'options': options,
        'dataset_frame': {'dataset': str(geometry_dataset), 'index': 0},
        'masses': masses, 'velocities': [[0., 0., 0.] for _ in masses],
        'dynamics_units': 'EvAngstromFsU', 'timestep_fs': .0001,
        'steps': 5, 'sample_every': 1}, 'md'))
    assert md[-1]['kind'] == 'completed'

    public = read(args.public_request)
    public_settings = dict(settings, force_tolerance_ev_per_angstrom=.01,
        energy_tolerance_ev=1e-5, max_displacement_angstrom=.05, max_iterations=200)
    public_results = {}
    for backend, model_options in [('capi', public['options']),
                                  ('native', {'native_model': str(args.public_native_model)})]:
        data = records(call('minimize', {'options': model_options, 'frame': public['frames'][2],
            'minimization': public_settings}, 'public-' + backend))
        assert data[-1]['converged']
        public_results[backend] = data
    capi, native = [public_results[b][-2]['evaluation'] for b in ('capi', 'native')]
    parity = {key: float(np.max(np.abs(np.asarray(capi[key]) - np.asarray(native[key]))))
              for key in ('energy', 'forces', 'virial')}
    assert max(parity.values()) < 1e-9

    report = {'scope': 'bounded CPU tanh training and software integration; no chemical accuracy claim',
        'full_training': training_receipt, 'unified_training_resume_step': checkpoint['step'],
        'imported_periodic_frames': len(selected), 'batch_metrics': predicted[-1]['metrics'],
        'trained_minimization': portable(minimum[-1]),
        'md_steps': md[-1]['steps'], 'md_max_energy_drift_ev': md[-1]['max_absolute_energy_drift_ev'],
        'public_minimization': {b: portable(v[-1]) for b, v in public_results.items()},
        'public_final_max_absolute_backend_difference': parity,
        'engine_sha256': digest(args.binary), 'replay_sha256': digest(Path(__file__)),
        'checks': ['300-step learning and exact 150+150 restart', 'native export reload',
                   'unified CLI full-training checkpoint request', 'XYZ labels/geometry roundtrip',
                   'batch prediction', 'fixed-cell minimization and honest exit status',
                   'unlabeled structure export and exact reimport', 'five-step MD continuation',
                   'public C API/native converged minimum parity']}
    (args.work / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'passed': True, 'training_steps': 300,
                      'unified_resume_step': 301, 'public_parity': parity,
                      'trained_minimization_status': minimum[-1]['status']}))


if __name__ == '__main__':
    main()
