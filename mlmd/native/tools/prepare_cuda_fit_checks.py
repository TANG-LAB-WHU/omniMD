"""Prepare original CUDA cooperative-fit edge fixtures; no GPU calls/downloads.
Generated models and geometries are numerical fixtures, not chemical potentials.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from replay_cuda import frames, write
from replay_cuda_infra import water192


def boundary_model(base):
    model = deepcopy(base)
    serial = 0
    def layer(ni, no, residual, linear=False):
        nonlocal serial
        serial += 1
        return {'weights': [[.08/math.sqrt(ni)*math.sin(.17*(i*no+j+serial))
                             for j in range(no)] for i in range(ni)],
                'bias': [.01*math.cos(.23*(j+serial)) for j in range(no)],
                'activation': 'linear' if linear else 'tanh', 'residual': residual,
                'timestep': None if linear else [.8+.01*math.sin(j+serial) for j in range(no)]}
    widths = [1, 2, 4, 8, 16, 32, 64, 128, 64]
    model['descriptor']['axis_neuron'] = 2
    model['descriptor']['embedding'] = [[{'layers': [layer(a, b, 'concat2' if b == 2*a else 'none')
                for a, b in zip(widths, widths[1:])]} for _ in range(2)] for _ in range(2)]
    model['fitting'] = [{'layers': [layer(128, 128, 'identity') for _ in range(7)]
                         + [layer(128, 1, 'none', True)]} for _ in range(2)]
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--public-model', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    model = json.loads((root/'fixtures/full-tanh-v1.json').read_text())
    base = json.loads((root/'fixtures/frame.jsonl').read_text())
    base['cell'] = None
    single = deepcopy(base)
    single['positions'], single['species'] = [[0., 0., 0.]], ['O']
    swapped = deepcopy(base)
    swapped['species'][0], swapped['species'][1] = swapped['species'][1], swapped['species'][0]
    large = deepcopy(base)
    large['positions'] += [[x+10., y, z] for x, y, z in base['positions']]
    large['species'] += base['species']
    frames(output/'lifecycle.jsonl', [single, base, swapped, swapped, large, base])
    write(output/'boundary-model.json', boundary_model(model))
    frames(output/'boundary-frame.jsonl', [base])
    frame = water192()
    frames(output/'public-water192.jsonl', [frame])
    request = {'options': {'native_model': str(args.public_model.resolve()), 'device': 'cpu'},
               'frame': frame, 'masses': [15.999 if k == 'O' else 1.008 for k in frame['species']],
               'initialize_velocities': {'temperature_kelvin': 300., 'seed': 20261016,
                                         'rescale_to_temperature': True},
               'remove_center_of_mass': True, 'ensemble': {'kind': 'nve'},
               'timestep_fs': .1, 'steps': 20, 'sample_every': 1}
    for device in ('cpu', 'cuda'):
        r = deepcopy(request)
        r['options']['device'] = 'cpu' if device == 'cpu' else 'cuda:0'
        write(output/f'md-{device}.request.json', r)
    print(json.dumps({'gpu_executed': False, 'output': str(output),
                      'expected_first_type_upload_counts': [1, 2, 3, 3, 4, 5],
                      'boundary': {'embedding_width': 64, 'embedding_hidden_width': 128,
                                   'fitting_width': 128, 'depth': 8, 'B_components': 256}}))


if __name__ == '__main__':
    main()
