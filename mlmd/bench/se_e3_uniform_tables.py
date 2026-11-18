"""What a table of one step would lose on the embedding nets of an se_e3 descriptor.

    python se_e3_uniform_tables.py MODEL.native.json MIN_DISTANCE OUT.json

For every se_e3 descriptor of the model: the range of θ (the product of the normalized directions of two
neighbors) that neighbors from MIN_DISTANCE on reach, and for a quintic table with a knot every STEP of θ over
that range (DeePMD-kit's kind of table; its step for this descriptor is 0.1) the largest difference from the net
in value and in slope at 20,000 random θ, as fractions of the largest value and slope of the net there, and the
size such tables would have. This is why omniMD's tables for these nets have pieces of uneven length
(`native/src/adaptive_table.rs`). Needs numpy; the nets must be tanh or linear layers.
"""
import json
import sys

import numpy as np

STEPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


def descriptors(d):
    if d.get('kind') == 'hybrid':
        for part in d['list']:
            yield from descriptors(part)
    else:
        yield d


def jet(net, x):
    """Value, slope and curvature of every output at the inputs x: [n] -> three of [n][width]."""
    y, dy, ddy = x[:, None], np.ones((len(x), 1)), np.zeros((len(x), 1))
    for layer in net['layers']:
        w, b = np.array(layer['weights']), np.array(layer['bias'])
        z, dz, ddz = y @ w + b, dy @ w, ddy @ w
        if layer['activation'] == 'tanh':
            v = np.tanh(z)
            s = 1 - v * v
            c = -2 * v * s
        elif layer['activation'] == 'linear':
            v, s, c = z, np.ones_like(z), np.zeros_like(z)
        else:
            raise SystemExit(f"activation {layer['activation']}")
        scale = np.array(layer['timestep']) if layer.get('timestep') is not None else 1.0
        out, dout, ddout = v * scale, s * scale * dz, scale * (c * dz * dz + s * ddz)
        if layer['residual'] != 'none':
            repeats = out.shape[1] // y.shape[1]
            out, dout, ddout = out + np.tile(y, repeats), dout + np.tile(dy, repeats), ddout + np.tile(ddy, repeats)
        y, dy, ddy = out, dout, ddout
    return y, dy, ddy


def quintic(step, v, d, dd):
    c0, c1, c2 = v[:-1], d[:-1] * step, dd[:-1] * step * step / 2
    d0 = v[1:] - c0 - c1 - c2
    d1 = d[1:] * step - c1 - 2 * c2
    d2 = dd[1:] * step * step - 2 * c2
    return np.stack([c0, c1, c2, 10 * d0 - 4 * d1 + d2 / 2, -15 * d0 + 7 * d1 - d2, 6 * d0 - 3 * d1 + d2 / 2], -1)


def main():
    path, minimum, out_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    model = json.load(open(path, encoding='utf-8'))
    rng = np.random.default_rng(5)
    parts = []
    for d in descriptors(model['descriptor']):
        if d['kind'] != 'se_e3':
            continue
        widths = [n['std'][1] for n in d['normalization']]
        t = max((minimum - d['rcut_smth']) / (d['rcut'] - d['rcut_smth']), 0.0)
        radial = (t * t * t * (-6 * t * t + 15 * t - 10) + 1) / minimum
        reach = (radial / min(widths)) ** 2
        ntypes = len(d['sel'])
        width = len(d['embedding'][0][0]['layers'][-1]['bias'])
        worst = {step: [0.0, 0.0] for step in STEPS}
        largest_weight = 0.0
        for t in range(ntypes):
            for u in range(t, ntypes):
                net = d['embedding'][t][u]
                largest_weight = max(largest_weight, float(np.abs(np.array(net['layers'][0]['weights'])).max()))
                sample = rng.uniform(-reach, reach, 20000)
                value, slope, _ = jet(net, sample)
                for step in STEPS:
                    intervals = int(np.ceil(2 * reach / step))
                    coefficients = quintic(step, *jet(net, -reach + step * np.arange(intervals + 1)))
                    position = (sample + reach) / step
                    i = np.minimum(position.astype(int), intervals - 1)
                    x = (position - i)[:, None]
                    c = coefficients[i]
                    table = c[..., 0] + x * (c[..., 1] + x * (c[..., 2] + x * (c[..., 3] + x * (c[..., 4] + x * c[..., 5]))))
                    table_slope = (c[..., 1] + x * (2 * c[..., 2] + x * (3 * c[..., 3] + x * (4 * c[..., 4] + x * 5 * c[..., 5])))) / step
                    worst[step][0] = max(worst[step][0], float(np.abs(table - value).max() / np.abs(value).max()))
                    worst[step][1] = max(worst[step][1], float(np.abs(table_slope - slope).max() / np.abs(slope).max()))
        nets = ntypes * (ntypes + 1) // 2
        parts.append({
            'rcut': d['rcut'], 'sel': d['sel'], 'embedding_outputs': width,
            'widths_of_the_directions': widths,
            'theta_reaches': reach,
            'largest_first_layer_weight': largest_weight,
            'by_step': {str(step): {
                'largest_value_difference_over_largest_value': worst[step][0],
                'largest_slope_difference_over_largest_slope': worst[step][1],
                'float32_megabytes_for_all_nets': int(np.ceil(2 * reach / step)) * width * 6 * 4 * nets / 1e6,
            } for step in STEPS},
        })
        print(f"se_e3 to {d['rcut']} A: theta within +-{reach:.0f}, first-layer weights to {largest_weight:.2f}")
        for step in STEPS:
            entry = parts[-1]['by_step'][str(step)]
            print(f"  step {step}: value {worst[step][0]:.1e}, slope {worst[step][1]:.1e}, "
                  f"{entry['float32_megabytes_for_all_nets']:.0f} MB in float32")
    result = {'model': path.replace('\\', '/').split('/')[-1], 'neighbors_from_angstrom': minimum,
              'what': 'quintic tables with a knot every step of theta over the range neighbors from that distance reach, '
                      'against the nets at 20,000 random theta of each net: the worst net of each se_e3 descriptor',
              'descriptors': parts}
    with open(out_path, 'x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, indent=1)
        handle.write('\n')


if __name__ == '__main__':
    main()
