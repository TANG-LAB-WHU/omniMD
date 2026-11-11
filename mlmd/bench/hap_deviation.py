"""Model deviation of a prepared ensemble through `omnimd-mlmd`, timed per frame (host).

    python bench/hap_deviation.py CASE_DIR <target>/release RESULT.json [--other <target>/release]
        [--repeat N] [--modes MODE ...]

`CASE_DIR` comes from `hap_prepare.py`; the engine must be built with
`--features native-gpu`. `model-deviation` runs with every model of the case on
the GPU runtime over the frames of the case, once as they are and once
repeated `--repeat` times. The whole-process time of a run is mostly loading
the models, so the time per frame is the difference of the two runs divided by
the frames they differ by. The deviations of the first run are compared with
those computed directly from DeePMD's reference forces
(`deepmd-ref-model-K.jsonl`): the largest, smallest and mean over atoms of the
standard deviation of the force vector over models.

`--other` times a second build the same way (for example the one before a
change) and reports the largest difference between the deviations of the two.
"""
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

MODES = {'float64': {}, 'float32': {'precision': 'float32'},
         'float64-compressed': {'compress_min_distance': 0.5},
         'float32-compressed': {'precision': 'float32', 'compress_min_distance': 0.5}}


def engine(release):
    exe = release / 'omnimd-mlmd'
    return exe if exe.exists() else exe.with_suffix('.exe')


def run(exe, request, work, name):
    path, output = work / f'{name}.json', work / f'{name}.jsonl'
    path.write_text(json.dumps(request))
    start = time.perf_counter()
    proc = subprocess.run([str(exe), 'model-deviation', str(path), str(output)], capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise SystemExit((proc.stderr.strip().splitlines() or ['engine failed'])[-1][:400])
    records = [json.loads(line) for line in output.read_text().splitlines()]
    output.unlink()
    return elapsed, records


def repeated(dataset, target, times):
    """The frames of `dataset` `times` times over, with indices and IDs of their own."""
    lines = dataset.read_text().splitlines()
    frames = [json.loads(line)['data'] for line in lines[1:-1]]
    with target.open('w') as out:
        out.write(lines[0] + '\n')
        for copy in range(times):
            for data in frames:
                data = dict(data, index=copy * len(frames) + data['index'], id=f"{data['id']}#{copy}")
                out.write(json.dumps({'kind': 'frame', 'data': data}) + '\n')
        out.write(json.dumps({'kind': 'completed', 'frames': times * len(frames), 'source_sha256': {}}) + '\n')
    return times * len(frames)


def reference_deviation(case_dir, models):
    """Per frame (max, min, mean) over atoms of the force std over models."""
    refs = [[json.loads(line) for line in (case_dir / f'deepmd-ref-model-{k}.jsonl').read_text().splitlines()]
            for k in models]
    out = []
    for frame in zip(*refs):
        per_atom = []
        for atom in zip(*[f['forces'] for f in frame]):
            mean = [sum(f[a] for f in atom) / len(atom) for a in range(3)]
            per_atom.append((sum(sum((f[a] - mean[a]) ** 2 for a in range(3)) for f in atom) / len(atom)) ** 0.5)
        out.append((max(per_atom), min(per_atom), sum(per_atom) / len(per_atom)))
    return out


def forces(records):
    return [r['statistics']['force_ev_per_angstrom'] for r in records if r.get('kind') == 'deviation']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('result', type=Path)
    parser.add_argument('--other', type=Path, help='a second build to time and compare')
    parser.add_argument('--repeat', type=int, default=25)
    parser.add_argument('--modes', nargs='*', default=['float64', 'float32-compressed'], choices=list(MODES))
    args = parser.parse_args()
    manifest = json.loads((args.case_dir / 'manifest.json').read_text())
    models = manifest['models']
    expected = reference_deviation(args.case_dir, models)
    builds = {'this': engine(args.release)}
    if args.other:
        builds['other'] = engine(args.other)
    report = {'models': len(models), 'frames': len(manifest['frames']), 'repeat': args.repeat, 'modes': {}}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        once = (args.case_dir / 'dataset.jsonl').resolve()
        many = work / 'many.jsonl'
        many_frames = repeated(once, many, args.repeat)
        for mode in args.modes:
            entry = {}
            for name, exe in builds.items():
                def request(dataset):
                    options = lambda k: {'native_model': str((args.case_dir / f'model-{k}.native.json').resolve()),
                                         'device': 'cuda:0', 'gpu_runtime': MODES[mode]}
                    return {'schema': 'omnimd-model-deviation-v1', 'dataset': str(dataset),
                            'models': [{'id': f'model-{k}', 'options': options(k)} for k in models]}
                brief, records = run(exe, request(once), work, f'{name}-{mode}-once')
                full, long_records = run(exe, request(many), work, f'{name}-{mode}-many')
                ours = forces(records)
                assert len(ours) == len(expected) and len(forces(long_records)) == many_frames
                worst = max(max(abs(o['max'] - e[0]), abs(o['min'] - e[1]), abs(o['mean'] - e[2]))
                            for o, e in zip(ours, expected))
                entry[name] = {'evaluation': records[0].get('evaluation', 'member by member, frame by frame'),
                               'whole_process_seconds': [brief, full],
                               'ms_per_frame': (full - brief) * 1e3 / (many_frames - len(ours)),
                               'max_difference_from_deepmd_force_deviation': worst,
                               'deviations': ours}
                print(f"{mode:19s} {name:5s} {entry[name]['ms_per_frame']:7.3f} ms per frame of {len(models)} models  "
                      f"({brief:.1f} s, {full:.1f} s)  vs DeePMD deviations {worst:.1e} eV/A  "
                      f"[{entry[name]['evaluation']}]", flush=True)
            if 'other' in entry:
                entry['max_difference_between_builds'] = max(
                    abs(a[key] - b[key]) for a, b in zip(entry['this']['deviations'], entry['other']['deviations'])
                    for key in ('max', 'min', 'mean'))
                entry['speedup'] = entry['other']['ms_per_frame'] / entry['this']['ms_per_frame']
                print(f"{mode:19s} builds differ by {entry['max_difference_between_builds']:.1e} eV/A; "
                      f"{entry['speedup']:.2f} times the frames per second", flush=True)
            for build in entry.values():
                if isinstance(build, dict):
                    build.pop('deviations', None)
            report['modes'][mode] = entry
    args.result.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
