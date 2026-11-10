"""Time the omniMD GPU se_e2_a evaluator on the benchmark cases (host, release
build with `--features gpu`) and check it against DeePMD-kit reference E/F/V.

    python bench_omnimd_gpu.py cases <target>/release results/omnimd-gpu-<tag>.csv \
        [--reference deepmd_ref.jsonl]

Per case, network precision (float64, float32) and embedding mode (exact
networks, then tables from `--compress`) the evaluator runs one warmup, K
timed in-process evaluations (`--bench K`) and one profiled evaluation; rows
use the same columns as bench_deepmd.py. The reference file
comes from `deepmd_reference.py` (DeepMD-kit CPU float64); the error columns
go to a JSON file next to the CSV.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

FIELDS = ['engine', 'route', 'device', 'precision', 'natoms', 'ms_per_eval',
          'us_per_atom_step', 'ns_per_day', 'peak_mem_mb', 'status', 'note']


def errors(e, ref):
    df = [a - b for fa, fb in zip(e['forces'], ref['forces']) for a, b in zip(fa, fb)]
    dv = [a - b for ra, rb in zip(e['virial'], ref['virial']) for a, b in zip(ra, rb)]
    return {'energy_per_atom': abs(e['energy'] - ref['energy']) / len(ref['forces']),
            'force_max': max(abs(x) for x in df),
            'force_rms': (sum(x * x for x in df) / len(df)) ** 0.5,
            'virial_max': max(abs(x) for x in dv)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('output_csv', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--compress', default='0.5',
                        help='shortest neighbor distance the embedding tables cover, in angstrom')
    args = parser.parse_args()
    manifest = json.loads((args.case_dir / 'manifest.json').read_text())
    model = args.case_dir / f"{manifest['models']['float64']}.native.json"
    exe = args.release / 'omnimd-native-gpu-eval'
    if not exe.exists():
        exe = exe.with_suffix('.exe')
    refs = {}
    if args.reference:
        for line in args.reference.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                refs[r['natoms']] = r
    details = []
    with args.output_csv.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for case in manifest['cases']:
            name, natoms = case['name'], case['natoms']
            frame = (args.case_dir / f'{name}.jsonl').read_text()
            bench = max(10, min(100, 250000 // natoms))
            float64 = None
            modes = [(precision, compressed) for compressed in (False, True)
                     for precision in ('float64', 'float32')]
            for precision, compressed in modes:
                extra = ['--compress', args.compress] if compressed else []
                proc = subprocess.run([str(exe), str(model), '--precision', precision, '--bench', str(bench),
                                       '--profile', *extra], input=frame, capture_output=True, text=True)
                row = {'engine': 'omnimd', 'route': 'gpu-se_e2_a-cli' + ('-compressed' if compressed else ''),
                       'device': 'gpu', 'precision': precision, 'natoms': natoms}
                if proc.returncode != 0:
                    note = (proc.stderr.strip().splitlines() or [''])[-1]
                    row.update(status='oom' if 'out of memory' in note.lower() else 'error', note=note[:200])
                else:
                    info = [json.loads(l) for l in proc.stderr.splitlines() if l.startswith('{')]
                    timing = next(i for i in info if 'ms_per_eval' in i)
                    stages = next(i for i in info if 'stages_ms' in i)['stages_ms']
                    ms = timing['ms_per_eval']
                    row.update(status='ok', ms_per_eval=ms, us_per_atom_step=ms * 1e3 / natoms,
                               note=f"{timing['evaluations']} in-process evaluations after 1 warmup; "
                                    f"network {precision}, geometry and forces float64; "
                                    f"{stages['chunks']} chunk(s)"
                                    + (f"; embedding tables down to {args.compress} A" if compressed else ''))
                    result = json.loads(proc.stdout.strip().splitlines()[-1])
                    detail = {'natoms': natoms, 'precision': precision, 'compressed': compressed,
                              'ms_per_eval': ms, 'stages_ms': stages}
                    if natoms in refs:
                        detail['vs_deepmd_float64'] = errors(result, refs[natoms])
                    if precision == 'float64' and not compressed:
                        float64 = result
                    elif float64 is not None:
                        detail['vs_omnimd_float64'] = errors(result, float64)
                    details.append(detail)
                writer.writerow(row)
                handle.flush()
                print(json.dumps(row), flush=True)
    args.output_csv.with_suffix('.json').write_text(json.dumps(details, indent=1) + '\n')


if __name__ == '__main__':
    main()
