"""Batches of frames and committees of models on omniMD's GPU se_e2_a path (host).

    python bench/hap_batch.py CASE_DIR <target>/release RESULT.csv [--check REPORT.json]
        [--systems NAME ...] [--batches N ...] [--members N ...] [--modes MODE ...]

`CASE_DIR` comes from `hap_prepare.py`. Timing: for every system (its first
frame), mode, number of models and batch size, `omnimd-native-gpu-eval` times
passes over a batch of copies of the frame (`--bench-batches`: all batch sizes
in one process); the CSV holds microseconds per pass and per frame and model,
the least of `--repeats` runs (the card is shared with a desktop). One model
and one frame is the plain evaluation.

`--check`: all models as one committee evaluate all frames of the case in
batches of 16, and every member's energies, forces and virials are compared
with DeePMD's own for that model, mode by mode.
"""
import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

MODES = {'float64': [], 'float32': ['--precision', 'float32'],
         'float64-compressed': ['--compress', '0.5'],
         'float32-compressed': ['--precision', 'float32', '--compress', '0.5']}


def evaluator(release):
    exe = release / 'omnimd-native-gpu-eval'
    return exe if exe.exists() else exe.with_suffix('.exe')


def environment():
    env = dict(os.environ)
    if env.get('CUDA_PATH'):
        env['PATH'] = os.path.join(env['CUDA_PATH'], 'bin') + os.pathsep + env['PATH']
    return env


def model_flags(case_dir, models):
    flags = [str(case_dir / f'model-{models[0]}.native.json')]
    for k in models[1:]:
        flags += ['--member', str(case_dir / f'model-{k}.native.json')]
    return flags


def time_passes(exe, case_dir, models, mode, frame_line, batches, evaluations):
    """Microseconds per pass over each batch size, in one process (the models load once)."""
    command = [str(exe), *model_flags(case_dir, models), *MODES[mode], '--bench', str(evaluations),
               '--bench-batches', ','.join(map(str, batches))]
    proc = subprocess.run(command, input=frame_line * max(batches), capture_output=True, text=True,
                          env=environment())
    times = {}
    for line in proc.stderr.splitlines():
        if '"ms_per_eval"' in line:
            record = json.loads(line)
            times[record['frames']] = record['ms_per_eval'] * 1000
    if proc.returncode != 0 or sorted(times) != sorted(batches):
        raise SystemExit(f"{' '.join(command)}\n{proc.stderr}")
    return times


def check(exe, case_dir, manifest, report_path):
    models = manifest['models']
    frames_text = (case_dir / 'frames.jsonl').read_text()
    references = [[json.loads(line) for line in
                   (case_dir / f'deepmd-ref-model-{k}.jsonl').read_text().splitlines()] for k in models]
    report = {'frames': len(manifest['frames']), 'models': len(models), 'batch': 16, 'modes': {}}
    for mode, flags in MODES.items():
        proc = subprocess.run([str(exe), *model_flags(case_dir, models), *flags, '--batch', '16'],
                              input=frames_text, capture_output=True, text=True, env=environment())
        if proc.returncode != 0:
            raise SystemExit(proc.stderr)
        predictions = [json.loads(line) for line in proc.stdout.splitlines()]
        assert len(predictions) == len(manifest['frames']), (len(predictions), mode)
        worst = [{'energy_per_atom': 0.0, 'force': 0.0, 'virial': 0.0} for _ in models]
        for meta, members, *wanted in zip(manifest['frames'], predictions, *references):
            assert len(members) == len(models)
            for record, got, want in zip(worst, members, wanted):
                record['energy_per_atom'] = max(record['energy_per_atom'],
                                                abs(got['energy'] - want['energy']) / meta['natoms'])
                record['force'] = max(record['force'], max(
                    abs(a - b) for fa, fb in zip(got['forces'], want['forces']) for a, b in zip(fa, fb)))
                record['virial'] = max(record['virial'], max(
                    abs(a - b) for ra, rb in zip(got['virial'], want['virial']) for a, b in zip(ra, rb)))
        notes = sorted({line for line in proc.stderr.splitlines() if line.startswith(('warning', 'note'))})
        report['modes'][mode] = {'max_difference_from_deepmd_per_member': worst, 'messages': notes[:5]}
        print(f"committee {mode:19s} dE/atom {max(w['energy_per_atom'] for w in worst):.1e} eV  "
              f"dF {max(w['force'] for w in worst):.1e} eV/A  dV {max(w['virial'] for w in worst):.1e} eV  "
              f"{len(notes)} message(s)", flush=True)
    report_path.write_text(json.dumps(report, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('result', type=Path)
    parser.add_argument('--check', type=Path, help='committee against the DeePMD references; report path')
    parser.add_argument('--systems', nargs='*', help='systems to time (default: three of different size)')
    parser.add_argument('--batches', nargs='*', type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--members', nargs='*', type=int, help='numbers of models (default: one and all)')
    parser.add_argument('--modes', nargs='*', default=['float32-compressed', 'float64'], choices=list(MODES))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--evaluations', type=int, default=2000, help='timed passes for one frame and model')
    args = parser.parse_args()
    manifest = json.loads((args.case_dir / 'manifest.json').read_text())
    exe = evaluator(args.release)
    if args.check:
        check(exe, args.case_dir, manifest, args.check)
    frame_lines = (args.case_dir / 'frames.jsonl').read_text().splitlines(keepends=True)
    first = {}
    for meta, line in zip(manifest['frames'], frame_lines):
        first.setdefault(meta['system'], (meta['natoms'], line))
    systems = args.systems
    if systems is None:
        # The smallest, a middle and the largest system of the case.
        by_size = sorted(first, key=lambda name: first[name][0])
        systems = [by_size[0], by_size[len(by_size) // 2], by_size[-1]]
    members = args.members or sorted({1, len(manifest['models'])})
    with args.result.open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['system', 'natoms', 'mode', 'models', 'frames', 'us_per_pass', 'us_per_frame_and_model'])
        for system in systems:
            natoms, line = first[system]
            for mode in args.modes:
                for count in members:
                    runs = [time_passes(exe, args.case_dir, manifest['models'][:count], mode, line, args.batches,
                                        args.evaluations) for _ in range(args.repeats)]
                    cells = []
                    for batch in args.batches:
                        best = min(run[batch] for run in runs)
                        writer.writerow([system, natoms, mode, count, batch, f'{best:.1f}',
                                         f'{best / (batch * count):.1f}'])
                        cells.append(f'{batch}: {best / (batch * count):.0f}')
                    out.flush()
                    print(f"{system.split('__')[0]:<18} {natoms:>4} atoms  {mode:<19} {count} model(s)  "
                          f"us per frame and model at batch  " + '  '.join(cells), flush=True)


if __name__ == '__main__':
    main()
