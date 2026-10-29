"""Time omniMD native evaluation on the benchmark cases (run on the host).

Each eval CLI reads JSONL frames from stdin. Per-evaluation time is
(min t[K frames] - min t[1 frame]) / (K - 1) over a few repeated runs, which
removes process start-up and model loading (the minimum keeps file-cache and
scheduler noise from making the difference negative) but still includes JSON
parsing of each frame. Rejections (for example
the CUDA subset's size limits) are recorded as rows with status "rejected".
"""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

FIELDS = ['engine', 'route', 'device', 'precision', 'natoms', 'ms_per_eval',
          'us_per_atom_step', 'ns_per_day', 'peak_mem_mb', 'status', 'note']


def timed(exe, model, frame_line, count, timeout):
    start = time.perf_counter()
    proc = subprocess.run([str(exe), str(model)], input=frame_line * count, capture_output=True,
                          text=True, timeout=timeout)
    return time.perf_counter() - start, proc


def bench(exe, route, device, model, case_dir, case, natoms, frames, repeats, timeout):
    row = {'engine': 'omnimd-mlmd', 'route': route, 'device': device, 'precision': 'float64',
           'natoms': natoms}
    line = (case_dir / f'{case}.jsonl').read_text()
    singles, manys = [], []
    try:
        for _ in range(repeats):
            seconds, proc = timed(exe, model, line, 1, timeout)
            if proc.returncode != 0:
                message = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ['']
                return {**row, 'status': 'rejected', 'note': message[0][:200]}
            singles.append(seconds)
            seconds, proc = timed(exe, model, line, frames, timeout)
            if proc.returncode != 0:
                message = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ['']
                return {**row, 'status': 'error', 'note': message[0][:200]}
            manys.append(seconds)
    except subprocess.TimeoutExpired:
        return {**row, 'status': 'timeout', 'note': f'over {timeout} s'}
    per_eval = (min(manys) - min(singles)) / (frames - 1)
    return {**row, 'status': 'ok', 'ms_per_eval': per_eval * 1e3,
            'us_per_atom_step': per_eval * 1e6 / natoms,
            'note': f'(min of {repeats}: {frames} frames - 1 frame)/{frames - 1}; '
                    f'single-frame run incl. load {min(singles):.2f} s; includes JSON parsing'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('bin_dir', type=Path, help='directory with release omnimd-native-*eval binaries')
    parser.add_argument('output_csv', type=Path)
    parser.add_argument('--cpu-max-atoms', type=int, default=1536)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    manifest = json.loads((args.case_dir / 'manifest.json').read_text())
    model = args.case_dir / f"{manifest['models']['float64']}.native.json"
    suffix = '.exe' if (args.bin_dir / 'omnimd-native-eval.exe').exists() else ''
    routes = [('native-cpu', 'cpu', args.bin_dir / f'omnimd-native-eval{suffix}'),
              ('native-cuda', 'gpu', args.bin_dir / f'omnimd-native-cuda-eval{suffix}')]
    with args.output_csv.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for case in manifest['cases']:
            name, natoms = case['name'], case['natoms']
            for route, device, exe in routes:
                if device == 'cpu' and natoms > args.cpu_max_atoms:
                    continue
                frames = 6 if device == 'gpu' or natoms <= 192 else 3
                row = bench(exe, route, device, model, args.case_dir, name, natoms, frames,
                            args.repeats, args.timeout)
                writer.writerow(row)
                handle.flush()
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
