"""Time DeepMD-kit 3.1.3 on the benchmark cases (run inside the DeepMD image).

Two routes per case: the Python DeepPot API (one E/F/V evaluation per call) and
LAMMPS `pair_style deepmd` NVE dynamics. Failures such as out-of-memory are
recorded as rows, not raised.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

FIELDS = ['engine', 'route', 'device', 'precision', 'natoms', 'ms_per_eval',
          'us_per_atom_step', 'ns_per_day', 'peak_mem_mb', 'status', 'note']


def repeats_for(natoms):
    return max(3, min(50, 20000 // natoms))


def deeppot_worker(case_dir, model, case, device):
    """Child process: one model, one case, one device. Prints a JSON row."""
    import numpy as np
    from deepmd.infer import DeepPot
    import torch
    data = np.load(case_dir / f'{case}.npz')
    coords = data['coords'].reshape(1, -1)
    cell = data['cell'].reshape(1, 9)
    atype = data['atype'].tolist()
    pot = DeepPot(str(case_dir / model))
    for _ in range(3):
        pot.eval(coords, cell, atype)
    if device == 'gpu':
        torch.cuda.reset_peak_memory_stats()
    n = repeats_for(len(atype))
    start = time.perf_counter()
    for _ in range(n):
        pot.eval(coords, cell, atype)
    elapsed = (time.perf_counter() - start) / n
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == 'gpu' else ''
    print(json.dumps({'ms_per_eval': elapsed * 1e3,
                      'us_per_atom_step': elapsed * 1e6 / len(atype),
                      'peak_mem_mb': peak, 'note': f'{n} timed calls after 3 warmups'}))


def run_deeppot(case_dir, model, case, natoms, device, precision, threads):
    env = dict(os.environ)
    if device == 'cpu':
        env.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS=str(threads),
                   DP_INTRA_OP_PARALLELISM_THREADS=str(threads),
                   DP_INTER_OP_PARALLELISM_THREADS='1')
    row = {'engine': 'deepmd-3.1.3', 'route': 'python-deeppot-pt', 'device': device,
           'precision': precision, 'natoms': natoms}
    proc = subprocess.run([sys.executable, __file__, '--worker', str(case_dir), model, case, device],
                          env=env, capture_output=True, text=True)
    if proc.returncode == 0:
        row.update(json.loads(proc.stdout.strip().splitlines()[-1]), status='ok')
    else:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ['']
        row.update(status='oom' if 'out of memory' in proc.stderr.lower() else 'error', note=tail[0][:200])
    return row


LAMMPS_INPUT = """units metal
boundary p p p
atom_style atomic
read_data {data}
mass 1 15.999
mass 2 1.008
pair_style deepmd {model}
pair_coeff * * O H
velocity all create 300.0 {seed} mom yes rot yes
timestep 0.0005
fix 1 all nve
thermo 100
run 20
run {steps}
"""


def write_lammps_data(path, case_npz):
    import numpy as np
    data = np.load(case_npz)
    coords, atype, cell = data['coords'], data['atype'], data['cell']
    lines = ['water benchmark', '', f'{len(atype)} atoms', '2 atom types',
             f'0 {cell[0, 0]} xlo xhi', f'0 {cell[1, 1]} ylo yhi', f'0 {cell[2, 2]} zlo zhi', '',
             'Atoms # atomic', '']
    lines += [f'{i + 1} {t + 1} {x} {y} {z}' for i, (t, (x, y, z)) in enumerate(zip(atype, coords))]
    path.write_text('\n'.join(lines) + '\n')


def run_lammps(case_dir, work_dir, model, case, natoms, precision):
    steps = max(50, min(1000, 200000 // natoms))
    data = work_dir / f'{case}.data'
    write_lammps_data(data, case_dir / f'{case}.npz')
    script = work_dir / f'{case}-{precision}.in'
    # LAMMPS runs with cwd=work_dir, so every path it reads must be absolute.
    script.write_text(LAMMPS_INPUT.format(data=data.resolve(), model=(case_dir / model).resolve(),
                                          seed=12345, steps=steps))
    row = {'engine': 'deepmd-3.1.3', 'route': 'lammps-pair-deepmd', 'device': 'gpu',
           'precision': precision, 'natoms': natoms}
    # The image runs as root, which Open MPI refuses unless told otherwise.
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT='1', OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1')
    proc = subprocess.run(['lmp', '-in', str(script), '-log', str(script.with_suffix('.log'))],
                          capture_output=True, text=True, cwd=work_dir, env=env)
    perf = re.findall(r'Performance:\s*([\d.]+) ns/day', proc.stdout)
    loop = re.findall(r'Loop time of ([\d.eE+-]+) on .* for (\d+) steps with', proc.stdout)
    if proc.returncode == 0 and perf and loop:
        seconds, nsteps = float(loop[-1][0]), int(loop[-1][1])
        row.update(status='ok', ns_per_day=float(perf[-1]),
                   ms_per_eval=seconds / nsteps * 1e3,
                   us_per_atom_step=seconds / nsteps * 1e6 / natoms,
                   note=f'NVE 0.5 fs, {nsteps} timed steps after 20 warmup steps')
    else:
        text = proc.stdout + proc.stderr
        lines = [l.strip() for l in text.splitlines() if l.strip().strip('-')]
        errors = [l for l in lines if 'error' in l.lower()]
        note = (errors or lines or [''])[0 if errors else -1]
        row.update(status='oom' if 'out of memory' in text.lower() else 'error', note=note[:200])
    return row


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        _, _, case_dir, model, case, device = sys.argv
        deeppot_worker(Path(case_dir), model, case, device)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('output_csv', type=Path)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--cpu-max-atoms', type=int, default=1536)
    parser.add_argument('--skip-lammps', action='store_true')
    parser.add_argument('--skip-deeppot', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.case_dir / 'manifest.json').read_text())
    work_dir = args.case_dir / 'lammps'
    work_dir.mkdir(exist_ok=True)
    # Write each row as it completes so a crash keeps the finished measurements.
    with args.output_csv.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        def record(row):
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)

        for precision, stem in manifest['models'].items():
            model = f'{stem}.pth'
            for case in manifest['cases']:
                name, natoms = case['name'], case['natoms']
                if not args.skip_deeppot:
                    record(run_deeppot(args.case_dir, model, name, natoms, 'gpu', precision, args.cpu_threads))
                if not args.skip_deeppot and natoms <= args.cpu_max_atoms:
                    row = run_deeppot(args.case_dir, model, name, natoms, 'cpu', precision, args.cpu_threads)
                    row['note'] = f"{row.get('note', '')}; {args.cpu_threads} threads".strip('; ')
                    record(row)
                if not args.skip_lammps:
                    record(run_lammps(args.case_dir, work_dir, model, name, natoms, precision))


if __name__ == '__main__':
    main()
