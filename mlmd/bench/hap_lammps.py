"""Time LAMMPS `pair_style deepmd` on an exploration directory of the HAP
pipeline (run in the DeePMD-kit image; the directory may be mounted read-only).

    python bench/hap_lammps.py EXPLORE_DIR OUT_DIR results/hap-lammps-<tag>.csv RUN [RUN ...] [--steps 2000]

`EXPLORE_DIR/RUN/conf.lmp` is the starting structure and
`EXPLORE_DIR/models/model-K.pb` are the four models the pipeline explores
with. Each run is timed with one model and with all four (model deviation
every 50 steps), using the pipeline's protocol: NVT at 300 K, 0.5 fs, skin 0
with a rebuild every step, 2 OpenMP threads.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

FIELDS = ['engine', 'route', 'device', 'precision', 'natoms', 'ms_per_eval',
          'us_per_atom_step', 'ns_per_day', 'peak_mem_mb', 'status', 'note']
INPUT = """units metal
boundary p p p
atom_style atomic
read_data conf.lmp
mass 1 40.0780
mass 2 30.9738
mass 3 15.9990
mass 4 1.0080
mass 5 12.0110
pair_style deepmd {models}{deviation}
pair_coeff * *
neighbor 0.0 bin
neigh_modify every 1 delay 0 check yes
velocity all create 300.0 20261329 mom yes rot yes dist gaussian
fix 1 all nvt temp 300.0 300.0 0.05
timestep 0.0005
thermo 50
run 100
run {steps}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('explore_dir', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('output_csv', type=Path)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--steps', type=int, default=2000)
    args = parser.parse_args()
    models = [str((args.explore_dir / 'models' / f'model-{k}.pb').resolve()) for k in range(4)]
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT='1', OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1', OMP_NUM_THREADS='2',
               DP_INTRA_OP_PARALLELISM_THREADS='2', DP_INTER_OP_PARALLELISM_THREADS='1')
    with args.output_csv.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for run in args.runs:
            for count in (1, 4):
                work = (args.out_dir / f'lammps-{run}-{count}').resolve()
                work.mkdir(parents=True, exist_ok=True)
                shutil.copy(args.explore_dir / run / 'conf.lmp', work / 'conf.lmp')
                deviation = ' out_freq 50 out_file model_devi.out' if count > 1 else ''
                (work / 'in.lmp').write_text(INPUT.format(models=' '.join(models[:count]), deviation=deviation,
                                                          steps=args.steps))
                proc = subprocess.run(['lmp', '-in', 'in.lmp', '-log', 'log.lammps'], capture_output=True,
                                      text=True, cwd=work, env=env)
                loop = re.findall(r'Loop time of ([\d.eE+-]+) on .* for (\d+) steps with (\d+) atoms', proc.stdout)
                perf = re.findall(r'Performance:\s*([\d.]+) ns/day', proc.stdout)
                row = {'engine': 'deepmd-3.1.3', 'route': f'lammps-tf-{count}-model', 'device': 'gpu',
                       'precision': 'float64', 'natoms': '', 'note': run}
                if proc.returncode == 0 and loop and perf:
                    seconds, steps, natoms = float(loop[-1][0]), int(loop[-1][1]), int(loop[-1][2])
                    row.update(status='ok', natoms=natoms, ms_per_eval=seconds / steps * 1e3,
                               us_per_atom_step=seconds / steps * 1e6 / natoms, ns_per_day=float(perf[-1]),
                               note=f'{run}; NVT 300 K 0.5 fs, skin 0, {steps} timed steps after 100 warmup steps')
                else:
                    lines = [l.strip() for l in (proc.stdout + proc.stderr).splitlines() if 'error' in l.lower()]
                    row.update(status='error', note=f"{run}: {(lines or ['failed'])[0][:160]}")
                writer.writerow(row)
                handle.flush()
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
