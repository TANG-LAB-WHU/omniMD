"""Check omniMD's GPU se_e2_a path against DeePMD on a prepared ensemble (host).

    python bench/hap_check.py CASE_DIR <target>/release REPORT.json [--recorded SEALED_TEST.json]

`CASE_DIR` comes from `hap_prepare.py`. For every model and every mode
(float64 or float32 networks, exact or compressed embedding) all frames are
evaluated with `omnimd-native-gpu-eval` and compared with DeePMD's own energy,
forces and virial. The report also holds the errors against the dataset labels
per system (energy RMSE per atom, force RMSE), computed identically from
DeePMD's and from omniMD's predictions, and `--recorded` compares them with a
pipeline record of the form {model: {system: {energy_rmse_eV_atom,
force_rmse_eV_A, frames}}}.
"""
import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

MODES = [('float64', []), ('float32', ['--precision', 'float32']),
         ('float64-compressed', ['--compress', '0.5']),
         ('float32-compressed', ['--precision', 'float32', '--compress', '0.5'])]


def label_errors(manifest, labels, predictions):
    """Per system: energy RMSE per atom and force RMSE against the labels."""
    energy, force = defaultdict(list), defaultdict(list)
    for meta, label, prediction in zip(manifest['frames'], labels, predictions):
        energy[meta['system']].append((prediction['energy'] - label['energy']) / meta['natoms'])
        force[meta['system']].extend(a - b for fa, fb in zip(prediction['forces'], label['forces'])
                                     for a, b in zip(fa, fb))
    rmse = lambda values: (sum(v * v for v in values) / len(values)) ** 0.5
    return {system: {'energy_rmse_eV_atom': rmse(energy[system]), 'force_rmse_eV_A': rmse(force[system]),
                     'frames': len(energy[system])} for system in energy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--recorded', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.case_dir / 'manifest.json').read_text())
    exe = args.release / 'omnimd-native-gpu-eval'
    if not exe.exists():
        exe = exe.with_suffix('.exe')
    frames_text = (args.case_dir / 'frames.jsonl').read_text()
    labels = [json.loads(line)['data']['labels'] for line in (args.case_dir / 'dataset.jsonl').read_text().splitlines()
              if '"kind": "frame"' in line]
    recorded = json.loads(args.recorded.read_text()) if args.recorded else {}
    report = {'frames': len(manifest['frames']), 'atoms': sum(f['natoms'] for f in manifest['frames']),
              'models': {}}
    for k in manifest['models']:
        reference = [json.loads(line) for line in
                     (args.case_dir / f'deepmd-ref-model-{k}.jsonl').read_text().splitlines()]
        deepmd_errors = label_errors(manifest, labels, reference)
        entry = {'deepmd_label_errors': deepmd_errors, 'modes': {}}
        if f'model-{k}' in recorded:
            entry['recorded_vs_deepmd_reference_max_relative'] = max(
                abs(deepmd_errors[system][key] - values[key]) / values[key]
                for system, values in recorded[f'model-{k}'].items() if system in deepmd_errors
                for key in ('energy_rmse_eV_atom', 'force_rmse_eV_A'))
        for mode, flags in MODES:
            start = time.perf_counter()
            proc = subprocess.run([str(exe), str(args.case_dir / f'model-{k}.native.json'), *flags],
                                  input=frames_text, capture_output=True, text=True)
            if proc.returncode != 0:
                entry['modes'][mode] = {'status': 'error', 'note': proc.stderr.strip().splitlines()[-1][:300]}
                print(k, mode, entry['modes'][mode], flush=True)
                continue
            predictions = [json.loads(line) for line in proc.stdout.splitlines()]
            worst = {'energy_per_atom': 0.0, 'force': 0.0, 'virial': 0.0}
            for meta, got, want in zip(manifest['frames'], predictions, reference):
                worst['energy_per_atom'] = max(worst['energy_per_atom'],
                                               abs(got['energy'] - want['energy']) / meta['natoms'])
                worst['force'] = max(worst['force'], max(abs(a - b) for fa, fb in zip(got['forces'], want['forces'])
                                                         for a, b in zip(fa, fb)))
                worst['virial'] = max(worst['virial'], max(abs(a - b) for ra, rb in zip(got['virial'], want['virial'])
                                                           for a, b in zip(ra, rb)))
            ours = label_errors(manifest, labels, predictions)
            metric_shift = max(abs(ours[system][key] - deepmd_errors[system][key])
                               for system in ours for key in ('energy_rmse_eV_atom', 'force_rmse_eV_A'))
            notes = sorted({line for line in proc.stderr.splitlines() if line.startswith(('warning', 'note'))})
            entry['modes'][mode] = {'status': 'ok', 'max_difference_from_deepmd': worst,
                                    'label_errors': ours, 'max_label_error_shift': metric_shift,
                                    'messages': notes[:5], 'wall_seconds': time.perf_counter() - start}
            print(f"model {k} {mode:19s} dE/atom {worst['energy_per_atom']:.1e} eV  dF {worst['force']:.1e} eV/A  "
                  f"dV {worst['virial']:.1e} eV  label-error shift {metric_shift:.1e}  {len(notes)} message(s)",
                  flush=True)
        report['models'][f'model-{k}'] = entry
    args.report.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
