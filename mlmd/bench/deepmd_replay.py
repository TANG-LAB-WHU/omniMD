"""Replay a DeePMD-kit training run in omniMD, frame for frame (host, needs numpy).

    python bench/deepmd_replay.py DEEPMD_RUN_DIR DATASET.jsonl INITIAL.native.json <target>/release OUT_DIR
                                  [--record FILE] [--precision float64] [--seed 5] [--part 5000] [--test TEST.jsonl]

The two trainers draw their own random numbers, so the models they train
from one request differ like two seeds of one trainer do. This takes the
randomness out: omniMD starts from DeePMD-kit's initial model and is given
the frames in the order DeePMD-kit took them, with the schedule and the loss
of that run. If the trainers compute the same step, omniMD then arrives at
DeePMD-kit's trained model (as far as rounding lets it: a long stretch at a
large learning rate amplifies the last digits).

- `DEEPMD_RUN_DIR` is what `deepmd_train.py` left of a run on systems that
  `dataset_to_deepmd.py` wrote from `DATASET.jsonl`: `input.json`,
  `train.log` and `model.native.json`, the trained model converted for
  omniMD.
- The order of the frames is DeePMD-kit's own record of it,
  `Sample_rank_0.txt`: the system and the frame of every step, which it
  writes with the environment variable `SAMPLER_RECORD=1`. It is read from
  the run, or from `--record FILE`: DeePMD-kit's order follows from the
  training seed, the systems and the device, not from the model, so the
  record of any run with those (a small model, for a run that was not
  recorded) is the order of this one. The first step tells: the two
  trainers must have the same errors there, or the replay stops.
- `INITIAL.native.json` is DeePMD-kit's model before the first step: the
  same request with `numb_steps` 1 and both learning rates at 1e-12, run
  through `deepmd_train.py` (DeePMD-kit draws the same networks and takes
  the same statistics for the same seeds).
- The new `OUT_DIR` receives the requests and learning curves of the
  replay, omniMD's model, and `replay.json`: how far DeePMD-kit moved the
  parameters of every part of the model, how far omniMD's are from
  DeePMD-kit's afterwards, the errors of the steps DeePMD-kit displayed in
  both, and with `--test` the errors of both models on those frames
  (`omnimd-native-gpu-test`).

`omnimd-native-gpu-train` visits its frames in shuffled epochs and loads at
most 10,000 of them. So the replay goes in parts of `--part` steps. The
dataset of a part holds one frame per step, placed so that the shuffle of
the part's one epoch (SplitMix64 and Fisher-Yates from the training seed
and the part's first step, pinned by a unit test of the trainer) visits
them in DeePMD-kit's order. A part ends with the snapshot the trainer writes
there (`save_freq`); the next continues from that snapshot and its optimizer
state, and the trainer of a part is stopped once its snapshot is written.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MASK = (1 << 64) - 1
GOLDEN = 0x9e3779b97f4a7c15


def first_epoch(count, seed):
    """The order in which `omnimd-native-gpu-train` visits `count` frames in the first epoch of a run."""
    state, order = seed & MASK, list(range(count))
    for i in range(count - 1, 0, -1):
        state = (state + GOLDEN) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & MASK
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & MASK
        z ^= z >> 31
        j = z % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def numbers(value, out):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, list):
        for item in value:
            numbers(item, out)
    elif isinstance(value, dict):
        for item in value.values():
            numbers(item, out)


def flat(value):
    out = []
    numbers(value, out)
    return np.array(out)


def parts(model):
    """The trained numbers of a native model by what they belong to."""
    descriptor = model['descriptor']
    listed = descriptor['list'] if descriptor['kind'] == 'hybrid' else [descriptor]
    found = {f"embedding nets of descriptor {number} ({part['kind']})": flat(part['embedding'])
             for number, part in enumerate(listed)}
    found['fitting nets'] = flat(model['fitting'])
    return found


def written(path):
    """Whether a file is there and no longer grows."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    size = path.stat().st_size
    time.sleep(1.0)
    return path.stat().st_size == size


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('initial', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--record', type=Path, help="DeePMD-kit's Sample_rank_0.txt, when the run has none")
    parser.add_argument('--precision', default='float64', choices=['float64', 'float32'])
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--part', type=int, default=5000, help='steps per part, at most 10000')
    parser.add_argument('--test', type=Path)
    args = parser.parse_args()
    # The order the trainer's unit test pins (`frame_order_is_a_reproducible_permutation`).
    assert first_epoch(12, 5) == [7, 1, 8, 10, 6, 9, 0, 5, 11, 3, 4, 2]
    if not 0 < args.part <= 10000:
        raise SystemExit('--part is 1 to 10000 steps: the trainer loads at most 10000 frames')

    header, frames = None, {}
    with args.dataset.open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            if record['kind'] == 'header':
                header = record
            elif record['kind'] == 'frame':
                frames[record['data']['id']] = line
    # DeePMD-kit numbers the systems in the order deepmd_train.py gave them, sorted by path, and the
    # frames of a system as dataset_to_deepmd.py wrote them: in the order of the dataset.
    systems = sorted({name.split('/')[0] for name in frames})
    of_system = {system: [name for name in frames if name.split('/')[0] == system] for system in systems}
    config = json.loads((args.run / 'input.json').read_text(encoding='utf-8'))
    total = config['training']['numb_steps']
    steps = []
    record_file = args.record or args.run / 'Sample_rank_0.txt'
    for line in record_file.read_text(encoding='utf-8').splitlines():
        match = re.match(r'Step (\d+): sample system(\d+)\s+frame\[(\d+)\]', line.strip())
        if not match:
            raise SystemExit(f'not a line of one frame per step: {line!r}')
        steps.append(of_system[systems[int(match.group(2))]][int(match.group(3))])
    if len(steps) < total:
        raise SystemExit(f'{len(steps)} recorded steps for a run of {total}')
    steps = steps[:total]

    args.out_dir.mkdir(parents=True, exist_ok=False)
    out = args.out_dir.resolve()
    exe = args.release / 'omnimd-native-gpu-train'
    if not exe.exists():
        exe = exe.with_suffix('.exe')
    rate = config['learning_rate']
    request = {'precision': args.precision,
               'learning_rate': {key: rate[key] for key in ('start_lr', 'stop_lr', 'decay_steps')},
               'loss': {key: value for key, value in config['loss'].items() if key != 'type'}}
    ours, model, state = {}, args.initial.resolve(), None
    for number, first in enumerate(range(0, total, args.part)):
        last = min(first + args.part, total)
        # The trainer seeds the frame order of a run from its seed and its first step, and visits file
        # position order[k] at the k-th step of the run: that position holds the frame of that step.
        order = first_epoch(last - first, args.seed ^ ((first * GOLDEN) & MASK))
        placed = [None] * (last - first)
        for step, position in enumerate(order):
            placed[position] = steps[first + step]
        dataset = out / f'part-{number}.jsonl'
        with dataset.open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(header) + '\n')
            for index, name in enumerate(placed):
                record = json.loads(frames[name])
                # An id of its own for every line; the group (up to the first '/') stays the system.
                record['data'].update(index=index, id=f'{name}#{index}')
                handle.write(json.dumps(record) + '\n')
            handle.write(json.dumps({'kind': 'completed', 'frames': last - first, 'source_sha256': {}}) + '\n')
        output = out / ('model.native.json' if last == total else f'part-{number}.native.json')
        training = {'numb_steps': total, 'batch_size': 1, 'seed': args.seed, 'disp_freq': 1,
                    'disp_file': (out / f'part-{number}.lcurve.out').as_posix()}
        if state is not None:
            training['optimizer_state'] = state.as_posix()
        if last < total:
            training['save_freq'] = last - first
        body = dict(request, model=model.as_posix(), dataset=dataset.as_posix(), output=output.as_posix(),
                    training=training)
        (out / f'part-{number}.json').write_text(json.dumps(body), encoding='utf-8')
        snapshot = Path(f'{output}.step-{last}')
        state = Path(f'{snapshot}.adam')
        with (out / f'part-{number}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([str(exe), str(out / f'part-{number}.json')], stdout=log, stderr=log)
            if last == total:
                process.wait()
            else:
                # The snapshot of the end of the part: its model, then its optimizer state.
                while process.poll() is None and not written(state):
                    time.sleep(1.0)
                process.kill()
                process.wait()
                # (The trainer reserves its output when it starts: a stopped part leaves it empty.)
                if output.exists() and output.stat().st_size == 0:
                    output.unlink()
        done = output if last == total else state
        if not done.exists():
            raise SystemExit((out / f'part-{number}.log').read_text(encoding='utf-8', errors='replace')[-1500:])
        dataset.unlink()
        for line in (out / f'part-{number}.lcurve.out').read_text(encoding='utf-8').splitlines():
            fields = line.split()
            # (A stopped part has gone on for some steps past its end: those are not of the replay.)
            if fields and fields[0].isdigit() and int(fields[0]) < last:
                ours[int(fields[0])] = (float(fields[1]), float(fields[2]))
        model = snapshot
        print(f'steps {first} to {last} of {total}', flush=True)

    # DeePMD-kit's log has the errors of the steps it displayed (its batch N is step N - 1 here).
    shown = []
    log = (args.run / 'train.log').read_text(encoding='utf-8', errors='replace')
    for match in re.finditer(r'Batch\s+(\d+): trn: .*?rmse_e = ([0-9.eE+-]+), rmse_f = ([0-9.eE+-]+)', log):
        step = int(match.group(1)) - 1
        if step in ours:
            shown.append({'step': step, 'deepmd_rmse_e_per_atom': float(match.group(2)),
                          'deepmd_rmse_f': float(match.group(3)), 'omnimd_rmse_e_per_atom': ours[step][0],
                          'omnimd_rmse_f': ours[step][1]})
    if not shown or shown[0]['step'] != 0:
        raise SystemExit("DeePMD-kit's log has no line for its first batch")
    first = shown[0]
    for theirs, mine in ((first['deepmd_rmse_e_per_atom'], first['omnimd_rmse_e_per_atom']),
                         (first['deepmd_rmse_f'], first['omnimd_rmse_f'])):
        # DeePMD-kit prints three digits.
        if abs(mine - theirs) > 6e-3 * abs(theirs):
            raise SystemExit(f'the first step differs ({mine} against {theirs}): not the same initial model or frame')

    deepmd = json.loads((args.run / 'model.native.json').read_text(encoding='utf-8'))
    omnimd = json.loads((out / 'model.native.json').read_text(encoding='utf-8'))
    initial = json.loads(args.initial.read_text(encoding='utf-8'))
    report = {'deepmd_run': args.run.as_posix(), 'record': record_file.as_posix(), 'steps': total,
              'steps_per_part': args.part, 'precision': args.precision, 'learning_rate': request['learning_rate'],
              'loss': request['loss'], 'displayed_steps': shown, 'parameters': {}}
    start = parts(initial)
    for (name, theirs), mine in zip(parts(deepmd).items(), parts(omnimd).values()):
        moved, apart = np.abs(theirs - start[name]), np.abs(theirs - mine)
        report['parameters'][name] = {'numbers': int(theirs.size), 'mean_change_by_deepmd': float(moved.mean()),
                                      'mean_difference': float(apart.mean()), 'largest_difference': float(apart.max())}
        print(f'{name}: DeePMD-kit moved {theirs.size} numbers by {moved.mean():.3e} on average; omniMD is '
              f'{apart.mean():.3e} away on average, {apart.max():.3e} at most')
    print('errors of the displayed steps (energy per atom, force), DeePMD-kit and omniMD: '
          + '; '.join(f"step {row['step']}: ({row['deepmd_rmse_e_per_atom']:.3g}, {row['deepmd_rmse_f']:.3g}) "
                      f"({row['omnimd_rmse_e_per_atom']:.3g}, {row['omnimd_rmse_f']:.3g})"
                      for row in (shown if len(shown) <= 6 else shown[:3] + shown[-3:])))
    if args.test:
        tester = args.release / 'omnimd-native-gpu-test'
        if not tester.exists():
            tester = tester.with_suffix('.exe')
        report['test'] = {'dataset': args.test.as_posix()}
        for name, path in (('deepmd', args.run / 'model.native.json'), ('omnimd', out / 'model.native.json')):
            scored = json.loads(subprocess.run([str(tester), str(path), str(args.test)], capture_output=True, text=True,
                                               check=True).stdout)
            report['test'][name] = {key: scored['all'][key] for key in ('force_rmse', 'energy_rmse_per_atom', 'frames')}
            print(f"{name}: test force RMSE {scored['all']['force_rmse']:.6f} eV/A, "
                  f"energy RMSE {scored['all']['energy_rmse_per_atom'] * 1e3:.4f} meV/atom")
    with (out / 'replay.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, indent=1)
        handle.write('\n')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
