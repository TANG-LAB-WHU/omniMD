"""Time omniMD and DeePMD-kit on the water benchmark in one session, alternated.

    python bench_alternated.py RELEASE OUT.json [--blocks table,bullets,agreement,once]
        [--rounds-table 5] [--rounds-bullets 3] [--seconds 2] [--busy-check gpu_busy.ps1]
        [--build-label TEXT] [--work DIR] [--sizes 192,...]

RELEASE is the directory of a release build of `omnimd-native-gpu-eval` (feature
`gpu`), OUT.json the record, rewritten after every slot so that a stopped run keeps
what it measured. The evaluator is copied into the work directory first and the
copy is what runs and is hashed. DeePMD-kit runs `bench_deepmd.py --skip-deeppot
--held` in the DeePMD-kit image (by its image ID, never pulled) through Docker
(`docker run --gpus all`), omniMD the evaluator on the host. The cases are those
of `make_water_case.py` (`cases`, `cases/se-e3`, `cases/hybrid`, `cases/hybrid_e3`).

Blocks:
- `table`: the se_e2_a model at 192, 1,536, 12,288 and 98,304 atoms in float64 and
  float32, with the exact embedding nets and with tables (`--compress 0.5` in omniMD,
  `dp compress` in DeePMD-kit); DeePMD-kit up to 12,288 atoms. 5 rounds.
- `bullets`: se_e3, the hybrid of se_e2_a and se_e2_r, and the hybrid of se_e2_a and
  se_e3, in float32 (se_e3 also in float64), the sizes the README quotes. 3 rounds.
  DeePMD-kit's tables of se_e3 do not reproduce its model, so it is timed with its
  nets only; at 1,536 atoms in float64 its se_e3 takes seconds a step, so that case
  is in `once`.
- `once`: one run each of what DeePMD-kit does at 98,304 atoms (se_e2_a with and
  without tables, the hybrid) and of its se_e3 in float64 at 1,536 atoms. A run in
  which the card was found busy afterwards is marked and run once more.
- `agreement`: untimed; every model, precision and table setting at the sizes of
  `deepmd-ref-float64.jsonl` (DeePMD-kit's CPU float64 result), against it and
  against omniMD's own float64 with the exact nets.

A round of a timed block has three slots, `omnimd`, `deepmd` and `omnimd again`,
the order reversed every other round; `omnimd again` repeats `omnimd` as the
control (what the two differ by is what the machine adds) and is left out of
omniMD's times. An omniMD sample is one process: the evaluator's mean over K
evaluations of one frame after two untimed ones (`--bench K`), K chosen in a pilot
pass so that the K evaluations take about SECONDS. A DeePMD-kit sample is one LAMMPS
run: the loop time of the timed run divided by its steps (1,000, 130 and 50 at 192,
1,536 and 12,288 atoms, after 20 untimed steps), with the atoms held (a time step of
1e-6 fs: every step evaluates the frame of the case, and LAMMPS never rebuilds its
neighbor list, where omniMD searches the neighbors at every evaluation). Of each
case the least of its samples is kept and the median recorded beside it.

Before every slot and every pilot pass the card is checked for other work: no
container may be running (a container's GPU work does not show as its own process
on Windows), and `--busy-check`, a command that prints nothing when the card is
free (on this project's machine `tools/ht/gpu_busy.ps1` of the build directory),
must print nothing; the seconds waited are recorded, and the record names the
check and its hash (`busy_check`). The busy check runs again right after
every slot, and the samples of a slot after which it printed something are kept
but marked `disturbed` and left out of the summary. `nvidia-smi` samples memory,
clocks, temperature and utilization every 500 ms through the session; each slot
records them for its time. The host side is not held to any cores here: start
this script through an affinity wrapper to hold the evaluator (DeePMD-kit runs in
Docker's virtual machine, which that does not reach); the record keeps the mask.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

BENCH = Path(__file__).resolve().parent
MLMD = BENCH.parent
IMAGE = 'ghcr.io/deepmodeling/deepmd-kit:3.1.3_cuda129'
CONTAINER = 'omnimd-bench-deepmd'
# descriptor: (case directory under bench/, model of omniMD)
MODELS = {
    'se_e2_a': ('cases', 'water-se_e2_a-float64.native.json'),
    'se_e3': ('cases/se-e3', 'water-se_e3-float64.native.json'),
    'hybrid': ('cases/hybrid', 'water-hybrid-float64.native.json'),
    'hybrid_e3': ('cases/hybrid_e3', 'water-hybrid_e3-float64.native.json'),
}
SIZES = (192, 1536, 12288, 98304)
F64, F32 = 'float64', 'float32'
EVALUATOR_TIMEOUT = 900


def case(model, natoms, precision, tables):
    return {'model': model, 'natoms': natoms, 'precision': precision, 'tables': tables}


def deepmd(model, tables, precisions=None, min_atoms=None, max_atoms=None, timeout=600):
    return {'model': model, 'tables': tables, 'precisions': precisions, 'min_atoms': min_atoms,
            'max_atoms': max_atoms, 'timeout': timeout}


BLOCKS = {
    'table': {
        'omnimd': [case('se_e2_a', n, p, t) for n in SIZES for p in (F64, F32) for t in (False, True)],
        'deepmd': [deepmd('se_e2_a', False, max_atoms=12288), deepmd('se_e2_a', True, max_atoms=12288)],
    },
    'bullets': {
        'omnimd': [case('se_e3', n, p, t) for n in (192, 1536) for p in (F64, F32) for t in (False, True)]
        + [case('se_e3', 98304, F32, t) for t in (False, True)]
        + [case('hybrid', n, F32, t) for n in SIZES for t in (False, True)]
        + [case('hybrid_e3', n, F32, t) for n in (192, 1536, 12288) for t in (False, True)],
        'deepmd': [deepmd('hybrid', False, F32, max_atoms=12288), deepmd('hybrid', True, F32, max_atoms=12288),
                   deepmd('hybrid_e3', False, F32, max_atoms=1536), deepmd('se_e3', False, F32, max_atoms=1536),
                   deepmd('se_e3', False, F64, max_atoms=192)],
    },
    'once': {
        'omnimd': [],
        # se_e3 first: the runs at 98,304 atoms are the ones that may take Docker down.
        'deepmd': [deepmd('se_e3', False, F64, min_atoms=1536, max_atoms=1536, timeout=1500),
                   deepmd('se_e2_a', False, min_atoms=98304, timeout=900),
                   deepmd('se_e2_a', True, min_atoms=98304, timeout=900),
                   deepmd('hybrid', False, F32, min_atoms=98304, timeout=900),
                   deepmd('hybrid', True, F32, min_atoms=98304, timeout=900)],
    },
}


def now():
    return datetime.datetime.now().isoformat(timespec='milliseconds')


def log(text):
    print(f'[{now()}] {text}', flush=True)


def run(args, **kwargs):
    return subprocess.run(args, capture_output=True, encoding='utf-8', errors='replace', **kwargs)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def key(model, natoms, precision, tables):
    return f'{model} {natoms} {precision} {"tables" if tables else "nets"}'


def host():
    """The CPU and the cores this process (and so the evaluator) may run on."""
    facts = {}
    if os.name == 'nt':
        import ctypes
        import winreg
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        if kernel.GetProcessAffinityMask(ctypes.c_void_p(kernel.GetCurrentProcess()),
                                         ctypes.byref(process), ctypes.byref(system)):
            facts['affinity_mask'] = hex(process.value)
            facts['system_mask'] = hex(system.value)
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
                facts['cpu'] = winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    elif hasattr(os, 'sched_getaffinity'):
        facts['cpus'] = sorted(os.sched_getaffinity(0))
    return facts


class Session:
    def __init__(self, args):
        self.args = args
        exe = args.release / 'omnimd-native-gpu-eval'
        if not exe.exists():
            exe = exe.with_suffix('.exe')
        if not exe.exists():
            sys.exit(f'no omnimd-native-gpu-eval in {args.release}')
        self.work = args.work
        if self.work.exists() and any(self.work.iterdir()):
            sys.exit(f'{self.work} is not empty; give a new --work')
        (self.work / 'bin').mkdir(parents=True, exist_ok=True)
        # Run a copy, so that a build into RELEASE during the session cannot change what is timed.
        self.exe = self.work / 'bin' / exe.name
        shutil.copy2(exe, self.exe)
        self.env = dict(os.environ)
        if os.name == 'nt' and 'CUDA_PATH' in self.env:
            self.env['PATH'] = os.path.join(self.env['CUDA_PATH'], 'bin') + os.pathsep + self.env['PATH']
        self.out = args.out
        self.sampler = None
        self.samples_path = self.work / 'nvidia-smi.csv'
        self.busy_command = None
        if args.busy_check:
            check = str(args.busy_check.resolve(strict=True))
            self.busy_command = (['powershell', '-NoLogo', '-NonInteractive', '-NoProfile', '-ExecutionPolicy',
                                  'Bypass', '-File', check] if check.endswith('.ps1') else [check])
        self.image = None
        self.record = {'what': __doc__, 'started': now(), 'build': args.build_label or exe.name,
                       'omnimd': {'exe': exe.name, 'sha256': sha256(self.exe)}, 'host': host(),
                       'busy_check': ({'name': args.busy_check.name, 'sha256': sha256(args.busy_check)}
                                      if args.busy_check else None),
                       'deepmd': {'image': args.image}, 'blocks': {}}

    def save(self):
        tmp = self.out.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps(self.record, indent=1) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(20):
            try:
                os.replace(tmp, self.out)
                return
            except PermissionError:
                # A reader holds the record open; the next save carries the data if this one cannot.
                time.sleep(0.5)
        log(f'could not replace {self.out}; the data is in {tmp}')

    # --- the machine ---------------------------------------------------------------------------
    def describe(self):
        smi = run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,clocks.max.sm',
                   '--format=csv,noheader'], timeout=60)
        self.record['device'] = smi.stdout.strip()
        image = run(['docker', 'image', 'inspect', '--format', '{{.Id}}|{{json .RepoDigests}}', self.args.image],
                    timeout=120)
        if image.returncode != 0:
            sys.exit(f'the image {self.args.image} is not present (docker image inspect failed); pull it first')
        self.image, digests = image.stdout.strip().split('|', 1)
        self.record['deepmd'].update(image_id=self.image, repo_digests=json.loads(digests))
        leftover = run(['docker', 'ps', '-a', '--filter', f'name=^/{CONTAINER}$', '--format', '{{.State}}'],
                       timeout=120).stdout.split()
        if 'running' in leftover:
            sys.exit(f'a container named {CONTAINER} is running; let it finish (docker ps) and start again')
        if leftover:
            run(['docker', 'rm', CONTAINER], timeout=120)
        if self.busy_command:
            check = run(self.busy_command, timeout=120)
            if check.returncode != 0:
                sys.exit(f'the busy check fails: {check.stderr.strip()[-300:]}')

    def start_sampler(self):
        handle = open(self.samples_path, 'a', encoding='utf-8')
        self.sampler = subprocess.Popen(
            ['nvidia-smi', '--query-gpu=timestamp,memory.used,clocks.sm,clocks.mem,temperature.gpu,'
             'utilization.gpu,power.draw', '--format=csv,noheader,nounits', '-lms', '500'],
            stdout=handle, stderr=subprocess.DEVNULL)

    def stop_sampler(self):
        if self.sampler is not None:
            self.sampler.terminate()
            self.sampler.wait(timeout=30)
            self.sampler = None

    def window(self, start, end):
        """What the card did between two times: memory in use, SM clock, temperature, utilization."""
        rows = []
        try:
            lines = self.samples_path.read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError:
            return None
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 7:
                continue
            try:
                stamp = datetime.datetime.strptime(parts[0], '%Y/%m/%d %H:%M:%S.%f').isoformat(timespec='milliseconds')
                memory, clock = float(parts[1]), float(parts[2])
                temperature, utilization = float(parts[4]), float(parts[5])
            except ValueError:
                continue
            if start <= stamp <= end:
                rows.append((memory, clock, temperature, utilization))
        if not rows:
            return None
        return {'samples': len(rows), 'memory_mib_min': min(r[0] for r in rows),
                'memory_mib_max': max(r[0] for r in rows),
                'sm_mhz_median': statistics.median(r[1] for r in rows), 'sm_mhz_min': min(r[1] for r in rows),
                'temperature_c_max': max(r[2] for r in rows),
                'utilization_median': statistics.median(r[3] for r in rows),
                'utilization_max': max(r[3] for r in rows)}

    def containers(self):
        for attempt in range(3):
            try:
                listing = run(['docker', 'ps', '-q'], timeout=120)
                if listing.returncode == 0:
                    return listing.stdout.split()
            except subprocess.TimeoutExpired:
                pass
            time.sleep(10)
        sys.exit('docker ps fails; stopping')

    def busy(self):
        """What the busy check prints (nothing: the card is free)."""
        if not self.busy_command:
            return ''
        for attempt in range(3):
            try:
                check = run(self.busy_command, timeout=120)
                if check.returncode == 0:
                    return check.stdout.strip()
            except subprocess.TimeoutExpired:
                pass
            time.sleep(10)
        sys.exit('the busy check fails; stopping')

    def wait_for_card(self):
        """Seconds waited until, in one pass, no container runs and the busy check prints nothing."""
        waited, container_waited, seen = 0, 0, []
        while True:
            running = self.containers()
            busy = '' if running else self.busy()
            if not running and not busy:
                return waited, seen[:5]
            if running:
                if container_waited >= 600:
                    sys.exit('a container has been running for ten minutes; stopping (see docker ps)')
                seen.append('container running')
                time.sleep(10)
                waited += 10
                container_waited += 10
            else:
                seen.append(busy.splitlines()[0][:120])
                log(f'card busy: {busy.splitlines()[0][:120]}; waiting')
                time.sleep(30)
                waited += 30

    # --- omniMD ---------------------------------------------------------------------------------
    def omnimd(self, c, bench):
        directory, model = MODELS[c['model']]
        frame = BENCH / directory / f'water-{c["natoms"]}.jsonl'
        command = [str(self.exe), str(BENCH / directory / model), '--precision', c['precision'],
                   '--bench', str(bench)] + (['--compress', '0.5'] if c['tables'] else [])
        with open(frame, encoding='utf-8') as stdin:
            proc = run(command, stdin=stdin, env=self.env, timeout=EVALUATOR_TIMEOUT)
        if proc.returncode != 0:
            raise RuntimeError(f'{key(**c)}: {proc.stderr.strip().splitlines()[-1:]}')
        lines = proc.stderr.splitlines()
        timings = [json.loads(l) for l in lines if l.startswith('{') and 'ms_per_eval' in l]
        if not timings:
            raise RuntimeError(f'{key(**c)}: no bench line in the output of the evaluator')
        timing = timings[0]
        if c['tables'] and timing.get('table_fallbacks', 0) != 0:
            raise RuntimeError(f'{key(**c)}: the frame left the tables ({timing["table_fallbacks"]} fallbacks)')
        header = [l for l in lines if not l.startswith('{')]
        return timing, header

    # --- DeePMD-kit -----------------------------------------------------------------------------
    def stop_container(self):
        try:
            run(['docker', 'stop', '-t', '60', CONTAINER], timeout=180)
        except subprocess.TimeoutExpired:
            log(f'docker stop {CONTAINER} did not return')

    def deepmd(self, spec, label, folder):
        directory = MODELS[spec['model']][0]
        out_dir = self.work / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f'{label}.csv'
        mount = str(MLMD).replace('\\', '/')
        work = str(self.work.resolve()).replace('\\', '/')
        options = ['--skip-deeppot', '--held', '--timeout', str(spec['timeout'])]
        if spec['tables']:
            options.append('--compressed')
        if spec['precisions']:
            options += ['--precisions', spec['precisions']]
        if spec['min_atoms']:
            options += ['--min-atoms', str(spec['min_atoms'])]
        if spec['max_atoms']:
            options += ['--max-atoms', str(spec['max_atoms'])]
        command = ['docker', 'run', '--rm', '--name', CONTAINER, '--gpus', 'all', '-v', f'{mount}:/m',
                   '-v', f'{work}:/out', '-w', '/m', self.image, 'python', 'bench/bench_deepmd.py',
                   f'bench/{directory}', f'/out/{folder}/{csv_path.name}'] + options
        sizes = [n for n in SIZES if (spec['min_atoms'] or 0) <= n <= (spec['max_atoms'] or max(SIZES))]
        runs = len(spec['precisions'].split(',')) if spec['precisions'] else 2
        budget = runs * len(sizes) * spec['timeout'] + 600
        started = time.time()
        try:
            proc = run(command, timeout=budget)
        except subprocess.TimeoutExpired:
            # The client is gone, not the container: stop it before anything else uses the card.
            self.stop_container()
            raise RuntimeError(f'{label}: docker run did not return in {budget} s; container stopped')
        except BaseException:
            self.stop_container()
            raise
        (out_dir / f'{label}.stdout.txt').write_text(proc.stdout + '\n--- stderr\n' + proc.stderr, encoding='utf-8')
        if proc.returncode != 0 or not csv_path.exists():
            raise RuntimeError(f'{label}: docker run failed ({proc.returncode}): {proc.stderr.strip()[-300:]}')
        with open(csv_path, encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        tag = ('-compressed' if spec['tables'] else '') + '-held'
        results = []
        for row in rows:
            natoms, precision = int(row['natoms']), row['precision']
            log_path = BENCH / directory / 'lammps' / f'water-{natoms}-{precision}{tag}.log'
            facts = {}
            if log_path.exists() and log_path.stat().st_mtime >= started - 5:
                shutil.copy2(log_path, out_dir / f'{label}-water-{natoms}-{precision}.log')
                facts = lammps_facts(log_path.read_text(encoding='utf-8', errors='replace'))
            status = row['status']
            # LAMMPS aborts before its screen output is flushed; the log says what the CSV could not.
            if status == 'error' and facts.get('out_of_memory'):
                status = 'oom'
            results.append({'model': spec['model'], 'natoms': natoms, 'precision': precision,
                            'tables': spec['tables'], 'status': status, 'status_csv': row['status'],
                            'ms': float(row['ms_per_eval']) if row['ms_per_eval'] else None,
                            'note': row['note'], 'log': facts})
        return results

    # --- blocks ---------------------------------------------------------------------------------
    def selected(self, cases):
        sizes = self.args.sizes
        return [c for c in cases if not sizes or c['natoms'] in sizes]

    def deepmd_selected(self, specs):
        # --sizes is a contiguous run of SIZES (checked in main), so a range selects exactly those sizes.
        sizes = self.args.sizes
        kept = []
        for spec in specs:
            if sizes:
                low = max(spec['min_atoms'] or 0, min(sizes))
                high = min(spec['max_atoms'] or max(SIZES), max(sizes))
                if low > high:
                    continue
                spec = dict(spec, min_atoms=low if low > min(SIZES) else None,
                            max_atoms=high if high < max(SIZES) else None)
            kept.append(spec)
        return kept

    def slot(self, entry, record_slot, measure):
        """Run one slot: check the card, measure, check it again, and keep what was measured."""
        waited, seen = self.wait_for_card()
        start, samples, error = now(), [], None
        try:
            measure(samples)
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            end = now()
            after = self.busy() if error is None else ''
            disturbed = bool(after)
            for s in samples:
                s['disturbed'] = disturbed
            entry['samples'] += samples
            entry['slots'].append(dict(record_slot, start=start, end=end, waited_s=waited, seen=seen,
                                       seen_after=after.splitlines()[:3], complete=error is None, error=error,
                                       card=self.window(start, end)))
            if 'summary' in entry:
                entry['summary'] = summarize(entry['samples'])
            self.save()
        return disturbed

    def timed_block(self, name, rounds):
        block = BLOCKS[name]
        cases = self.selected(block['omnimd'])
        specs = self.deepmd_selected(block['deepmd'])
        entry = self.record['blocks'][name] = {'rounds': rounds, 'seconds': self.args.seconds, 'pilot': {},
                                               'bench': {}, 'slots': [], 'samples': [], 'summary': {}}
        entry['pilot_waited_s'], entry['pilot_seen'] = self.wait_for_card()
        log(f'{name}: pilot of {len(cases)} omniMD cases')
        for c in cases:
            timing, header = self.omnimd(c, 3)
            ms = timing['ms_per_eval']
            entry['pilot'][key(**c)] = {'ms': ms, 'header': header}
            entry['bench'][key(**c)] = max(3, min(20000, round(self.args.seconds * 1e3 / ms)))
        self.save()
        for r in range(1, rounds + 1):
            order = ['omnimd', 'deepmd', 'omnimd again']
            if r % 2 == 0:
                order.reverse()
            for slot in order:
                if (slot == 'deepmd' and not specs) or (slot != 'deepmd' and not cases):
                    continue
                log(f'{name}: round {r}, {slot}')

                def measure(samples, slot=slot, r=r):
                    if slot == 'deepmd':
                        for i, spec in enumerate(specs if r % 2 else list(reversed(specs))):
                            label = (f'{name}-r{r}-{spec["model"]}-{"tables" if spec["tables"] else "nets"}'
                                     f'-{spec["precisions"] or "all"}-{i}')
                            for result in self.deepmd(spec, label, f'{name}-r{r}'):
                                samples.append(dict(result, engine='deepmd', slot=slot, round=r))
                    else:
                        for c in cases:
                            timing, _ = self.omnimd(c, entry['bench'][key(**c)])
                            samples.append(dict(c, engine='omnimd', slot=slot, round=r, status='ok',
                                                ms=timing['ms_per_eval'], at=now(),
                                                fields={k: v for k, v in timing.items()
                                                        if k not in ('ms_per_eval', 'frame', 'natoms', 'precision')}))

                self.slot(entry, {'round': r, 'slot': slot}, measure)

    def once_block(self):
        specs = self.deepmd_selected(BLOCKS['once']['deepmd'])
        entry = self.record['blocks']['once'] = {'slots': [], 'samples': []}
        for i, spec in enumerate(specs):
            for attempt in (1, 2):
                label = (f'once-{spec["model"]}-{"tables" if spec["tables"] else "nets"}'
                         f'-{spec["precisions"] or "all"}-{i}-a{attempt}')
                log(f'once: {label}')

                def measure(samples, spec=spec, label=label, attempt=attempt):
                    for result in self.deepmd(spec, label, 'once'):
                        samples.append(dict(result, engine='deepmd', slot='deepmd', round=1, attempt=attempt))

                if not self.slot(entry, {'label': label}, measure):
                    break

    def agreement_block(self):
        entry = self.record['blocks']['agreement'] = {}
        for model, (directory, native) in MODELS.items():
            ref_path = BENCH / directory / 'deepmd-ref-float64.jsonl'
            if not ref_path.exists():
                continue
            refs = {r['natoms']: r for r in (json.loads(l) for l in ref_path.read_text().splitlines() if l.strip())}
            for natoms in sorted(refs):
                if self.args.sizes and natoms not in self.args.sizes:
                    continue
                frame = BENCH / directory / f'water-{natoms}.jsonl'
                results = {}
                for precision in (F64, F32):
                    for tables in (False, True):
                        command = [str(self.exe), str(BENCH / directory / native), '--precision', precision] \
                            + (['--compress', '0.5'] if tables else [])
                        with open(frame, encoding='utf-8') as stdin:
                            proc = run(command, stdin=stdin, env=self.env, timeout=EVALUATOR_TIMEOUT)
                        if proc.returncode != 0:
                            raise RuntimeError(f'agreement {model} {natoms}: {proc.stderr.strip()[-300:]}')
                        if tables and 'left the embedding table range' in proc.stderr:
                            raise RuntimeError(f'agreement {model} {natoms} {precision}: the frame left the tables')
                        results[(precision, tables)] = json.loads(proc.stdout.strip().splitlines()[-1])
                exact64 = results[(F64, False)]
                for (precision, tables), result in results.items():
                    item = {'vs_deepmd_float64': errors(result, refs[natoms])}
                    if (precision, tables) != (F64, False):
                        item['vs_omnimd_float64_nets'] = errors(result, exact64)
                    entry[key(model, natoms, precision, tables)] = item
                self.save()


LOOP = re.compile(r'Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps with (\d+) atoms')
PAIR = re.compile(r'Pair\s*\|.*\|\s*([\d.]+)\s*$')
BUILDS = re.compile(r'Neighbor list builds = (\d+)\s*$')
CUTOFF = re.compile(r'master list distance cutoff = (\S+)\s*$')
WALL = re.compile(r'Total wall time: (\S+)\s*$')
THERMO = re.compile(r'(\d+)\s+(\S+)\s+\S+')
ALLOCATE = re.compile(r'Tried to allocate ([\d.]+ \w+)')


def lammps_facts(text):
    """What the log of a held run says about its timed run (the second `run`) and its warm-up."""
    runs = [{}]
    facts = {'out_of_memory': 'out of memory' in text.lower()}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('LAMMPS (') and 'lammps' not in facts:
            facts['lammps'] = s
        elif s.startswith('source commit:') and 'deepmd_kit_commit' not in facts:
            facts['deepmd_kit_commit'] = s.split(':', 1)[1].strip()
        elif (m := CUTOFF.match(s)):
            facts['neighbor_cutoff'] = m.group(1)
        elif (m := WALL.match(s)):
            facts['wall'] = m.group(1)
        elif (m := ALLOCATE.search(s)) and 'tried_to_allocate' not in facts:
            facts['tried_to_allocate'] = m.group(1)
        elif (m := LOOP.match(s)):
            runs[-1]['loop_s'], runs[-1]['steps'] = float(m.group(1)), int(m.group(2))
            runs.append({})
        elif (m := PAIR.match(s)):
            runs[-2 if len(runs) > 1 else -1]['pair_percent'] = float(m.group(1))
        elif (m := BUILDS.match(s)):
            runs[-2 if len(runs) > 1 else -1]['neighbor_list_builds'] = int(m.group(1))
        elif (m := THERMO.match(s)):
            try:
                temperature = float(m.group(2))
            except ValueError:
                continue
            runs[-1]['last_step'] = int(m.group(1))
            runs[-1]['last_temp'] = temperature if math.isfinite(temperature) else None
    done = [r for r in runs if 'loop_s' in r]
    facts['runs_completed'] = len(done)
    if done:
        facts['warmup'] = done[0]
    if len(done) >= 2:
        facts['timed'] = done[1]
    return facts


def errors(e, ref):
    df = [a - b for fa, fb in zip(e['forces'], ref['forces']) for a, b in zip(fa, fb)]
    dv = [a - b for ra, rb in zip(e['virial'], ref['virial']) for a, b in zip(ra, rb)]
    return {'energy_per_atom': abs(e['energy'] - ref['energy']) / len(ref['forces']),
            'force_max': max(abs(x) for x in df),
            'force_rms': (sum(x * x for x in df) / len(df)) ** 0.5,
            'virial_max': max(abs(x) for x in dv)}


def summarize(samples):
    """Per case and engine: the least and the median of the samples, and omniMD's control."""
    groups = {}
    for s in samples:
        if s['status'] != 'ok' or s['ms'] is None or s.get('disturbed'):
            continue
        slot = 'omnimd again' if s['slot'] == 'omnimd again' else s['engine']
        groups.setdefault(key(s['model'], s['natoms'], s['precision'], s['tables']), {}) \
            .setdefault(slot, []).append(s['ms'])
    summary = {}
    for k, slots in groups.items():
        item = {}
        for slot, values in slots.items():
            item[slot] = {'least': min(values), 'median': statistics.median(values), 'n': len(values)}
        if 'omnimd' in item and 'omnimd again' in item:
            item['control'] = item['omnimd again']['least'] / item['omnimd']['least'] - 1
        if 'omnimd' in item and 'deepmd' in item:
            item['deepmd_over_omnimd'] = item['deepmd']['least'] / item['omnimd']['least']
        summary[k] = item
    return summary


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('release', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--blocks', default='table,bullets,agreement,once', help='the blocks, in this order')
    parser.add_argument('--rounds-table', type=int, default=5)
    parser.add_argument('--rounds-bullets', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=2.0, help='about how long the K evaluations of a sample take')
    parser.add_argument('--busy-check', type=Path, help='a command that prints nothing when the card is free')
    parser.add_argument('--build-label', help='what the record calls the build (default: the file name)')
    parser.add_argument('--image', default=IMAGE)
    parser.add_argument('--work', type=Path, help='raw outputs (default: results/raw/<name of OUT>)')
    parser.add_argument('--sizes', help='only these sizes, by comma, a contiguous run of 192,1536,12288,98304')
    args = parser.parse_args()
    if args.sizes:
        args.sizes = sorted({int(s) for s in args.sizes.split(',')})
        if any(n not in SIZES for n in args.sizes):
            parser.error(f'--sizes takes sizes of {SIZES}')
        first = SIZES.index(args.sizes[0])
        if list(SIZES[first:first + len(args.sizes)]) != args.sizes:
            parser.error('--sizes must be a contiguous run of 192, 1536, 12288, 98304')
    args.work = args.work or BENCH / 'results' / 'raw' / args.out.stem
    blocks = args.blocks.split(',')
    unknown = [b for b in blocks if b not in ('table', 'bullets', 'once', 'agreement')]
    if unknown:
        parser.error(f'unknown block {", ".join(unknown)}')
    if args.out.exists():
        parser.error(f'{args.out} exists')
    session = Session(args)
    session.describe()
    session.start_sampler()
    try:
        for name in blocks:
            if name == 'table':
                session.timed_block('table', args.rounds_table)
            elif name == 'bullets':
                session.timed_block('bullets', args.rounds_bullets)
            elif name == 'once':
                session.once_block()
            else:
                session.agreement_block()
            log(f'{name} done')
    except BaseException:
        session.record['error'] = traceback.format_exc()[-4000:]
        raise
    finally:
        session.record['ended'] = now()
        session.stop_sampler()
        session.save()


if __name__ == '__main__':
    main()
