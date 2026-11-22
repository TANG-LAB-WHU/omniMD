"""The tables of a record of `bench_alternated.py`, in Markdown.

    python alternated_report.py RECORD.json

Prints the se_e2_a table (least of the samples in ms per evaluation, DeePMD-kit
with the atoms held), the numbers of the other descriptors, the controls and
spreads of every case, what the card did in the slots, and the agreement of
omniMD with DeePMD-kit's CPU float64 result and with its own float64.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

SIZES = (192, 1536, 12288, 98304)


def fmt(ms):
    """Three significant figures, as the README writes them."""
    if ms is None:
        return '-'
    if ms >= 100:
        return f'{ms:,.0f}'
    if ms >= 10:
        return f'{ms:.1f}'
    if ms >= 1:
        return f'{ms:.2f}'
    return f'{ms:.3f}'


def key(model, natoms, precision, tables):
    return f'{model} {natoms} {precision} {"tables" if tables else "nets"}'


def once_results(record):
    """DeePMD-kit's single runs: of each case the last attempt the card did not disturb, else its last."""
    out = {}
    for s in record['blocks'].get('once', {}).get('samples', []):
        k = key(s['model'], s['natoms'], s['precision'], s['tables'])
        if s.get('disturbed') and k in out and not out[k].get('disturbed'):
            continue
        out[k] = s
    return out


def deepmd_cell(record, k, fallback):
    item = record['_summary'].get(k, {})
    if 'deepmd' in item:
        return fmt(item['deepmd']['least']), item['deepmd']['least']
    s = fallback.get(k)
    if s is None:
        return '-', None
    if s['status'] == 'ok':
        return fmt(s['ms']) + (' (one run, card busy after it)' if s.get('disturbed') else ' (one run)'), s['ms']
    if s['status'] == 'oom' or s.get('log', {}).get('out_of_memory'):
        return 'out of memory', None
    return s['status'], None


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('record', type=Path)
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding='utf-8'))
    summary = {}
    for name in ('table', 'bullets'):
        summary.update(record['blocks'].get(name, {}).get('summary', {}))
    record['_summary'] = summary
    once = once_results(record)

    print(f'Record: {args.record.name}; {record["build"]}; {record["started"]} to {record.get("ended")}')
    print(f'Device: {record.get("device")}; host: {json.dumps(record.get("host"))}')
    print(f'DeePMD-kit image: {record["deepmd"].get("image_id")}')
    if record.get('error'):
        print('\nTHE RUN STOPPED WITH AN ERROR:\n' + record['error'])

    print('\n## se_e2_a, ms per evaluation (least of the samples)\n')
    print('| Atoms | Precision | DeePMD-kit | omniMD | DeePMD-kit, tables | omniMD, tables |')
    print('| ---: | --- | ---: | ---: | ---: | ---: |')
    for n in SIZES:
        for p in ('float64', 'float32'):
            cells = []
            for tables in (False, True):
                k = key('se_e2_a', n, p, tables)
                cells.append(deepmd_cell(record, k, once)[0])
                cells.append(fmt(summary.get(k, {}).get('omnimd', {}).get('least')))
            print(f'| {n:,} | {p} | ' + ' | '.join(cells) + ' |')

    print('\n## Every timed case\n')
    print('| case | omniMD least | median | n | again least | control | DeePMD-kit least | median | n | ratio |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for k in sorted(summary, key=lambda k: (k.split()[0], int(k.split()[1]), k.split()[2], k.split()[3])):
        item = summary[k]
        o, a, d = item.get('omnimd', {}), item.get('omnimd again', {}), item.get('deepmd', {})
        control = f'{item["control"]:+.1%}' if 'control' in item else '-'
        ratio = f'{item["deepmd_over_omnimd"]:.1f}' if 'deepmd_over_omnimd' in item else '-'
        print(f'| {k} | {fmt(o.get("least"))} | {fmt(o.get("median"))} | {o.get("n", "-")} | {fmt(a.get("least"))} '
              f'| {control} | {fmt(d.get("least"))} | {fmt(d.get("median"))} | {d.get("n", "-")} | {ratio} |')
    controls = [item['control'] for item in summary.values() if 'control' in item]
    if controls:
        print(f'\nControls: {min(controls):+.1%} to {max(controls):+.1%} over {len(controls)} cases; '
              f'beyond 3%: {", ".join(k for k, i in summary.items() if abs(i.get("control", 0)) > 0.03) or "none"}')
    spreads = {(k, s): i[s]['median'] / i[s]['least'] - 1 for k, i in summary.items()
               for s in ('omnimd', 'omnimd again', 'deepmd') if s in i and i[s]['n'] > 1}
    if spreads:
        worst = max(spreads, key=spreads.get)
        print(f'Median over least: at most {spreads[worst]:+.1%} ({worst[1]}, {worst[0]})')

    print('\n## DeePMD-kit single runs\n')
    for k, s in once.items():
        facts = s.get('log', {})
        timed = facts.get('timed', {})
        print(f'- {k}: {s["status"]}, {fmt(s["ms"])} ms, attempt {s.get("attempt")}, disturbed {s.get("disturbed")}; '
              f'note {s.get("note")!r}; log: runs {facts.get("runs_completed")}, timed {json.dumps(timed)}, '
              f'out of memory {facts.get("out_of_memory")}, tried to allocate {facts.get("tried_to_allocate")}')

    print('\n## The card in the slots\n')
    for name, block in record['blocks'].items():
        slots = [s for s in block.get('slots', []) if isinstance(s, dict) and s.get('card')] \
            if isinstance(block, dict) else []
        if not slots:
            continue
        clocks = [s['card']['sm_mhz_median'] for s in slots]
        print(f'- {name}: {len(slots)} slots; SM clock median {min(clocks):.0f} to {max(clocks):.0f} MHz; '
              f'temperature at most {max(s["card"]["temperature_c_max"] for s in slots):.0f} C; '
              f'memory at most {max(s["card"]["memory_mib_max"] for s in slots):,.0f} MiB; '
              f'waited {sum(s["waited_s"] for s in slots)} s; '
              f'disturbed {sum(1 for s in slots if s.get("seen_after"))}')
    pilots = {}
    for name in ('table', 'bullets'):
        for k, p in record['blocks'].get(name, {}).get('pilot', {}).items():
            pilots[k] = p['header']
    if pilots:
        first = next(iter(pilots))
        print(f'\nHeader of {first}:')
        for line in pilots[first]:
            print(f'    {line}')

    agreement = record['blocks'].get('agreement', {})
    if agreement:
        print('\n## Agreement (largest over the sizes)\n')
        print('| model | precision | embedding | sizes | energy/atom vs DeePMD-kit | force vs DeePMD-kit '
              '| force vs omniMD float64 nets |')
        print('| --- | --- | --- | --- | ---: | ---: | ---: |')
        groups = {}
        for k, item in agreement.items():
            model, natoms, precision, emb = k.split()
            groups.setdefault((model, precision, emb), []).append((int(natoms), item))
        for (model, precision, emb), items in groups.items():
            sizes = ', '.join(f'{n:,}' for n, _ in sorted(items))
            e = max(i['vs_deepmd_float64']['energy_per_atom'] for _, i in items)
            f = max(i['vs_deepmd_float64']['force_max'] for _, i in items)
            own = [i['vs_omnimd_float64_nets']['force_max'] for _, i in items if 'vs_omnimd_float64_nets' in i]
            print(f'| {model} | {precision} | {emb} | {sizes} | {e:.2g} | {f:.2g} | '
                  f'{(f"{max(own):.2g}" if own else "-")} |')


if __name__ == '__main__':
    main()
