"""Rebuild audited profiles only from the two exact recorded fixture binaries.

This is NOT a way to approve a new architecture. New profiles require review of
all graph semantics, an explicit mapping, and cross-backend acceptance evidence.
"""
import hashlib
import json
from pathlib import Path
import sys
from export_frozen import FrozenGraph, PROFILE_DIR, profile_specs

KNOWN = ['fbe846eed0e503411cac336d76ac81fabfd9d0d195a03c96dd9fe00fecacca1f',
         '3fc0e101964433200ff2ac17b5c4e3038d3eb791ae6d157c9f365864084ae78f']

def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: make_frozen_profiles.py PUBLIC.pb SYNTHETIC.pb')
    PROFILE_DIR.mkdir(exist_ok=True)
    for path, expected, profile in zip(sys.argv[1:], KNOWN, profile_specs()):
        graph = FrozenGraph(path)
        if hashlib.sha256(graph.raw).hexdigest() != expected:
            raise ValueError('fixture SHA differs; refusing to bless an unreviewed graph')
        rows = graph.canonical(profile)
        output = PROFILE_DIR/(profile['id']+'.jsonl')
        text = ''.join(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for r in rows)
        output.write_text(text,encoding='utf-8')
        print(profile['id'],len(rows),len(text),hashlib.sha256(text.encode()).hexdigest())

if __name__ == '__main__': main()
