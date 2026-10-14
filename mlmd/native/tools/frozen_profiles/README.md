# Audited frozen graph profiles

SPDX-License-Identifier: LGPL-3.0-or-later
Copyright: DeepModeling Team / DeePMD-kit contributors.

These JSONL files are readable, weight-free canonical graph data, not models
that can be executed. They pin the exact supported topology and metadata. Only
explicit parameter and effective-cutoff placeholders may change. See
../../FROZEN.md and ../../THIRD_PARTY_NOTICES.md for source provenance,
modifications, semantic mapping, limitations, and numerical evidence.

- public-deeppot-1.1-v1.jsonl: 1202 checked nodes; source binary SHA-256
  fbe846eed0e503411cac336d76ac81fabfd9d0d195a03c96dd9fe00fecacca1f.
- synthetic-tf-3.1.3-v1.jsonl: 1291 checked nodes; source binary SHA-256
  3fc0e101964433200ff2ac17b5c4e3038d3eb791ae6d157c9f365864084ae78f.

`../make_frozen_profiles.py` refuses any other binary hashes. Never replace a
profile merely to suppress an unsupported-model error.
