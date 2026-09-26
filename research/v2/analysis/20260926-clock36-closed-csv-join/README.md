# Exact full36 closed-CSV join — local preparation only

Standalone stdlib analyzer for an **already retrieved, final** full36 results bundle. No host/SDK/compiler access, training/model import, cache traversal/mutation, queue, data or evaluator changes. Actual full36 final retrieval remains pending; no native match is claimed here.

## Use after normal final retrieval

Run locally, supplying the archive path and its separately pinned immutable SHA:

```text
python3 -I -B research/v2/analysis/20260926-clock36-closed-csv-join/join_closed_csv.py --bundle /absolute/local/path/to/results-bundle.json --expected-sha256 EXACT_ARCHIVED_BUNDLE_SHA256
```

The bundle must be the exact full36/trn22/root/source binding, complete terminal/outer-exit/train-harness records, and all four closed rank graph_stats tables. It authenticates every contained text/base64 artifact's declared bytes/SHA, optional bundle totals and explicit expected bundle SHA. These are archive consistency checks, not a replacement for Operator provenance/result acceptance.

Hard local input bounds:16MiB serialized/decoded artifacts,256 files,1000 rows per rank. JSON duplicate keys/nonfinite constants, corrupted hashes, malformed numeric fields or mismatched sizes refuse. No files are written by the analyzer; JSON is stdout.

The exact raw A/B module SHA constants are pinned to previously authenticated observation commits bdde21c9/9b525f5f and receipt SHA values. Only those64hex values join graph_id prefixes. Separate directory names and compiler IR signatures are reported but never substituted as keys. All matches retain rank/name/nodecount/full graph_id/key suffix/timestamp and positive compile time. Zero remains UNKNOWN_NOT_CACHE_HIT. No cross-rank time summation or startup-causality inference.

The output includes the **original harness values** for process wall, uncapped fifth boundary, capped allowance, charged-through-exit, step count and checkpoint, not reconstructed timing. A complete mapping requires one consistent matching row for each capture in each of four ranks. Missing/partial/nonunique keys, incomplete finality or missing timing return UNKNOWN, retaining only raw observations.

Exit codes:
-0: MATCHED_CLOSED_FULL36_INPUTS_ONLY.
-3: UNKNOWN (including nonfinal archive or missing exact keys).
-2: REFUSED_INVALID_ARCHIVE.

None means a warm cache hit, path-only causality, timing win, quality acceptance or launch authority.

## Tests and known cold replay

Eleven tiny synthetic parsing/identity tests pass normal/-O: positive matches/zero ambiguity, text/base64 equivalence, corrupted artifact and outer SHA, duplicate JSON/counts, missing rank/harness/value, nonfinal status, misleading IR-signature key, nonfinite time and duplicate same-rank match. Synthetic graph names are deliberately not native labels.

Known immutable cold36 archive e08afdec, SHA6c45190b3c1db2242073fdb04a1a5ca997ec2eb51c7abf655e70488b0b0f7e18, authenticates20 files/21 graph rows. Both captured full-run hashes have **zero matches**, as expected. Output is UNKNOWN/exit3 because this is the cold packet, not final full36, and both mappings are absent. It retains cold harness fifth845.6273377500474, wall850.0966789470986, charged4.469341197051108 without relabeling them as full-run timings. Normal/optimized cold outputs are byte-identical (SHAe1cdc73541180f5d4b68d3b719d801e676a9a4d76d3de6194105dd448b1399d1).

This prepares the bounded next offline check from83d2e59a; it does not request another capture or experiment.
