# Independent W1 result authentication and final full36 CSV join

Disposition: **AUTHENTICATED_DIAGNOSTIC_ONLY — WARM_TARGET_MISSED**.
This Analyst receipt does not change a source, ledger, queue, threshold, cache,
submission, or launch permission. No host access, tensors, dataset reads, or
checkpoint download occurred.

## W1 versus original S1

The predeclared useful target is an uncapped fifth boundary **<=120 seconds**.
W1 measured **120.3379416820826 seconds**, missing it by
**0.3379416820826009 seconds**. Rounding to 120 is not a pass.

S1 measured **845.5936225099722 seconds**. The observed operational reduction
is **725.2556808278896 seconds (85.76882104137872%; 7.0268247× ratio)**.
This is not a training-throughput speedup or full-budget quality result.

The independently parsed rank-0 printed rows are:

| Step | S1 loss | W1 loss | S1 lrm | W1 lrm |
|---|---:|---:|---:|---:|
| 0 | 9.0101 | 9.0101 | 1.000 | 1.000 |
| 1 | 8.7779 | 8.7779 | 0.739 | 1.000 |
| 2 | 8.4722 | 8.4047 | 0.739 | 1.000 |
| 3 | 8.1402 | 8.0712 | 0.739 | 1.000 |
| 4 | 8.1285 | 7.7683 | 0.738 | 1.000 |

Only four-decimal loss/three-decimal LR values are available, not full-precision
or per-rank equality. The different elapsed-time schedule is an observed
nuisance: identical source and argv did not imply identical update trajectories.
Different checkpoint file hashes are also recorded; file hashes alone do not
locate tensor differences.

Authenticated evidence:

- W1 raw fae79d5f/canonical81ca7dbe: 28 files, 123536 decoded bytes. Outer bundle
  SHA e50418bffad38eaabf4d2b38be6c7d2e4d511bbe82b5591a5feec77d02436b5e.
  Every inner size and hash was independently recomputed.
- Original S1 archive190ccf4a: 26 files, bundle SHA
  6424cfcfbb311a49a6e39080a88afa6d05b027ce951a09a506b2547e49742b0c.
  Its PAIR_REFUSED_OR_INCOMPLETE / MATCHED_PRINCIPAL_COMPILER_BYTES_MISSING and
  absent S2 remain unchanged.
- Exact3e79 source blob at b4d83b5d, protected and adjacent prepare, harness,
  frozen52c31b73 packet/manifest, actual staged pins, and proposal-first
  ae77965c were checked. The sole proposal-first/frozen proposal difference
  is its prepared_utc correction; scientific content/target are identical.
- Executed train.py, expected adjacent prepare, harness, protected prepare and
  stable directory match the original S1 identity and both W1 pre/post records,
  including mode, inode, device and exact integer nanoseconds. All eight W1
  identity records are unchanged; package copies are separately pinned.
  Recorded immutable file modes are 0444 and stable directory0555.
  This authenticates recorded filesystem identities, not actual Python import
  origin: that remains unobserved.
- Entire scientific argv equals S1. All six retained cache paths and compile
  URL=DIR match S1, binding and four live worker environments. Only label and
  metrics path differ in the recorded environment. New output/cwd/rendezvous,
  time and SDK-managed cache state remain nuisances. S1 fresh-six-cache receipt
  is retained; W1 explicitly did not request fresh caches.
- Exactly five finite ordered boundaries and five printed finite losses,
  child/outer exit0, and the checkpoint stat/hash receipts agree.
  W1 process wall124.35529707605019 minus min(fifth,900) gives
  **4.017355393967591 charged seconds**, within source1800. Child watchdog is
  1800+60 seconds; neither that nor the source budget is the 120-second target.
  Checkpoint453048594 bytes, SHA
  858f826e7ecf929335a93deb40400e1cc5f2ee80366faabed362951c04da5673.
- Lifetime lock and pre/post ps+neuron-ls receipts passed. Ten IDs in the live
  snapshot exactly equal the explicit postflight list. All absent at
  00:30:48.008886 UTC; 257 unique unfiltered process-identity rows have no
  matching PID/PPID/PGID/SID or compiler-named comm. Device0/four cores/LNC2
  idle, original lockdevice66305/inode832589 free, no kills/changes.
  Bounded snapshots cannot prove no compiler ever ran between observations.

All four W1 graph CSVs contain the same two principal names/keys/node counts
as S1 (666 forward /1169 backward by their literal recorded names). Matching
principal compiler input bytes are missing. Zero compile counters, same keys,
same pathname and retained caches do **not** establish a native cache hit,
NEFF reuse or path-only causality. These remain **UNPROVEN**.

Research implication: the same immutable source/cache setup has a large
observed operational startup reduction worth accounting for prospectively.
Any consistent-path/cache strategy must be selected in a new explicit
proposal; this observation does not authorize an automatic production flip,
repeat, full run, adoption or retroactive reinterpretation of S1/S2.

## Final full36 closed archive join

Unchanged reviewed joiner SHA
9ae32dcc10b7263cac8bee6cafc8beb72c35f901a7ecf3102fbac57340bbab16
was replayed normal/-O against final110054fe bundle SHA
2557fbc01b642425201eb077fe84a1967ac4b518b52a9f9bf6ac1c0f8be2cb99.
Both outputs are byte-identical; **MATCHED_CLOSED_FULL36_INPUTS_ONLY**:
26 authenticated files, two captured module hashes × four closed CSV rows.

- Capture A SHA5a0c39fd… maps to model_default_0_forward_1,666 nodes on all ranks.
  Positive compile record: rank1 **201.175 seconds**.
- Capture B SHAf21636e7… maps to model_default_0_backward_3,1169 nodes on all ranks.
  Positive compile record: rank3 **472.263 seconds**.
- Original final training harness: wall2597.199031607015,
  fifth807.6938910679892, charged1789.505140539026,3630 boundaries.
- These are per-row positive recorded compile times, not rank sums or a full
  elapsed-time decomposition. Graph key, compiler IR signature and debug
  directory identities are kept distinct. Other zero counters are unknown,
  not cache hits. The previous training-only join's final-status unknown is
  resolved by the final successful bundle, not by inventing cache causality.
  No quality comparison or BPB verdict is performed here.

## Reproduction and limitations

From any checkout sharing these immutable Git objects:

```sh
python3 -I -B research/v2/verification/20260926-w1-warm-result-independent/audit.py
python3 -O -I -B research/v2/verification/20260926-w1-warm-result-independent/audit.py
```

Each mode passes **351 checks**, including eight negative/boundary probes:
inner/outer hash corruption, one-nanosecond change, contradictory charged
budget, nonfinite boundary, surviving descendant, exact120 boundary and
nonfinite target input. No Python assert controls acceptance. Initial audit
overstrictly demanded identical proposal timestamps; that failed attempt and
the exact one-field correction are preserved in initial-audit-attempt.json.

The final join's exact command (normal and with -O) was:

```sh
python3 -I -B research/v2/analysis/20260926-clock36-closed-csv-join/join_closed_csv.py --bundle research/v2/operator/20260926-trn22-startup-clock-full/completed/results-bundle.json --expected-sha256 2557fbc01b642425201eb077fe84a1967ac4b518b52a9f9bf6ac1c0f8be2cb99
```

Audit SHA6d9ea6d082b854675ed3c0fcadfb8e57c58da9dbdb56f23156e17e7b4e95d363.
Normal result SHAcd7dc28ccae74c23cd719224d7f11ed1355f737daf53a42e9fa6454342c8945d.
Optimized result SHA8105c7eb6f269503d119cadc158e4e49abce721374b5a71bec46f62295e03af5.
Both final joins SHAed9381bb9fa246b5dceedda47d41cace53c4cb0c9d0cf0ecfff2c62a00819fef.
ledger-candidate.json is **diagnostic-only and not inserted**; no selector
control, validation BPB, source adoption, native correctness qualification or
submission claim is supplied.
