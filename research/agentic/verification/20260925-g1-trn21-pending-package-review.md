# Independent review: pending trn21 G1 public20M calibration package

Target: `d63ae1065382bcd843e8c9c2f0bcec983117fde8`,
`research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r1/`.
Request SHA-256: `b39f3e5573793f5b9eb9d559f0d977c17d1f3f2dde304edf27c806f2222e9a4a`.

Verdict: **static identity/CLI package PASS; dispatch HOLD**. This review is neither
G1 PASS for trn21 nor approval of a finalized request or host launch. I contacted
neither host and ran no training, evaluation, AWS, cleanup, or breaker action.

## Static evidence

- The target commit adds only `request.pending.json` and `RUNBOOK_PENDING.md`.
  `jq empty` and `git diff --check` pass. Manual review and a credential-pattern
  scan of both added files found no private key, credential, auth token, or `.env`
  content. Operational paths and the private hostname are present, but no
  authentication material is embedded.
- At pinned source commit `347f81a8572d6af8876067cd7c777a341e019929`,
  the **nested trained** `submissions/dense6x1024_direct_review_matrixlr015/train.py`
  hashes to `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`;
  `prepare.py` hashes to
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  Both match the historical trn21 confirmation Operator receipt. No protected
  evaluator, tokenizer, data, or training source is changed by the package.
- The retained checkpoint path, 469,831,009-byte length, and SHA-256
  `5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9`
  match that receipt. Its **current** existence, file type, holder state, and hash
  are expressly unverified and remain pre-dispatch checks; it must be preserved.
- The argv is one `timeout`-bounded, fresh Python `prepare.py eval-public`
  process using the pinned nested train file and retained checkpoint. It fixes
  `--device neuron --batch-size 1 --seq-len 1024 --eval-tokens 20971520
  --timeout-seconds 1200`; the outer bound is 1800 seconds. These flags match
  the unchanged evaluator CLI and the historical public20M invocation, apart
  from absolute paths. The default CLI token count is only 2,097,152.
  `prepare.py` selects `public_val`, computes cross-entropy from logits, and
  prints JSON containing the requested BPB fields.
- The named environment pins the historical optlevel-1 compiler flags,
  `NEURON_LOGICAL_NC_CONFIG=2`, native Python, and the staged-root
  `PYTHONPATH`. All six Neuron/Inductor compiler-cache paths exactly match the
  historical trn21 `eval-rehearsal.json`; their **present** existence has not
  been proved. The request requires absent rank/distributed, step-log, and
  alternate data/tokenizer overrides before launch. The success parser must
  check the full JSON, not just its BPB regex.
- Historical `full20m.metrics.json` reports 20,971,520 tokens, 83,462,186
  bytes, 57,379,240 nats and BPB `0.9918353324459732`.
  `57379240 / (83462186 * ln(2))` independently reproduces that value.
  This is an evaluator calibration reference, not a new model result or
  advancement control.

## Dispatch hold and finalization checks

The request says `PENDING_NOT_DISPATCHABLE`, `launch_authorized=false`,
`expected_commit=null`, unverified staged-source hashes, and unverified current
checkpoint hash. At the target commit, `research/v2/state.json` also has
`launch_authorized=false` and `trn21.launch_allowed=false`. No matching label
appears in the tracked controller run/dispatch material before this package;
host-side uniqueness is unknown.

The most recent archived trn21 inventory in `state.json` is **6,901,788,672
bytes free at 2026-09-24 21:50:53 UTC**. It is 77,533,184 bytes below the
immutable 6.5-GiB floor and 2,225,016,832 bytes below the package's
9,126,805,504-byte prelaunch gate (floor plus 2 GiB). The request's
`last_archived_trn21_free_bytes` and the runbook's “most recent” value are
6,901,800,960, stale by 12,288 bytes. Correct those historical fields when
finalizing; the discrepancy does not change the blocked verdict or the need
for a fresh `df` after human volume expansion and guest filesystem growth.

Before any separate one-label authorization, an Operator must stage and prove
a clean checkout at the pinned commit, rehash both source files and the
unheld checkpoint, verify all six cache directories, establish a clean
single-job/Neuron-holder/label/breaker preflight, and provide a guarded
process with continuous disk monitoring and stable quiescent postflight.
For exact environment identity, the final preflight should also record the
effective user/`HOME` (which selects `prepare.py`'s default data/tokenizer
cache) and inspect the full `NEURON_COMPETITION_R1_*` namespace, including
tokenizer URL and SHA overrides, rather than relying only on the six explicitly
listed unset variables. `prepare.py` can refresh an invalid tokenizer cache;
checking its fixed hashes first avoids an unintended mutation under this
no-cleanup request. A missing cache or changed identity/environment/argv pin
requires a revised request and independent review. A failed live preflight
holds dispatch until fresh evidence passes; the disk gate must be recalculated
against free space after staging. Any attempted dispatch consumes this label.
No automatic scored run follows a successful calibration.
