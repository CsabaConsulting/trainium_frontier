# bcbd final-result audit preparation

Status: **ONLY_PENDING_REAL_RAW**. No bcbd score, result PASS, selector verdict,
ledger append or adoption is claimed. This is archive-only Analyst preparation;
no training source, evaluator, authoritative ledger/state, queue or host changes.

Pinned run: fs-20260926-basebcbd-initial-cache-trn21-r1-full.
Packet7da2559f03cfda04af3a7858e1b2d4037f8b676b, release40b7d40084de50b12a1502c06c1c524406a4998c.
Sourceb635c8b0/SHAbcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522.
Exact6ef7 Recursive6x1024 science plus initial-cache-refresh only.

## Ready commands

After retrieval, substitute the **full40-character immutable final archive commit**
and its **64-character bundle SHA**:

```sh
python3 -I -B research/v2/analysis/20260926-basebcbd-final-audit-preparation/audit.py --raw-commit FULL_COMMIT --bundle-sha BUNDLE_SHA
```

Only after inspecting that authentication output, run the unchanged selector
as a separate read-only invocation:

```sh
python3 -I -B research/v2/analysis/20260926-basebcbd-final-audit-preparation/audit.py --raw-commit FULL_COMMIT --bundle-sha BUNDLE_SHA --decide
```

The second call reauthenticates evidence, runs exact decide.py
SHAfa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac from
Git1c879c2b on its immutable ledger snapshot plus an **in-memory** candidate,
and prints the result. It never edits the actual selector or ledger.
If newer genuine controls/rules are selected, Supervisor must explicitly review
and update these snapshot pins; this is not a silent latest-ledger service.

No arguments prints ONLY_PENDING_REAL_RAW with null candidate/selector fields.

## Required final inputs

At the supplied raw commit, under the fixed directory
research/v2/operator/20260926-trn21-basebcbd-full/completed/:

- results-bundle.json: completed training, both exact launches/environments,
  frozen binding/manifest, train/eval receipts/logs, checkpoint receipt, official
  step log, both lifetime-lock guards, all four closed graph CSVs, exits and
  final successful terminal/outer records.
- explicit-process-postflight.json: all observed training/compiler/evaluator
  identities, full current PID/PPID/PGID/SID census, idle device and original
  trn21 lock66305:596823. The first ten known training PIDs are independently
  pinned to launch archive7aa13bd0/SHAeefd25b1…, not invented.
- post-eval-checkpoint-hash.json: actual unchanged checkpoint hash/stat after
  fresh20M, held original lock and idle checks, with timing/size evidence.

Missing files, schema differences, incomplete evaluation, hash drift or failed
mechanical/numeric gates return HOLD_MISSING_OR_INVALID_FINAL_EVIDENCE, exit2.
Do not fabricate records or reuse another run merely to satisfy the parser.
Additional compiler/evaluator PID completeness still needs human-readable
cross-check against final live evidence; a parser cannot discover unarchived
processes. Expected final file names/layout follow the existing full36 schema.

## Checks and comparison

The parser hashes every archived file, authenticates exact source/prepare/
harness/support/manifest, effective argv/environment and isolated sandbox
receipts, five-boundary allowance capped900, finite monotonic steps, source
over_budget=False and independently computed charge through process exit<=1800.
Optimizer time need not equal1800: checkpoint reserve is valid. Step count is
reported, never a verdict gate. It checks fresh public20971520/B1/context1024,
recalculates BPB from total_nats/(num_bytes*ln2), eval timeout/exit, checkpoint
handoff/post-eval identity, all4 CSVs and explicit process/device/lock quiescence.
CSV counters do not imply cache hits; retained caches are not a cold-full claim.

Formal same-host control is the genuine kind=control6ef7 r2:
**0.9899528600562257**, label fs-20260925-l3-base-repeat-trn21-r2-full.
Its exact ledger row and raw eval log are pinned and BPB independently recomputed
from57270336 nats/83462186 bytes. The older0.9900560897366009 schedule_arm remains
a secondary historical contrast; the frozen packet is not rewritten. No control
is retroactively relabelled. Faster startup alone is never an initial-cache win.

## Preparation evidence

Own tests use closed full36 archive110054fe **only as a parser fixture**, never
as bcbd evidence. Normal/-O outputs are byte-identical:441 checks, one positive
archived fixture and17 expected refusals (including wrong source/run, partial
eval, bad totals, missing rank CSV, budget contradiction, altered checkpoint
and surviving process). Two additional unchanged-selector budget negatives
exercise only the authentic historical control. No bcbd selector outcome ran.

```sh
python3 -I -B research/v2/analysis/20260926-basebcbd-final-audit-preparation/test_audit.py
python3 -O -I -B research/v2/analysis/20260926-basebcbd-final-audit-preparation/test_audit.py
```

Audit SHA1ec138987f32d050448d2318702f06632f6ca8e1d531c9c3e1142812730dc770.
Test SHAb38f7e8d54947116097e2003501c92eddcdf20c350bbfa0c288f9afa91cc8e33.
Both test receipts SHAb4d07c2f51670e2f97ba791f8a3bab9e08a010c4eda39d9190bcaa918741a10a.
Source/native correctness is not independently requalified by this preparation;
only final result authentication/comparison is being readied. No tensor imports,
training/validation data reads, network access, host actions or authoritative
writes occurred.
