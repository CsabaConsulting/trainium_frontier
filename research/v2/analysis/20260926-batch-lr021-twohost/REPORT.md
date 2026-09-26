# Joint batch262144 / matrix-LR0.021: actual two-host confirmation

Analyst-only result review, 26 September 2026. **The unchanged selector accepts
the two-host improvement** with the exact output `ADOPT_PENDING_HUMAN_ACK`.
This is eligible for research-base adoption under the existing standing human
approval; the Supervisor records that action separately. This task changes no
canonical ledger, queue, state, scientific source or host. No new gate or run.

## Matched actual results

| Host | Genuine e1e2 control | Batch/LR arm | Exact BPB delta | Arm/control steps |
| --- | ---: | ---: | ---: | ---: |
| trn21 | 0.9880953480915388 | 0.9869854734569898 | -0.001109874634548924 | 1921 / 3631 |
| trn22 | 0.9879041000701606 | 0.9867730678453942 | -0.001131032224766404 | 1921 / 3640 |

- Candidate public-20M mean: **0.986879270651192**.
- Matched control mean: 0.9879997240808497.
- Exact mean delta: **-0.001120453429657664**.
- Exact host-delta disagreement: **0.00002115759021747987**.
- Both deltas are negative, mean is <= -0.0010, and disagreement <= 0.0015.
  No big-win HOLD applies. Two actual arm runs and two actual prospectively
  declared controls are used, with original labels/roles and no duplication.

The selector rounds each host delta before its final arithmetic. Preserve its
raw values: mean_delta=-0.0011205, disagreement=0.0000211,
`verdict=ADOPT_PENDING_HUMAN_ACK`, reason “both hosts negative, mean <= -0.0010,
hosts agree”. `selector.json` is unedited stdout, not a reimplemented decision.

The 0.5291/0.5277 step ratios are diagnostic, not rejection gates. Tokens per
step doubled; nominal total trained tokens are 1.05811/1.05549 times their
controls. Both actual training budgets pass. No assertion that all 1921 loss
values were logged: receipts show 1921 sequential completed step boundaries;
only loss lines actually emitted by the logger were checked finite.

## Exact research-base identity

Adoptable configuration is the accepted **eight-layer e1e2 architecture** with
the existing ordinary schedule and the prescribed JOINT changes: total batch
262144 and matrix LR0.021. It is not a causal isolation of batch or LR alone.

Source SHA-256:
`8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee`

Frozen local source:
`research/v2/proposals/20260926-sprint-p0-clean-batch/variants/batch-lr14-trn22/package/train.py`

Both hosts ran exactly:

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --total-batch-size 262144 --matrix-lr 0.021
```

Seed42, source training budget1800 seconds, compiler flags
`--optlevel=1 --auto-cast matmult --auto-cast-type bf16`, context1024, pack32.
No K2, tables, or Recursive schedule is included in this confirmed configuration.

Recursive schedule alone on eight layers was 0.9880239239389417 versus the
actual trn21 control, delta **-0.00007142415259708823**, selector DISCARD in the
ambiguous low-prior band. Its historical six-layer gain cannot be added to
this mean. Stacking schedule onto the confirmed batch/LR base is a new test.
Separate K2 net-throughput/startup criteria do not invalidate this actual
fixed-budget batch/LR quality result.

## Newly verified trn22 archive

- Raw: `research/v2/operator/20260926-final-sprint-successors/trn22-batch-lr14-completed/results-bundle.json`.
- SHA-256: `c05eaa57272719cb7ba3e2d24dae67f11ba6ca6e2de170ba7d0fbffe163bd91f`.
- **39/39** captured text files rehashed with matching byte lengths.
- Original proposal, exact source/launch file, actual train/eval source hashes,
  command/environment bindings and retained source/prepare/harness identities
  match. The actual argv matches the previous trn21 arm exactly.
- Training exit0, 1921 consecutive boundaries; wall **2599.903523843037 s**;
  measured startup **811.3222134970129 s**, charged-through-exit
  **1788.581310346024 s**, `over_budget=False`.
- Research cache reused from the same-host batch/LR.015 arm. Neither a genuine
  cold start nor actual cache hits are claimed from namespace reuse.
- Fresh separate `prepare.py eval-public` process starts **10:13:23.116502 UTC**;
  actual context1024, batch1, **20,971,520 tokens**, no timeout, exit0.
- Metric independently recomputed: **57,086,380 nats** /
  (ln(2) × **83,462,186 bytes**) = **0.9867730678453942 BPB**.
- Eval terminal: **10:25:10.115361 UTC**.
- Checkpoint **453,044,626 bytes**, SHA-256
  `124a72a9889bf14880bfb2abd66f43217dc25f35b5a21f2f369e86beb630960f`.
  Rehearsal/pre-eval/post-eval/stable retained identity agree. Checkpoint remains
  remote: this authenticates archived hash/size receipts, not a new local copy.
- Original lifetime lock device66305/inode832589 preserved throughout train
  and attached eval. Postflight **10:30:24.781659 UTC** reports all observed
  identities absent, clean ps, idle device, fuser return1 and same lock free;
  outer driver exit0. This review takes no action on later jobs.

The previously audited trn21 arm and both genuine controls remain their actual
ledger records; their original kinds are unchanged. No favorable old replay
was relabeled as a control. Scientific cleanliness is scoped to authenticated
staged bytes, not a blanket surrounding-checkout claim. `causality_passed=null`
means no newly reported causality test, not a fabricated PASS.

## Selector reproduction and remaining limits

`audit.py` is read-only/stdlib-only. It authenticates the new archive and
constructs a temporary four-record input from the three actual existing
matching records plus the new observed arm. `selector-input.jsonl` is an
analysis snapshot, not another canonical ledger. It contains all currently
applicable controls and both actual runs, each once. Reproduce with:

```bash
python3 workflow/tools/decide.py --ledger research/v2/analysis/20260926-batch-lr021-twohost/selector-input.jsonl --arm e1e2-batch262144-lr021
```

Unchanged selector SHA-256:
`fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac`.
`result.json` contains the actual new record for the Supervisor to append ONCE;
this Analyst did not modify the canonical ledger or record an adoption.

This is a two-host **same-seed research improvement**, not a new official score,
an independent-seed result, or submission clearance. The matched gain of
about0.00112 does **not** meet the >=0.003 official-gain upload bar. Both arm
runs still lack genuine cold full proof for this exact configuration. The
official accepted base remains human-reported0.9878 until an actual new upload
is approved and scored; all submission actions remain human-only.
