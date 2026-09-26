# Actual trn21 e1e2 control and matched verdicts

Analyst-only review, 26 September 2026. No host action, source change, queue
change, selection-rule change, or adoption. All checks below use the retrieved
receipts; no Torch, NKI, compiler, or other local heavy test was run.

## Outcome

The genuine trn21 control is **0.9880953480915388 public-20M**, not a candidate
relabeled as a control. The original archived proposal explicitly assigned
`kind=control` to this exact label before launch, and its hash is in the actual
under-lock preflight. The existing historical cold replay remains its original
kind. Exactly one new actual control row was appended to `research/v2/ledger.jsonl`;
all prior ledger bytes are preserved.

| Arm | Actual public-20M | Delta vs actual trn21 control | Unchanged selector |
| --- | ---: | ---: | --- |
| Batch 262144 + matrix LR 0.021 | 0.9869854734569898 | -0.001109874634548924 | NOMINATE; other-host confirmation required |
| Recursive schedule on e1e2 | 0.9880239239389417 | -0.00007142415259708823 | DISCARD; ambiguous band, low-prior kind |

The schedule's older six-layer gain is not a demonstrated gain on this
eight-layer base. A future schedule/batch/LR stack is an experiment, not an
assumed additive improvement. The trn22 batch/LR confirmation had not produced
a scored ledger row at this analysis; no confirmed adoption or new upload is
claimed. Best official remains the human-reported 0.9878.

The batch arm's 1921/3631 = 0.529055 step ratio triggers the diagnostic alarm,
not a BPB gate. Its batch doubles from 131072 to 262144 tokens, so its nominal
trained-token ratio is 1.058111, not a comparable 47% reduction in work. Both
runs finished within budget. Cold startup remains unproved for the batch arm.

## Receipt verification

- Raw archive: `research/v2/operator/20260926-final-sprint-successors/trn21-e1e2-control-completed/results-bundle.json`.
- Archive SHA-256: `9289bf8a94f164dccedbc1898408e7b8a02a4e3ed7cb927a44ad501b203e22d9`.
- All **39/39** embedded text members rehashed with matching byte lengths.
- Source: `e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43`,
  identical to the accepted candidate; source also matches training, evaluation,
  preflight, and retained-input identity receipts.
- Launch-file SHA: `b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd`.
  Actual training argv matches the archived accepted command. Seed 42,
  LNC=2/four ranks, 1800-second declared budget, and compiler flags
  `--optlevel=1 --auto-cast matmult --auto-cast-type bf16` match the binding.
- Training: exit 0, 3631 consecutive step boundaries, all printed losses finite;
  wall 2588.8803238560213 s, observed startup 799.1598010229645 s,
  charged through process exit **1789.7205228330567 s**, `over_budget=False`.
  This is a retained research-cache namespace, not a proved cold start or cache hit.
- Fresh separate evaluation starts 09:41:13.673926 UTC after training exit;
  actual command is `prepare.py eval-public`, batch 1, context 1024,
  20,971,520 tokens, Neuron device, exit 0, no timeout. Terminal time is
  **09:52:54.694218 UTC**.
- Exact metric: 57,162,876 nats / (ln(2) × 83,462,186 bytes) =
  **0.9880953480915388 BPB**.
- Checkpoint: 453,044,626 bytes, SHA-256
  `b3e3f061775fc03c7d5fee8bca16cf2e7c06afb225fafaea8259709e1cdbf780`.
  Training rehearsal, pre-eval, post-eval, and stable retained identity agree.
  The checkpoint stays on the host; this is verification of archived remote
  hash/size receipts, not a newly downloaded checkpoint or independent load.
- Original lifetime lock device 66305/inode 596823 is preserved through
  training and attached evaluation. Fresh postflight at 09:53:58.689560 UTC
  reports all observed PIDs absent, idle device, clean ps, fuser return 1,
  and the same lock available. Outer driver exit 0.

One inherited descriptive `NATIVE_BINDING.json` field says
`scientific_patch: recursive`. It is stale template text: the explicit
`kind=control`, original proposal, full train.py bytes, launch command, and
actual source receipts all identify the unchanged e1e2 control. No artifact
was altered to hide that discrepancy. Scientific source authentication is
scoped to those exact staged bytes, not a claim that the surrounding checkout
was globally clean. No new causality test is claimed (`causality_passed=null`).

## K2 comparison reference, unchanged prospective window

The retained control supports the already chosen `35*131072/(t44-t9)` statistic:

- t9 = 801.6183728299802 s; t44 = 818.829054338974 s.
- Base = **266550.74626781716 tok/s**.
- Three-percent threshold = **274547.2686558517 tok/s**.

`result.json` also retains every logged loss through step 200. Examples:
step 0 = 9.0101, step 50 = 5.2341, step 100 = 4.3222,
step 150 = 3.8286, step 200 = 3.6780. These are rounded logged values,
not bit-exact unrounded losses. No faster timing window was selected.

## Reproduction and selector integrity

`audit.py` is stdlib-only and prints the verified result without writing files.
It verifies hashes, roles, bindings, arithmetic, checkpoint and idle receipts.
`result.json` is its archived output.

The existing `workflow/tools/decide.py` remains SHA-256
`fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac`.
These exact commands produced the unedited selector output files:

```bash
python3 workflow/tools/decide.py --ledger research/v2/ledger.jsonl --arm e1e2-recursive
python3 workflow/tools/decide.py --ledger research/v2/ledger.jsonl --arm e1e2-batch262144-lr021
```

Outputs: `recursive-selector.json` and `batch-lr021-selector.json`.
Selector rounding is preserved; the table above gives the unrounded subtraction.
No extra review, experiment, or upload gate is introduced by this result audit.
