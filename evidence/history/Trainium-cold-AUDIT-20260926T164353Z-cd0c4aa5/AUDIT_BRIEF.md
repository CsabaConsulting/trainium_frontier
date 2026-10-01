# Claude audit handoff — NOT upload approval

This folder contains exactly five attachments. Attach all five to the reviewer.
Candidate train.py SHA256: cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d
Accepted parent SHA256: e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43

## Actual cold result (supersedes pending-cold prose in historical request below)

Public20M/context1024: 0.986890471728; seed42; 1919 steps.
Cold startup: 846.251979s.
Startup excess: 0.000000s.
Charged through process exit: 1789.602241s; over_budget=False.
Archive SHA256: 0af569836f3432ccdde6c2a5bcc689135716ac7feb595f57dacf78d76762d13f
This is public-dev, NOT a new official score or an acceptance guarantee.
Historical confirmed research mean:0.986879270651192 (different research source).
The cold result must be used as measured, not replaced with that mean.

## Exact unchanged launch command

```bash
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```

Only upload train.py if Poojith approves after audit and final result checks.
Do NOT upload accepted_parent.py or cold-evidence.json. No automatic submission.
The ordinary batch/LR stack is included; no Recursive, momentum150, K2 or tables.
Seed43 confirmation was cancelled by Poojith on 2026-09-26. No seed43 result
is claimed or included. The authenticated seed42 cold replay is the native
submission evidence; Claude audit remains pending. Submission keeps seed42.

## Original scoped source-review request

# Source-audit request — current final-slot fallback, not upload clearance

Poojith will provide this to Claude. No automatic external review or upload
has been performed. Please independently examine the actual candidate source
and competition README; treat every statement below as a claim to verify.

Candidate: train.py SHA256
cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d.
Accepted parent: workflow/candidates/confirmed-clock36-116188/train.py SHA256
e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43.
Human reports parent sub-86d307b4 SCORED0.9878 official and its Claude audit
accepted. That history is not automatic acceptance of this candidate.

Claimed complete delta from accepted parent:

```diff
-TOTAL_BATCH_SIZE = 131072
+TOTAL_BATCH_SIZE = 262144
-MATRIX_LR = 0.015
+MATRIX_LR = 0.021
```

Launch command is byte-identical to parent's launch-command.txt, SHA256
b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd.
No --total-batch-size or --matrix-lr launch flags are needed because both
values are baked. No Recursive, K2, tables or momentum150 is included.

Please verify:

1. Only those two literals differ. Independently check effective batch,
   two gradient-accumulation microsteps, LR routing and optimizer groups.
2. Training uses allowed data from scratch and exactly the permitted budget.
   No changes to evaluator, prepare.py, tokenizer, scoring or startup accounting.
3. load_for_eval loads the actual trained model with weights_only=True;
   no alternate eval-only model, adapters, hidden information or validation fit.
4. Existing custom NKI/autograd paths have a clear training purpose; inspect
   import side effects, serialization and any reviewer-sensitive constructs.
5. Distinguish SOURCE-AUDIT ACCEPT/REFUSE from full upload readiness. Report
   exact lines and executable counterexamples for concerns; do not infer
   native correctness or a cold result from a small CPU test.

Evidence available in this folder: CPU_RECEIPT.json and test_package.py.
The executed test uses reduced8-layer/width128/context32/vocab256 BF16 CPU,
actual Muon/AdamW and two accumulation microsteps. Three-step loss/gradient/
model/optimizer states match research8c6 under its batch262144/LR.021 argv.
It uses test-only import shims and is NOT native/distributed/full-size evidence.

Research-source seed42 public20M/context1024 scores are .9869854734569898
ontrn21 and .9867730678453942 ontrn22; mean .986879270651192. Both research
runs reused caches. Candidate-specific cold full/fresh20M, second-seed checks
and this source audit remain pending. No official score for this file exists.
Read READINESS.md for later updates; never replace missing evidence with estimates.

If limited to five attachments: candidate train.py; competition README.md;
accepted parent's train.py (rename accepted_parent.py for attachment clarity);
this request; CPU_RECEIPT.json. Actual later cold logs must be supplied before
any final runtime-readiness conclusion. Poojith alone decides and performs upload.


The five attachments here supersede that request's older attachment list:
train.py, accepted_parent.py, README.md, AUDIT_BRIEF.md, cold-evidence.json.
The evidence JSON includes the CPU receipt and actual cold runtime receipts.
Do not follow historical launch or pending-state prose over the measured block.
