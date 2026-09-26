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
