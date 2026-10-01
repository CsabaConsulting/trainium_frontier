# CLAUDE AUDIT COPY — NOT READY TO SUBMIT

Prepared 26 September 2026, approximately 09:55 IST. This folder is for an
independent source/compliance review, NOT for leaderboard upload. Full cold
training and fresh public20M now pass; native consistency and this independent
compliance audit are still pending. Do not interpret this handoff as a passing audit.

## Latest verified result — 26 September 11:30 IST

This exact clean file completed its full cold run and fresh-process public20M:
**0.9883890652263234 BPB**, 3,628 training steps, actual cold startup
859.4735238229623 seconds and official through-exit charge1789.768582729972
seconds (over_budget=false). The unchanged public evaluator scored20,971,520
tokens at context1024/batch1 in720.338654589 seconds, timeout=false.
Checkpoint4d024cce5120189010d0017fcbf6aaac22a064506eaeb8ca98b453972997d2f2
remained unchanged. Complete terminal/43-artifact archive and702-observed-PID
postflight were authenticated by81 checks in normal and optimized Python.
This closes the full-cold/BPB/budget items listed as pending in older updates
below. It does NOT close native batch-route consistency or Claude compliance.
The official leaderboard score is still the last human-reported0.9901.

## Request to Claude

Independently audit the attached full `train.py` against the attached competition
`README.md`. Default to HOLD when evidence is missing. We want compliant ordinary
language-model training, not merely code that avoids a reviewer keyword. Do not
run the training file in a cloud environment, download weights/data, submit,
change the evaluator or use private validation. Read-only source review and
small self-contained CPU reference tests are sufficient for this audit.

Please inspect:

1. Whether training and `load_for_eval` implement the same learned function,
   including precision, rotary/mask handling, caches, custom convolution/MLP
   kernels, train/eval branches and batch-dependent behavior. Do not assume that
   removing or renaming an evaluation adapter proves compliance.
2. Whether checkpoint loading is ordinary `weights_only=True`, with no external
   weights, validation-derived state, hidden data, special scoring behavior or
   reconstruction that changes the trained model.
3. Causality, initialization, forward/backward correctness and whether every
   custom autograd/NKI wrapper has an identifiable training purpose. Separate
   a legitimate fallback from an unsupported numerical-equivalence assertion.
4. Import-time environment changes/monkeypatching, filesystem/network behavior,
   unused or suspicious feature branches, and any likely human-review concern.
5. Training accounting: 1800-second budget, capped startup allowance, stop
   predictor, learning-rate clock, boundary reporting and checkpoint time.
   The code must not evade official accounting. Do not infer compliance only
   from a locally reported wall-clock value.
6. Whether the exact launch command below matches this file and the rules.

Return a concise verdict (SOURCE-AUDIT ACCEPT, HOLD, or REJECT), followed by
specific file/line findings, severity, reasons and smallest compliant fixes.
Clearly separate actual source defects, unproved numerical/runtime claims and
optional cleanup. If you execute a reference test, include its code and raw
output. Do not invent missing runtime results or guarantee organizer acceptance.

## Exact candidate and command

Candidate SHA-256:
`e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43`

Competition README SHA-256:
`4e4d39d5ef9d4bd69bc2048821292ce74f2ea99b684a823f14f3b01de2b11316`

Launch-command file SHA-256:
`b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd`

```
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```

This command is deliberately unchanged from the previously scored submission
vehicle. Do not replace it with `python3 train.py` or append research flags.
Any proposed submission-command change still requires Poojith's approval.

## Context and evidence limits

The intended architecture is eight 1024-wide layers, attention/convolution in
layers 0,2,4,6,7, four U-Net scalar skips, value embedding only in layer7,
matrix LR0.015, batch131072, seed42, four ranks/LNC2 and1800seconds. No Recursive
schedule or new embedding kernel is included. Training source constants bake
the adopted research configuration into an ordinary single-file submission.

The research source (SHA36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb)
has two fresh public20M/context1024 scores:0.9882310747242088 and
0.9883821509811542, mean0.9883066128526815. These are NOT this clean candidate's
measured scores and NOT official scores. Earlier0.986340 did not confirm and
is not used as the claimed result. The last human-reported official score is
0.9901; no new official result is claimed.

Completed CPU evidence, whose raw archive remains with Codex: full113M-parameter
reference/candidate standard-G2 pair, all65 parameter records and12 losses
matching; ordinary checkpoint helper with65 full parameter hashes,30 paired
forward records at B1/B2/B3 andT256/T257, causal-prefix and state/input-invariance
checks. Fresh independent review ran188 own receipt/mask/input checks. CPU
shims/fallbacks were used; this does NOT establish native Neuron equivalence.

The research loader splits multi-row inputs into separate calls; the clean
candidate instead uses ordinary GPT on the supplied batch, as required by its
clean-port contract. CPU tests match for the tested shapes. Native B1/B2
consistency/causality still needs explicit evidence; a B1 public eval alone is
not a substitute for that broader check.

Update at10:19 IST: the clean cold-five completed with five finite steps,
uncapped startup859.167909355s (below the900s cap), process wall863.638870780s
and charged4.470961425s. Its checkpoint and complete process/device/lock
postflight were authenticated. This establishes cold startup only, not BPB.

Still pending: full cold1800 replay with fresh unchanged public20M/context1024;
native loaded-model consistency/causality; full-run budget/checkpoint evidence
and this independent compliance audit. The cold-five run started09:44:22 IST.
Update at11:01 IST: the separate full cold replay started10:24:43 IST. Its
actual10:54:17 observation had1855 completed step markers and printedstep1850
at265783tokens/s, all four original workers live and no observed training error.
This is progress only: final accounting, native consistency and20M remain
pending. No clean-candidate BPB or submission approval is inferred.
A first local staging attempt failed
before training because its folder was outside the unchanged runner's required
directory; the corrected retry changes only that folder/label, not this file.

Historical organizer feedback was truncated after mentioning `_sub...` and
several constructs needing human inspection. We do not possess full rejection
text in this handoff, and later user reports said older entries were queued for
human review. Do not infer that the adapter was the only concern or that any
particular rename will pass. The ordinary vehicle had previously scored; this
new source still requires independent audit and runtime validation.

Only Poojith may upload. Keep this folder marked NOT READY until Codex supplies
the completed runtime evidence and the human accepts the audit.
