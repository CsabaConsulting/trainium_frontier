# BF16 residual candidate — request for Claude source/compliance audit

Audit this exact train.py against the included accepted cd0c4aa5 parent.
Do not assume acceptance from the performance result. Poojith alone uploads.

Candidate SHA-256: 54c835df60eb5d17ff49c6b79aa05aafd277027313fe6a07ed2955bdfaaebe3b
Parent SHA-256: cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d
Change: only the two scalar use-site casts in Block.forward. FP32 master
parameters/gradients remain; residual activations stay BF16. No architecture,
optimizer, data, evaluator, load hook, kernel, launch, or hyperparameter change.
The inherited header still says FP32 residuals; that description is stale.
It is deliberately not edited after the measured run, preserving exact bytes.

## Exact launch command (unchanged accepted vehicle)

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```

Upload only train.py, after Claude accepts this SHA. Do not use a bare python
command or append any research timing/table/flash flags.

## Measured evidence

- Public20M/context1024: 0.9843338604346282; seed42, trn21.
- Same-host cold parent: 0.9868904717283661; delta -0.0025566112937379.
- 1,988 steps; observed fifth-step startup 828.8808253350435s, charge0.
- Charged through process exit1788.8580229139188s; budgetPASS, over_budgetFalse.
- Training checkpoint c50a0e035c1a66e0d70a136f51bc9261c5c09bc613beb4d38555c17ea6f93da8.
- Attached fresh-process public20M succeeded; same source/checkpoint receipts.
- Empty-cache preflight and exact harness/private-cache inventory included.
  See the evidence-review caveat/conclusion in cold-evidence.json. An empty
  rehearsal fresh_empty_sdk_caches dictionary alone is not a cold-run receipt.
- One measured BF16 run, not a replicated mean. No BF16 official result exists.
- Poojith reports parent official0.9866; estimated BF16~0.9838 is a planning
  estimate, not guaranteed calibration or a leaderboard result.

## Audit focus

Verify same trained model in load_for_eval, weights_only loading, no evaluator
or data access changes, legal full-chip launch and budget. Review inherited
NKI/custom-autograd/import-time behavior as well as the tiny diff; parent
acceptance does not guarantee this upload. Author CPU and original independent
source review receipts are included. The new evidence rechecker previously
implemented the casts and is NOT a fresh independent source auditor.

No upload has been performed. Human reports quota15/week,5used; check portal
before upload. Freeze Sep29 18:00IST. This package changes no host process.
