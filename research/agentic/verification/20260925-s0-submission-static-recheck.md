# Independent S0 submission static recheck

Verdict: **no proven rule violation in the S0 source; submission and warm dispatch remain HOLD**.
This is a source/package audit only. I did not edit train.py or prepare.py, use a host, train,
evaluate a checkpoint, reset a breaker, or authorize a submission. Automated-review acceptance
and Neuron equivalence cannot be guaranteed by static inspection.

## Exact scope

- Controller snapshot: 753787ae5e7710218538a5c0dae5f992c3301630.
- S0: workflow/candidates/S0/train.py and submissions/candidates/S0/train.py both SHA-256
  ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb.
- Parent: 05-train-submission-clean-116188.py SHA-256
  116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d.
- Trusted prepare.py SHA-256 5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb.
- Warm S0 science parent c9bc9517166cd38cba23764af2756aab3f53d5a4 has the exact S0 root
  train.py and unchanged prepare.py bytes. Pending request commit 008e9bd3101c8301e5bf2af4ea10511cf8a3b566.

## Findings, highest priority first

1. **P0 — Runtime submission gates have not passed.** S0 EVIDENCE.md still requires native
   training/checkpoint loading, 100-step or equivalent Neuron lockstep, batch-independence and
   causality, two cold bare-command replays, and fresh-process public-20M evaluation of S0's own
   checkpoint. The pending warm request explicitly says dispatch_authorized=false and would not
   substitute for either cold replay. With the latest recorded trn22 free space 6,448,951,296 B,
   its 8,833,482,752 B pre-dispatch threshold is short by 2,384,531,456 B; the host breaker is
   tripped. Do not upload or dispatch from this report.

2. **P1 — Training/import compiler defaults differ.** At S0 lines 37–40, executing train.py as
   main defaults NEURON_CC_FLAGS to optlevel 1 plus BF16 matmul auto-cast, while importing it via
   prepare.py's scorer defaults only to optlevel 1. The pending warm request explicitly exports
   the training flags, but the official scorer environment is not established by this request.
   This is not proof of a different learned model: the checkpoint weights are strictly loaded
   unchanged. It is, however, an unclosed Neuron inference-numerics/compile risk and the first
   unresolved item from the report's section 7.4. Compare the same S0 checkpoint's logits/BPB in
   fresh processes with both compiler environments, and document the result before submission.

3. **P1 — Native evaluator batch shape takes a different NKI lane variant.** S0 lines 581 and
   92 select two lanes for the even training batch (PACK_FACTOR=32) and one for the scorer's
   default batch size 1. Both are intended to implement the same causal convolution; the source
   contains no batch-content special case. Archived CPU logits and lockstep checks do not execute
   either Neuron lane variant. Prove same-token logits are batch-independent on Neuron and that
   batch-1 fresh-process loading/evaluation succeeds.

4. **P2 — Eval loader has visible, justified shape handling that may still draw review.**
   At lines 1203–1222, load_for_eval enlarges only the non-persistent rotary-buffer capacity to
   8192, initializes the model, then strict-loads all checkpoint parameters, freezes it and
   returns the same GPT class. The code explains the purpose and CPU evidence shows identical
   FP32/BF16 logits to 116188 through 2048 tokens; this is not an observed eval-only model.
   Nonetheless, an automated judge may focus on the eval-specific config assignment or the
   redundant-looking init_weights call. Preserve the explanation and include the Neuron
   same-checkpoint equivalence result; simplify only through a separately verified new hash.

5. **P3 — Broad optional-NKI exception is a readability risk, not an identified bypass.**
   Lines 180–598 catch every Exception and set _HAS_NKI=false. On Neuron the causal-convolution
   path then raises instead of silently changing the scored model, but the broad catch hides the
   original import/setup error and may invite closer review. No source change is requested in
   this verifier receipt.

## Checks completed and favorable evidence

- Full 116188-to-S0 diff read: 91 insertions, 212 deletions. Config is baked to 6x1024 and
  pack factor 32; dead GQA/coalesced branches, absent-feature flags, in-process validation and
  optimizer-resume code were removed. The same GPT/Block/MLP forward is used during training and
  by load_for_eval; no self.training branch or _SubmissionEvalGPT-like class remains.
- GPT.forward uses causal SDPA, a past/current-token two-tap convolution, per-token norms/MLP
  and current-token value embeddings. No collectives, file reads or rank/batch-content branches
  occur in the model forward. Distributed operations are training-side optimizer/gradient work.
- No script-side public/private validation read, network import, monkeypatch, or forbidden
  eval/exec dynamic-execution builtin,
  or pretrained/external data reference found. Training calls make_dataloader(..., 'train', ...).
  The only torch.load uses weights_only=True; state_dict loading uses strict=True. The checkpoint
  is model-only and written under out/final.pt with the bare launch defaults.
- The submission launch file is exactly
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py:
  full chip, legal shape, no extra train.py flags. The actual pending warm job adds a timeout,
  isolated environment, step log and warm cache paths outside that science command; it is not a
  cold or official-environment replay.
- The archived four-rank CPU comparisons at workflow/evidence/lockstep3-116188-vs-S0.json and
  research/v2/g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/remote-run/comparisons/116188-vs-s0.json
  both report identical plan, three losses and 62/62 parameter hashes. The CPU eval-path receipt
  reports bit-identical FP32/BF16 logits on four input shapes. These receipts are not Neuron proof.
- I re-ran the warm package's local static test on its isolated branch: PASS. Direct
  prepare.py check could not start in this workstation Python because pyarrow is absent. I
  executed its exact AST-only check_submission and launch-warning definitions instead; warnings
  were empty. This does not replace the check on the official environment.

The previous reviewer concern about unsafe torch.load is contradicted by the literal
weights_only=True call. Custom NKI kernels and autograd wrappers remain, but each is tied to
ordinary training math and no evaluation-specific adapter is present. A reviewer may still ask
for native evidence; static cleanliness is not a promise of acceptance.
