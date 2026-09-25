# Disabled checkpoint-only behavioural diagnostic

Assessment: a conservative, bounded diagnostic is feasible on unchanged sources. This is **not** reversal, repair, relabelling or renewed model approval of the fresh `EVAL_BATCH_WRAPPER_OMITTED` REFUSE in f15d9239. No reviewer follow-up or provider retry is proposed. Full training, quality claims, source-gate acceptance and upload remain HOLD.

## Why this is a useful separate question

README.md:219–223 requires a consolidated single-device replicated trained model whose forward is pure/stateless/causal, independent of batch size/content/world size/rank, frozen, without hooks or forward collectives. It specifies observable behaviour, not a required subclass or per-row implementation. The existing 116188-style ordinary-GPT loader is deliberately retained by the clean candidate. The fresh reviewer correctly proves that it differs structurally from research3e79's per-row wrapper, while observing zero CPU logit error in its small tests. Neither inference — “different hook count means measured invalid logits” nor “small CPU equality proves native batch invariance” — is warranted.

The narrower falsifiable question is whether the unchanged ordinary-GPT loader, using the **same retained trained weights**, exhibits correct B1/reference parity and batch/state/causal invariance at the prospective context1024 on CPU and the actual native backend. Passing would supply new behavioural evidence, not make the structural omission disappear or retrospectively change that review's specification/verdict.

## Immutable inputs

- Clean commit `bd9e33eee671ee489a1201485faf748bf0a35dea`, `workflow/candidates/architecture3e79-116188/train.py`, SHA256 `3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f`.
- Research commit `b4d83b5db66971f3c348837f04f53cba65aa6d33`, `workflow/proposals/research-v2/train.py`, SHA256 `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`.
- Trained reference checkpoint, retained on **trn22**: `/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-primed-trn22-r2-full-packet/train/out/final.pt`; SHA256 `068749c2559f7c68f6acc76d558caff9d3806d85273fa8562483d87bbad50d25`, 453,048,594 bytes. Retrieved training receipt is `research/v2/operator/20260926-trn22-arch3e79-full/completed-training/checkpoint.json`, raw commit8d9ee4b0. Rehash locally on that host before and after this read-only diagnostic; do not edit, move, overwrite or download it to the laptop.
- Unchanged prepare.py SHA256 `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`; README SHA256 `4e4d39d5ef9d4bd69bc2048821292ce74f2ea99b684a823f14f3b01de2b11316`.

No training/update/save of a new checkpoint, public/private data, tokenizer call, trusted scorer change or val_bpb evaluation. Use deterministic synthetic in-vocabulary tokens only. Each source's real `load_for_eval(checkpoint, device)` loads the identical consolidated113,256,724-parameter/65-state-tensor checkpoint strictly. Authenticate state names/shapes/dtypes and bytes; log effective architecture/config, model class, compute dtype, output dtype and source/checkpoint pins. Do not strip reference metadata, transplant outputs, patch globals, alter the table floor, install a candidate eval adapter, monkeypatch forward, attach hooks, wrap row execution, or change either source.

## Fixed minimal test matrix

One context **T=1024** (thus actual NKI-eligible lengths), B=1 and B=2 only. Deterministically construct distinct rows A/B/C of1024 valid tokens using seed42; retain their exact bytes/hashes. A′/B′ are copies with only the suffix starting at `cut=max(2,T//2)` reversed, matching the unchanged prepare.py causality transformation. No padding, data-dependent batching, or source-specific input differences.

For each source and backend, in a fixed recorded order, call the actual frozen model on:

1. A; B.
2. [A,B]; [B,A]; [A,C]; [A,A].
3. [A′,B′].
4. [A,B] again, then A again after that unrelated call history.

Nine calls/source/backend, reusing captured outputs for all comparisons. Do not extend the context/batch matrix or retry automatically after an error.

Compare same-backend candidate vs reference B1 outputs; candidate B2 rows vs its separately evaluated B1 A/B; reversed batch after inverse permutation; unchanged A row under different partner B/C and duplicate A; repeat A/AB after intervening calls; and candidate/reference outputs for the corresponding matrix cases. All shapes/dtypes/finiteness must agree. Record exact byte equality, element equality, max absolute/relative error (with explicit denominator), relative RMS, mismatch count and representative/max-error coordinates/values. Relative/cosine summaries are diagnostic only, not substitute acceptance thresholds.

For batch/reference/repeat invariance, propose **strict same-backend byte equality** as this diagnostic's conservative success threshold. Any nonzero/byte difference is HOLD for interpretation, not silently accepted using a new tolerance and not automatically declared an official rules violation. README gives no numeric batch-invariance tolerance. Do **not** repurpose prepare.py's causality1e-2 threshold for batch or cross-source parity.

Separately reproduce prepare.py:663–684's causal-prefix measurement exactly on synthetic inputs: compare original vs suffix-reversed outputs through `:cut-1`, in float32, maximum absolute difference <=**1e-2**. Label this a synthetic diagnostic following the unchanged function's arithmetic, **not** an invocation of the official real-data causality_check. Also report the full error, including whether exact equality happened. Do not monkeypatch prepare.py's loader to substitute synthetic tokens.

Before and after the matrix, fingerprint parameters and registered buffers, assert eval mode, no trainable parameters and empty model hook dictionaries, and check repeated-output independence. No source hook is added to inspect call counts. A changed state or nondeterministic output is a concrete HOLD; fail closed and retain evidence. Static review must confirm no world/rank-conditioned forward or collectives. Native workers use the same tokens on every rank and compare rank outputs outside forward; this provides rank0–3/world4 evidence, **not** universal world-size proof.

## Two bounded stages; no other work on the host

CPU first: one single-thread process on trn22, ordinary CPU loader dtype with **no BF16 monkeypatch**, source models loaded sequentially. Maximum600s+15s termination grace. This executes full-parameter inference only, not the pending four-rank G2 update test. Compare CPU to CPU, not CPU FP32 to native BF16.

Only after independently accepted CPU receipts and a separate scoped Operator assignment, native: four LNC2 workers on the full chip, using the existing native evaluation initialization/mapping and actual loader dtype. Source models may run sequentially in each worker; no training or forward collectives. Maximum900s+15s grace for the whole native diagnostic, including imports/compile/load/cases/cleanup. Existing cache roots may be reused without mutation/reset; record all effective cache bindings, SDK versions, source/class/dtype and actual process wall. Cache warmth and cold-startup costs are unproved and irrelevant to a performance verdict because none is sought. Timeout or unavailable API means **UNRESOLVED**, never permission to expand the job or proceed to full training.

Use the existing common lifetime flock `/home/ubuntu/.cache/trainium_freshstart/device0.lock`, inherited FD through child exit, fresh ps AND neuron-ls under lock before each phase, fail closed on busy/error/unknown. Require no other host job, no STOP/breaker, current sole-Operator assignment, at least64GiB available host RAM and6.5GiB free disk, one-thread CPU environment, and bounded artifact output. No laptop full tensors or native work from the Verifier. Proposed labels/driver/argv/source/support pins and the complete proposal still need Implementer/independent mechanical review; this plan is not a launcher.

Keep scalar/hash receipts for every rank/case, exact synthetic inputs, raw stdout/stderr/exit codes, source/checkpoint/config identities, native timing and final quiescence. Cap total new artifact data at256MiB; at most one selected full mismatch pair across the entire diagnostic, with other ranks/cases retaining deterministic inputs/checkpoint pins and bounded mismatch slices rather than every64MiB logit tensor. A cap breach stops the diagnostic with explicit incomplete-evidence status; no automatic rerun. Original source/checkpoint and prior REFUSE archive remain untouched.

## Interpretation

Possible terminal results are `BEHAVIOURAL_DIAGNOSTIC_EXACT_ON_TESTED_SCOPE`, `HOLD_NUMERIC_OR_STATE_DIFFERENCE`, or `UNRESOLVED_EXECUTION_OR_EVIDENCE`. None equals fresh-source PASS, full training authority, candidate BPB0.986340, or submission approval. CPU-only success cannot authorize native or establish it; successful native T1024/B1/B2 cases still do not prove all contexts/world sizes. After evidence retrieval, the Supervisor/compliance process must explicitly reconcile the functional contract question with the still-true structural REFUSE; do not alter its original report or solicit a reworded verdict. Existing fullCPU/update, true-cold/startup, full-budget quality and final submission gates remain separate and unpassed.

No code, tensors, providers, hosts or model source were executed/changed to prepare this plan. Only local README, immutable source, unchanged prepare.py, archived review and checkpoint receipts were inspected.
