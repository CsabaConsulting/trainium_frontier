# S0: review-clean dense candidate (numerics identical to 116188)

**Prepared by agents. Only a human submits.**

| Item | Value |
|---|---|
| `train.py` SHA-256 | `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb` |
| Launch command | `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py` (bare; the configuration is in the file) |
| Built from | `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d` by `make_s0.py` (exact-match block edits, each asserted unique) |
| Expected score | same model as 116188: public-20M ≈ 0.99195 (two 116188 runs) → official ≈ 0.991 (±0.002, one calibration pair) |

## What changed (reviewer-facing; no arithmetic changes)
1. The docstring now states the launch command, the full chip (LNC2 × 4), the budget logic, that training
   reads only the `train` split, and the `load_for_eval` contract.
2. The configuration is baked in (`SEQ_LEN=1024`, `N_EMBD=1024`, `PACK_FACTOR=32`). All "must agree" flags
   are removed, including `--ngram-ve` and `--qk-shift`, which advertised absent features.
3. **One MLP forward for training and evaluation.** The old `self.training`/shape switch is gone. The
   custom autograd function only changes what the backward saves.
4. **No validation data in the training script.** The in-process `evaluate_bpb(split='public_val')` and
   the causality check are removed, and so are their imports.
5. Dead code removed:
   - GQA and coalesced-projection branches;
   - constant-folded validations for batch-ramp, weight-EMA and byte-WTE (features that don't exist);
   - `LOSS_SCALE = 1.0` and `* 1.0`, which are exact no-ops;
   - `depth = None`;
   - unused imports;
   - the unused optimizer resume path.
6. Wording: "ownership probe" → a descriptive error; the "NKI CE adapter" helper → `_ce_lanes`, defined
   before its kernel; the load-balancing timing constants are documented.
7. Kept on purpose, because it's how 116188 was trained: attention reads the block input from before the
   local convolution. It's now commented rather than hidden behind `q_in = k_in = v_in = x`.

## Proof so far (CPU, gloo, 4 ranks, the file's real `main()`; receipts in `workflow/evidence/`)
| Check | Result | Receipt |
|---|---|---|
| G2 with the bare launch command | PASS; checkpoint strict-loads with deterministic logits | `lockstep3-S0-bare.json` |
| Training lockstep against 116188 (3 real steps) | **identical**: ownership plan (`b89d6222…`), all ranks' losses, 62/62 parameter hashes | `lockstep3-116188-vs-S0.json` |
| Eval path: the same checkpoint through 116188's and S0's `load_for_eval` | **bit-identical logits** in FP32 and BF16 at shapes (1,1024), (4,1024), (2,2048), (3,200); identical state dicts | `S0-eval-path-equivalence.txt` |
| Reviewer grep | no `adapter`, `probe`, `evaluate_bpb`, `public_val`, `self.training`, `MethodType`, `_SubmissionEval`, `NEURON_COMPETITION_R1_*` | this file |

## Remaining gates (Neuron; Operator plus an independent Verifier)
1. **Neuron 100-step lockstep** against 116188 (116188 with its flag argv, S0 bare). Steps 0–1 identical;
   later steps within 1e-4 relative. The NKI kernels (rope+norm, local conv, softcap CE) and the
   `_FusedMLP` path on Neuron are not exercised on CPU.
2. **Two cold replays** of S0: fresh `NEURON_COMPILE_CACHE_DIR`, the bare command, public-20M within
   ±0.0008 of 0.99195, startup ≤ 840 s, `over_budget=False`.
3. **Eval-path check on Neuron:** evaluate one S0-trained checkpoint through both S0's and 116188's
   `load_for_eval` at public-2M. The BPB must be identical. S0's `GPTConfig` drops two unused fields, so
   S0 cannot load an *old* 116188 checkpoint; load S0-trained checkpoints through both files instead.
4. **Verifier checklist C** (`workflow/roles/verifier.md`), then the human submits and records the result
   in `research/v2/calibration.json`.
