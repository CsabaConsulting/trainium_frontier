# Claude audit result: D2 CONTINUE package (train.py b758450b)

Date: 2026-09-29 03:40 IST. Auditor: Claude (session with Poojith). Scope: source audit of train.py
SHA256 b758450b599d4acc5b1fc17d37cccb87421d22bb99c213f6d8ed29fdede13fa8 (99,709 bytes), two cold runs
0.9726966327 (trn21) and 0.9724059578 (trn22), mean 0.9725513, against the uploaded b585 (official 0.9768).

## Verdict: CONDITIONAL ACCEPT. Upload allowed ONLY for the derived file described in "Required fix".

The recipe, the model path and the evidence are acceptable. The frozen file itself is NOT safe to upload
as is, because its command-line default selects the losing FREEZE variant (0.9761) and the scoring runner
may execute train.py with no arguments.

## Required fix before upload (one derived file, no retraining)
1. Change the argparse default: `--tbl_freeze_tokens` default 286000000 -> 0 (line 1567), so a bare
   `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py` runs exactly the scored
   CONTINUE recipe. Keep the flag itself; with 0 passed explicitly the behaviour is unchanged.
2. Fix the two stale header sentences: "Parameters and optimizer state are FP32" (the table and its RMS
   state are BF16) and "the output directory is the only CLI option" (the freeze flag exists). Comments only.
3. Codex must verify: (a) the byte diff versus b758 contains only those lines; (b) `build_config()` from a
   bare parse yields tbl_freeze_tokens == 0 and the same config JSON as the scored runs; (c) AST of the model
   and training code otherwise identical; (d) record the new SHA256 here. No new full runs are needed:
   both scored runs executed this code path with the same value, and the eval path (load_for_eval + forward)
   does not read the flag at all.
4. Upload payload: the derived train.py only. Launch command to declare: the bare command above.

## What was checked (lines refer to b758)
- Model path shared by train and eval: `forward` computes the bigram hash from the input ids
  (row-local, position 0 uses the token itself, lines 987-991) and reads the persistent BF16 buffer
  `value_bigram_table` (register_buffer persistent=True, line 877) in both modes; `table_context` is
  supplied only by the training loop and only turns the looked-up rows into a gradient leaf; the eval
  call passes no context, so the compiled/eager core sees identical inputs. No `self.training` branch
  exists anywhere in the model.
- load_for_eval unchanged versus b585: weights_only=True, strict=True, model.eval(), no eval-only class.
- record_step_boundary(step) is called once per optimizer step after synchronize(device) (line 1681); the
  table update (every 2 steps, synchronized across ranks) happens inside the step, before that call.
- No prepare.* attribute access or monkey-patching; prepare imported for the documented names only.
- No new file, network, environment or subprocess calls in the diff versus uploaded b585.
- Table trained only on training-loader tokens (hash of input ids); no validation data touched.
- Cold startup 645.2 s and 676.2 s (< 900); charged 1789.2 s and 1790.0 s (< 1800); both zero excess.
- Checkpoint 588,313,401 bytes (table 128 MiB included); no size rule exists in the public rules.
- Seed 42 hardcoded; two same-byte runs 0.0003 apart, consistent with the b585 spread.

## Eight-item checklist (source scope, my reading of the brief's rows)
1. Only train.py changed: PASS (organizer clean-box equivalence unproved, same as every prior package).
2. Strict load, shared causal train/eval path: PASS.
3. No validation/external/network dependence: PASS (the tokenizer URL is inherited from the provided
   prepare.py, unchanged since b585).
4. Cold <= 900 s, charged <= 1800 s: PASS on both runs.
5. Cumulative writes unmeasured: non-blocking, same as the parent audits.
6. Bare command runs the scored recipe: FAIL on b758 as frozen; PASS after the required fix.
7. Fixed seed, declared time-based behaviour only: PASS.
8. Readable source: two stale header sentences and unreachable injection alternatives remain; the header
   is fixed by the required fix, the dead alternatives are acceptable for this upload.

## Notes for Poojith
- This file is the Tuesday-morning reviewer probe: expected official 0.9722-0.9730, about 0.0044 better
  than the standing 0.9768, still above the top-10 line (0.9699).
- Write-up sentence for the submission form: "Adds one hashed bigram lookup table (262,144 x 256, BF16),
  an ordinary trainable buffer indexed by a hash of the previous and current token ids, trained only on the
  provided training data with RMSprop; its rows feed the input embedding and the layer-7 value path through
  learned gates. Train and eval use the same forward pass; evaluate_bpb and record_step_boundary are
  untouched."

## Codex R66 implementation receipt — Sep 29, 03:36 IST

This is a Codex execution record, not a rewrite of Claude's verdict above.
Required derivative SHA256:
`6d197b3cd08a4e107020e5bb03a946a06a361ed36c6d7dfc12b35ca1484be14c`
(99,747 bytes), exported as `train_UPLOAD_6d197b3c.py` in this folder.
The original scored `train.py` remains byte-identical b758450b, as directed by
the latest Round66 file; upload the derivative as train.py, not the original.

All requested source conditions verified: exactly the three authorized text
replacements; bare argparse freeze0; complete model_config equals BOTH scored
logs; normalized module/model/training AST otherwise identical. Explicit0
still selects the same CONTINUE path. `build_config`'s direct default was not
changed because main passes the parsed value. No evaluator/model change,
retraining, new score or Codex upload occurred.

Author structural receipt SHA256:
`2f3020a86911765f8414618839e366171547e2fccce0f36e0ac5fa2cf37da4c3`.
Fresh independent reviewer authored and executed its own stdlib AST/parser
reference: ACCEPT for this derivative/export; receipt
`135e59a323e9e06549ef7891b7b36d2ed2f18394846794b7bb8b9257b947693c`,
review `6f15bd2e2ff000b04682a9283cf84b64524f3138400c12e7ccbaa68d0d84bfc2`.
Copies are in UPLOAD_PROOF/. Original Claude audit before this appended receipt
was `55be15b181201ef89a420d6048081a60c417e55836a51d5e8d7a7a7bfa745df9`.

Declared upload command (per Claude's required fix):
`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`.
The two measured scores remain .9726966326753679 and .9724059578084576
(mean .9725512952419127); they belong to scored b758 with explicit0, and
support this derivative by the authorized equivalence proof, not new cold runs.

## Derived upload file verified (2026-09-29 03:50 IST)
train_UPLOAD_6d197b3c.py, SHA256 6d197b3cd08a4e107020e5bb03a946a06a361ed36c6d7dfc12b35ca1484be14c, 99,747 bytes.
Byte diff versus b758: exactly the approved lines (header lines 11-12 and 20-21 corrected; line 1567 argparse
default 286000000 -> 0). main() passes args.tbl_freeze_tokens into build_config, so a bare command yields
tbl_freeze_tokens = 0, the scored CONTINUE recipe. AST parses. Nothing else changed.

## Verdict: SOURCE-AUDIT ACCEPT (upload allowed): upload train_UPLOAD_6d197b3c.py renamed to train.py.
Declare the bare command: NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py
Expected official: 0.9722 to 0.9730. Planned slot: Tuesday morning (reviewer probe), unless a better
two-run file exists by then.
