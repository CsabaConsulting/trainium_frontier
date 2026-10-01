# Claude audit result: A3 package (upload train.py 29e5cd2c, scored source c5ef8414)

Date: 2026-09-29 09:55 IST. Auditor: Claude (session with Poojith).

## Verdict: SOURCE-AUDIT ACCEPT (upload allowed). Upload this folder's train.py as train.py.
SHA256 29e5cd2cfd35bb52e5a955057a1661786239d0812d9a797c85dc0df7fe2d7446 (100,795 bytes).
Command to declare: NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py
Scores (same bytes c5ef8414, seed 42, three cold runs): 0.9719336 / 0.9716332 / 0.9715861, mean 0.9717176,
0.0008 better than D2's two-run mean (0.9725513). Expected official: 0.9712 to 0.9722.

## What was verified on the laptop copies
- Byte diff scored c5ef8414 -> upload 29e5cd2c: exactly the three authorized edits (header lines 11-12 and
  20-21 corrected; line 1577 argparse default 286000000 -> 0). main() passes args.tbl_freeze_tokens into
  build_config (line 1587), so the bare command runs the scored CONTINUE recipe. AST parses.
- Scored A3 vs D2 (SCORED_VS_D2.diff): proj7 (learned 128->1024) removed and replaced by a fixed repeat of the
  128-wide slice across the 8 heads before the existing per-head 2*sigmoid gate; a config assertion
  (128 channels per head); and an end-of-run rank-consistency diagnostic on a cloned table (broadcast +
  max all-reduce, charged inside the budget, no state change). Fewer parameters than D2, same forward for
  train and eval, eval path (load_for_eval, weights_only=True, strict=True) unchanged.
- No prepare.* monkey-patching, no evaluate_bpb reimplementation, record_step_boundary after every step,
  no new file/network/env calls, table hashed from input ids inside forward, table is a persistent buffer.
- Cold starts 664-687 s (< 900), charged 1792-1794 s (the harness's own accounting; over_budget False on all
  three receipts), rank table max diff 0 on all three runs.

## Eight-item checklist (source scope)
1 only train.py: PASS (clean-box equivalence unproved, as for every prior package).
2 strict load, shared train/eval path: PASS.  3 no validation/external data: PASS.
4 cold/charged budgets: PASS x3.  5 cumulative writes: unmeasured, non-blocking.
6 bare command = scored recipe: PASS (default 0).  7 fixed seed 42: PASS.
8 readable source: header fixed; dead injection alternatives remain, acceptable.

## Notes for Poojith
- Order of uploads: D2 (6d197b3c) first as the reviewer probe if not yet uploaded; A3 (this file) next.
  If both cannot go up today, A3 alone is fine: it is the same design shape with fewer parameters and a
  better score.
- README.md in this folder carries the write-up framing.
