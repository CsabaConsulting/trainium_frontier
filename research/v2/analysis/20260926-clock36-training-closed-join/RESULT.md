# Closed training-phase compiler-input join — exact 36a862

Scope: local archived evidence only. The full run's **training phase completed**, but this archive was retrieved while its attached fresh evaluation was running. No BPB, quality, final job/quiescence or cache-hit verdict is made. No host, source, cache, SDK, evaluator or gate changes.

Input: Git `6b292621`, `research/v2/operator/20260926-trn22-startup-clock-full/training-completed/results-bundle.json`, SHA-256 `34cf54017543997d7d5673e023601d02ecb6069cfb3a17219970bcf2cee08911`. All 22 files / 408,769 decoded bytes authenticate. All four closed graph CSVs authenticate against their embedded hashes. `RESULT.json` preserves the exact row keys, per-file hashes and original harness timing.

## Exact principal mapping

| Captured raw module | Closed CSV graph name | Nodes | Matching ranks | Positive recorded compile time |
| --- | --- | ---: | --- | --- |
| A: `5a0c39fda4d4962afd222db766c1b2eda6ad6ed7d0b0fd1889afc42edc9a4e3c` | `model_default_0_forward_1` | 666 | 0, 1, 2, 3 | rank 1: 201.175 s |
| B: `f21636e78487960b13e9682fbfd8cc987854ffc2da4d7df7ad314474ec467848` | `model_default_0_backward_3` | 1,169 | 0, 1, 2, 3 | rank 3: 472.263 s |

Each capture has exactly one matching row per rank. Identity is the actual captured binary MLIR SHA-256 equaling the CSV `graph_id` prefix—not the different compiler IR signature or debug-directory name. The remaining ranks' zero compile metrics are **unknown, not cache hits**. Durations are individual positive row values, not summed across ranks or equated to elapsed startup. The live observations independently recorded real compiler processes; this is not an inference from zero metrics.

The prior exact-source cold archive `e08afdec` has the same named graphs / node counts on all four ranks but different principal raw-key prefixes: forward `9836e351be15c436667378517bb45d19d032facf0d2a9c5e283c3e63ac0ad6c2` (202.864 s, rank 3) and backward `74f4bcee2eb7b52306f5b86a3ce3bcf9d6175e7b6707674531e9bd2dc005d03f` (470.027 s, rank 0). Both runs share the recorded key suffix `cd47b8171ac56a23cc896e6ceb62fa0bfd9e8cf0bf9b08bd2a965094c3965433`. Equal names/node counts do not establish byte or semantic identity. Different source path / other captured input metadata may contribute; this does not isolate path as the sole cause.

## Timing and explicit cache evidence

The unchanged final **training** harness reports return code 0, 3,630 boundaries, wall time 2,597.199031607015 s, uncapped fifth boundary and startup allowance 807.6938910679892 s, and charged-through-process-exit 1,789.505140539026 s (`within_budget: true`). Checkpoint SHA-256 is `f22d8287c6fb29a7d97344d6c3b3d63492bb7d4b58fc4fbf91039e483bf6e20e`, 453,048,594 bytes. These are retrieved training evidence, not independent final experiment acceptance.

No explicit persistent-cache hit/miss, execution-cache-key, or NEFF read/write/load/save message matched in the complete retrieved 56,192-byte training log or either 65,536-byte captured compiler-log prefix. Those prefixes originally came from logs of 3,569,926 and 715,045 bytes; unretained later messages are unknown. No actual NEFF read/write identity is recorded. Thus matching cache roots did not prevent the observed positive principal compilation, but the cache decision / cause remains unproved. The earlier note at `../20260926-clock36-live-input-evidence/ANALYSIS.md` retains the actual flags and source-path evidence without treating either as a sole explanation.

## Reproduction and limits

`python3 -I -B research/v2/analysis/20260926-clock36-training-closed-join/audit_training_phase.py` and the same command with `-O` both passed 64 explicit stdlib checks and emitted byte-identical JSON. No Python `assert` is used for gating. The existing bounded join tool is **unchanged**: on this training-only bundle it deliberately returns exit 3 / `UNKNOWN`, with sole reason `FINAL_SUCCESSFUL_FULL_BUNDLE_UNPROVEN`. The diagnostic authenticates completion of training separately; it does not bypass that final gate. Existing tiny malformed/hash/parsing tests remain unchanged.

The next smallest local action is rerunning the unchanged final join when the completed evaluation/outer receipts are archived. Establishing actual cache reuse would still require explicit cache decision / NEFF identity evidence; none is fabricated here. Quality comparison belongs to the Supervisor after the separate fresh evaluation is retrieved.
