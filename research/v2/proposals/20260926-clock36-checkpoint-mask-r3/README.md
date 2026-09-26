# Disabled mask-aware checkpoint-only retry

Proposal `d8683fff`; helper frozen separately at `b7650cfca279c89fac7693f972539db17cf837b1`. This is an Implementer packet, not independent approval. Original r2 qualification remains **FAILED**. Actual corrected checkpoint qualification is unexecuted; cold/full/native/submission remain HOLD and require separate release.

The only scientific-checking repair is the new helper. It validates the exact root `mask_full` and `mask_short` buffers against the restored loader sequence length and restored/saved `SHORT_WINDOW_FRAC`: causal visibility `j<=i && j>i-window`, FP32, shape1×1×L×L, exact positive-zero/negative-infinity bytes. It rejects NaN, positive infinity, wrong positions and signed-zero changes. All parameters (including a parameter named mask_full), nested masks, other buffers and outputs remain finite-only. Raw hashes of every parameter and nonpersistent buffer, repeat/input checks and synthetic prefix tolerance1e-2 remain. Scratch mask comparison is64 rows at a time; no full mask oracle is materialized beyond the model's own mask.

All four ordinary loaders remain unpatched. The previously existing reference/default/OFF BF16 CPU comparison is retained and labeled separately; it is not used to claim unpatched-loader success or ON/OFF equivalence. Source bytes, prepare, model weights and evaluator are unchanged.

## Immutable retained inputs

`INPUTS.json` pins the16 original request/source/checkpoint-identity/checkpoint files across all four arms under the r2 root. Checkpoint paths/hashes are additionally compiled into the frozen helper. The four trained checkpoints are read only, not copied, regenerated or overwritten. No G2 rerun is included. The driver rehashes all16 inputs before and after the child, and pins prepare plus copied helper/guard/dispatch. Private new tmp and disabled bytecode writes prevent incidental old-tree bytecode writes. These protections do not claim a filesystem sandbox against arbitrary malicious source; exact trusted scientific bytes remain mandatory.

## Disabled one-child mechanics

New root `/home/ubuntu/trainium_clock36_recursive_checkpoint_20260926_trn22_r3`, label `fs-20260926-clock36-recursive-checkpoint-trn22-r3`. Only the separately assigned sole Operator may stage and invoke the existing shared dispatch after source/helper and packet review. One helper child is bounded by timeout600s+60s TERM grace, one CPU thread,64GiB available RAM and10.5GiB free disk floors. No installation, data/eval/training dispatch, existing cache deletion or second automatic attempt occurs.

The original common trn22 exclusive flock is held through the whole synchronous helper lifetime, with its FD passed into timeout. The unchanged shared guard authenticates the inherited flock and obtains BOTH fresh ps/neuron-ls checks before work, immediately before spawn and after child completion. Driver additionally authenticates exact hostname and original lock device66305/inode832589, checks STOP and preserves refusal/error logs. There is one Popen site; no queue or new orchestration. The ancestor is r2's same guarded CPU-checkpoint phase, now isolated without G2 phases; the driver is not claimed byte-identical because it adds explicit retained-input and original-inode checks.

All outcomes require independent Operator retrieval and actual observed PID/starttime/PPID/PGID/SID/descendant/compiler/device/original-lock postflight. The timeout leader's exit or this driver's post guard alone does not prove descendant quiescence. Missing/ambiguous/alive means HOLD and no next job. Raw stdout/stderr, exit, launch/environment, pre/post input identities and terminal records are retained under the new root; old receipts remain untouched.

## Preparation evidence

Helper:45 bounded case executions normal and `-O` (three repeated1×1 cases), maximum64 tensor elements, Torch2.13CPU and one thread. The original legitimate-mask failure is reproduced, not relabeled. Packet:42 stdlib checks normal and `-O`, including original-lock replacement, immutable-input and malformed-result negatives. The first packet fixture forgot to mock the named lock's local lstat in one negative; its two raw failures are retained and only the fixture was corrected. No full model, source import, trained checkpoint, SDK or host action occurred locally. These are own tests, not independent verification.

Review inputs are the full helper diff, neutral spec and raw receipts. Do not give the fresh reviewer Implementer test code or this narrative. Raw original failed92-file bundle remains `research/v2/operator/20260926-trn22-clock36-recursive-cpu-r2/completed/results-bundle.json`, SHA256d2298e9cafa63d98e9aa54c8b324e42a5ac345f20f4144b811112900ba2b2a08.

The literal full-context `.diff` artifact has single-space empty context lines by patch syntax; git whitespace checks report those context lines only. Runtime Python/JSON have no such exception.
