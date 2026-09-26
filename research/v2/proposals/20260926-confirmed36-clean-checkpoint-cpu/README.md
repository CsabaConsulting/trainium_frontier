# Disabled confirmed36 clean retained-checkpoint CPU helper

Completed Supervisor proposal0930f2ae is unchanged. This implementation is
preparation only in a separate worktree; no source, evaluator, harness, host,
queue, real data or previous evidence was changed. No full113M model or real
checkpoint was imported or allocated locally. Real pair/checkpoint hashes
remain PENDING. No launcher is included and this is not self-review.

## Exact scope

Two frozen sources: reference7419927c/SHA36a86274 and clean7b9c20c6/SHAe1e2f686.
Both retained checkpoints must come from the separately reviewed full113M G2
pair8cc56def. The helper authenticates exact source/request/tool/prepare pins,
all four G2 ranks and three finite losses, standard comparison, clean exits
and checkpoint identity. It independently compares all G2 loss/digest/plan
records, then all65 saved parameter tensors/113256724 elements by complete
bytes. Standard G2's sixteen-hex digest prefixes are provenance only: full
checkpoint tensor comparison is not truncated.

Both files use their actual CPU load_for_eval without any dtype/global/forward
override. The clean object must be exactly its module.GPT, not a subclass or
adapter. Loaded tensors must match saved tensors; frozen/eval/no-forward-hooks
conditions are checked. Only the original research reference has mask_full and
mask_short; its unchanged mask helper independently validates every byte from
the restored window/length formula. Exact clean GPT has only cos/sin buffers:
all clean buffers must be finite, with no negative-infinity exemption. The old
mask helper and its previous failure remain unchanged. A checkpoint selecting
the document/tokenizer loader route is refused before ordinary loader use.

Six fixed synthetic shapes are B1/B2/B3 × T256/T257, token IDs1..8191 from a
local seed7 generator. For each source/shape: original, repeat, each independent
row, and a per-row nondegenerate suffix reversal. Complete output bytes must
match repeat, rowwise and cross-source comparisons. The only nonzero tolerance
is the pre-existing synthetic causality formula from prepare.py663–684:
cut=max(2,T//2), reverse suffix[cut:], FP32 prefix[:cut-1] maxabs<=1e-2. Exact
prefix equality is also reported. No organizer data/causality_check is called.
Any numerical mismatch is REFUSED, with actual/expected shape,dtype,full SHA,
maxabs and relative RMS; no post-hoc tolerance. Signed zeros remain distinct.

All parameters and persistent/nonpersistent buffers plus input bytes and all
module train/eval modes are checked around every call. Output snapshots are
cloned immediately to defeat scratch-buffer aliasing. This is finite synthetic
evidence, not a proof about every input or arbitrary unregistered Python state.

## Pending input binding and execution

INPUTS.pending.json is deliberately non-executable and fails REAL_INPUTS_PENDING.
After the actual pair completes, a separate authority must freeze a new INPUTS
receipt and its exact SHA. Do not edit this pending template into a claimed result.

The future receipt uses status=FROZEN_REAL_INPUTS, the exact fixed private pair
root, exactly reference/candidate arms with checkpoint_path/checkpoint_sha256/
checkpoint_bytes, and a files map from absolute path to {sha256,bytes}. Exactly
these files are required: original prepare; staged standardG2 tool; pair terminal,
standard-compare JSON and exit; each arm's train.py, request, G2, terminal and
checkpoint-identity; and its exact independently identified tmp/g2-*/out/final.pt.
No checkpoint discovery/newest selection. All paths must be existing regular
files with no symlink component. Hashes/lengths are checked before and after,
including when model checks fail. Source/request/G2/prepare pins are immutable.
The caller must authenticate the INPUTS SHA; a merely self-created receipt is
not independent acceptance or launch permission.

Prospective invocation shape only, NOT a launch command/release:

    python -B compare_checkpoints.py --inputs <separately-frozen-absolute-INPUTS.json> --inputs-sha256 <reviewed-SHA>

The already reviewed environment must expose the exact original prepare module;
the helper does not inject a path or shim and checks its actual import origin.
An eventual separately reviewed Operator binding must retain original trn21
lock66305:596823, fresh ps/neuron-ls/fuser/STOP/breaker checks, >=64GiB available
RAM and >=10.5GiB disk plus footprint, timeout600s+60s and explicit observed
PID/descendant/compiler/device/lock postflight. No automatic retry or native step.

Full-host memory estimate, NOT measured:113M FP32 weights ~453MB, research masks
~537MB, six retained reference outputs ~101MB for FP32; models, payload and
temporary operations add overhead. Keep64GiB preflight. Full runtime and600s
sufficiency remain unmeasured; repeated all-state/mask scans are intentionally
not skipped. Laptop tests use only <=108 elements per tensor; Python3.14.4/
Torch2.13 CPU evidence does not establish target Torch2.11 or native behavior.

## Own tests and limitations

Own tiny tests cover mask/no-mask positives, odd/even batches and contexts,
cross-source equality, noncausal and aliased outputs, signed zero, mutable
parameter/buffer/input/mode, hooks, wrong mask values/dtype/shape, all nonfinite
classes, full-size-count rejection, source/checkpoint/path/manifest drift,
symlinks, actual tiny file mutation and malformed G2 negatives. Source/AST tests
authenticate the unmodified proposal, both sources, request pins, weights_only
loaders and verbatim reused mask functions. Raw outputs are in raw/.

Two test-fixture failures are preserved: the first future-leak negative used
a suffix whose last token did not change; its input was corrected. The first
static test assumed the reference called torch.load directly; the pinned source
actually delegates to its weights_only=True torch_load_weights wrapper. No
scientific source, helper oracle or tolerance was changed to hide either result.
Interim passing receipts are retained as interim, not final evidence.

Fullsize pair, real checkpoint identity/output, fresh independent review,
native loader/cold full/fresh20M and Claude via Poojith remain separate gates.
No quality score, source adoption or upload-readiness claim.
