# Confirmed36 clean vehicle review specification

Proposal: research/v2/proposals/20260926-confirmed-clock36-clean-116188/EXPERIMENT_PROPOSAL.json.
This is a NEW port with a revised, explicit contract. Review from the full source
diffs, this specification, original refusal and raw test receipts. Fresh reviewer
must execute their own exactness reference; implementer tests are not independent
verification. Default REFUSE until the applicable evidence is complete.

Original vehicle SHA256: 116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d.
Confirmed research commit: 7419927c7417b5666a984ec9c2b849feadabaaa9;
workflow/proposals/research-v2/train.py SHA256:
36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb.
Candidate SHA256: e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43.
LONG command file SHA256: b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd.
Both patches contain full context and permit reconstruction, not excerpt review.

Required model: eight 1024-wide layers; attention/convolution at0,2,4,6,7;
four scalar skips3→4,2→5,1→6,0→7; VE only7; exact initialization,
parameter ordering and optimizer grouping. Preserve original enabled kernels,
FP32 masters/BF16 compute policy, matrixLR.015, batch131072, seed42,1800s.
Initial differentiable cast refresh is within the counted first step; later
refreshes remain. Predictor clears excluded startup samples at boundary5.
Bake exact confirmed36 startup_adjusted_schedule_elapsed helper and route both
LR and WD progress consumers through it; do not introduce a new clock read or
alter real startup cap, stop timing, checkpoint charging or record_step_boundary.

Required loader is the proven vehicle's ordinary GPT/weights_only=True loader.
No eval row adapter, equivalent renamed adapter, evaluator change, scoring
change, external data or launch-command change is permitted. The research loader
uses per-row calls for B>1; the candidate executes the supplied batch directly.
This is an acknowledged route discrepancy, not a byte-identical effective batch
implementation. Mathematical/numerical consistency must be checked for B1/B2,
odd/even batches and native-eligible convolution contexts. Do not infer native
equivalence from small CPU equality. If this contract is noncompliant or ambiguous,
park and request human review; do not add the prohibited adapter.

OLD_REFUSAL.md is copied verbatim from the original independent review. Its
REFUSE applies to the old3f5035 draft and old effective-route-preservation
contract, remains valid historical evidence, and is neither erased nor claimed
fixed. The new request explicitly forbids its suggested adapter correction.

Required later gates: fresh independent own-reference CPU test, full113M CPU
identity on an approved large-memory machine, native B1/B2 consistency/causality,
child-specific true cold five (uncapped fifth<=900s, target870), separately bound
cold full1800 and fresh unchanged public20M/context1024, then Claude compliance
audit via Poojith and human-only upload. No source or CPU result authorizes launch.
Research mean0.9883066128526815 is not this candidate's measured quality.

The supplied self-check reuses a historical review test as IMPLEMENTER evidence.
It uses synthetic D128/V256 eight-layer models, one CPU thread, BF16 emulation,
test-only prepare/pyarrow stubs, eval rotary/mask floor32, no compile backend and
no collectives. It exercises three updates, tiny B1/B2 logits/causal suffixes,
isolated256-token MLP backward and detached-cache/wrong-skip negative controls.
Actual clock constants5/900 and prescribed step-start samples replace the old
test's placeholder constants; 420 scalar helper cases cover steps0..6. No native
convolution-context claim follows from these tests. Full-size and native gates
remain pending. Preserve failed attempts and exact errors when reviewing receipts.

Attempt3 adds CPU context256 B1/B2/B3 ordinary trained/loaded/reference comparisons,
independent-row equality and changed-suffix causality. It records actual block
input shapes, including research row splitting versus candidate supplied batch.
This exercises CPU fallback at native-eligible sequence length; native lane-grid
parity and numerical consequences remain unmeasured. The 64MiB memory watcher
covers the reused core probe only; the additional tests retain the same tiny
width/vocabulary and run sequentially without any full-model allocation.
