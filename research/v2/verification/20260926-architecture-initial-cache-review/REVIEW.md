# Independent architecture initial-cache-refresh review

**PASS — exact source and independently repeated tiny CPU checks only.**
No concrete source bug found. Exact8x1024 CPU and native compilation/startup/
quality remain pending. This supplies source review for an assignable CPU
packet; it does not launch or clear a native rerun, adopt the architecture, or
certify the separate Recursive port.

Architecture source commit `b4d83b5db66971f3c348837f04f53cba65aa6d33`, SHA256
`3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`.
Base is exact93ae at `1e21c3d797eed764f108c41546b270065ad5650c`, SHA256
`93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5`.
Shared test/receipt packet `b635c8b0cf2a10e1bcdd983e86192672ed6ad56f` is stored
on the separate Recursive branch; **do not stage that branch's train.py** for
the architecture job. Extract the complete3e79 architecture file.

## Source review

Exactly four executable lines are added: a default-False BooleanOptionalAction
and a guarded first-step loop over the existing `cached_linears`. Removing
the new argument and the complete guarded block reproduces exact93ae bytes.
Default/explicitOFF preserve all prior argument defaults and execution paths.
ON calls the existing refresh in normal grad mode after synchronization/t0 and
zero_grad, before the first micro-forward. There is no no_grad, detach, dummy
step or work moved outside accounting. It executes once at step0; existing
post-optimizer refreshes remain unchanged.

The cached tensor is the ordinary differentiable cast of the same FP32 master.
`Linear`, `OwnedOptimizer`, `lr_multiplier`, `save_checkpoint` and
`load_for_eval` are AST-identical to93ae; all other bytes outside the four
insertions are unchanged. Thus optimizer/ownership, loss normalization, LR/WD
formula, skip/attention topology, data, compiler settings, startup cap900,
launch margin30 and checkpoint reserve are unchanged. This is standalone
8x1024, not an architecture+Recursive or FP8 stack. The training-only flag need
not be saved/restored by frozen evaluation: the checkpoint model format and
strict load path are unchanged, and caches are not model-state keys.

Priming changes the first model call from an empty to populated differentiable
cache path. It plausibly removes one forward/backward specialization, but CPU
evidence cannot establish this on Neuron. It may only move work or leave another
specialization cause (e.g. optimizer state). Do not call zero compile_us a cache
hit, sum cross-rank compiler seconds as wall time, or promise native first50-step
identity: faster startup changes the elapsed values consumed by the unchanged
schedule. Graph/startup effect remains a hypothesis.

## Independent CPU evidence

All23 packet manifest entries rehashed successfully. The five architecture
source/receipt tests passed normally and with optimized Python. I then repeated
the four architecture variants sequentially in my isolated worktree:
exact93ae, candidate default, explicitOFF and ON, each four CPU ranks, three
real optimizer steps, depth8/width128, actual synthetic2x128 input per rank.
Each run exited0 (9.5/9.8/9.5/9.7seconds). Five post-replay tests passed.
No full-size laptop tensors, native imports/execution or host calls occurred.

Independent raw receipts are retained in `independent-tiny/`; all four match
the builder's corresponding losses and every per-rank/per-step gradient,
update/optimizer-state and checkpoint observation, not just the final norm.
All65 live Parameter gradient names/missing statuses before and after clipping,
all updates and serialized owned optimizer states are identical. First-step
missing-gradient lists are empty. All four arms serialize the identical
checkpoint SHA256
`01ce0f7b8a9ecd76db109fc2510af1efdb058d47ac388a54f319e705bb2617be`;
unchanged load_for_eval checks pass. Rank ownership plan SHA256 is
`83b06c658067902fe9e255779ce792e5117006563c932b16bc1f4bba6f6595d5`.

Every one of38 initial ON caches equals the BF16 cast of its FP32 master and
has a live autograd connection: grad(cache.sum(),master) returns ones while
leaving Parameter.grad unchanged. Default/OFF initial caches areNone. Later
caches/forwards match the reference; ON has one extra refresh per cached Linear.
Eight FusedMLPBackward nodes per loss graph (24 across three forwards) confirm
the custom MLP path is represented, not a count of instrumented native callbacks.

Scope remains the packet's disclosed CPU shims: fake clock, synthetic tokens,
stub prepare with two excluded steps, BF16 CPU/no compile, import-only PyArrow
stub and Neuron-only CE bypass. It proves fixed-clock tiny arithmetic, not
native compiler equivalence or production131072-token updates. Tiny architecture
flags use width128 components, not the width-forcing `--unet-8x1024` preset.

## Concrete next CPU binding, not native authorization

The supplied `g2_observe.py` SHA256
`a1b9108c3aa3af22cd819e6c0553e6c5c2bb7f3887bc0d3c1b8b4e6d856db03d`
wraps unchanged G2 (`6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`)
without replacing optimizer/gradient operations. Its root-relative staging
requirement is real: preserve support-tree layout or the existing audited G2
staging arrangement. Do not copy that file alone to an arbitrary directory.

On an idle suitable-memory host, bind original93ae and exact3e79 separately,
real unchanged prepare5807, the proposal's original production argv (including
`--unet-8x1024 --attention-layers 0,2,4,6,7`), three steps/four CPU ranks,
synthetic2x256,3600-second timeout, one OMP/MKL/OpenBLAS thread and unique output.
Only candidateON receives `--initial-cache-refresh`; original93ae has no such
argument. Check exact actual8x1024 config and compare all observed gradients,
updates/state, losses/ownership and checkpoint identity. Candidate default can
use the packet's additional default-control check; this review adds no further
CPU/native ceremony. All full-size checks remain pending, not inferred from
the tiny receipt. The next native job, if assigned after its gates, is the one
human-selected compile-fix rerun; a source PASS is not an adoption decision.

Reviewer replay helper SHA256
`d70254a75e37623798d520a5eb65806fea2171ca32d730ac95494d431740ca18`.
Summary SHA256 `97bb9fbbdc00af06d8f3a6478357079be6741be35de28ed6370319bcd9b02125`.
No candidate, support script, evaluator, queue or ledger was edited.
