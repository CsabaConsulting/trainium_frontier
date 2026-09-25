# Raw table rows outside the compiled GPT, 2026-09-25

Implemented behind default-OFF `--ngram-gather-outside-compile`, on shared
timer-fixed parent SHA-256
`6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.
The isolated implementation starts at canonical commit
`795ff5d4ad5a30a7cc8a2c669621c1499e81f8d1`.
Candidate `workflow/proposals/research-v2/train.py` SHA-256:
`9bf9568d1b6fe861e4fca0826229e4644527fcd4a7296be982449a18617b85aa`.
`train.patch` is the complete zero-context production diff against that parent;
apply it with `git apply --unidiff-zero` only after checking the parent hash.

## Boundary and invariants

The ordinary eager training caller computes the original on-device int32
hashes, gathers `weight[rows].to(COMPUTE_DTYPE)`, and passes a fresh fixed-order
tuple into the existing compiled GPT. Table projection remains inside GPT at
the original layer. No table Parameter, detached leaf, custom backward, row
cache, sparse update, CPU hash or model wrapper is introduced. Both normal
training and compile-only use this caller. `model` and `orig_model` keep the
same optimizer/debug/checkpoint roles and attributes. OFF uses the old model
call directly. ON supports `--no-compile` for the same CPU arithmetic and rejects
Engram or absent tables. OFF preserves both configurations.

The flag is training execution only and adds no checkpoint toggle or tensor.
Ordinary `GPT.forward(idx)` and the existing eval hook continue to hash/gather
from the original weights. The source-scope check proves AST identity for all
code outside the declared GPT helpers/forward and the main flag/call wiring,
including timing, schedules, optimizer, reductions, data, eval and SDK paths.
No protected file changed.

## Implementer self-checks

These are development checks, not an independent Verifier receipt or launch
authorization. Torch is `2.13.0+cpu`, using the existing interpreter
`/home/poojith-devan/venv/bin/python`; no package installation, host access or
external provider was used.

- `cpu_checks.json`: original parent, candidate OFF and ON match bit for bit
  for logits, losses, every dense parameter gradient, parameters and optimizer
  state through three steps with two microbatches. Cases cover zero-init and
  explicitly nonzero BF16 tables, narrow projected and full-width tables,
  actual distinct hash collisions for every bigram/trigram key, repeated
  microbatches, untouched rows, original parameter IDs/order/state keys and
  one RMS update per table per step. The model-only checkpoint reload passes
  ON/OFF, repeat, batch decomposition, future-token causality, state and RNG
  invariants. Geometry is four layers, width 128, vocab 64, T16, multiplier 4.
- The same file records a separate CPU AOTAutograd counting-backend probe:
  one forward capture and one backward capture, stable over three calls.
  Inputs and intermediates contain no full table-shaped tensor; core graphs
  contain no advanced-index or scatter table operation. Raw row inputs have
  nonzero gradients that propagate back into the original table Parameters.
  Token/unigram embedding operations remain. This is CPU frontend evidence,
  not Neuron graph count, layout, compile time or speed evidence.
- `tiny-g2-parent.json`, `tiny-g2-off.json`, `tiny-g2-on.json`: all four CPU/gloo
  ranks complete the real main/adapter/optimizer loop for three steps, with
  finite losses, rank-identical parameters, changed tables, and checkpoint
  reload. All use A2 layers 1,-1, width 128, depth 6, multiplier 1, dimension 16,
  pack32/T1024 and a synthetic forward microbatch prefix 1x128. The default
  OFF flag is omitted, testing its actual default. `flag_off_identity.json`
  and `flag_on_identity.json` show identical losses and all 74 parameter
  digests; `rms_trace_identity.json` shows identical traces on every rank,
  three RMS calls and twelve table checks per rank.

G2 appends `--no-compile` and shrinks idx/targets inside `GPT.forward`.
`preflight_rows.py` is a lane-local synthetic adapter that also slices each
raw tuple element with **exactly** `row[: ns.micro_rows, : ns.micro_tokens].contiguous()`.
It keeps the gather and slice autograd-connected; the original hashes at a
causal prefix equal hashes computed after the same token-prefix slice.
`g2_adapter.patch` records its four guarded substitutions and each G2 receipt
records adapter/canonical-harness hashes and the disclosed shim. The canonical
harness is untouched. The existing PyArrow import stub rejects data reads;
the old Torch 2.6 Gloo shim is not loaded. Gloo AVG is native in this runtime.
Do not deploy this test adapter or stub with the candidate.

The first draft of the graph test omitted the selected `COMPILE_SDPA_DIRECT`
setting, so fullgraph capture rejected the existing disabled `nki_attention`
fallback. Correcting the test configuration to the selected profile passed;
no production workaround was added. This was not a native experiment.

## Reproduce and next gates

Run `check_boundary.py --base <exact-parent.py> --out <receipt.json>` with
the interpreter above. The test verifies the parent hash before execution.
For G2, use `preflight_rows.py` with the exact argv/settings saved in each
receipt, `--stub-prepare --steps 3 --nproc 4 --micro-rows 1 --micro-tokens 128
--trace-rmsprop --keep`.
Set OMP/MKL/OpenBLAS thread counts to 1. Compare receipts with unchanged
`workflow/tools/preflight_dryrun.py --compare`. The graph test uses small
synthetic tensors; no local full-size A2 allocation was attempted.

Independent review and exact-size A2 G2 remain required. The native Operator
must then establish the actual boundary and numerical behavior, distinct graph
inventory, cache evidence, stable whole-step cost, and measured cold startup.
The old four-rank steady-state reshard hypothesis was already falsified; this
patch tests cold compilation complexity only. A reduction is plausible but
unmeasured. Eager gather/backward may itself compile or cost more; tuple
materialization/lifetime may increase memory. A2's raw rows occupy 32 MiB at
pack32/T1024/dim128, while dense gradients, reduction and RMS costs persist.

The mechanism target remains at least 10% less matched cold startup and at most
3% stable step regression. The planning target is 870 s; official startup cap
remains 900 s, with excess fully charged. A valid matched cold 1800 s run plus
fresh public-20M/context-1024 evaluation is needed for quality claims. Warm
research results cannot establish cold eligibility. No performance, BPB,
resharding, submission or independent-PASS claim is made by this patch.
