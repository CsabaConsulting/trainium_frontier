# L2: 8x1024 U-Net / partial-attention architecture

Update: the subsequent narrow startup-predictor port is documented in
`startup-predictor-reset/IMPLEMENTATION.md`. Current source SHA-256 is
`93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5`.
The model receipts and original architecture handoff below remain bound to 52e234e9.

Implementation candidate, **not host-launch approval**. The Supervisor assigned this
local Codex L2 lane under Poojith's Final Push directive and exact Lane 2 prompt.
Only `workflow/proposals/research-v2/train.py` changes model behavior. The evaluator,
data/tokenizer, training loop, budget clock, kernels and submission file are unchanged.

Base SHA-256: `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
Candidate SHA-256: `52e234e90f33fc2ccfcc22edb7afe101be25089020a58c352afda4470ac5a7c1`.
The branch starts at the prior architecture draft `326a9396`; review the combined
model diff from its parent `b18fb358`, whose research-v2 file is the exact c5c base.
The old draft's per-channel skip checkpoints are intentionally not compatible.

## Interface and behavior

`--unet-8x1024` defaults OFF. ON forces depth 8, width 1024, U-Net skips and partial
attention, regardless of separate depth/width flags. `--attention-layers` selects
distinct zero-based indices; the default is `0,2,4,6,7`. Layers 1,3,5 are MLP-only.
The existing draft's separate `--unet-skip` and `--partial-attn-8` remain default-OFF
ablation controls. The latter requires depth 8; the former requires even depth >=2.
Invalid lists/depths fail before parameter allocation. Partial attention rejects
parallel blocks, whose coalesced attention/MLP implementation needs both branches.

Four learned FP32 scalar parameters (one-element tensors) are initialized to zero.
The output of layer i feeds the input of layer 7-i: 0->7, 1->6, 2->5, 3->4. The
addition occurs before the destination block's normal residual/x0 mixing. Encoder
outputs retain their gradients. Scalars use the existing AdamW scalar group, its
unchanged SCALAR_LR/betas and zero weight decay. No new learning-rate policy is added.

MLP-only layers allocate no attention, local-convolution, QK-shift, value-embedding
or n-gram parameters. VE retains the base alternating parity rule among active
attention layers. Therefore the default profile has VE only at layer 7. Explicit
enabled n-gram selections targeting removed attention sites fail with an explanatory
error. Use `--no-ngram-ve` for the standalone architecture arm; `--ngram-ve-layers=-1`
is compatible with its default pattern. An A2 `1,-1` stack needs an explicitly
different attention list including layer 1 and a separate declared stack proposal.
An empty table-layer list means all eligible *active* VE sites. Checkpoint toggles
persist the actual attention list and skip settings; the existing strict reload path
reconstructs the same model and rejects missing parameter keys.

## Attribution and assumptions

Read on 2026-09-25: [Ritwika Kancharla, Model and Kernel Co-Design under a 30-Minute
Training Budget](https://ritwikareddykancharla.github.io/projects/trainium-of-thought.html).
Its configuration describes roughly width 1024, eight layers, attention in a subset,
and U-Net-style skips. It reports jointly developed kernel and architecture changes;
it does not isolate an architectural gain transferable to our current c5c model.
No source code or weights were copied.

The exact width, attention indices, scalar zero initialization and i->n-1-i pairing
come from Poojith's Lane 2 task, not an unreported detail of the public write-up.
The following are implementation assumptions where that source is silent:

- Plain additive skips with no projection, normalization, extra gate nonlinearity or
  scaling; native residual dtype; scalar optimization uses the existing scalar group.
- MLP-only means retaining the base MLP and residual/x0 mixing, while removing the
  entire attention branch and its attention-specific convolution and embeddings.
- Keep the base parity rule for VE on surviving attention layers. Do not move removed
  tables to another layer or allocate trainable parameters with no forward use.
- Keep all existing batch, optimizer, schedule, data, precision and kernel choices.
  The source's other recipe choices are separate lanes, not part of this patch.

Zero-init skips preserve step-zero logits relative to the *same eight-layer partial
model without skips*. They do not make the changed eight-layer architecture identical
to the six-layer base. Flag-OFF does preserve the base exactly in the recorded test.

## Evidence and limits

All CPU work was sequential with OMP/MKL/OpenBLAS thread limits of 1, using the
pre-existing `/home/poojith-devan/venv/bin/python` (Python 3.14.4, torch 2.13.0+cpu).
No packages were installed. The first ON attempt failed at import with
`ModuleNotFoundError: No module named 'pyarrow'`; `g2-on-small.json` preserves it.
An import-only test stub now rejects every attempted Parquet read. It is for synthetic
tests only and must not be copied into a training/evaluation environment.

- `g2-on-small-r2.json`: unmodified `preflight_dryrun.py`, four Gloo ranks, exact
  candidate bytes, 8x128, three steps, 1x16 synthetic training micro-batches; PASS.
  Every rank agrees on all parameters; rank-0 checkpoint strict reload is deterministic.
- `g2-base-small.json` and `g2-off-small.json`: exact c5c and candidate, same 6x128
  argv and three steps; both PASS. `lockstep-small.json` records identical losses,
  all 62 final parameter digests and optimizer-ownership plan.
- `self-check-small.json`: preset parsing forces 8x1024 without allocating it;
  four scalar gates/optimizer coverage; step-zero skip identity; three finite steps
  with gradients and learned skips; strict reload logit parity; missing-key rejection;
  causal prefixes, deterministic and batch-independent frozen eval; odd-depth/list
  rejection; tiny-table allocation and configurable-pattern checks all PASS.

These are **Implementer self-checks only** and use reduced model dimensions. Exact
8x1024 four-rank G2 and its matching flag-OFF/base lockstep remain required on an
authorized machine with enough memory, followed by independent diff/numeric review
before a host run. No full-size table was allocated on the laptop. There has been
no host access, native compilation, NKI execution, benchmark, real training or BPB.
The model unit checks set the existing eval minimum sequence length to 32 in the
test process only; the G2 receipts use the unchanged eval helper without that override.

Reproduce from this worktree (the G2 receipts include their complete argv):

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=research/v2/l2-unet8x1024/test_support:workflow/tools/stub_prepare \
/home/poojith-devan/venv/bin/python research/v2/l2-unet8x1024/check_small.py \
  --train-py workflow/proposals/research-v2/train.py \
  --out research/v2/l2-unet8x1024/self-check-small.json
```

For a *later independently cleared* full-chip architecture-only job, the proposed
model argv is the established c5c base plus `--unet-8x1024`:

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015 --seed 42 --max-train-seconds 1800 --unet-8x1024
```

An Operator must add a unique approved label/output directory and follow fresh host
checks. This is an architecture package comparison: extra MLP capacity, less attention
and VE, plus skips change together by explicit task instruction. It is not an isolated
skip ablation. A later FP8 or table combination is a separate stack experiment.

## Expected effect and risks

No gain is claimed. Eight MLPs replace six while five attention blocks replace six;
the throughput balance depends on compiled whole-step behavior. Fewer VE sites may
hurt quality; additional skip dependencies may increase activation lifetime and
compiler pressure. The source's FP8/kernel improvements are absent here. Strict
30-minute public-20M/context-1024 evaluation must decide whether the package helps.
The historical standalone zero-init skip arm lost (1.008391 vs 1.005734 on the older
stack); see the 2026-09-23 decision-log entry. The new human directive explicitly
reopens a different combined architecture. That prior loss remains a negative prior.
