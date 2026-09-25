# Exact 93ae initial-cache-refresh port

Implementer self-check only; independent review, full-size CPU G2 and native
gates are pending. No host access, launch, queue, ledger or submission change.

Base commit: `1e21c3d797eed764f108c41546b270065ad5650c`.
Base train SHA-256: `93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5`.
Candidate train SHA-256: `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`.
Branch: `codex/20260925-architecture-initial-cache-refresh`.

Only four executable lines were added to `workflow/proposals/research-v2/train.py`:
the default-False BooleanOptionalAction `--initial-cache-refresh`, then an
initial refresh of the existing cached Linear list when `step == 0`. It runs
in normal grad mode after the existing first-step `t0`, before its first
forward. Existing post-update refresh, model, optimizer, architecture,
Recursive-OFF state, evaluator, compiler flags and accounting are unchanged.
Explicit OFF is `--no-initial-cache-refresh`.

This is the architecture source, NOT a stack with adopted Recursive6ef7.
Extract the complete file from this branch/commit; applying only this patch
to a different base would not reproduce this source hash.

Shared test/proposal/evidence packet is on
`codex/20260925-recursive-initial-cache-refresh` in
`research/v2/lanes/initial-cache-refresh/`, with the architecture-specific
receipts under `architecture/`. It compares exact93ae, candidate omitted
flag, explicit OFF and ON: 4 CPU ranks, 3 steps, width128/depth8, actual2x128
synthetic inputs and real custom MLP backward. All65 live Parameter gradients,
missing-gradient statuses, updates, optimizer state and serialized checkpoint
bytes match. Every one of38 initial ON caches has the exact BF16 master cast
and a verified autograd path to its FP32 master. Five source/receipt tests
pass normally and with `-O`. This is NOT exact8x1024 or Neuron proof.

The full preset `--unet-8x1024` is not used on the laptop because it forces
width1024. Tiny tests instead use existing `--depth 8 --n-embd 128 --unet-skip
--partial-attn-8 --attention-layers 0,2,4,6,7`; the native proposal retains the
exact original preset and production shape.

Native hypothesis: remove the extra initial model forward/backward
specialization. No graph-count or compile-time reduction is claimed yet.
Faster startup can change the unchanged elapsed-based schedule; fixed-clock
CPU identity does not promise native first50-step identity. Initial refresh
never runs outside official timing, and no dummy update/prewarming is added.
