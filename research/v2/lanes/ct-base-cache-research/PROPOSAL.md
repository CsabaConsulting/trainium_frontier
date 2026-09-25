# CT base cache refresh: select the existing exact port, not a new implementation

2026-09-25 20:12 UTC / 26 September 01:42 IST. **Research proposal only;
no production change, tensor run, host action or launch authority.** Recommend
selecting the already-prepared default-OFF Recursive port for its remaining
base-specific review/CPU/cold gates. Its source-supported mechanism is shared
with architecture 3e79; the base's actual cold improvement remains unknown.

## Exact identity and four-line scope

| Vehicle | Commit | Source SHA256 |
|---|---|---|
| Adopted Recursive base | `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b` | `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3` |
| Existing, unadopted base port | `b635c8b0cf2a10e1bcdd983e86192672ed6ad56f` | `bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522` |
| Cold-tested architecture port | `b4d83b5db66971f3c348837f04f53cba65aa6d33` | `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c` |

All source paths are `workflow/proposals/research-v2/train.py`. The port was
prepared earlier but is not the canonical research source. Do not cherry-pick
its large implementation/evidence commit blindly into the base or combine it
with architecture, batch, FP8, K2 or optimizer changes. Reuse its pinned bytes
only if the Supervisor selects it. K2/e4d93 remains frozen and REFUSE.

The existing patch at `research/v2/lanes/initial-cache-refresh/train.patch`
adds exactly this parser line and guarded three-line block:

```python
    parser.add_argument('--initial-cache-refresh', action=argparse.BooleanOptionalAction, default=False, help='Refresh differentiable weight casts inside the timed first step before its first forward; default OFF.')
        if step == 0 and args.initial_cache_refresh:
            for m in cached_linears:
                m.refresh_weight_cache(COMPUTE_DTYPE)
```

The block is after existing gradient zeroing, after the first-step `t0`, and
before the first micro-forward. No detached/no-grad cast, optimizer warmup,
external prewarming, timer reset, compiler option or per-step refresh change.
The two placements are separate; the snippet is not one contiguous function.
`audit_source.py` independently re-read all four source objects (including
original93ae) and confirmed stripping these four lines restores each original
byte-for-byte. Default and explicit OFF execute no added tensor work.

## Same concrete specialization trigger; causal proof still incomplete

These line numbers refer to exact6ef7:

- `USE_WEIGHT_CACHE=True` at427; `Linear._cached_weight=None` at1461.
  `Linear.forward` selects the cache at1468–1469, otherwise casts its FP32
  master inside forward at1471. Refresh at1464–1465 is differentiable `.to`.
- The genuine installed MLP also selects the cache in `make_forward` at
  4482–4486. `_install_submission_training` at4870 installs the native-forward,
  custom-backward MLP with default `weight_mode=0`; the packed-weight refresh
  override at4536 is therefore inactive. This is not a stale alternative MLP.
- `torch.compile(model, dynamic=False)` is created at3862 before execution;
  `cached_linears` collects the actual model's Linear modules at3916. Existing
  post-update refresh at4038–4039 makes subsequent forwards see BF16 tensors.
  The first-step timer is3966, first forward3977, backward3980.
- The base has40 cached module keys:24 attention projections,12 MLP
  projections, `transformer.h.{1,3,5}.attn.ve_gate`, and `lm_head`. Full names
  are retained from the earlier real-loop tiny receipts in `source_audit.json`.

The `Linear`, `make_forward`, `_submission_mlp_install`,
`_install_submission_training` and `OwnedOptimizer` ASTs are identical between
6ef7 and original93ae. This establishes the shared **None-to-Tensor branch
transition**. Priming makes the first model trace encounter the cache-present
path already used in steady state, moving its differentiable cast outside the
model forward but inside the counted first step. FP32 masters remain the
gradient destinations. The target is the avoidable first model forward/backward
specialization, not Muon's shape keys, lazy optimizer states, evaluation graphs
or every compiler subgraph. Existing replacements of cache tensors after each
update remain unchanged.

Native evidence supports testing this mechanism, not declaring it proven:

| Retrieved observation | Meaning and limit |
|---|---|
| 6ef7 P rank0 steps0/1:448.562645/357.518251s | Two heavy startup steps; wrapper capture targeted step20. This is not a count of startup graph keys or proof of their guards. |
| 6ef7 true-cold trn22 fifth boundary:855.071994s | Same Recursive profile, seed42, existing optlevel1; step0/1 logged470.25/355.94s. Source capped estimate852.8s is a different clock. |
| Architecture3e79 true-cold fifth boundary:844.353010s | Qualified cold result; step1 is0.51s after step0's812.91s. Original93ae had startup lower bound1544.745s. This historical contrast is not a base effect estimate or perfectly matched cold A/B. |

The original architecture archive has two distinct model forward/backward
specializations on every rank (`initial-cache-refresh/graph-inventory.json`).
The retrieved P/base-cold packet paths inspected here contain no startup
`graph_stats` files or guard trace. Thus the actual base model graph keys and
the guard responsible remain **unobserved**, despite a strong source-supported
hypothesis. Zero SDK `compile_us` must never be relabeled a cache hit.

## CPU identity requirements and existing evidence

The historical base-port self-check already records four-rank, three-step
6x128 real-loop runs with synthetic2x128 inputs and a fixed clock: base,
default, explicit OFF and ON. The new stdlib audit rechecked source hashes,
all per-rank losses, per-step raw/reduced gradients, parameters, owned optimizer
state and checkpoint payload/file equality in those receipts. Each rank has
62 live Parameters; default/OFF refresh120 times versus160 for ON. This is
receipt reinspection, not a new tensor test or independent verifier PASS.

Base-specific acceptance still needs the normal independent gate: omitted/OFF
byte reconstruction and execution identity; ON first loss, **every** Parameter
gradient including missing status, clipping/update and three-step optimizer
identity under fixed inputs/clock; each cache equal to the BF16 master cast
and differentiably connected; unchanged RNG, parameter identity, checkpoint
schema/bytes and load-for-eval behavior. Include the actual custom MLP path,
not only ordinary Linear. The architecture review does not automatically
approve this different full source/profile. Exact117M/world4 CPU G2 remains
separately required on a memory-qualified host, never the constrained laptop.
Earlier tiny shims (BF16 CPU, stub prepare/imports, synthetic inputs, fake
clock, no Neuron CE/compile) are not native or official-timing evidence.

## Selected experiment, if approved

Reuse the reviewed cold five-step observer, real unchanged prepare.py and
the base cold argv in
`research/v2/operator/20260926-ct-base-startup-only/EXPERIMENT_PROPOSAL.json`;
the scientific delta is **only** source bcbd plus `--initial-cache-refresh`.
Keep6x1024/117453580 Parameters, context1024/pack32/world4/LNC2, batch131072,
RecursiveON, dense LR0.015, seed42, budget1800 and the already-set optlevel1
compiler flags. Supervisor/Operator must bind a new unique label and reviewed
current common-lock/idle-check route; the old packet's launcher is not new
authority. No queue entry or label allocation is made by this research note.

Measure uncapped launch-to-fifth-boundary startup, source capped estimate and
charged excess separately. **Base CT target<=600s**; preserve the existing
controller rules, including an offset before a full run if base startup>870s.
Retrieve all-rank graph IDs/nodes and actual compiler arguments if available;
test whether the second model specialization disappears. Retain failures or
null results. Three postcompile steps are not a robust steady-speed benchmark.

Expected steady arithmetic is unchanged, with one extra40-cache initial
refresh only; real steady tokens/s and cold/peak-memory cost are unmeasured.
The first step may now hold the whole cache set earlier. Do not assume the
architecture's relative startup reduction transfers numerically to6x.
The current base's855.07s startup is already below the900s cap (also below it
after the source's30s margin), so reducing it to600 does **not** directly buy
more charged optimizer time. It provides startup headroom/shorter wall time,
not an automatic BPB gain. A steady slowdown cannot be justified by claiming
to recover startup excess that this control did not incur.

The unchanged wall-clock schedule at3248–3257 and4010–4022 sees different
elapsed times if startup improves, especially before the fifth-boundary reset
at4046–4053. Consequently fixed-clock CPU identity does not promise native
first50-step weight identity or final quality. Do not alter LR/WD/time inputs
to hide that effect. Native correctness, any eventual full1800/public20M
comparison, unchanged replication/selection and submission gates remain.

Night-mode decision: preserve/reuse the existing exact port as a proposal,
not duplicate implementation. Reason: the new native observation strengthens
an already-scoped mechanism; base benefit is not yet measured. Undo: decline
selection; no production, host or queue state was changed.
