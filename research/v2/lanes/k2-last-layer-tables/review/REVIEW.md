# Independent review — FP32 last-layer hashed tables

Verdict: **PASS_FOR_SHORT_NATIVE_CHECK**.

This is one independent review of the supplied patch, with tiny CPU checks. It
permits progression through the existing short native diagnostic gate only. It
is not launch authorization, a native correctness pass, timing qualification,
quality evidence, adoption approval, or submission approval. No candidate edit,
native operation, SSH/AWS operation, training run, or second review was performed.

## Scope and input integrity

The only scientific inputs were `SPEC.md`, `parent.diff`, and `cpu-result.json`
in this folder. Their SHA256 hashes were checked against the assigned values:

| Input | SHA256 |
| --- | --- |
| SPEC.md | b5547916655fe0b5643cf51069e3e4ecd6f0c648f715ea356ad832c2761e5a3e |
| parent.diff | 7b5bff632a616b84ae94f3d9fe17b1b5fc88802485bfdab1728300b9d8e3afea |
| cpu-result.json | 4d50830392a7735298e08438542736a191f01ca0d4cc795a08f0c7432bbf11a1 |

The candidate SHA reported in the assignment and supplied receipt is
`b4728d17d082b44018290d2f465b5956f99ecbebe61da8b81508365c7dc53190`.
I could not recompute it because the full candidate file was deliberately not
provided. This verdict attaches to the reviewed diff; the existing Operator
must bind its exact full candidate to that SHA before any allowed native check.

Additional context disclosed: the role assignment, user-provided repository
contract, general tool instructions, local UTC time, and the existing runtime
executable path `/home/poojith-devan/venv/bin/python` supplied by the supervisor
after system Python lacked Torch. No repository, implementer worktree, test
source, implementation notes beyond the assigned SPEC, or task history was
read. No external documentation was consulted. All created files are here.

## Independently executed evidence

Command: `/home/poojith-devan/venv/bin/python independent_check.py`.
Final execution exited 0 using Torch `2.13.0+cpu` and NumPy `2.5.1`.
The detailed output is `independent-result.json`.

* Extracted the actual new `_p2_hashes` source from the diff and executed it
  with Torch. A separately written scalar Python signed-int32 oracle matched
  exactly for three batch rows at lengths 1, 2, 3, 4, 17, and 97, plus signed
  integer boundary inputs. Same-shape prefix and batch independence passed.
* Extracted the actual new consumer source and changed only its two geometry
  assignments through the AST: V/T/C = 16/8/4 and partition/features/groups =
  4/2/4. Ran its actual loops, resets, address expressions, selection, and
  stores through a small independently written NumPy interpreter. Registered
  the actual new forward/setup/backward functions using Torch's custom-op
  factory. Compared against ordinary `F.embedding(FP32 master).to(BF16)`.
* Output bytes and FP32 gradient bytes matched ordinary Torch after each of
  two microsteps for historical BF16 cancellation, both FP32 ordering
  witnesses, positive/negative BF16 subnormals, and normal cancellation.
  The `[1, 2^-8, -1]` collision produced `2^-8` per microstep; after two identical
  microsteps the relevant gradient was `2^-7`, as the amended reference requires.
* Interleaved duplicate IDs, inactive lanes, unused rows, repeated calls,
  disjoint feature owners, and nonzero rows in multiple runtime groups were
  covered. Every output element was stored exactly once per call in the
  interpreter, including all-unused groups. No giant table was allocated.
* Extracted the actual new RMS update method and compared three updates with
  an independently written explicit multiply/add second-moment recurrence.
  Parameters and state stayed FP32, steps incremented correctly, and observed
  maximum absolute errors were zero. The RMS assertion tolerance was explicitly
  `atol=1e-7, rtol=1e-6`; this check does not claim general bit identity of two
  differently expressed floating-point recurrences.

Test-source SHA256:
`7956b5f6e648c163461985c7786882289da19f03d4f7fcecf8e1167a5cb2ad4f`.
Independent-receipt SHA256:
`3bde62971fa3c8d25a874050bf323a01d4ce731ebae6933f71be9fd689958b41`.

## Source review findings

References below are line numbers in `parent.diff`, not candidate-file lines.

**OFF identity:** new modules, initialization, lookups, and optimizer group are
flag-guarded; the configuration default and environment fallback remain OFF
(150, 158–164, 217–233, 241–251, 260–273, 344). Passing `ngrams=None` adds no
tensor arithmetic. An empty table exclusion set preserves main parameter order.
The RMS ownership branch is not taken with tables OFF. The supplied receipt
reports three byte-identical OFF steps against the parent. I did not rerun a
full model, so that model-level result remains supplied evidence.

**FP32 arithmetic and routing:** both forward routes gather from the master
before conversion (108, 270). The custom backward rejects a non-BF16 incoming
tangent, produces dense FP32 output, and returns only the weight gradient
(112–122). Each occurrence is widened and accumulated sequentially in FP32.
The table group has zero momentum/decay, beta2=.999, bias correction, FP32 state,
and the specified width-scaled LR expression (251, 282–301). Tables receive
AVG reduction before their joint clip, separately from other parameters
(316–326, 381–392). The plan excludes RMS tables from packed ownership while
each rank receives the complete table group and dispatches one RMS update
(400–425). These branches are coherent. Real collective behavior, scheduler
handling of `no_lr_decay`, and complete gate membership in the unchanged
optimizer grouping were not independently executed or fully exposed by the diff.

**Model and checkpoint consistency:** one config flag controls the same tables,
projections, and layer-7 injection in ordinary and registered routes. Bigram
and trigram gates read normalized V channels 32:64 and 64:96, with unigram,
bigram, then trigram injection before the Q-dtype conversion (178–203).
Hashing uses only current/prior tokens with 8192 padding; native hashing takes
the K2 snapshot and creates separate metadata per table (262–268). The diff
adds no alternate evaluator or model-only evaluation surrogate. Existing model
checkpoint code is unchanged and config has a default-false field. Strict ON
and old-parent checkpoint roundtrips are supplied CPU evidence, not a new
independent checkpoint test. Hybrid optimizer resume intentionally raises.

**NKI geometry, lifetime, and liveness:** 4096 row groups × 128 rows covers
524288 rows. Two owners × 64 features covers 128 channels without overlap.
Accumulator zeroing precedes every group's occurrence loop; token/value
scratch resets precede skipped DMA lanes; inactive lanes retain their previous
accumulator. Every group has a final dense store, even when its maximum is zero
(18–92). The source uses runtime loops rather than 4096 Python-expanded bodies
and has bounded named scratch. Saved tensors keep each call's metadata live
through backward (114, 119). These properties passed the tiny interpretation,
but compiler lifetime/allocation and actual producer ownership are not proven.

**Precision of the RNG claim:** constructor `fork_rng` scopes and late projection
initialization support preserving common parameter initialization, as the
supplied receipt reports. The projection `uniform_` calls at diff line 231 are
outside a `fork_rng` scope, so global RNG state after `init_weights()` is not
shown to be identical between ON and OFF. Do not read SPEC's “RNG-neutral” as a
whole-program RNG-state guarantee. This does not defeat OFF identity or block
a diagnostic native check; controlled training must retain the existing
seed/data-stream comparison requirements.

## Limits that the native gate must resolve

The interpreter is neither NKI simulation nor native compilation. It assumes
the intended indirect row-address semantics and BF16-to-FP32 DMA conversion.
Its metadata producer is independently written tiny Torch code; the unchanged
K2 producer source was unavailable. Neither these assumptions nor native
stable sort/searchsorted support were measured here.

Runtime-group slices, register-driven nested loops, dynamic indirect DMA,
mixed input/output DMA dtypes, `nl.where` allocation/liveness, shared output
ownership, hardware subnormal/rounding behavior, graph integration, and
compilation duration remain native uncertainties. They are not observed
compiler/runtime failures. A native failure remains a failed gate and must not
be reclassified as a promising result.

The two tables add 512 MiB each for total weights, dense gradients, and RMS
state per rank: 1.5 GiB before temporary updates, metadata, and activations.
Checkpoint growth is 512 MiB. Complete-step measurements must include both
metadata producers, both table backwards, FP32 reduction/RMS, and checkpoint
costs. No speed, memory-fit, BPB, or 30-minute-budget claim follows from this review.

The supplied optional cross-shape prefix check failed its stated tolerance at
one logit, maximum absolute difference `7.62939453125e-6`. I preserve that as a
measured diagnostic failure. Shape-dependent arithmetic is a plausible
explanation, not a demonstrated cause. Supplied fixed-shape prefix/batch tests
passed by bytes; my hash-level fixed-shape check also passed. None of these is
native model causality evidence, and the cross-shape difference is not erased
or converted into an exactness pass.

The existing proposal, authority separation, machine preflight, unique label,
artifact retrieval, and native causality/correctness gates remain intact.
No unconditional full-run permission is granted by this verdict.
