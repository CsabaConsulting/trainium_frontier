# K1 residual audit: remote-only parameter writeback, conditional on P

Researcher-only proposal. No candidate implemented, tensor run, host call,
threshold change or launch. Exact6ef7 source remains unchanged. K2 and its
full-size packet are untouched. Current measured cost/speedup is **unknown**.

## Exact source and remaining work

Source commit a4aa4351789115a2d0aa947d9ca00ff689f9bd0b,
`workflow/proposals/research-v2/train.py`, SHA
`6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.
Line references below are to that exact file, not another source or architecture.
The existing K1 audit/plan is `research/v2/lanes/k1-muon-batching/`.

`setup_optimizer:2213` groups exact shape/orientation. `_muon_step:2873` keeps
persistent state/stacks;2887/2895 stage gradients/weights,2916 calls batched
compiled math once per local group,2920 copies results into live masters.
`_muon_math:2351` through2373 is independent per matrix;2356/2360 iterate five
NS steps, not matrices. Literal same-shape batching is already present.

| Muon group | Real total | Rank0 real→executed | Rank1 | Rank2 | Rank3 |
|---|---:|---:|---:|---:|---:|
| 8x32 VE gate | 3 | 3→3 | 0→0 | 0→0 | 0→0 |
| 1024x1024 attention | 24 | 6→6 | 5→5 | 5→5 | 8→8 |
| 1024x4096 MLP output | 6 | 0→0 | 2→2 | 2→2 | 2→2 |
| 4096x1024 MLP input | 6 | 1→2 | 2→2 | 2→2 | 1→2 |

There are39 real matrices,41 batched rows including two zero tall dummies, and
three nonempty local Muon groups per rank. Six distinct mathematical compile
keys exist across the four processes: gate batch3; square batches5/6/8; wide
batch2; tall batch2. The twelve local cache entries do not prove twelve actual
native compiles: cross-process cache behavior is unmeasured here.

The dummies are required by the current native exactness evidence, not casual
padding. `OwnedOptimizer:4603` creates them; they never enter the real parameter
exchange. Historical wave61 singleton tall groups changed second moments and
weights; wave62 batch2 padding repaired strict native equality. Removing them
is not this proposal. Their two96.64G-operation tall rows represent about193.27G
matrix-multiply algebra operations over five NS iterations; this is shape math,
not observed device FLOPs, elapsed time, or a9% whole-step opportunity.

`make_plan:4559`/4572 uses old per-matrix profile constants, then4578 greedily
assigns whole parameters. Actual owner counts are:

| Rank | Parameter tensors | FP32 elements | Old planner cost, ms | Cost plus linear dummy proxy, ms |
|---|---:|---:|---:|---:|
| 0 | 17 | 27,265,796 | 12.947406 | 14.444491 |
| 1 | 16 | 30,414,851 | 12.939731 | 12.939731 |
| 2 | 17 | 30,412,805 | 12.948754 | 12.948754 |
| 3 | 12 | 29,360,128 | 12.947728 | 14.444813 |

The planner's0.009023ms spread is only0.0697% of its mean. It does not price the
dummy rows or batch geometry. Adding the same old tall-per-matrix weight for a
dummy suggests ranks0/3 could take longer, but **none of these columns is a
current timing**. Parameter bytes, compute FLOPs and critical-rank time are not
interchangeable. Reassigning whole tall pairs could reduce aggregate work yet
leave the slowest rank unchanged or worsen ownership balance; no such second
candidate is selected here.

One all-gather is already used (`OwnedOptimizer:4644`), not a gather per matrix
or shape. It reconstructs486,639,616bytes/rank for469,814,320 live bytes:
16,825,296 padding bytes,3.5813% of live payload. The selected proposal neither
changes that gather nor claims to recover its padding/own-chunk traffic.

## One distinct reversible candidate

Proposed flag name `--owned-remote-writeback`, default OFF, **not implemented**.
At construction, retain paired lists of live Parameters and received views only
where `entry['owner'] != self.rank`. At line4645, ON copies that remote-only
list; OFF retains the exact full-list `_foreach_copy_` statement. Everything
before it—including owner plan, dummy policy, optimizer math/state, packing and
the same single all-gather—stays unchanged. No persistent-storage alias redesign.

Source-supported redundancy: at4637–4641 the owner already updates its live
FP32 parameters in place. At4643 it packs those exact values into send views;
the gather returns that same own chunk among all chunks. Line4645 then copies
the owner's identical FP32 values back into those same parameters, as well as
the necessary remote updates. Only the own copy is omitted. On skipped AdamW,
the owner already holds the unchanged correct value. None/zero-gradient Muon
semantics are handled before this point and must not change.

Per rank removable payload is109,063,184 /121,659,404 /121,651,220 /117,440,512
bytes. Across the chip this omits469,814,320bytes of copied values, or
939,628,640bytes of read+write traffic if both are counted. It is exactly25% of
the aggregate final full-master-copy volume. It saves **no collective**, no
optimizer arithmetic, no send packing, no dummy math, and does not imply25%
wall-clock savings. The foreach call still exists and may have fixed overhead.

Numerical hypothesis is exact values, but removing an in-place write also
changes version-counter activity and may affect backend scheduling, graph
guards, or weight-cache interactions. Therefore static argument is not approval.
Native proof must cover refreshed BF16 caches, every live master, state,
None/zero gradients, skipped AdamW, multi-step updates and checkpoint reload.
No optimizer checkpoint schema/ownership-plan change is expected or authorized.

## Cost bound and decision

Current expected milliseconds and whole-step delta are **unknown** pending P.
A counterfactual bound, holding other execution costs fixed, is that removing
these copies cannot save more than their current critical-path time. If the *entire* final writeback occupies
less than5% of whole step, this candidate cannot meet5% even if all of that
phase vanished. Under an additional linear-bandwidth assumption, removing about
one-quarter of its bytes would require writeback to occupy roughly one-fifth
of the step to reach5%; that assumption is not a measured backend model.

Prior evidence is discouraging for a large gain: at
`785c294e:research/fresh-start-optimizer-ownership-2026-09-18.md` wave60, gather
alone was about4.69ms and full pack/gather/copyback about6.40ms. Their roughly
1.71ms residual includes both pack and copy, on an older diagnostic profile.
It is not a current6ef7 bound and cannot be combined with a new timing as a
matched result. The historical5.06245% full-step ownership win is already in
the base. No new>=5% estimate is defensible from these old numbers.

Recommendation: **do not implement or spend a native slot before P shows a
sufficient relevant critical-path cost**. This is a concrete distinct reversible
candidate, not an assertion it is worthwhile. If P exposes only a small final
copy, record K1 literal batching as already done and prioritize another measured
bottleneck. Do not rebrand the existing mechanism or lower the5% requirement.

Cheapest falsifier: inspect P's actual6ef7 steady trace for final foreach-copy
device work, host gaps and rank wait, distinguished from send packing and gather.
If P cannot separate them, report unknown; don't infer copy time from aggregate
optimizer duration. Only if selected: independent proposal review, default-OFF
implementation, CPU reference/three-step OFF identity/exact-size G2, independent
adversarial review, native correctness and cold accounting, matched whole-step
>=5% gate, then any fixed1800/public20M run. Existing legality/locks untouched.

## Reproducibility / decision record

`audit_residual.py` executes only the exact existing make_plan AST against
metadata, authenticates the existing plan, and computes counts/byte volumes.
No Torch import, tensors or production edits. `counts.json` retains the full
rank inventory and null native timing/speedup fields. The normal and `-O`
outputs match. Narrow history searches for remote-only writeback identifiers
found no earlier implementation; this is not an exhaustive proof of novelty.

2026-09-26 01:26 IST /2026-09-25 19:56UTC: retain this as a profile-conditional
Researcher proposal only. Reason: redundancy is real,>=5% benefit unsupported.
Undo: discard this isolated proposal branch; no runtime behavior was changed.
