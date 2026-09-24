# G0 ownership-stride root-cause analysis

Date: 2026-09-24  
Role: independent Analyst/Researcher (read-only source and evidence analysis)  
Verdict: **the live A1 ownership plan is correct; the accepted comparator froze the
dense plan and is stale for A1**

This report does not change the production candidate, gate implementation, proposal,
authorization, or evidence. It does not authorize G1, any further hardware action, or
full A1 training.

## Exact evidence boundary

- Authorized request-bearing head:
  `03bfeabf0f4293e30dc86eecf3aa9cce49e9735b`.
- Accepted gate implementation:
  `7528ca8dd9d9915097929217c2a2b0d9cfc17107`.
- Production source:
  `submissions/dense6x1024_public0992851/train.py`, SHA-256
  `9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae`.
- Immutable G0 evidence commit:
  `37aa3644e63e20a1427cfa69b44fd6de7e25cd81`.
- Independent G0 verifier receipt commit:
  `01d3cc04b3d55ed4d453fc1aff0e0eddad581682`.
- G0 label:
  `fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1`.

The G0 receipt correctly reported a protocol FAIL against the accepted package. This
analysis resolves which side of that failed comparison is wrong; it does not rewrite
the historical FAIL.

## Exact parameter accounting

The frozen dense plan has 62 owned parameters and no n-gram tables, n-gram projection
matrices, or n-gram-specific gates. Its owner totals and stride, repeatedly present in
the historical dense ownership evidence, are:

```text
elements_by_owner = [27,265,796, 30,414,851, 30,412,805, 29,360,128]
owned total       = 117,453,580
owner maximum     = 30,414,851
128-rounding      = ceil(30,414,851 / 128) * 128 = 30,414,976
```

The authorized A1 flags create a last-layer bigram table and a last-layer trigram
table with dimension 128 while the model width is 1024. Source construction and
`setup_optimizer()` therefore add the following regular FP32 Muon parameters, in
addition to the two BF16 RMSProp tables:

| A1-only owned parameter | Shape | Elements | Live group/position |
| --- | ---: | ---: | ---: |
| `transformer.h.5.attn.ve_gate_bi.weight` | `8 x 32` | 256 | `6/3` |
| `transformer.h.5.attn.ve_gate_tri.weight` | `8 x 32` | 256 | `6/4` |
| `ngram_proj.b5.weight` | `1024 x 128` | 131,072 | `7/0` |
| `ngram_proj.t5.weight` | `1024 x 128` | 131,072 | `7/1` |
| **Total** | | **262,656** | |

This is exact, not an approximation:

```text
117,453,580 + 262,656 = 117,716,236
```

The live rank-0 inventory at evidence commit `37aa3644` contains 66 owned entries
totalling exactly `117,716,236` elements. Its complete group accounting is:

| Group | Kind | Parameter count | Owned elements | Meaning |
| ---: | --- | ---: | ---: | --- |
| 0 | AdamW | 1 | 8,388,608 | token embedding |
| 1 | AdamW | 1 | 8,388,608 | language-model head |
| 2 | AdamW | 12 | 12 | residual/x0 scalars |
| 3 | AdamW | 3 | 25,165,824 | value embeddings |
| 4 | RMSProp | 2 | 0 | replicated n-gram tables, excluded from ownership |
| 5 | AdamW | 6 | 12,288 | local convolutions |
| 6 | Muon | 5 | 1,280 | three dense VE gates plus two A1 n-gram gates |
| 7 | Muon | 2 | 262,144 | A1 128-to-1024 projections |
| 8 | Muon | 24 | 25,165,824 | attention matrices |
| 9 | Muon | 6 | 25,165,824 | MLP projection matrices |
| 10 | Muon | 6 | 25,165,824 | MLP input matrices |

The two replicated RMSProp parameters are separately and exactly reported as
`ngram_embeds.b5.weight` and `ngram_embeds.t5.weight`, each BF16 shape
`[524288,128]`, `67,108,864` elements. Their combined `134,217,728` elements do not
appear in the owned total. Thus the mismatch is not caused by accidentally owning a
table.

## Allocator replay and stride

Production `make_plan()` sorts owned entries by decreasing estimated update cost,
then element count, then original index. It assigns each entry to the rank minimizing
`(accumulated_cost, accumulated_elements, rank)`. It finally rounds the largest owner
count up to a multiple of 128.

Replaying that rule from the immutable G0 entry list gives the live result exactly:

```text
elements_by_owner = [27,405,059, 30,540,035, 30,408,963, 29,362,179]
owned total       = 117,716,236
owner maximum     = 30,540,035
128-rounding      = ceil(30,540,035 / 128) * 128 = 30,540,160
total padded bytes = 30,540,160 * 4 ranks * 4 bytes = 488,642,560
```

Removing only the four A1-only owned tensors above, renumbering entries in their
source order, and replaying the same allocator reproduces the dense plan exactly:

```text
elements_by_owner = [27,265,796, 30,414,851, 30,412,805, 29,360,128]
estimated_ms      = [12.947406063447303, 12.939731226905751,
                     12.948754427971444, 12.947727783333331]
shard_elements    = 30,414,976
```

The added tensors have total estimated cost `0.1757 ms` under the planner. Because
allocation is greedy by accumulated cost, adding them changes later tie/load choices;
the per-owner deltas are consequently
`[+139,263, +125,184, -3,842, +2,051]`, rather than simply placing the four new
tensors on otherwise unchanged owners. The maximum remains rank 1, and both plans
have 125 padding elements, so the stride difference is exactly the rank-1 count
difference:

```text
30,540,160 - 30,414,976 = 125,184 elements
gather output delta = 4 * 125,184 = 500,736 elements
padded storage delta = 500,736 * 4 bytes = 2,002,944 bytes
```

## Source and adapter disposition

The live behavior follows the authorized source:

1. `GPT.__init__` constructs one bigram and one trigram table for A1 and, because
   `ngram_dim != n_embd`, one `Linear(128,1024,bias=False)` projection per table.
2. The layer-5 attention module constructs bigram and trigram VE gates, each
   `Linear(32,8,bias=False)` in stored `[8,32]` shape.
3. `setup_optimizer()` puts table parameters in the RMSProp group, projection weights
   in `matrix_params`, and the n-gram gate weights in the attention matrix scan.
4. `make_plan()` records RMSProp parameters only in
   `replicated_rmsprop_groups`; it creates exchange entries for AdamW/Muon parameters.
5. `OwnedOptimizer` verifies that the exact optimizer parameter set is partitioned
   into owned AdamW/Muon plus replicated RMSProp. Each update steps the full RMSProp
   group on every rank, updates local AdamW/Muon ownership shards, then gathers the
   owned parameters.

The immutable G0 evidence agrees: all four ranks report the same 66-entry live plan,
the two tables are excluded from owner entries, there are exactly five owner gathers
with FP32 input `[30540160]` and output `[122160640]`, RMS call counts advance once per
update, replicated table/RMS state agrees across ranks, persistent model digests agree,
and the verifier found no dynamic-plan overlap or omission.

There is therefore no evidence of an ownership implementation defect in G0. In
particular, forcing the exchange back to the dense stride would make the buffers too
small for the authorized A1 parameter set and would omit or overlap legitimate model
parameters.

## Comparator root cause

Accepted comparator commit `7528ca8d` introduced:

```text
# Exact 35c9 dense ownership plan. Replicated RMSProp tables are excluded from
# this plan, so A1 must retain the already-frozen dense exchange stride.
OWNERSHIP_SHARD_ELEMENTS = 30_414_976
```

The first sentence is accurate and the inference in the second is not. Replicating
the two table tensors excludes those tensors from ownership, but A1 also creates the
four ordinary FP32 Muon parameters listed above. Those four must remain owned and
exchanged. The package verification at `aeb85c2c` authenticated the comparator's
collective-shape checks, but it repeated the stale dense premise rather than deriving
the expected inventory for the actual A1 configuration.

The accepted comparator also validates generic entry shape/coverage and the RMS table
schema, but it does not bind the complete named A1 owned-parameter manifest or replay
the allocator from that independently frozen manifest. Merely replacing the numeric
constant would fix this observed rejection but would leave that broader authentication
gap.

## Required correction and tests

The cheapest scientifically valid resolution is a gate-package correction, not a
production-source change and not a hardware rerun:

1. An Implementer should replace the dense ownership premise with an independently
   derived, A1-specific frozen manifest: exact ordered group kinds and parameter names,
   shapes/elements/dtypes, the two replicated table schemas, all 66 owned entries,
   owner/offset assignments, owner counts/costs, stride `30,540,160`, and padded bytes
   `488,642,560`.
2. The comparator should authenticate each rank against that manifest before accepting
   the five FP32 gathers of `[30540160] -> [122160640]`. It must not trust a stride
   merely because the live report is internally self-consistent.
3. A fresh independent Verifier should reconstruct the manifest from the pinned source
   and A1 flags, review the package diff, and issue a new receipt. Required negative
   tests should reject:
   - the old dense stride/plan;
   - either missing or altered projection;
   - either missing or altered n-gram gate;
   - either RMS table entering ownership;
   - missing, duplicated, reordered, renamed, reshaped, re-owned, overlapping, or
     offset-drifted entries;
   - changed counts, costs, rounding, total padded bytes, dtype, world size, gather
     shape/count/order, or extra/missing gathers.
4. The corrected, independently verified comparator should be applied to the immutable
   G0 archive at `37aa3644`; a new evidence-verification receipt must preserve the
   original FAIL and explicitly supersede only blocker
   `B1_FROZEN_OWNER_EXCHANGE_CONTRACT_MISMATCH`.

## Advancement decision

- **Implementation versus comparator:** live implementation correct; frozen comparator
  stale for A1.
- **Current protocol state:** G0 remains FAIL under the accepted package until the
  correction and independent re-adjudication above are durable. This report alone is
  not a PASS.
- **G0 rerun:** **not required** if the corrected exact-manifest comparator can validate
  the complete immutable archive, as the existing archive already contains the live
  inventories, collectives, five updates, checkpoint seal, and clean postflight. Rerun
  only if the independently reconstructed A1 manifest disagrees with the archive or a
  required observable is absent.
- **G1:** remains locked. Do not prepare or authorize G1 until a corrected package has
  passed independent verification and G0 has a durable superseding PASS receipt under
  that exact package and source/config pin set.

