# Architecture initial-cache refresh: numeric and disabled binding review

**PASS for GPT technical CPU verification and the disabled native packet only.** The new ALL IN cross-family adversarial gate remains **UNMET** after the reported Opus provider refusal. This report is not cross-family evidence, a waiver, a native assignment or launch authorization. No host/API calls, candidate edits or tensors were used here.

## Exact-size CPU results

Independently audited operator archive `abe8c373` at `research/v2/operator/20260926-trn22-arch-initial-cache-cpu-r1/results/`. All 28 files in the retrieval manifest match SHA-256 and size, including all eight separate rank receipts and both torchrun logs. Source is exact `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`, G2 remains `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`, and plan remains `fb622153f463f3fdb67456380a01c61ff804fa4d0a90c2a1e008ee913a3f634c`. Prepared scope was independently reviewed in `b48cdf4a`; source/tiny gradient/update identity was reviewed in `74889f48`.

| Arm | Raw G2 SHA-256 | Result |
| --- | --- | --- |
| Explicit OFF | `47b7b33f942670f88594e9167960f02f1929b6af5b7045fe7f08a0714d23633d` | PASS, exit 0, 21.2 seconds |
| Explicit ON | `52cd0aba1efbec8ab23c756d0132316a3fc69ef3278aa529ac09965675914101` | PASS, exit 0, 21.2 seconds |

The original reference remains authenticated `c2fee697d02322198d61951363861fb31201bd7b68d4b91f159dbd44001392b0` from the original 93ae architecture. Each new arm has all 12 rank-step losses, all 65 final parameter digests and the ownership plan bit-identical to that reference. Every rank reports 65/65 parameters changed, no cross-rank mismatches and three finite losses. Rank-0 losses are `[9.009672164916992, 9.021651268005371, 9.053120613098145]`. Raw rank records exactly match their aggregate entries; checkpoint strict-load and deterministic finite logits checks pass.

I independently ran the unchanged `preflight_dryrun.py --compare` for both pairs; both returned exit 0, identical true, 65 parameters compared, and match the archived comparison results. Actual launches match the prepared requests, differ only by explicit cache-refresh flag, use the same runtime/prepare/shims as reference, and are serial: OFF terminal 18:48:07 UTC precedes ON launch 18:48:44 UTC. Final retrieval reports no active processes and an idle device.

Both retained, not-downloaded 453,048,594-byte checkpoints have SHA-256 `2e3d8e442e8c7ddfbb29308787fcd6f0ccba9f8c3f3bc051331f7a2b2a01fd3b`. This is **OFF-versus-ON checkpoint file identity**; the audit does not invent a full original-reference checkpoint hash comparison absent its hash.

Scope: full **8x1024 parameter size**, four CPU ranks, three steps, physical synthetic **2x256** micro-batches (2048 tokens/update), CPU/no-compile, BF16 compute shim, Neuron CE bypass and fake clock. It is not production-token/full-chip gradient proof. The standard harness does not capture all intermediate gradients/cache connectivity; the separately labeled tiny review does. No tiny observer is repurposed and no additional full-size observer is requested.

## Disabled native packet

Reviewed preparation `28ec1ae0a6ef710fa48fe0c7de5e16d0f5d0cfc5`, installed-reader evidence `49ae7632e7dec2486cd2fa6a1a8c5ef467071ca2`, and the exact binding bytes present at `abe8c373`:

- Binding SHA-256: `1dda3adcb209787d2250b3c076598f2690ca6c77355f91e1250e359c898ffaa4`.
- Driver SHA-256: `650735eb662268cbc51f3d27d1724fce7f069bcb8010d9e51229feafbe8e4452`.
- Unique label: `fs-20260926-arch93ae-initial-cache-trn22-r1-full`.

No concrete binding bug found. AST comparison confirms the existing manual driver's process/phase/timeout logic is unchanged: only ROOT/source/label/cache/FLAGS literals and the added `NEURON_COMPILE_CACHE_URL` mapping differ. The exact ON CPU argv is the planned native argv, retaining seed 42, full four ranks/LNC2, 1800 seconds and the original architecture flags, without Recursive or FP8. The old source text in the driver docstring is not the binding; the executable TRAIN_SHA is exact 3e79.

The AST-only environment evaluation matches the binding exactly. Compiler URL and DIR aliases target the same fresh compiler root; training and evaluation use separate roots; evaluation inherits neither training metrics destination nor training-budget variables. Twelve private cache roots must be newly created and empty at actual preflight. Existing data/tokenizer cache remains unchanged. Compiler flags remain exactly optlevel 1/BF16 matmul; no `--cache_dir` override or cache deletion is authorized. The installed SDK excerpts support explicit NEFF/local/HLO roots and rank-specific Inductor descendants; those excerpt-file whole-source hashes are Operator observations, not independently recomputed full installed-file hashes.

The shared nonblocking lock remains in the outer command; it does not use Python optimization that would remove the inherited driver's assertions. Fresh machine/STOP/breaker/source/import/disk/idle checks and owned queue pause remain Operator duties. The driver by itself is not a replacement for that preflight. No automatic old fallback follows; completion returns to the Supervisor's current priorities. The binding explicitly leaves stage/launch unauthorized and independent acceptance null; this review does not flip them.

Coldness, cache effectiveness, compile speed and budget are **not** proved by planned paths. Actual fresh-root/effective-process-environment/artifact evidence is still required to claim cold scope. Source estimates and process wall remain separate; this reused driver has no exact fifth-step observer, so do not invent a precise startup timestamp. Through-exit compliance must use available bounds honestly. The original 93ae result remains a DISCARD; unequal cache provenance prevents a causal compile speedup claim merely by contrasting startup totals. The proposed sole human-selected rerun is not adoption or submission readiness.

## Independent checks

`audit.py` passed **92 checks in normal and optimized Python**. It uses explicit exceptions, no model imports, and executes only the extracted environment-building AST with fake path objects. The two comparator reruns are stdlib-only. `AUDIT_RESULT.json` preserves exact hashes/results/bounds.

```text
python3 -B research/v2/verification/20260926-architecture-cache-numeric-review/audit.py
python3 -B -O research/v2/verification/20260926-architecture-cache-numeric-review/audit.py
```

Overall: CPU/narrow binding technical PASS; cross-family review and native/cold/performance evidence remain pending. Only the human/Supervisor may resolve the provider-blocked review-policy choice, and only a separately assigned Operator may launch.
