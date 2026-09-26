# Independent actual K2 result analysis

Conclusion: the frozen run shows a **required ordinary-control failure, not an observed K2 candidate correctness failure**. The gate remains **FAILED**. A control/compiler subnormal-handling explanation is consistent with the evidence, but the precise mechanism (flushing, lowering, or backward-path behavior) is not established by host-level receipts.

## Authenticated evidence

- Frozen raw commit: `30630c8441249f7db894a80c19200f67003bd18d`.
- Bundle SHA256: `a210263a3f0cd67f4896cb76c55ae67198331053220432800639b4483b1d2563`. All 31 embedded files authenticate: 7,682,695 decoded bytes. Non-scientific bundled prose was not interpreted.
- Receipt scientific pins match the previously frozen checker, adapter, kernel, and reference. Runtime is native Torch `2.11.0+cpu`, NKI `0.5.0+28631259367.ga768afa6`; this analysis itself uses only local stdlib Python, not Torch/native execution.
- All four ranks actually completed all 20 cases, the nonzero FP32 sentinel, and both changed-tangent replays. Each rank has 130 exact direct/repeated/registered-eager/registered-compiled comparisons, with unchanged input records. The sole required-route mismatch per rank is `compiled_ordinary_embedding`, `fp32_subnormal`, FP32 gradient. Its forward output is exact.
- In that case the upstream roundtrip and CPU dense oracle are exact. All four candidate routes return the exact expected gradient. The compiled ordinary control returns finite all-positive-zero bytes: nonzero count 0, SHA `fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471`, versus expected 1408 nonzeros, SHA `402743e587f15111ef1d3b927df591942db4bd96038726e4edc134a622e44085`.
- Old observations are not erased: each rank retains 8 original-Function-compiled mismatches and 13 exact observations; original-Function-eager is exact in all 21 nonempty observations. These remain separate from candidate results.
- Terminal evidence says `DIAGNOSTIC_CHILD_FAILED`, return code 1, `complete: false`, and no automatic next job.

## Own CPU exactness reference

`analyze.py` independently reconstructs the specified fixture using signed integer units of the least positive binary32 subnormal, `2^-149`: token row is `token % 11`, and sign is negative exactly when `(token + channel) % 3 == 0`. It does not call the supplied fixture, dense accumulator, or byte encoder for this oracle. Integer accumulation and explicit IEEE-754 bit encoding yield 683 entries of 7 units, 384 of 8 units, 341 of 9 units, and 31,360 zeros. Thus there are exactly 1408 nonzeros and maximum zero-output error `9 * 2^-149 = 1.2611686178923354e-44`. Both the gradient SHA above and upstream SHA `8e79a64b9db2dd89e7bac15c068285a6aafb12db7dcd6f208bb683e3d08cf78d` match all four raw receipts.

This proves the nonzero expectation is meaningful: it is not a mistaken reference count or a zero fixture. The candidate preserving these bytes while the ordinary compiled route loses them rules out universal device-transfer flushing in this run. It does not identify which operation inside the ordinary compiled backward loses them. No tolerance adjustment is justified.

## Postflight

Explicit postflight SHA256: `5ae851a4abd4417aed5ba85c371e95fb551958a32f3559b42b3d884c1ccb1f89`, timestamp `2026-09-26T01:50:07.178839+00:00`. Its 11 observed dispatcher/timeout/harness/worker/compiler PIDs are absent; related process rows are empty; device processes are empty; the original lock (device 66305, inode 832589) is free. This is authenticated historical cleanup evidence, not a fresh host preflight or new launch permission.

## Minimal distinct next diagnostic recommendation

Prepare a separately labeled, independently reviewed **ordinary-embedding subnormal localization** proposal, not a K2 retry and not a retrospective alteration of this gate. Run a fresh control-only process under the same pinned native runtime, shape, rank binding, and lifecycle. Import no K2 adapter/custom registration/kernel. Compare ordinary eager and ordinary compiled embedding backward on the exact frozen repeated-ID FP32-subnormal fixture, with the same nonzero normal warmup and reused callable as the frozen checker. Read back upstream bytes immediately before backward; retain complete forward/gradient/input/error records.

Include only the necessary paired witnesses: the original `2^-149` tangent and the same signed pattern scaled exactly to `2^-126`, plus a negated subnormal replay through the same callable. Use an independently encoded strict-byte CPU oracle for every witness. These distinguish subnormal-specific loss from generic backward failure and stale tangent capture without loosening candidate checks. If the ordinary compiled-only subnormal loss reproduces without any K2 code while eager/normal witnesses are exact, it is independent control-platform evidence; if it does not, investigate process/order interaction before any new candidate gate. Retain every mismatch/error; do not make the diagnostic's reproduction criterion a candidate qualification criterion.

Any later candidate qualification needs a new explicit proposal and independent review while preserving this failed run, all candidate strict-byte requirements, and all old failures. No automatic retry, validator change, tolerance change, source adoption, performance/BPB inference, or launch authority follows from this analysis.

## Execution and failures

Executed once, exit 0, with complete raw output in `raw-analysis.log`:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/poojith-devan/venv/bin/python /tmp/trainium-k2-subnormal-review-25yvH0/analyze.py
```

No failed analysis attempt occurred. Only an isolated review-output directory was written; candidate, validator, gate/state, host, and native environment were unchanged. Root handles canonical integration.
