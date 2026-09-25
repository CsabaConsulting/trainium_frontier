# K2 symmetric trn21 packet: PASS, mechanics only

Reviewed proposal-first `dfdcfdd12eff3791f75ee6a600074955a88104d2` and frozen packet `710ea6772d5525d97d990ecdf7f835f7f65d8380`, path `research/v2/proposals/20260926-k2-symmetric-native-trn21/`. This reviewer did not author that packet. No host, SDK/native execution, tensors, source edit or provider call occurred.

**Disposition: PASS for the disabled mechanical binding only.** This is neither the separate fresh input-only source review nor permission to stage/launch. Exact-source compilation/numerical correctness remains unproved. The old d0f0/r3 compiler failure remains parked and unchanged; this packet binds a distinct source, not a rearm of that route.

## Independently checked identities and scope

Own `audit.py` passes **127 explicit checks under normal and optimized Python** with identical final output. `FINAL-normal.json` and `FINAL-optimized.json` retain results; the earlier 105-check passes are also preserved. Final fixtures add22 assertions that each negative affects only rank3, modelling separately serialized rank files rather than shared in-memory objects.

The entire driver, rank entry and bound-command text equal parent `84d9f94f6b0341efcb7ddcd8ccaf2944481dc1e3` after only project/host, label/root, consumer and derived rank-entry hash substitution. All original packet files remain byte-identical. Validator bytes are unchanged; all manifest, driver, support and source Git-object pins authenticate. Exact source `9ce2e6a037212aaa76bd85b9b4ab7d311a481757` resolves to consumer2c8295; checker89189024 and reference14572563 equal the prior source's bytes. No oracle, tolerance, timing count, case set or compiler option was silently changed.

| Bound artifact | SHA256 |
| --- | --- |
| host_driver.py | ae4d1a328c868fdbfffe33d084e1525684821e1d1577c30e01afc19b982706c9 |
| NATIVE_BINDING.disabled.json | 926c49136160f8056776ed4678aa31d00714682c114afe0204aa0790d961eae6 |
| MANIFEST.sha256.json | b85d9c91907d368a7884dbeee3c4f437e7b0f96081b57b005ff7f689a7462132 |
| consumer_nki.py | 2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae |
| check_native.py | 89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850 |
| reference.py | 14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f |
| validate_receipts.py | c551f7933d410d27eda6755d197c1e83152c72ad0f7285ebf8869a3e4d514b40 |

## Own completion and refusal fixtures

The independent fixture explicitly builds four sets of20 cases (10 names times BF16/FP32),150 exact comparisons per rank (18 nonempty cases times8, two empty cases times3), and four groups of3 positive component timing samples. The unchanged validator accepts that complete synthetic schema only as `TINY_DIAGNOSTIC_COMPLETE_PENDING_INDEPENDENT_ARTIFACT_REVIEW`, with source adoption/full authorization/accepted speedup false.

Own22 isolated rank3 negatives refuse: missing rank; REFUSE status; wrong kernel, environment, argv or label; nonzero wrapper exit; missing or STARTED case; missing backward; nonfinite gradient; signed-zero-style differing bytes despite zero absolute error; false exactness; tiny nonzero error; relaxed tolerance; wrong dtype; invalid tensor hash; zero/NaN/fewer timings; claimed speedup; infinite elapsed time. No tensors are instantiated. The byte-difference case exercises validator contradiction handling, not an actual native signed-zero computation.

The Implementer's separate85-check binding suite and4 mechanical test groups also pass normal/-O; full outputs are retained in `supplied-test-replay.json`. They are supplemental, not substituted for the own reference fixtures.

Reproduce:

```text
python3 -I -B research/v2/verification/20260926-k2-symmetric-binding-independent/audit.py
python3 -I -B -O research/v2/verification/20260926-k2-symmetric-binding-independent/audit.py
```

## Execution and cleanup boundaries

One synchronous `timeout --foreground --signal=TERM --kill-after=60s 600s` child wraps four-rank torchrun/LNC2. Fixed V256/C128/T256, seed42, optlevel1/BF16 compiler flags, one CPU thread settings, >=10.5GiB disk floor and six absent/empty per-label SDK cache roots are unchanged. Compiler URL equals DIR. The 600+60 ceiling covers the child, not preflight/receipt processing or an official1800-second quality run. There is no new RAM or aggregate output quota claim.

The existing `/home/ubuntu/.cache/trainium_freshstart/device0.lock` remains the only lock, passed by authenticated inherited FD to the child. Unchanged guard `adafb54a...` requires the exact outer flock parent and a live exclusive inode, then fresh `ps` AND `neuron-ls`; busy/error/unknown refuses. The historical **device66305/inode596823** must separately be authenticated by the sole Operator before dispatch. The driver does not hardcode that historical inode; the packet says so explicitly. Hostname is independently checked at runtime.

Rank entry records whitelisted environment/source/argv before imports, preserves checker exceptions/SystemExit and records wrapper status when finally can run. The unchanged checker destroys its process group in nested finally after entering its case block, including receipt serialization failures. Pre-case initialization errors and externally killed workers can lack full receipts or cleanup execution; these cannot pass the all-four-rank validator. Child nonzero produces `DIAGNOSTIC_CHILD_FAILED`, and exit zero still must pass the full validator. No automatic next job is available.

**Driver exit, released parent lock and name-filter emptiness are not universal descendant cleanup.** The existing driver does not itself guarantee reaping every possible compiler/worker descendant. After *every* exit, the mandatory Operator artifact `postflight-explicit-pids.json` must prove all actually observed dispatch/flock/driver/timeout/torchrun/worker/compiler identities and PPID/PGID/SID descendants absent, with unfiltered process evidence, fresh `ps`/`neuron-ls`, and the original lock free. Missing/ambiguous/live evidence parks the route and forbids the next launch. This external obligation is part of the reviewed mechanics, not already-satisfied evidence or a new framework.

## Remaining prerequisites and interpretation

First the current full75b9 plus attached fresh20M must be terminal, retrieved and explicitly quiescent. Separately assigned P decoder27768550 must then be terminal with bounded artifacts retrieved and explicit process/device quiescence, including if P fails. The running genuinely fresh source review must conclude within its own scope. Root must independently select/release this packet to the sole Operator; all fresh host/STOP/disk/hash/cache/lock checks still apply. No hardware readiness follows from this receipt.

Three component repeats for unique/all_same in each dtype are an explicit preservation of the inherited binding, despite the source proposal's illustrative zero-timing command. They exclude routing and transfer and cannot establish whole-step speed or accepted speedup. Even successful native artifacts would establish only this tiny synthetic consumer's exactness within the prescribed cases, not production routing, full model autograd, BPB, adoption or submission authority.
