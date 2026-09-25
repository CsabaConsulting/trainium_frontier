# Successor disposition: PASS, offline mechanics only

Frozen successor: `b529468f29d6a8f763cf772d9a607b3b6ad12286`. The original `8429c1a2` HOLD and its real defect remain unchanged in commit `0b49593d`, `REVIEW.md` and the two original raw receipts. This successor resolves that defect; it does not retrospectively pass the original.

The same independent `audit.py`, without edits, passes **311 checks normal and optimized**, including the exact late-detached-child reproducer. All six successor runs now report `REFUSED_OR_INCOMPLETE` with `UNEXPECTED_DECODER_DESCENDANT_AT_FINAL_CLEANUP`, retain the authenticated descendant identities and restore quiescence. The ordinary child still succeeds; combined-stream overflow still refuses and retains exactly the test cap. The 86-file synthetic layout/copy/identity and resource tests are unchanged. Raw evidence is in `repair-normal.json` and `repair-optimized.json`.

Own `repair_scope.py` passes **14 checks normal and optimized with identical output**. It proves exactly seven helper lines were added and all other helper bytes remain identical. Driver bytes differ only in the helper hash. Proposal/argv/inputs/limits are byte-identical. Binding changes are matching hashes plus repair metadata; the actual driver pins resolve to exact staged files. Original preparation/test evidence is retained. `repair-scope-results.json` contains exact support hashes.

The separately supplied 17-test suite also passes in both modes; its complete replay output is retained in `repair-supplied-test-replay.json`. No candidate edit, host/SDK access, actual trace decode, tensor work or provider request occurred during this independent review.

```text
python3 -I -B research/v2/verification/20260926-p-offline-decoder-independent/audit.py --commit b529468f29d6a8f763cf772d9a607b3b6ad12286 --expect-fixed
python3 -I -B -O research/v2/verification/20260926-p-offline-decoder-independent/audit.py --commit b529468f29d6a8f763cf772d9a607b3b6ad12286 --expect-fixed
python3 -I -B research/v2/verification/20260926-p-offline-decoder-independent/repair_scope.py
python3 -I -B -O research/v2/verification/20260926-p-offline-decoder-independent/repair_scope.py
```

| Successor artifact | SHA256 |
| --- | --- |
| host_driver.py | 36e456ab0bf777ece5426799ee7f1bf6e396be9bfb60923b94da58c7f1cde844 |
| decoder_support.py | 96aebc1e9a43843f57691e67053c75ecbe904f2cec6899688a9e3036c863efd9 |
| OFFLINE_BINDING.disabled.json | f5ddd27cfa7e43ee6238ea91f5f77b8495d5d148db00db4e3aa02235f618b299 |
| MANIFEST.sha256.json | c12bf98ba7baebe28bab72f09041e4e640d871f27f01fe941c01e65f8ddb2219 |

The accepted scope remains one fixed offline summary attempt against independently copied full 86-file/80-NEFF input layout, without hardlinks or original writes. Original lifetime flock and both fresh idle checks remain required throughout the exclusive slot. Ceilings remain 600 s total, 300 s decoder plus 10 s grace, 300 CPU-seconds with stricter subdivisions, one CPU/`GOMAXPROCS=1`, AS 16 GiB, file 64 MiB, combined retained streams 16 MiB, sampled scratch 256 MiB, RAM floor 24 GiB and free-disk floor 12 GiB. No install, concurrent job, fallback or automatic retry is introduced.

This PASS is **mechanics only**. Sole Operator assignment must separately await the current trn21 full job **and attached fresh evaluation**, then authenticate host/root/pins, original lock and fresh quiescence. This review grants no staging/launch authority. Actual ingestion, nested NEFF matching and coverage remain unknown; no device attribution, dtype/MFU, throughput, quality or adoption claim follows. Per-process AS limits and sampled scratch are not cgroup/filesystem quotas, and read-only modes are not a hostile-executable sandbox; the original review's explicit limitations remain applicable.
