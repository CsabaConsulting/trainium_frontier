# Bounded architecture cache investigation

Read-only window: 2026-09-25 21:48:30–21:58:30 UTC. No host edits, new jobs, compiler execution, decoder, SDK imports, recursive cache traversal, cache clearing or checkpoint transfer. Both active training jobs and attached evaluations remain protected.

## Conclusion

The nominally warm trn22 full architecture run really recompiled its principal graphs. Changing compilation keys, not just slow artifact loading, is established. Absolute source-path churn is a concrete hypothesis, not a proven isolated root cause. No cache fix or hit is claimed.

Exact source: 3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c. Same-host cold and full runs share all six cache roots listed in authenticated-comparison.json and the same compiler profile. Cold fifth boundary: 844.35300997796 s. Full fifth boundary: 814.6809981259285 s.

| Graph | Cold key prefix | Full key prefix | Nodes | Positive full compile seconds |
| --- | --- | --- | --- | --- |
| forward | 3b08e870f3032e057b51964a70500391bb8b6322f1a7e8c4f6ec45292309c68c | 53a92b049370b5b633a75b78f6628d62e5e30c92e9312ced879072255916c486 | 666 | 202.390 (rank 2) |
| backward | dce743d7cb608694c0807dd91a160a09422f44d7b74ba094db2c5408195b6039 | 97682f6d3f3bc59622a56cc557660dcb7df167b28f21c53473d866e336beddfd | 1169 | 477.604 (rank 0) |

The major full compile total is 679.994 s. Cold positive compilation was 200.236/468.847 s on rank 3. All four ranks share the principal keys within each run. Per-rank row counts are unchanged: 5,6,5,5 (rank order 0,1,2,3); there is no observed graph-count explosion. Every key shares suffix cd47b8171ac56a23cc896e6ceb62fa0bfd9e8cf0bf9b08bd2a965094c3965433; its interpretation is not independently established, so it is called key_suffix, not a flags hash.

## Authenticated MLIR evidence

Existing exact-byte archive: research/v2/operator/20260926-trn22-arch3e79-full/live-compiler-201309/exact-byte-bundle.json. The base64 binary module is 20777 bytes and SHA-256 97682f6d3f3bc59622a56cc557660dcb7df167b28f21c53473d866e336beddfd, exactly the full backward graph-key prefix. It embeds this absolute source path:

/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-primed-trn22-r2-full-packet/train/train.py

The sibling module.mlir text is lossy and is NOT used for byte equality or hashing. Matching authenticated cold MLIR bytes are missing. Consequently we cannot show that source-path metadata is the only changed compiler input, nor map these major graph IDs to sampled persistent NEFF filenames.

Installed metrics source already archived at research/v2/operator/20260925-trn22-l2-93ae-cpu-r1/native/metrics-source-readonly.json documents graph_id as the C++ compilation-cache key and compile time from the matching entry. Zero also means a missing entry: zero is NOT cache-hit proof. Graph CSV export is exit-only.

## Current jobs and bounded metadata

At 21:48:35 UTC trn21 pure architecture confirmation was at step 1360, 268451 tok/s, 1133 s remaining, with all four workers and the original lock present. It uses another host's existing dense research roots, not a proven populated architecture cache. First five logged step durations were 779.52/.48/.49/.49/.49 s. The uncapped harness fifth boundary is pending final receipt.

At 21:49:14 UTC trn22 LR0.012's existing compiler log shows a fresh forward compile, same compiler profile and a new unique LR source pathname. The copied prefix is capped at 16384 bytes from an approximately 3.55 MB log. The debug directory name is NOT treated as a cache key. Current complete graph-stat CSVs are absent because the process has not exited. Both jobs remain active; no new host work was triggered by this investigation.

Compiler command observed: neuronx-cc compile module.mlir --framework XLA --target trn2 --lnc 2 -O2 --model-type transformer --auto-cast=none --optlevel=1 --auto-cast matmult --auto-cast-type bf16 --output module.neff. Version 2.26.6360.0+6f180f47.

Raw metadata records exact sampled paths, sizes and timestamps. Sampling was limited to 40 top entries per layer, four nested directories times 20 entries, plus four named sharded directories. No recursive walk or NEFF/HLO body read. Sampled NEFF files prove artifacts exist, not that the major graph was reused; atime is not a hit receipt. The compiler root was empty while other caches contain sharded artifacts. No missing-write conclusion follows.

## Next work is design-only

PATH_DIAGNOSTIC_DESIGN.md proposes one reversible controlled design; it grants no implementation, staging or launch authority. Current jobs and the next CPU checkpoint assignment have priority.
