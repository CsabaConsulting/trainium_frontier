# Anchored exact36 control: completed, independent result review pending

The single read-only boundary observation at 2026-09-26 04:49:54 UTC found both phases terminal. No successor was started and no host files, caches, workers or configuration were changed.

- Label: `fs-20260926-clock36-anchored-warm-control-trn21-r1`.
- Training return code 0; 3,631 steps. Actual uncapped fifth boundary 110.20984183403198 seconds; process wall 1899.47586020804 seconds; charged through exit 1789.2660183740081 seconds, within 1800.
- Fresh public evaluation started 04:38:09.257567 UTC and ended 04:49:51.607522 UTC. Context 1024, batch 1, 20,971,520 tokens, 83,462,186 bytes, 57,164,348.0 total nats, public BPB **0.9881207925137612**. Return code 0; eval_seconds 687.4916540000122, eval_over_timeout false.
- Outer command exited 0 at 04:49:51.986287 UTC. Terminal error null and final_errors empty. This is a prospective control, not a newly adopted change or submission qualification.
- Post-eval checkpoint rehash: 453,048,594 bytes, SHA-256 `8b2df035d505d6953cf5f2ff9bd718c7132de80a3baaed3068e6039dcf281ae1`; stable device/inode/size/mtime/ctime during read and identical to the training checkpoint record. No checkpoint was downloaded.
- Explicit final postflight at 04:50:53 UTC: all **13 actually observed** process identities absent; complete 254-row PID/PPID/PGID/SID/starttime census had no matching lineage or job/compiler rows. No transient unobserved identities were invented. Device idle, fuser exit 1 with empty output, original lock device 66305/inode 596823 freely acquired and immediately released.
- Free disk 13,067,284,480 bytes. No cleanup or resource-gate reduction; this is below a new 13.5-GiB-reserve launch requirement.
- 39 complete text artifacts (827,288 bytes), including train/eval logs and rehearsals, all four closed graph-stat tables, source-anchor before/after receipts, source-copy identities and exact budget/accounting records are preserved in `results-bundle.json`.

Stable-path cache reuse and path-only causality remain **unproven**, despite the shorter measured startup. Actual Python import origin was not instrumented. All raw evidence remains available for independent review before control use.

Bundle SHA-256: `8d48db60dea8dee0d591cd2a51c7649ae617d4594334ee89f2dd2bd39e4766ea`.
