# Exact bcbd full-quality launch

Sole Operator dispatched the frozen 7da2559f packet after the corrected authoritative top-level trn21 state in 5f7c2b19. Assignment 40b7d400, exact source bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522. No source, science argv, evaluator, cache, guard, driver or budget changes.

- Dispatch: 2026-09-26 00:15:00.880619 UTC; native training launch 00:15:00.993796 UTC.
- Dispatch/flock/driver PIDs: 3102316 / 3102317 / 3102318.
- Timeout/harness/torchrun: 3102334 / 3102335 / 3102353.
- Four rank workers: 3102368, 3102369, 3102370, 3102371, each with recorded actual rank and own PGID/SID.
- Common device0 lock inode596823/device66305 retained. Under-lock ps and neuron-ls both PASS. Historical queue PID2983496 absent; PAUSE_NEXT preserved.
- Prior K2 terminal and outer exit1 verified, all eight observed IDs absent and no PID/PPID/PGID/SID lineage remains. K2 failure is preserved, not reset or adopted.
- Prior K2 outer end 00:09:18.571653 UTC; idle until this training launch was342.422143seconds (5m42s).
- Free disk at first worker evidence00:15:39.802520 UTC:19,624,853,504bytes, above10.5GiB launch gate but below25GB preference. No cleanup performed.
- Research caches are retained existing paths, not proof of hits or a cold full replay. Train1800s source budget, actual capped-first-five process-through-exit accounting; fresh20971520-token/context1024/B1 eval follows under the same lifetime lock with fresh idle checks.

First raw observation adaptation had a one-item tuple typo; preserved separately. It did not affect the native job. A subsequent tool display was truncated, so the raw receipt was retrieved into a local immutable artifact before printing a compact summary. No duplicate dispatch occurred. This label is consumed; never rerun stage-dispatch.py.

No automatic next job, native retry, adoption, submission or AWS action.
