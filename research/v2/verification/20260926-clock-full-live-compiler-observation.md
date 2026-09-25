# Bounded live compiler-input observation; no new job

Supervisor selection2026-09-25 23:31 UTC. Sole Operator may collect read-only
evidence while the already authorized full36a862 runs on trn22. Preserve its
source, process, original lifetime lock, cache contents and all settings.
This is not a second launcher, compiler invocation or permission for S1/S2.

Use actual current compiler argv/PID and recorded full36a862 worker/job lineage
to identify a working directory ONLY under this exact configured debug root:
`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch36a8-startup-clock-cold-trn22-r1-packet/train/caches/debug/neuronx-cc/`.
No guessed directory, recursive scan, broad environment dump or other-job read.
If provenance is ambiguous, record UNKNOWN and stop that observation.

At most two distinct principal-graph compiler directories during this run;
only their exact `module.mlir` (regular non-symlink, <=4MiB) and the first64KiB
of `log-neuron-cc.txt` if present. At most4files/9MiB total bytes and30s per
observation, with no busy polling. Record actual paths, argv, UTC, byte sizes,
stat-before/after and SHA256; refuse or mark unstable if an input changes.
MLIR may be binary: transport base64/exact bytes, never a lossy decoded view.
Logs are bounded prefixes, explicitly not whole-log hashes. Retain locally in
a uniquely named observation directory; do not create host copies or write
inside caches. No tensors, SDK import, model/data reads or compiler changes.

These tiny observations do not claim identical graphs, cache hits or speedups.
Closed graph_stats and independent analysis must establish any later mapping.
Original S1 pair remains refused and all three directories absent at its later
bounded lookup remain a separate result. Undo by skipping further observation;
the valid full run and attached evaluation continue unchanged.
