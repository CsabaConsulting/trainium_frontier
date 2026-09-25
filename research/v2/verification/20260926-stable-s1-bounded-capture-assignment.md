# One read-only S1 compiler-file observation — Operator assignment

Supervisor selection, 2026-09-25 23:20 UTC / 04:50 IST. This is a bounded
existing-file observation, not a training launch, failed-pair retry or S2 release.
Preserve current trn22 cold36a862 and trn21 full75b9 plus its attached evaluation.

Sole Operator may perform this once on trn22 after cold36a862 has naturally
exited and its terminal artifacts and explicit descendant quiescence are
retrieved. Hold the existing common host flock and require fresh ps plus
neuron-ls idle checks. If busy/unknown, skip; never interrupt a current job.
Do not delay a ready qualified full job by more than this 30-second observation.

Read only this exact prefix:

`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-stablepath-trn22-r1-pair/s1/train/caches/debug/neuronx-cc/`

Only these three immediate children, already named by archived live compiler argv:

- `96b199d519a331288b0a67881f092b12`
- `ea30e57eba554980a991fca1b737d5ac`
- `860bfc7a33ecceebfbabbde99c9f2e27`

For each, no-follow stat before/after `module.mlir`, `module.neff` and
`log-neuron-cc.txt`. Reject symlinks or unstable identities. Retain each MLIR
only if <=4 MiB; stream-hash each NEFF only if <=16 MiB (do not copy its
contents); retain log head/tail at most64 KiB each. At most32 immediate names
per named directory if needed, no recursion or other cache roots. Preserve
missing, oversize, race and hash-nonmatch observations explicitly.

Hard bounds: 30 seconds, 60 MiB total file reads, 16 MiB retained artifact
bytes. No SDK import, compiler, tensors, device work, cache writes/deletion,
data access or widened fallback. Do not follow guessed persistent-cache
filenames; this assignment covers only the named debug files above. Store
transport evidence locally under a new uniquely named Operator receipt path;
there is no need to modify the host's existing artifacts.

Compare any retained MLIR SHA with archived S1 principal keys:
forward `aa53f72d8a4c45dbc0ba1784f3536f8803525417f6de4d3f8fb82332aa2ef58e`,
backward `d81a59fd782a752806c228735a21928133bf2d21c5d59344e2e0e94fb3db8c25`.
Matching means existing input identity only, not a cache hit. Absence now
cannot prove absence at the earlier capture. Do not reverse the original
`MATCHED_PRINCIPAL_COMPILER_BYTES_MISSING` refusal or launch S2.

Reason: diagnose the missing compiler evidence without another blind cold run.
Undo: revoke or skip the unstarted observation; no source/runtime changes to undo.
