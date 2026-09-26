# Actual trn22 batch262144 / LR0.021 launch

Supervisor release `a6e0af03`, assignment SHA
`efcf29b1287bd973500b6871bae1860fd18657869128b73780e52d28d5dcf378`.
The release prose header was later than the actual dispatch; the Supervisor's
explicit immediate release controlled. Actual timestamps below are from the host.

- Dispatch: 2026-09-26 09:30:00.148440 UTC, PID3373963.
- Training phase: 09:30:00.272625 UTC.
- Label: `fs-20260926-sprint-clean-batch-lr14-trn22-r1`.
- At09:30:17.822617 UTC, all ranks0–3 were PIDs3374014–3374017,
  each startticks102906407; original device66305/inode832589 lock held.
- Source8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee,
  unchanged packetbb5d0a79, proposal785bfe4e, driver6beaab12.
- Full1800 seconds, seed42, LNC2/four ranks, total-batch262144, matrixLR0.021,
  attached fresh-process public20971520/context1024 evaluation.
- Retained same-host batch-lr1 six-cache namespace reused exactly as prepared.
  This is research cache reuse, not proof of hits or a cold run.
- Actual predecessor K2 failed with zero steps and all53 observed identities
  gone; failure remains parked. Idle terminal-to-dispatch gap271.705434s.

Fresh exact-byte, unused-path, original-lock, STOP, ps/neuron-ls/fuser,
RAM64GiB and initial13.5GiB checks passed. Existing guardian remains responsible
for training, attached evaluation and descendant quiescence. Trn21 control was
not touched. No automatic adoption, upload or successor is authorized.
