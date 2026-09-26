# trn21 batch262144 / LR0.021 training-to-evaluation handoff

Observed 2026-09-26 08:45:18.767240 UTC, read-only. Raw `trn21-live.json`
SHA-256 `73a6c89a614693d326fc8ca63355e5c0ab74b3ae705eeb7a967f726d66ccfe49`.

- Label: `fs-20260926-sprint-clean-batch-lr14-trn21-r1`.
- Training exited 0 at 08:45:16.194952 UTC; 1,921 sequential timed boundaries.
- Uncapped fifth boundary 798.4203850750346 seconds; the same capped allowance
  applies. Actual process wall 2587.1740332130576 seconds; charged through exit
  1788.753648138023 seconds, within 1,800 seconds.
- Source estimates remain separate: training 1784.3 seconds, startup allowance
  estimate 796.3 seconds, charged estimate 1784.7 seconds, `over_budget=False`.
- Checkpoint 453,044,626 bytes, SHA-256
  `7ca06a5ca4ca7a148b9b65fa0802229478deaa5f7f837274e7b5eb81bb3cb5ea`.
- Fresh public20M/context1024/batch1 evaluation started 08:45:16.265049 UTC.
  Observed timeout 3255448, harness 3255449 and evaluator 3255522; original
  guardian 3231387 and original device66305/inode596823 lock remain held.
- Free root space 169,195,470,848 bytes after the separately authorized online
  root expansion. No fatal error or evaluation score in this observation.

Local authentication rehashed all 16 captured files, verified all 1,921 step
indices/timestamps and through-exit arithmetic, and checked stable PID/startticks/
PPID/PGID/SID for every observed process. The generic observer marks the guardian's
`identity_stable` false only because its normal process state changed S to R;
the actual identity and lineage keys are unchanged. This is not a free-host or
completed-evaluation receipt. No successor was launched.
