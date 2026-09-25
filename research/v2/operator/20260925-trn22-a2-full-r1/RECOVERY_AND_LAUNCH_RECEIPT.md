# trn22 laptop-restart recovery and A2 full launch

Operator observation and launch receipt, 2026-09-25. This proves a host process
launch; it does not yet prove completed training, an evaluation score or a win.

## Restart recovery

Read-only SSH observations at 15:08:45 and 15:09:41 UTC found no training,
evaluation, Neuron compiler, driver or monitoring process on trn22. The old
September-16 `tmux finalwrap` session contained an idle bash shell. The device had
no holder. Existing host logs independently confirmed these terminal jobs:

- c5c A0 r10 trained, then completed its separate evaluation at 14:16:06 UTC;
  evaluation return code 0, public-20M/context-1024 BPB 0.991513958330512.
- 116188 cold startup smoke finished at 14:36:26 UTC, exit 0; its five steps do
  not satisfy the earlier 20-step discriminator threshold.
- Exact-size A2 CPU G2 finished at 14:54:16 UTC, exit 0; independent numeric
  PASS is commit `1c24f3babcc6992737174777b75ace16091b123b`.

The exact A2 full-run label had no host path and had never launched. No completed
job or evaluation needed restarting after the laptop reboot. The host was idle
for more than 15 minutes after its CPU G2, and this was reported to Supervisor.

## Current authorization and experiment

Supervisor explicitly assigned sole trn22 launch ownership following the recovery
check, under Poojith's newest FINAL PUSH directive. This is an existing, reviewed
file's flag-only A2 profile. Historical proposal statements requiring another
native smoke or a one-use prelaunch review ceremony are overridden for this run.
The complete predeclared experiment is
`research/submission-validation/20260925-v2-c5c-a2-m64d128-trn22-r1/EXPERIMENT_PROPOSAL.json`;
its unchanged historical gate status is not the dispatch authority. Independent
postlaunch receipt/result review remains required before the score is used.

- Label: `fs-20260925-v2-c5c-a2-m64d128-trn22-r1-full`.
- Exact source SHA-256:
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
- Trusted unchanged prepare SHA-256:
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Same-host control: `fs-20260925-v2-c5c-a0-control-trn22-r10`,
  public-20M/context-1024 0.991513958330512, 3633 steps, independently accepted.
- Exact launch command is in `launch-command.txt`: dense1024/LR0.015 with
  bigram and trigram VE at layers 1,-1, multiplier64/dim128, BF16 RMSProp,
  flat table LR and separate table clipping. No model source edit.
- LNC2, four ranks/full chip, seed42, native training budget1800s. Research uses
  the six existing Sep24 warm cache paths recorded in `preflight.json`.

## Preflight, staging and detached host driver

The plain manual path copied the exact source, launch command, provenance,
proposal, CPU G2 and independent PASS receipt with scp. Every staged SHA-256
matched its controller copy. New package and work paths were absent beforehand.
The pinned Sep24 rehearsal harness SHA is
`7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b`.
It copies the exact source and unchanged evaluator into fresh isolated train/eval
directories, records actual argv and authenticated optimizer boundaries, and
checks the 1800-second charge through process exit.

At 15:12:50 UTC, repeated preflight proved no other training/eval process, no
Neuron device holder, no STOP or trn22/global breaker, shared device lock free,
exact source/evaluator/harness identities, and 53,800,849,408 free root bytes
(required minimum 6,979,321,856 bytes). Existing data and caches were retained.

The host-detached nohup driver launched at **2026-09-25 15:12:50 UTC / 20:42:50 IST**,
PID **3043794**, PPID1. At 15:13:02 UTC, torchrun PID **3043813** and four workers
**3043828–3043831** were alive with the exact A2 argv. The driver holds the shared
device lock across both training and evaluation. The logged source identity
matches c5c. No training step or BPB was claimed at this first snapshot.

Host package:
`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260925-v2-c5c-a2-m64d128-trn22-r1-full-source`

Train/eval sandboxes:
`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260925-v2-c5c-a2-m64d128-trn22-r1-full-work/{train,eval}`

After successful training, the same detached driver starts the unchanged harness
in a fresh Python process and fresh eval directory. It runs trusted
`prepare.py eval-public`, Neuron, batch1, context1024, public20971520 tokens,
evaluator timeout1200s. Training and eval exit codes and log tails on failure are
persisted on the host. This continuation does not depend on the laptop or SSH
session. A laptop restart will only require reconnecting monitoring and retrieving
receipts. The driver never appends a verdict or adopts the result.

The exact operator driver is preserved as `driver.sh`; it passed bash syntax
checking before transfer. `preflight.json`, `launch.json`, `attempt.started`,
`driver.log` and a live `train.log` snapshot are retrieved beside this receipt.
Final logs, checkpoint identity, measured steps/budget/BPB and independent review
remain pending. No AWS changes, cleanup, failed-label retry or submission occurred.
