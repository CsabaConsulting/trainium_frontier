# trn21 restart recovery and A1 dispatch

Read-only SSH inspection at 2026-09-25 15:08:36 UTC (20:38:36 IST) found trn21
up for 10 days, 50 GiB free, no training/eval/compiler/runner processes and no
Neuron device holders. A0 training and fresh public-20M eval had both exited zero;
the eval finished 14:26:47 UTC with BPB 0.9914739939934344. S0 warm compile smoke
exited zero at 14:40:00 UTC with five steps (inconclusive). A1 exact CPU G2 finished
14:56:34 UTC with aggregate PASS and exit zero. A1 full had never started.

The parent Supervisor then assigned this Operator the full A1 dispatch, using the
completed proposal `research/submission-validation/20260925-v2-c5c-a1-m64d128-trn21-r1/EXPERIMENT_PROPOSAL.json`
and the newest Final Push directive. Earlier prelaunch native-smoke/review clauses
in that proposal are superseded for this exact-source flag-only arm. The verified
source remains c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586.

The host driver holds the machine lock throughout training and its fresh-process
public-20M/context-1024 evaluation. It is started detached on trn21, so neither phase
requires a surviving laptop process. No science, prepare.py or evaluator edit.

Full A1 launched at **20:41:55 IST** (15:11:55 UTC). At 20:42:07 IST the detached
flock PID 2905396 was parented by PID 1, the host driver was PID 2905397, torchrun
PID 2905399 and all four training workers 2905414–2905417 were live. This is startup
evidence, not a completed training-step or BPB result. Host launch and exact training
environment receipts are adjacent. Latest preflight free space was 53,486,702,592 bytes.
