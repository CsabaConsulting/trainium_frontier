# trn21 dense Recursive schedule: native launch

Authorized label `fs-20260925-l3-timer-dense-trn21-r1-full` launched
**2026-09-25 16:26:55 UTC / 21:56:55 IST**. It uses exact source
`6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`
and the exact full-width dense Recursive-ON argv that passed CPU G2.
Supervisor checked proposal/manifest at canonical `681dc501` and explicitly
assigned launch after independent actual source/ON/OFF receipt review
`fb540549`. The earlier f9b6 source was not substituted.

The old **empty** waiter PID 2926930 was gracefully paused using an owned
`PAUSE_NEXT` marker and verified exited before deploying the reviewed runner
SHA `7fe160cf0b5eef110caeee963fa4e39f708f4b5aaa6d503a7cf2344f1ac91244`.
The owned marker alone was removed as documented. No active training was
stopped; no other file, checkpoint, cache or worktree was removed.

New host-detached runner **PID 2930000**, PPID 1; train-phase timeout PID
2930002. A distinct SSH postflight at **16:27:57 UTC** found exactly one
queue/controller and four live training workers **2930018–2930021**.
All four optimizer-ownership records appeared, source/evaluator hashes still
matched, and no terminal/error receipt existed. It was in startup/compilation,
not yet a scored result. The same shared lock spans training and fresh eval.

The complete exact launch command and sanitized environment are preserved in
`training-launch.json`. Existing A1 train/eval cache roots are unchanged;
this is a warm research run, not a cold-start or submission proof. The runner
will execute fresh-process organizer public-20M/context-1024 evaluation and
save terminal/result receipts automatically without the laptop.

Root host directory:
`/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1`.
Run directory:
`runs/fs-20260925-l3-timer-dense-trn21-r1-full` under that root.
Resolved manifest SHA:
`6ebc7a3889c370f9b5d113d5db99f08d2ccbaaf5416513c3f62e1a4c335f0571`.
Only this single job is armed. No old flag bank, table verdict or research-base
adoption was silently inferred from the still-pending A2 repeat.

## Recorded utilization incident

A1's evaluation finished at 16:09:41 UTC; this next native run started at
16:26:55 UTC: **17 minutes 14 seconds without a Neuron training/eval job**,
exceeding the 15-minute threshold. The interval included three serial required
CPU G2/identity probes (approximately 20 seconds each), their evidence
retrieval/independent review, exact proposal/runner-registry preparation and
Supervisor handoff. The threshold crossing was reported at 16:24:51 UTC;
launch followed the Supervisor assignment. This overhead is recorded rather
than counted as native hardware utilization. No launch/deployment failure
occurred. Final preflight was idle/device-free with 49,950,384,128 bytes free;
postlaunch free space was 49,469,980,672 bytes, above the 6.5-GiB floor.
