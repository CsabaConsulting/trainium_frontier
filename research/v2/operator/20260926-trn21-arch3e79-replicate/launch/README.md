# Pure3e79 trn21 confirmation — actual guarded launch

One-shot release91ee27d7 applied after seed43 training and attached fresh20M
ended naturally. Ten small seed43 final artifacts were retrieved and verified
before staging; archivecb7ad742. Old queue PID2983496 absent, original lock
inode596823/device66305 unchanged, fresh idle ps/neuron-ls and30.14GiB free.

Actual train launch2026-09-25T21:24:00.612118Z (26September02:54IST).
Dispatch3041448, flock3041449, driver3041450, torchrun3041481,
workers3041496/3041497/3041498/3041499. Snapshot21:24:11 confirms all four
workers and lifetime common lock held; in-lock guardPASS. No restart/overlap.

Source3e79a7c1, exactly reviewed pure8x1024/U-Net/partial attention/initial
cache-refresh, seed42, old schedule/LR.015, no Recursive flag, no port or FP8.
Same-host c5c comparator0.9914739939934344. Driver retains lock through
training and fresh20M/context1024/batch1 and checks ps/neuron-ls before eval.

Warm-intended existing cache directories are not proof of hits or cold cost.
Separate3e79 trn22 cold-five gate844.353s does not establish trn21 cold cost.
Preserve actual uncapped fifth boundary, wall-through-exit and charged budget.
This is a research confirmation only, with no automatic next job or submission.
