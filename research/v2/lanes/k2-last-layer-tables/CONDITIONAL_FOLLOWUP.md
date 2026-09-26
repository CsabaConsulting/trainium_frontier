# Conditional table follow-up, not a launch proposal

26 September2026. Source examined: b4728d17d082b44018290d2f465b5956f99ecbebe61da8b81508365c7dc53190.
Current K2 native and first P2 full results remain pending. Do not change the
first P2 experiment, claim a win, or launch another arm from this note.

The Researcher checked the actual P2 source while the hosts evaluated.
There is no table-freeze or table-cooldown flag. `train.py:1193` creates the
table RMS group with `no_lr_decay=True`; `train.py:1819` consequently keeps
its LR flat after warmup. Current table LR is approximately0.2598076211.
The ordinary scheduler at1663 already supports the baseline wall-clock
cooldown. This is source evidence, not proof that the flat LR harms BPB.

If the valid first P2 full result is weak/worse and reaches reported epoch2,
one possible single-factor follow-up is table-only LR cooldown: add
`--ngram-ve-flat-lr` defaultTrue and let its negative flag set only that
RMS group's `no_lr_decay=False`. Preserve every other parameter group,
gradient reduction/clipping, data, timing and ordinary checkpoint loader.
The baseline schedule is flat through450 schedule-seconds, then linearly
decays toward5% at1800. Do not implement/queue this until results select it.

FP32 master tables, accumulated gradients and RMS state remain required.
A future patch needs default/OFF three-step byte identity including times
past cooldown onset, FP32-master/BF16-gather reference updates at LR
multipliers1/.525/.05 with collisions and two microsteps, and ownership
propagation of the intended table LR on all mocked ranks. This is a test
outline for that hypothetical patch, not another test of current b4728d17.

Why not freeze precisely at printed epoch2: `prepare.py:435` counts each
rank's assigned row-group passes; `prepare.py:484` updates epoch on a
document-buffer refill. P2 `train.py:1812` fetches the next batch before
printing loader_state. Thus rank0's displayed epoch2 is not an exact common
first-repeated-token boundary. Never alter prepare.py to fix that reporting.
Historical six-layer A1 first logged epoch2 at step2380 and finished3051
with +0.0071572BPB; coexistence does not establish memorization as the cause.

Fixed1800/public20M/context1024 and unchanged selection rules remain.
Expected gain is unknown. K2/native and the first P2 full remain the priority.
