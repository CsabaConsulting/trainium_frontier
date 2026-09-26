# One compact K2 native201 diagnostic — Supervisor release

Issued 26 September2026,10:28UTC under the latest direct net-token amendment
and FINAL SPRINT. Sole Operator: `/root/trn22_queue_operator`.

Select trn22 ONLY, immediately after its current batch/LR training and attached
public20M evaluation naturally finish, final artifacts are retrieved, and the
normal fresh original-lock idle/source/resource admission passes. Do not
preempt that evaluation. Preserve the separately running662c test ontrn21.
No trn21 alternative, duplicate run, extra OFF, smoke or new review is released.

## Frozen source, proposal and command

- Label: `fs-20260926-k2-runtimegroups-native201-trn22-r1`.
- Source: `research/v2/lanes/k2-runtime-groups/train.py`.
- Source SHA256: `f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e`.
- Proposal: `research/v2/proposals/20260926-k2-runtime-groups-native201/EXPERIMENT_PROPOSAL.json`.
- Proposal SHA256: `e2c199370c1a0ed8e9a9b9d840c4b2d810bcd5522db38adcd3ef3641c5534947`.
- Exact command SHA256: `454ddb324b6b795ab4f05c46161a46d44e22c6e03e934d598a7e1256d496f23e`.

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --num-steps 201 --k2-embedding
```

Same production V8192/T32768/C1024, seed42, full chip/four ranks, batch131072,
1800-second schedule and official accounting. Only the reviewed outer-loop
representation differs from662c. At most201 completed steps; preserve normal
budget-driven earlier stopping, checkpoint reserve,2730-second wrapper and
60-second TERM grace. This one timing run IS the native test, not an extra
gate before another smoke. No full-quality or table successor is released.

## Evidence and measured continuation rule

Existing finite CPU/reference/OFF identity, exact production backend and two
official full-shape simulator receipts passed. One fresh independent review
`d6e385693` executed exhaustive addresses and ordered-FP32 reference cases.
Root `verification/20260926-k2-runtime-groups/COMPOSITION.md` authenticates
both full sources, the exact complete diff and113 unchanged other AST nodes.
Root executed38 read-only proposal/input/control-window binding checks.
The accepted unchanged native framework-forward evidence applies; no new gate.

Declared trn22 control: `fs-20260926-sprint-e1e2-control-trn22-r1`, exacte1e2.
t9=796.3216340140207, t44=813.5272218530299;
primary35*131072/(t44-t9)=266629.65793002315tok/s. Fixed window, no fastest
sample selection. Accepted cold reference859.473523823s charges0s; this is
separate from the warm control's startup, not a cold-cache claim about it.
Base net479933384.27404165; required net494331385.8022629tokens.

Report actual cold fifth-boundary startup, max(0,startup-900) charged excess,
total charged-through-exit, steady rate, net=max(0,1800-charge)*rate, percentage
delta, all emitted matched losses through200 and time-based-LR confounding.
Unknown measurements remain unknown. Net ratio>=1.03 is necessary for further
K2/table work; speed alone is insufficient. Neither a smaller standalone NEFF
nor this diagnostic grants a BPB win or an upload.

## Unchanged machine and artifact protections

Remote root:
`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-k2-runtimegroups-native201-trn22-r1-packet`.
Use the existing proven native201 guardian/driver with only mechanical
source/proposal/review/assignment/control/label/root pin substitutions.
Archive its final exact packet/driver/member hashes; verify those before use.
No semantic launcher rewrite or additional independent review is required.

Retain original `/home/ubuntu/.cache/trainium_freshstart/device0.lock`,
device66305/inode832589 for all children/compilers through postflight.
Fresh in-lock STOP/ps/neuron-ls/fuser/source/argv/approved-environment checks.
Disk>=max(14495514624B,11274289152B+actual conservative new footprint),
MemAvailable>=68719476736B. Create and prove six fresh empty private caches;
verify actual four-rank cache/compiler environment. No preload or warm reuse.
No AWS, cleanup, package, evaluator, tokenizer, data, launch-shape or budget edit.

After an actual failure, archive its literal error and release only after
original-lock/process/device quiescence. No automatic retry. Existing662c
full-control and414c table packets remain byte-identical and disabled. This
prospective source-specific compile diagnostic may use the free other host
without cancelling or waiting for current662c's terminal outcome.
