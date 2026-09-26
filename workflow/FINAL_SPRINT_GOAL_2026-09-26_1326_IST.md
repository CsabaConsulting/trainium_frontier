# Controlling pinned goal — FINAL SPRINT

Confirmed from the active app goal on 26 September 2026, 13:26 IST.
This replaces every earlier mission and conflicting priority, date, closed-lane,
quota or approval statement. Historical entries are evidence, not current authority.

FINAL SPRINT (authorized by Poojith). Replaces all earlier goals.
Goal: best official val_bpb by the Sep 29 18:00 IST freeze. Minimum: top 10 (cut ~0.970 and
falling). Base: e1e2f686 = 0.9878 official.
Rules unchanged: only train.py; no eval tricks, validation data, external data or weights;
load_for_eval returns the trained model.
Priorities: 1) K2 embedding scatter -> Trn2 timing -> full run. 2) Hashed tables on the 8-layer
base with the K2 backward, FP32 gradient accumulation. 3) Stack the winners: schedule, batch,
LR retune. 4) Kernels picked from the profile. Closed: K1, bigger models, gates, FP8.
Process: ask Poojith only about uploads, AWS spend and legality. Decide everything else, log it in
DECISIONS.md, continue. Evidence bar: CPU exactness + flag-off identity + one Trn2 timing
check, one 30-minute review. No host idle more than 10 minutes.
Submissions: quota unconfirmed (README says 5/week; Csaba is checking). Upload only confirmed
gains of >= 0.003 official (next: <= 0.9848) after cold replay + Claude audit. Poojith uploads.
Timeline: Sat 16:00 K2 timing. Sat night K2 + tables full runs. Sun stack. Mon LR retune +
second-seed replicate. Tue 18:00 freeze. Final upload by Sep 30 12:00 IST.
Report every 3 hours in 6 lines.

## Explicit table-precision amendment

Poojith decided FP32 accumulation for hashed-table gradients. CPU exactness must
compare to the FP32 reference, not reproduce the discarded historical BF16
collision-rounding path. FP32 master table parameters, gradients and RMS state
with BF16 gathered activations are the transparent implementation choice.
Reference: `F.embedding(fp32_master_table, rows).to(torch.bfloat16)`.
Do not cast the entire table to BF16 before lookup, and do not use hidden
evaluation adapters or change the trained model on load.

Off-host implementation may proceed while K2's native timing is pending. A full
tables run remains conditional on the K2 native result and the table patch's
CPU exactness, OFF identity, one review and native check. Target first full
tables launch tonight 20:00–22:00 IST; this is a conditional target, not a
measured readiness claim or guaranteed finish time.

## Preserved operational protections

One sole Operator, original lifetime lock, fresh process/device-idle and resource
checks, exact-source/argv receipts, fixed official budget, unchanged selection
rules and protected artifacts remain. No duplicate launcher, lowered disk floor,
unscored-artifact deletion, private data access or agent upload is authorized.
The earlier detailed Final Sprint applies only where compatible with this goal.
