# Latest direct amendment: net tokens and compile cost

Poojith,26 September2026,approximately15:49IST. This supersedes earlier
speed-only K2/table continuation gates. FINAL SPRINT legality, official budget,
one native timing, exactness, original locks and human-only uploads remain.

## Controlling continuation metric

Net tokens = max(0,1800 - charged startup excess) * steady tok/s.
Charged startup excess = max(0,actual cold startup seconds -900).
Reference base cold startup is approximately859s, hence zero startup charge.
Use the actual same-host control's preselected steady-window rate, not the
fastest sample; for current trn21 this is266550.74626781716tok/s.
Baseline net tokens=479791343.2820709; +3% threshold=494185083.580533.

Continue K2 and table follow-ups only when the applicable candidate's measured
net-token ratio is>=1.03 versus the accepted base. A >=3% steady-rate gain
alone no longer qualifies. Every table run must report actual cold startup and
charged seconds too. Warm research-cache startup is not a substitute for a
cold measurement. Report missing measurements as unknown, never zero.

Current native201 continues to its actual terminal or existing budget limit;
do not restart/kill solely because compile is long. Report cold startup, startup
excess charged, total charged-through-exit separately, steady tok/s, net tokens,
net delta and logged losses through200 with any time-based LR confound.
All existing disabled full/table transports remain disabled until this revised
gate and unchanged normal machine/source/correctness requirements pass.
This is the human's requested work estimate, not a BPB prediction or permission
to alter timing, checkpoint reserves, evaluator or final-quality selection.

## Parallel compile-time lane

Keep a dedicated compile-time agent. First diagnose actual base compile logs
for Muon shapes1024x1024,4096x1024,1024x4096 and8x32. Record per-graph time,
total and attribution confidence. Do not infer timings from aggregate startup.
Only the sole Operator collects remote evidence, with bounded read-only log
reads while existing jobs remain protected.

If measured Muon compile cost exceeds about60s total, consider stacking c_proj
transposed with c_fc using the existing transposed option, and moving tiny
ve_gate to AdamW to reduce four Muon shape specializations to two.
Before implementation, verify actual shape support/callers. Independently test
CPU transpose equivariance and full update mapping, including momentum/variance,
scaling and dtype tolerances. Orthogonalizer identity alone does not establish
optimizer-update identity. Moving ve_gate to AdamW is explicitly a scientific
optimizer change, not an exactness claim; keep its effect distinguishable.
Any selected patch needs the agreed evidence bar and one full budget run.
This narrowly reopens Muon COMPILE reduction, not closed K1 throughput racing.
No unconditional performance claim, new hardware, evaluator change or upload.
