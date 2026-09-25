# Architecture LR0.012 fallback — disabled preparation

Prepared at Supervisor request, not staged or launched. Checkpoint-only
clean3f5035/reference3e79 parity remains higher priority. A separate scoped
assignment is required; do not treat this file as dispatch authority.

Unique label fs-20260926-arch3e79-lr012-trn22-r1-full, exact3e79 source.
Only scientific change versus successful trn22 architecture run:
matrix LR0.015→0.012 (0.8×). Same8x1024/U-Net/attention0,2,4,6,7/initial-cache
refresh, seed42, old schedule, full1800s,LNC2x4,batch131072. No Recursive.

Primary LR comparator is same-host architecture0.9863403052402583/3629steps.
c5c0.991513958330512 belongs to actual label
fs-20260925-v2-c5c-a0-control-trn22-r10 (ledger verified), and is historical
architecture context only, NOT a fair causal LR control. The consumedr1 label
was a zero-step disk failure and must not be substituted.

The original c8957e5d full driver changes only label/output-path constants;
its control flow, guard, inherited lifetime lock FD, ps/neuron-ls before each
phase, prepare/harness, training budget and fresh20M evaluator are unchanged.
The learning-rate flag is in the packaged training argv, not an edited model.
Existing same-host research caches are preserved unchanged; names are not
hit evidence. Prior LR0.015 cold-five844.353s does not certify exactLR0.012.

No cleanup, old-runner restart, AWS action, eval modification, auto-next job,
research adoption or human submission is authorized. Own mechanical tests
are not independent review; all normal preflight and scoped release gates
must still pass. Preserve trn21's active pure-architecture confirmation.
