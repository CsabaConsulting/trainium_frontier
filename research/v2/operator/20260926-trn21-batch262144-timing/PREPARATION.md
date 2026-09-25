# Batch262144 prospective timing — not staged or released

Use exact reviewed a489f7 source and the adopted6ef7 Recursive profile. The only
science change is global batch131072→262144, implemented as two microbatches per
update; no automatic learning-rate compensation. Stop normally at105 updates:
five official startup boundaries plus100 steady updates. Source budget1800 stays.
There is no eval, automatic continuation, adoption or quality claim.

The existing unmodified rehearsal7a55cb87 is present on trn21 alongside exact
prepare5807c46e (read-only SHA confirmation19:44:57UTC). In warm-cache mode it
inherits the same six research cache roots as the completed6ef7 control. Its
50ms observer records the actual first-five startup and the100-update interval;
the result is not genuinely cold. No profiler wrapper is present in this run.

Both the candidate and its105-step cap use the existing source/parser/loop; no
train.py, prepare.py, timer, graph-shape, SDK or compiler-flag edits are made by
the Operator. The production accumulation2 native specialization remains unproven
by prior full-parameter CPU G2. Stage only after independent binding review and
new scoped release, after current P/CT work; ready K1/K2 can preempt this packet.

Predeclared same-host baseline is completed control
fs-20260925-l3-base-repeat-trn21-r2-full. At steps10,20,...100 its median printed
speed is268831tokens/s (all LR multipliers1.000). The5% threshold is282272.55.
Compare the identical printed window, not short-run BPB. Report observed100-step
candidate aggregate speed as a sanity diagnostic only: there is no same-observer
aggregate baseline, so it must not veto against the historical printed median.
Preserve divergence/stall/hang/budget evidence for review. Rank0 documented
STEP_LOG writes add real overhead absent from the historical control; disclose
it instead of treating sampled/aggregate definitions as interchangeable.

The offline analyzer verifies native/accumulation markers, finite logged samples,
105 sequential boundaries, checkpoint and budget receipt, expected LR window,
and the matched median speed check. It cannot launch, adopt or approve a full run. Any failure
archives the actual error. Source/cold/native/full1800+public20M gates remain.

Reuse the reviewed shared guardadafb54a and dispatcher c71474bc. The same common
host lock is authenticated before fresh ps+neuron-ls; authenticated FD passes to
the timeout/harness. Historical queues remain stopped/paused. Check descendants,
compilers and device quiescence after any failure; no unlocked retry or cleanup.
