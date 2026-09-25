# Human-authorized abort: A2 warm-r2

Disposition: **human_aborted_not_quality**. No BPB or scientific table verdict.

Poojith explicitly ordered this exact run stopped. Before signalling, the
Operator independently read PID, UID, start ticks, parent/process-group/session
IDs, argv and working directory for timeout3051234, torchrun3051235 and
workers3051250–3051253. They matched this label and its prior launch receipt.
The compiler subtree under worker3051251 was identified too; no unrelated
process group was selected.

At **2026-09-25 16:29:30.733940 UTC / 21:59:30 IST**, after creating the
owned `runs/PAUSE_NEXT` marker, the Operator sent **SIGTERM** only to the
verified parent group3051234 and worker groups3051250–3051253. No SIGKILL,
AWS action, source edit, cleanup or breaker reset was used. All expected
training PIDs had exited by **16:29:32.735 UTC**. The queue recorded
`TRAINING_EXIT_1` and exited; this is the expected induced abort, not evidence
of a model-quality failure. Its raw terminal record was preserved unchanged.

Only step00000 had completed: loss9.0112, first-step duration1040.11 seconds.
Wall time from phase launch to stop was approximately1656.56 seconds
(27m36.56s). There was no final startup/budget summary, no checkpoint and no
evaluation launch. No new evaluation was performed after the abort.

At **16:30:50.935 UTC**, the independent postflight found no live train,
torchrun, host queue, evaluator or compiler processes, no Neuron device holder,
and the common device lock free. Disk had49,798,868,992 bytes available.
All training logs, compiler/cache artifacts and the consumed run directory
remain in place. Exact signals, pre-stop process ownership, terminal log and
postflight are archived here with SHA-256 retrieval metadata.

The owned pause marker was subsequently removed only during the separately
authorized dense Recursive fallback launch, after its manifest/source/gates
and quiescent preflight passed. No aborted label was reused.
