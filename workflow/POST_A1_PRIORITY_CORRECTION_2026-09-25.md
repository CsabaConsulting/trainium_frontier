# Poojith's post-A1 correction, 25 September 2026

Newest direct human instruction; overrides the earlier table-first branch and
instruction to finish A2's warm rerun. Other legality, source/CPU checks,
budget, archival and human-only submission/AWS rules remain.

1. Stop the exact A2 warm-r2 run now. Preserve evidence and mark it human-aborted,
   never a quality loss or completed table screen. Do not start its eval.
2. trn21: Recursive schedule. trn22: checked8x1024 architecture as soon as ready;
   otherwise the Recursive schedule replicate. Keep installed serial queues.
3. FP8: finish verification, then matched whole-step timing at the first free
   slot. Only if at least3% faster, schedule a full-budget quality run.
4. Diagnose actual live-process compile-cache paths, including train.py's
   defaults, SDK-consumed paths, NEFF writes and real reuse. Directory reuse
   alone is not proof of cache hits. No speculative cache flags or deletion.
5. Tables now LOW PRIORITY. At most one later A1 experiment freezing tables at
   epoch2 (IMPL-2), after schedule/architecture/FP8 priorities. The gathered-row
   patch can be archived/reviewed, but native table work no longer preempts
   those arms. No A3/table-schedule/table-LR branch from failed or aborted A2.
6. Maintain automatic overnight jobs without the human. Checked architecture,
   FP8 tests and required replicates preempt unstarted fallback flag-bank jobs;
   never run two launchers or interrupt another active run to reorder the tail.

## Evidence versus hypotheses

A1 public20M BPB0.9986312057378084 versus same-host trn21 control
0.9914739939934344: delta+0.007157211744374; quality DISCARD. Correct same-host
step comparison is3051/3675, a16.98% reduction. The3633-step control belongs
to trn22. Step-window exemption remains applied; it does not change this loss.
Second-epoch table memorization is a hypothesis, not a demonstrated cause.
The later freeze arm tests it. User-reported official0.9901 and top-ten0.9824
are not newly measured results; no calibrated success probability is available
and speculative30–40% odds are not adopted as a fact.

## Actions already completed

trn21 Recursive6ef7 launched16:26:55UTC /21:56:55IST. The exact reviewed CPU
source/identity receipt isfb540549; launch evidence549cf7c6.
A2 warm-r2 stopped at16:29:30.734UTC /21:59:30IST with targeted SIGTERM.
By16:29:32.735UTC expected workers had exited and Neuron was clear. Subsequent
Operator inspection16:30:50UTC confirmed no live compiler/eval/queue process
and a free shared lock. One completed step, dt1040.11s, no checkpoint/eval.
The raw Operator receipt is archived at
`research/v2/operator/20260925-trn22-a2-warm-r2-human-stop/` and includes the
targeted signals, process identities, raw terminal error and empty-device
postflight. These are observed timestamps, not an invented result.

trn22 Recursive6ef7 launched16:32:57UTC /22:02:57IST. Both reviewed serial
queues now have six staged/armed scalar fallback jobs, as documented in
`research/v2/queue.json`. Source/timer gates for architecture93ae and FP89212
are partially complete, but their full-size CPU/native checks remain pending.

The six completed c5c scalar-flag proposals per host are authorized as a
fallback tail after active schedule jobs. They remain explicitly lower
priority than checked architecture/FP8 and retain exact prior source/argv;
no claim that all new-code arms are ready is permitted. A five-step fallback
would also be invalid quality, not a measured loss or win.
