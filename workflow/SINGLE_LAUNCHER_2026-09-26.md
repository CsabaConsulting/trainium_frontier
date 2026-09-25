# Mandatory single-launcher rule — direct instruction from Poojith

Effective immediately on 26 September IST for the 15-minute controller, main
session, every subagent and every manual fallback. This supplements NIGHT MODE;
it is not optional and does not authorize interrupting an existing job.

1. Before each training, native diagnostic or separately launched Neuron eval,
   acquire an exclusive nonblocking `flock` on the target host's existing
   `/home/ubuntu/.cache/trainium_freshstart/device0.lock`.
2. While holding that lock, obtain fresh `ps` and `neuron-ls` evidence proving
   no other torchrun/Neuron job or device holder is active. Both checks must
   succeed. Unknown, malformed, failed or busy observations are NOT an idle pass.
3. On lock contention or a failed check: do not launch, retain the actual error
   and timestamped observation, and move to other useful authorized work.
4. Retain the same lock through the complete child-process lifetime and any
   attached evaluation. Never release between checks and spawn. A later
   separately launched eval must acquire it and repeat the checks. For an
   attached eval, retain it and confirm training children have exited first.
5. Never delete, replace or rename the lock file, use a per-run alternative
   lock, bypass the rule via a fallback, or kill another job to obtain it.
   All launch paths must contend on the same persistent host-local inode.

The designated Operator remains the sole remote launcher. Other agents may
prepare/review packets but cannot independently start a job. Existing running
jobs are preserved; pending packets remain disabled until their lock/check
implementation is reviewed. The source, legality, disk, STOP, budget and
correctness gates remain unchanged.

By 00:57 IST the three existing automation prompts were updated in the app:
`trainium-safety-gated-controller`, `trainium-04-00-ist-self-audit`, and
`trainium-07-45-ist-morning-report`. Their schedules and task target did not
change. Operator was notified; pending-driver hardening and test receipts are
required before the next launch. Do not mistake prompt installation for a
completed native enforcement test.
