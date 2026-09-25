# Mandatory single-launcher rule — direct instruction from Poojith

Effective immediately on 26 September IST for the 15-minute controller, main
session, every subagent and every manual fallback. This supplements NIGHT MODE;
it is not optional and does not authorize interrupting an existing job.

## Latest timebox amendment

Poojith requires a manual fallback if guarded-launcher work takes more than
30 minutes. Count conservatively from 19:24 UTC when this safeguard work began:
**deadline 19:54 UTC / 01:24 IST on 26 September**. Do not restart the clock.
If the tested guarded launcher is not working by then, the Operator must
launch the next otherwise-ready queued job using the manual path: acquire the
same exclusive host lock, inspect fresh ps plus neuron-ls under the lock,
launch only on a clean result, and release after the job/attached eval exits.
Continue tooling repair in parallel. The manual path bypasses optional guard
tooling, not the lock, either idle check, exact source/argv or other scientific
and legal gates. Log the manual evidence and any refusal. If the host is busy
or inaccessible, do not manufacture a launch or bypass the checks.

Confirm the historical queue runner process has actually exited on both
hosts using fresh process identity evidence; an empty inbox is insufficient.
No new jobs may be published to an unguarded historical runner.

### Verified timebox outcome

The guarded path passed real launches at01:08:22IST on trn21 and01:09:16IST
on trn22, before the01:24deadline. Both diagnostics completed successfully;
manual fallback was not required. Independent repair review is preserved at
`research/v2/verification/20260926-idle-guard-repair-independent/REVIEW.md`.
Historical queue processes were confirmed absent on both hosts at01:04:26IST
(commitd9a0a247), not merely idle or out of queued jobs. New launches continue
to use the common lock/checks; this success does not waive future preflight.

The full architecture follow-up launched01:32:37IST with the same trn22 lock
inode832589; its in-lock checks and worker identities are archived under
`research/v2/operator/20260926-trn22-arch3e79-full/launch/` (e2cd3697).

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
jobs are preserved; pending automated packets remain disabled until their
lock/check implementation is reviewed. The deadline's explicitly authorized
manual fallback is the narrow exception to that tooling-review condition.
The source, legality, disk, STOP, budget and
correctness gates remain unchanged.

By 00:57 IST the three existing automation prompts were updated in the app:
`trainium-safety-gated-controller`, `trainium-04-00-ist-self-audit`, and
`trainium-07-45-ist-morning-report`. Their schedules and task target did not
change. Operator was notified; pending-driver hardening and test receipts are
required before the next launch. Do not mistake prompt installation for a
completed native enforcement test.
