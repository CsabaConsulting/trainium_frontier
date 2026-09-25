# Next-slot CPU packet review and assignment

Supervisor/reviewer: root, distinct from the candidate Implementers and host
Operators. This receipt authorizes the already-requested development CPU gates,
not a new human approval ceremony or a native performance claim.

## Architecture on trn22

Reviewed packet297c8044, source93ae1f92 / commit1e21c3d7, against the original
architecture review and exact timer reviewe9109dec. Read source manifest,
host-specific proposal, commands, unchanged G2 harness profile and legal native
shape. Independently reran check_packet.py: PASS_STATIC_PACKET_ONLY. Source,
reference, prepare and harness Git-object hashes match; candidate ON is8x1024,
partial attention0,2,4,6,7, no tables, seed42, LNC2/world4. Native argv and
CPU ON science argv match. CPU runs use synthetic2x256 with full parameters.

Assigned trn22 Operator to stage immutable bytes and run four serial CPU probes
only after its current Recursive train AND fresh20M eval complete: candidate
ON, candidate OFF, exactc5c and exact6ef7/Recursive-OFF. Require64GiB RAM,
6.5GiB disk plus footprint, fresh STOP/idle/source checks and the common lock.
The independent numeric review remains pending. Source93ae is not in the
current queue registry; a later native launch must use a reviewed addition
or the proven manual route after pausing the old queue between jobs.

## FP8 on trn21

Assigned trn21 Operator to stage exact921271d7 / commit50a90137 and run CPU
ON/OFF plus a matchingc5c reference after its current train AND eval finish.
Original source/reduced CPU reviewc15a0ac6 and exact timer reviewe9109dec apply.
Full6x1024 parameters, four ranks/three steps, synthetic1x16; raw losses,
parameter digests and ownership must match on the default-OFF comparison.

Timing packet30177711 supersedes the old packet's proportionality HOLD.
Independent reviewb880cc4a (canonical a43695e9) accepts one native component
check, one matched OFF100/ON100 pair (20 warmup +80 measured whole steps),
and an ON trained-checkpoint frozen-forward check. Repeats require concrete
measurement uncertainty, not a valid below-threshold result. No native FP8
launch is authorized by this CPU assignment. The full quality run still
requires at least3% measured whole-step speedup, with all correctness and
actual charged-time gates unchanged.

## Prepared manual routes and offline integration, 17:15 UTC

Both Operators have prepared the proven manual paths for the new source hashes,
so the existing queue registry is not a reason to build another launcher or
delay a job whose CPU receipts pass. Native remains pending those receipts.
The host queues stop only at the owned between-job boundary after fresh eval;
the six scalar fallback jobs per host remain intact for serial hand-back.

trn21 finished Recursive training at17:09:50 UTC with3,669 steps and entered
fresh20M evaluation. Its owned pause was placed at17:11:43 UTC after verifying
the exact staged ON/OFF/c5c CPU profiles. trn22's owned pause was placed at
16:59:49 UTC and likewise does not interrupt its active train/eval.

Architecture manual binding11e5f64a and FP8 immutable binding-v2 a46e562c
include optional SDK exit-only graph compilation CSV exports. Root and both
Operators independently inspected installed metrics.py c1702b4b: collection
is already unconditional, the settings affect only exit export. No source,
compiler/cache behavior, measured step window or charging is changed. Both
FP8 arms use the same logging policy. A zero compile time is not proof of a hit.

Combined architecture+FP8 sourcea2f6b188 (ab7a007e) is preserved on its own
branch and has independent source/reduced-CPU PASS14b035cc. This is offline
readiness only. It neither replaces6ef7 nor preempts the standalone hardware
checks. No native/performance/BPB result exists for the combined candidate.

The architecture's possible step-window exemption is an optional pending human
decision. Existing decide.py is unchanged; this does not prevent its run,
but a step ratio outside the existing window cannot be adopted without the
requested rule change. Do not relabel it as a throughput arm to evade the gate.

## Continuation

Both Operators must leave current work uninterrupted, use only their owned
between-job pause when ready, preserve all six fallback proposals and restore
the fallback route if blocked. No second launcher, new AWS action, cache flag,
evaluator change, cleanup or protected-artifact deletion is authorized.
Neither candidate is marked native-ready before its remaining gates pass.
