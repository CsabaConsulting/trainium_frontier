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

Timing packet17bc5748 is under independent proportionality HOLD91913505:
its source pins/CPU recipe pass, but mandatory multiple native conditioning
runs are being replaced with one matched whole-step OFF/ON pair and a repeat
only if ambiguous. No native FP8 launch is authorized by this CPU assignment.
The full quality run still requires at least3% measured whole-step speedup.

## Continuation

Both Operators must leave current work uninterrupted, use only their owned
between-job pause when ready, preserve all six fallback proposals and restore
the fallback route if blocked. No second launcher, new AWS action, cache flag,
evaluator change, cleanup or protected-artifact deletion is authorized.
Neither candidate is marked native-ready before its remaining gates pass.
