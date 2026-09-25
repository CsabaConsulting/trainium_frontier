# trn21 c5c A0 native request — disabled Implementer skeleton

This package preserves the exact science for the unique trn21 A0 label, but
**cannot launch**. `NATIVE_REQUEST.HOLD.json` has `dispatch_authorized=false`,
`host_action_authorized=false`, no executable launcher or command, and null
receipts for G2 numerical review, a source-matched safety checkout, live
monitors, fresh host preflight, and Supervisor one-attempt authority. It is
an Implementer handoff for a different Verifier, not a self-approval.

The independently accepted stage-r2 checkout is e65 at a new private path;
its `train.py` and unchanged `prepare.py` have the exact c5c/5807 hashes.
The historical r4 safety review passed a **different** 446b source locally.
Both commits have identical `train.py`, `prepare.py`,
`fresh_start_sync.py`, and `run_experiment.py` bytes, but e65's old launcher
does not check `dispatch_authorized` or require a live paired monitor. It
has no `fresh_start_monitor_interlock.py` or disk-monitor scripts, and its
guard uses an older process-group signal path. Thus a request with a full
command must **not** be pointed at the accepted e65 checkout. Its source
checkout PASS is not a native safety PASS.

The exact safety-source delta from e65 to reviewed 446b is six files: three
modified (`fresh_start_launch.py` 822d4317→9488efa9,
`fresh_start_guard.py` ac6f7d23→df902bec, and
`fresh_start_host.py` c4442b91→b34c368c) and three added
(`fresh_start_monitor_interlock.py` f2fcecd4, base monitor 91dae714,
paired monitor 9e2d6ed2). The full hashes are in the disabled request.
The 446b source Verifier accepted **local safety code only**; the trn22
monitor bindings are host/label-specific and cannot be reused on trn21.

The shortest safe path is a **new**, independently staged and reviewed clean
446b checkout on trn21, leaving the accepted e65 checkout untouched. A
separate Verifier must decide whether e65's eventual exact four-rank CPU G2
numeric result carries to 446b after comparing trained/eval bytes, argv,
effective environment, and harness. Neither matching `train.py` nor this
Implementer assessment is itself G2 acceptance. Later a final trn21-specific
native request and request→base→pair monitor bindings must be independently
reviewed, with the read-only boundary observer and two live monitors proved
ready before a one-attempt Supervisor assignment. No old label may be
relaunched after a failed attempt.

The last archived checkout postflight (03:17:38 UTC) supplies the warm-cache
device/inode identities and two protected checkpoint sizes/hashes in the
request. They are evidence to **recheck fresh**, not authority to clean or
prune. The 10-GiB prelaunch requirement, 6.5-GiB operational floor and
floor-plus-2-GiB stop threshold are unchanged. An independent disk monitor
and an unconditional separate postflight must remain part of any final plan.

The offline tests validate hashes and reject in-memory attempts to turn this
draft into a runnable request. They do not contact a host, run training,
establish numerical equivalence, or issue launch authority.
