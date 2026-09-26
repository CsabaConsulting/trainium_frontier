# Existing fixed-source support — local inspection only

The pinned rehearsal already accepts an absolute training script pathname in
the package launch string. No prepare.py, evaluator or accounting edit is
needed for that capability. This observation does not release a source-path
switch or change either prepared warmdown packet or live job.

The complete historical harness object
`00f2ca7f79696e65ce52458f074dd13b871da90c:research/scratch/submission_rehearsal.py`
was hashed locally to the current pinned value
`7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b`.
Lines 62–88 parse the exact manifest-matching launch string with shlex, keep
the unique sandbox/output/steps paths, and invoke the parsed command. Lines
100–111 retain the official fifth-boundary allowance capped at900 seconds
and charged-through-exit calculation. Lines72–78 retain ordinary fresh eval
of the package's byte-identical train.py at context1024/B1/public20M.

Executed W1 used an absolute path ending
`out/research-sources/sha256/3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c/train.py`
in `research/v2/proposals/20260926-stablepath-warm-observation-trn22/w1/package/launch-command.txt`.
Its unchanged rehearsal hash is recorded in WARM_BINDING.disabled.json and
its actual terminal archive. Its actual fifth was120.3379416820826 seconds;
that narrowly missed the declared120-second target. This was a five-step
operational observation, not a full1800 quality/eval qualification.

The existing W1 helper
`research/v2/proposals/20260926-stablepath-warm-observation-trn22/identity_checks.py`
has pinned SHA85eef95494f6a4da41ee5284d5089b5fe6df7c2555c5bea5bc16e86965d7a8f4.
It checks an immutable0555 directory containing only0444 train.py and an
unchanged colocated prepare.py, exact SHA/device/inode/integer-nanosecond
identities, no symlinks/hardlinks/import overrides, the selected executed
pathname, package equivalence and the protected prepare/harness. The W1
driver checks identities before and after the child. It does not claim
observed worker import origin or authenticated native cache hits.

The current full1800 driver can forward such a launch string, but it only
authenticates the package source. It does not currently authenticate a
different fixed executed pathname. A safe future mechanical binding would
therefore reuse the existing W1 source-identity checks under the current
lifetime guardian, preserve its exact original lock through train/eval,
and keep current full budgets/resource floors. The old W1 driver is not a
replacement for the current guardian and its five-step limit is not reused.
Creating a new source-SHA directory would have to preserve older immutable
directories; no rewrite, cache clearing, prepare edit or SDK change is needed.

A first4ec8 fixed-path run would not have established cache evidence merely
from existing8c6 directories or W1's different3e79 result. Cache reuse and
startup remain measured outcomes. Current warmdown commands retain relative
train.py exactly as approved; stage/launch remain disabled pending Supervisor.
