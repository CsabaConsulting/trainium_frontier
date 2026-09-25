# trn22 A2 exact-size CPU G2: read-only readiness snapshot

This is an Operator **read-only** machine snapshot at
`2026-09-25T14:43:08Z`. No A2 run root, marker, source, tool or result was
created; this is neither independent assignment review nor launch authority.
The assignment remained `PREPARED_HOLD_INDEPENDENT_ASSIGNMENT_REVIEW` with
`launch_authorized: false` when checked. Every gate must be freshly repeated
before any separately authorized one-use stage/launch.

- Required prior trn22 c5c A0 train/eval: driver has
  `EVALUATED_2026-09-25T14:16:06Z`; fresh eval receipt is
  `mode=eval`, `returncode=0`. The separate 116188 cold smoke has
  `TRAIN_FINISHED_2026-09-25T14:36:26Z`, `train.exitcode=0`, and a
  quiescent postflight; its five-step/cache verdict was INCONCLUSIVE and
  does not itself bar this CPU-only table gate.
- No torchrun/train/eval/G2/neuronx-cc process was listed; `fuser
  /dev/neuron0` was silent; the pre-existing host `device0.lock` was
  freely acquired non-blockingly. No host STOP file existed at the trusted
  checkout. Local controller STOP files were absent and controller
  `breakers.global_correctness.tripped` and `breakers.trn22.tripped` were
  both false.
- Exact A2 r2 unique root
  `/home/ubuntu/trainium_c5c_a2_exact_g2_20260925_trn22_r2` was absent,
  as were its `g2.json`, `tmp/`, `ATTEMPT_ONCE.json`, `launcher.log`,
  staged `source/train.py` and `tool/preflight_dryrun.py`; no pathname
  matching the exact label appeared under `/home/ubuntu` to depth three.
- Root free bytes: 54,810,624,000 (required >=34,359,738,368 planning
  floor and >=6,979,321,856 operational floor). `MemAvailable`:
  129,164,263,424 bytes (required >=68,719,476,736). Root free inodes:
  31,657,166 of 32,636,928. The private TMPDIR would use this same root
  filesystem; no smaller scratch filesystem was selected.
- Trusted unchanged host `prepare.py` SHA-256:
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  Local candidate/source SHA-256:
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
  Local exact G2 tool SHA-256:
  `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`.
  Local packet SHA-256:
  `23b357834922bf410f46d78d68f91b578536930c5528983317eb1988bdd6178f`.

Open protocol caveat flagged to Supervisor: the shared controller checkout
has unrelated, pre-existing untracked `research/run-logs/...dispatch.json`
and `runs/`, so it is not globally git-clean. The separate Operator
worktree is clean and contains the exact SHA-pinned candidate and tool;
the packet/assignment live in the controller checkout. No unrelated file
was touched. A distinct reviewer/Supervisor must decide source-origin
acceptability under the CPU-only packet before any write.
