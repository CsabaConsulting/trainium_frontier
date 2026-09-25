# Independent static PASS — trn22 r10 disabled stage r3 controller root

Verdict: **PASS for the exact disabled source correction only; dispatch HOLD.**
Reviewed subject `b3ec7406f0e7a3df6848698876e2106305dd283f`, relative to
corrected r2 `020bfa7ebd048f947d5207cbedaf028a29a4b159`. This review did
not contact trn22 or another host, consume a marker, transfer a bundle, make
a checkout, launch training/evaluation, reset a breaker, clean artifacts, or
authorize a submission.

The r3 diff changes only `REQUEST.HOLD.json`, `RUNBOOK.md`, `SHA256SUMS`,
`operator_once.py`, and `test_incremental_stage.py` in
`research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/`.
`remote_incremental.py` and `e65-to-r10.bundle` are byte-identical to r2.
The remote source SHA-256 remains
`da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`;
the bundle remains 109,800 bytes, SHA-256
`b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e`.
`git bundle verify` confirms sole prerequisite
`e65a3097a5b00f6178188a935bb2acb9a5042c6c` and HEAD
`22bfbab5a77d6feb670d2d737504f8b70b08ddc1` (15 descendant commits).
The target Git blobs are the intended `train.py` SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
`prepare.py` SHA-256
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
and launcher SHA-256
`1be19bc31d2bdcde8abe9fe90cc1ac884d6efd5f7be197d7bdbfe64c257dcff7`;
the model/evaluator were not changed by r3.

The repaired operator SHA-256 is
`a5fc2ebac3520f425d5ab0bdfb7ae0091a2e5db59b152dd8cd39eb67cff5beda`,
matching `REQUEST.HOLD.json` and `SHA256SUMS`; all eight manifest entries
verify. The request still says `dispatch_authorized=false` and
`source_stage_authorized=false`. Its trn22 source, new target, remote bundle,
label `fs-20260925-trn22-c5c-a0-r10-incremental-stage-r1`, and one-attempt
marker are unchanged. The enabled request, independent final PASS, and
Supervisor assignment are absent. Both normal and `-O` direct entrypoints
exit nonzero with `HOLD` before SSH or a marker; the named local marker was
absent at review time.

The earlier independent HOLD correctly found that r2 read the stale
`/home/poojith-devan/Documents/ChatGPT/Trainium/research/v2/state.json`.
r3 now reads the active
`/home/poojith-devan/Desktop/Trainium/research/v2/state.json` and checks
STOP under the active Desktop root, former Documents root, and package root.
The Desktop and Documents state files were observed as distinct regular
inodes (3584213 and 6075244); the former is not a controller-state fallback.
This is point-in-time local evidence, not a future launch preflight.

Independent checks on the exact r3 worktree:

- Supplied offline suite: **37/37 PASS** under normal Python and **37/37 PASS**
  under `python3 -O -B`; `git diff --check` clean.
- Separate adversarial probe: a tripped active Desktop breaker with stale-safe
  Documents state makes `stage_once` HOLD before SSH identity, SSH phase, or
  marker. A STOP only under the former Documents root also HOLDS before those
  seams. The operator never reads the stale Documents state.
- The real active Desktop state at review time had
  `launch_authorized=false`, `hosts.trn22.launch_allowed=false`, and both
  relevant breakers untripped. The four active/former STOP paths checked were
  absent at that instant. None of this substitutes for fresh host/local checks.

**Execution-root boundary:** the operator does not read the local package
worktree's root `train.py` or `prepare.py`. It reads its own exact source,
bundle, request/review/assignment and the absolute active Desktop state;
`HERE.parents[3]` is an additional STOP root. The checked-in suite uses that
Git root to inspect *pinned commit blobs*, not the working-tree science file.
Nevertheless the operator has no code-enforced requirement that it execute
from the original clean r3 branch/worktree rather than a cherry-picked
archival controller copy. In particular, the active Desktop working-tree
`train.py` was SHA-256
`52ac9e6560a40592255e3499f4d42ac6e7073ea82da1b428e03dc984e6d4baa8`,
not the staged target's `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
This static PASS does not validate an archival-root
science checkout or authorize using that working tree for a scored run. A
future enabled request/final review must identify and independently check
the exact clean package execution root and its bytes, and then separately
verify matching r10 Git HEAD on local and both hosts before scored controls.

Still HOLD: separately reviewed live r10 final-scan timing feasibility;
an exact enabled request, independent final review and one-attempt Supervisor
assignment; fresh STOP/breaker/host/source/target/lock/process/Neuron/disk
preflight; independent stage-result and G2 source-identity review; and a
separate native A0 request with paired monitors and containment. No BPB
result follows from this static PASS. The documented same-UID/alias and
post-final-scan races remain residual risks, not silently solved here.
