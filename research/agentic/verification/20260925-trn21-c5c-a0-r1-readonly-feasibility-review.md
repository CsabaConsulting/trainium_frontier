# Independent Verifier: trn21 c5c A0 r1 read-only feasibility

**Verdict: PASS for exact-source stage planning only; dispatch HOLD.** Reviewed the
Operator receipt introduced at `8fcdab3d` and preserved byte-for-byte on the
controller at `ed8acc12`, SHA-256
`90b77b5f220da9e72dbe1932b27b87b953ba85865793ea9ebc3318997f330df1`.
This review supplies no source-stage, G2, training, evaluation, AWS, breaker,
deletion, or leaderboard authority. The reviewer did not contact a host.

## Independent checks

- The receipt parses as JSON and adds only its own file; `git diff --check`
  passes. All nine cited controller-document hashes match the files at bound
  controller commit `c8ac28a21bfa65b69eb199e471626bd6809c66b3`, including
  the proposal, gate contract, runbook, ratification, static review, G1
  Operator and independent result receipts, `state.json`, and `g1.json`.
- Exact source commit `e65a3097a5b00f6178188a935bb2acb9a5042c6c` exists
  locally. Its root `train.py` hashes to
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and root `prepare.py` to
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The active controller's `submissions/research_v2/train.py` has the same c5c
  hash. The stage must still prove these identities at the new host checkout.
- The recorded read-only observation at `2026-09-25T00:12:54Z` reports trn21
  root-free bytes `58,676,293,632`, an unheld Neuron device and shared lock,
  no matching train/eval/guard/G2 process, unchanged identities of two
  protected checkpoints, six identified warm-cache directories, and absent
  provisional stage/output paths. These are point-in-time Operator
  observations, not a fresh pre-dispatch preflight.
- The source search is expressly limited to Git repositories directly under
  `/home/ubuntu` and one level below: 144 inspected, zero matches for the
  exact commit. It does **not** prove that the commit is absent everywhere on
  the host. The raw remote scan output and exact scan script are not archived
  here, so that count and host state cannot be independently reproduced from
  this receipt. The narrow conclusion remains sound: plan a separately
  reviewed clean transfer/stage of the locally available exact bytes rather
  than silently reusing an unverified host checkout. Recheck all live claims
  and exact paths before any creation or dispatch.
- The receipt explicitly records `launch_authorized=false`, no source copy,
  G2, training, evaluation, host-path creation, deletion, breaker reset, AWS
  action, or submission. Its next gate demands independent stage review,
  exact-environment four-rank G2 and numerical review, a final SHA-bound
  request, fresh host/disk checks, observer/monitor, and a separate one-shot
  authorization. No A0 control BPB or model improvement is claimed.

**Limit:** This PASS accepts the receipt solely to choose the next planning
step. It neither verifies the live host independently nor licenses the
transfer/stage. If later stage or preflight evidence conflicts with this
snapshot, stop and resolve the conflict; do not relax the 10-GiB prelaunch
gate or the 6.5-GiB operational floor.
