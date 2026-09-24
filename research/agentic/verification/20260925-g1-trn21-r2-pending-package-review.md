# Independent static review: trn21 G1 calibration r2 pending package

Reviewed at 2026-09-24 22:52 UTC. Target commit:
`9b4014230295bb6bac835d0acd0e51e3356f6214`. Request:
`research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/request.pending.json`,
SHA-256 `9804d3f76ac8b7495c919539e94d7022609162fc0d9dfc31377e46fb060a691d`.
Runbook SHA-256:
`48d9b2e39e25e0281d3178a4a87733a421f9f13744e8bf9cfd7c7f9c9566be67`.

**Verdict: static package PASS for a separate Operator to attempt staging only;
evaluator dispatch HOLD.** This does not finalize the request, pass trn21 G1,
authorize an evaluator, or establish a new model result. I did not contact a
host, stage a host checkout, run evaluation/training, clean artifacts, or reset
a breaker.

## Independently checked evidence

- The target commit adds only the pending JSON and runbook. `jq empty` and
  `git diff --check` pass. A pattern/name scan of the two new files found no
  credential, token value, private key, or `.env` contents. The request
  explicitly has `PENDING_NOT_DISPATCHABLE`, `launch_authorized=false`,
  `expected_commit=null`, and unverified staged hashes.
- The historical confirmation-r2 launch and Operator receipt name the actual
  trained commit `e7f11c38dc3b5e68b373322e10cfef580f1e8d36`. At that
  commit, the **nested**
  `submissions/dense6x1024_direct_review_matrixlr015/train.py` hashes to
  `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`,
  and `prepare.py` hashes to
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  Both are ordinary `100644` blobs. The prior r1 pin `347f81a8...` contains
  the same two file hashes, but was unavailable in the unrelated host repo;
  re-pinning to the trained commit removes that source-transfer blocker
  without changing either file.
- The retained confirmation-r2 Operator receipt identifies the exact
  checkpoint path in this request, 469,831,009 bytes and SHA-256
  `5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9`.
  The r1 stage-HOLD Operator rehashed it on trn21 and found the file regular,
  unheld and matching. That past observation is not a current r2 preflight.
- The r2 request preserves r1's checkpoint, expected source hashes,
  evaluation settings, success criteria and all six warm-cache paths. Its
  environment is identical apart from the clean-root `PYTHONPATH`; its
  evaluator argv is identical apart from the two clean-root paths. Inspection
  of unchanged `prepare.py` confirms `eval-public --train-py --checkpoint`
  with `--device neuron --batch-size 1 --seq-len 1024 --eval-tokens 20971520
  --timeout-seconds 1200`; `timeout` bounds the fresh process to 1,800 s.
  The historical result's 57,379,240 nats / (83,462,186 bytes × ln 2)
  independently recomputes `0.9918353324459732`. This is calibration only.
- The new r2 label is distinct from archived r1. A local search found no r2
  label in controller run logs, experiment outputs or the untracked `runs/`
  directory outside this pending package. Host-side uniqueness is unknown.
  The disk gate is arithmetically correct: 6,979,321,856-byte floor plus
  2,147,483,648-byte evaluator reserve equals 9,126,805,504 bytes. The
  request's last archived 58,907,746,304 bytes matches the r1 stage-HOLD
  receipt, but free space must be measured again **after** the clone.
- A local Git 2.53 disposable test cloned a detached-HEAD source containing
  an untracked marker with the proposed `file://`, depth-one, single-branch
  shape. After the runbook's origin-remote and dangling `origin/HEAD`
  symbolic-ref removal, target HEAD matched, it was shallow with one
  reachable commit, clean, remote-free and alternates-free; `git fsck`
  passed and the untracked marker was absent. This proves the command's
  intended isolation on this local Git, not on the host. The conditional
  symbolic-ref removal is necessary: omitting it produced an invalid-ref
  `git fsck` error in the test.
- The pinned commit tree has 6,583 tracked entries (146,035,747 uncompressed
  blob bytes), no Gitlinks or symlinks, no tracked `.env`, PEM/key or
  credential-named path by the checked filename patterns, and no exact AWS
  key-ID, private-key-header, OpenAI-project-key or GitHub-token signature
  by the checked content patterns. This heuristic scan is **not** a guarantee
  that arbitrary committed content is secret-free. The clone must use only
  committed objects, never dirty run outputs or source Git config/credentials.

## Mandatory holds before any evaluator dispatch

1. A distinct Operator receives a **staging-only** assignment. On trn21,
   recheck source HEAD `e7f11c38...`, source Git integrity and no alternates
   or suspicious config/hook/credential redirection; rehash the two committed
   blobs and prove neither required worktree/index path differs from HEAD.
   Prove the new target is absent and the r2 label unused. Do not mutate the
   retained source checkout or checkpoint.
2. Perform only the specified local `file://` clone. Reprove the host Git's
   detached-HEAD/shallow behavior, one reachable commit, exact staged HEAD,
   clean status including untracked paths, no remote, no alternates, Git
   integrity and exact staged `train.py`/`prepare.py` hashes. Any divergence
   is HOLD; do not substitute another checkout, copy dirty files or fetch
   from the network.
3. Reauthenticate checkpoint type/size/hash and unheld state; all six warm
   cache directories; native Python; effective UID/HOME and fixed
   tokenizer/data-cache identity. The JSON's six-name
   `environment_must_be_unset` list is **not exhaustive**: the finalized
   preflight must also prove no other rank/distributed (`LOCAL_WORLD_SIZE`,
   `GROUP_RANK`, `ROLE_*`, `MASTER_*`, `TORCHELASTIC_*`), `PYTHONHOME`,
   `NEURON_RT_*`, or `NEURON_COMPETITION_R1_*` override, including tokenizer
   URL/SHA or step-log values. Capture names/identities without printing
   secret values. No cold-cache substitution is allowed by this request.
4. Verify hostname, no train/eval/G2/torchrun or Neuron/device/checkpoint
   holder, closed applicable breakers, and at least 9,126,805,504 root-free
   bytes **after staging**. Reinstate an explicit continuous floor monitor in
   the finalized launch evidence; the r2 runbook still requires it even
   though r2's `disk_gate` object no longer has r1's monitor boolean.
5. The Operator writes a separate finalized request and stage receipt. A
   different Verifier checks those live facts. Only then may the Supervisor
   issue a separate one-label guarded evaluator authorization. Preserve full
   output, exact metrics, hashes, disk trace and two stable quiescent
   postflight checks. Any attempted r2 evaluator dispatch consumes r2;
   retries require a new label and diagnosis. Do not start A0 automatically.

No host score, BPB improvement, submission readiness, or trn21 G1 PASS is
established by this static receipt.
