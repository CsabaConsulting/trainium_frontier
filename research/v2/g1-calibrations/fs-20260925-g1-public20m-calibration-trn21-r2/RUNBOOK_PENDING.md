# trn21 G1 retained-checkpoint calibration r2 — pending, not dispatchable

This is a **new label and source-staging request**, not a rerun of r1. The r1 Operator stopped before staging or evaluator launch because commit `347f81a8...` was absent in the unrelated host controller repo and a quiet remote fetch exited 128. Its HOLD receipt remains historical. Do not use r1's planned checkout or label. This r2 package proposes a host-local Git clone from the **actual source commit that trained the retained checkpoint**. It changes no model, checkpoint, evaluator, metric, or compiler settings, and authorizes no host action by itself.

## Why this source is better bound than r1

Historical confirmation-r2 launch evidence records trained commit `e7f11c38dc3b5e68b373322e10cfef580f1e8d36`. A 2026-09-24 read-only trn21 probe found that `/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2` is a standalone Git repository with **detached HEAD exactly at e7f11c38**, not shallow, and `git fsck --connectivity-only` returning zero. Its six dirty/untracked entries are all under `research/run-logs/`; the two required tracked source paths have no index or worktree diff against HEAD. `git show HEAD:<path> | sha256sum` on the host and locally both gave the exact trained/evaluator hashes below. The retained checkpoint was separately rehashed in the r1 Operator stage-HOLD receipt. **All live facts must be rechecked before r2 staging.**

| Fixed identity | Value |
| --- | --- |
| New label | `fs-20260925-g1-public20m-calibration-trn21-r2` |
| New clean target | `/home/ubuntu/trainium_freshstart_20260925_g1_public20m_calibration_trn21_r2` (must be absent) |
| Local-clone source | `/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2` (read-only; do not evaluate or copy its dirty worktree) |
| Exact trained commit | `e7f11c38dc3b5e68b373322e10cfef580f1e8d36` |
| Nested trained train.py | `submissions/dense6x1024_direct_review_matrixlr015/train.py`; SHA-256 `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d` |
| Trusted prepare.py | SHA-256 `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` |
| Retained final.pt | Path in `request.pending.json`; 469,831,009 bytes; SHA-256 `5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9` |
| Historical public20M reference | `0.9918353324459732` BPB; 20,971,520 tokens; 83,462,186 bytes; 57,379,240 nats |

No commit reachable from the r1 pin `347f81a8...` exists in trn21's separate `/home/ubuntu/trainium_frontier` repository (all 652 ancestors were probed). A full-history transfer or 43.67-MB one-commit capsule is therefore unnecessary when the exact **trained** commit is already on the host. Do not silently substitute a source: r2 explicitly re-pins to e7f11c38 and needs its own independent review. The local e7f11c38 tree has no tracked `.env`, PEM, private-key, credential-named, `.pt`, `.parquet`, or `.safetensors` path; a filename-only scan for common credential shapes returned no hits. A separate Verifier must still check for secret-transfer concerns before any clone.

## Stage only after independent static review

A distinct Operator may use the following shape after a Verifier approves this pending package and the Supervisor assigns **staging only**. It is a local `file://` depth-one clone of committed Git objects, **not** a filesystem copy of the dirty run checkout, network fetch, or change to its `.git` metadata. The source is detached; a local scratch test confirmed Git can clone a detached HEAD. The Operator must prove the same behavior on trn21 and fail closed if it differs.

```bash
G1_SRC=/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2
G1_DST=/home/ubuntu/trainium_freshstart_20260925_g1_public20m_calibration_trn21_r2
G1_PIN=e7f11c38dc3b5e68b373322e10cfef580f1e8d36
test ! -e "$G1_DST"
test "$(git -C "$G1_SRC" rev-parse HEAD)" = "$G1_PIN"
git -C "$G1_SRC" fsck --connectivity-only --no-progress
git -C "$G1_SRC" ls-files --error-unmatch prepare.py submissions/dense6x1024_direct_review_matrixlr015/train.py >/dev/null
git -C "$G1_SRC" diff --quiet HEAD -- prepare.py submissions/dense6x1024_direct_review_matrixlr015/train.py
git -C "$G1_SRC" diff --cached --quiet HEAD -- prepare.py submissions/dense6x1024_direct_review_matrixlr015/train.py
git clone -q --depth=1 --single-branch --no-tags "file://$G1_SRC" "$G1_DST"
git -C "$G1_DST" remote remove origin
if git -C "$G1_DST" symbolic-ref -q refs/remotes/origin/HEAD >/dev/null; then
  git -C "$G1_DST" symbolic-ref -d refs/remotes/origin/HEAD
fi
test "$(git -C "$G1_DST" rev-parse HEAD)" = "$G1_PIN"
test "$(git -C "$G1_DST" rev-parse --is-shallow-repository)" = true
test "$(git -C "$G1_DST" rev-list --all --count)" -eq 1
test ! -e "$G1_DST/.git/objects/info/alternates"
test -z "$(git -C "$G1_DST" remote)"
git -C "$G1_DST" fsck --no-reflogs --connectivity-only --no-progress
test -z "$(git -C "$G1_DST" status --porcelain=v1 --untracked-files=all)"
sha256sum "$G1_DST/prepare.py" "$G1_DST/submissions/dense6x1024_direct_review_matrixlr015/train.py"
```

Before the clone, the Operator must also authenticate the **committed** `git show HEAD:<path>` hashes (not just the source worktree), inspect the pinned tree for secret-bearing tracked files without printing any values, and verify the source clone has no object alternates. A source mismatch, target collision, dirty **required source path**, missing Git object, unexpected clone contents, Git remote/config credential concern, or failed shallow/clean/hash check is HOLD; leave partial staging untouched for diagnosis and do not launch. Do not modify or clean the retained source checkout or its final.pt.

## Remaining launch gates are unchanged

The new request is `PENDING_NOT_DISPATCHABLE` with `launch_authorized=false`, `expected_commit=null`, and unverified staged hashes. After staging, the Operator must fill a **separate finalized request** and receipt with actual staged HEAD, clean-tree and source hashes; repeat the retained checkpoint's type/size/SHA and unheld check; verify all six historical eval-cache directories, native Python, effective UID/HOME, pinned tokenizer/data cache, full `NEURON_COMPETITION_R1_*` namespace and absence of rank overrides; prove host identity, unique r2 label, no active train/eval/G2/Neuron holder and closed applicable breakers; then measure root free space **after the clone**. It must remain at least **9,126,805,504 bytes** (immutable 6.5-GiB floor plus 2-GiB evaluator headroom). No cleanup or threshold lowering is allowed.

Only a different Verifier may approve that finalized request. The Supervisor then grants a separate **one-label** guarded evaluator authorization. Its argv/environment in `request.pending.json` are the same as r1 except the new clean root: one fresh Python `prepare.py eval-public` process, trusted trained train.py and retained checkpoint, `--device neuron --batch-size 1 --seq-len 1024 --eval-tokens 20971520 --timeout-seconds 1200`, 1,800-s outer guard, historical optlevel-1 compiler flags, and six warm caches. Continuous disk monitoring and two stable quiescent postflight checks remain mandatory.

Successful calibration means rc 0, finite JSON over exactly 20,971,520 tokens / 83,462,186 bytes, no timeout or surviving process/holder, unchanged source/checkpoint hashes, and BPB independently recomputed from nats/bytes within absolute 0.0002 of `0.9918353324459732`. A failed or attempted evaluator dispatch consumes r2; any retry needs r3 plus diagnosis. Archive an Operator receipt and independent review before trn21 G1 can pass. **Do not** call this a new model result, append it to the model ledger, start A0 automatically, reset any breaker, delete artifacts, or submit.
