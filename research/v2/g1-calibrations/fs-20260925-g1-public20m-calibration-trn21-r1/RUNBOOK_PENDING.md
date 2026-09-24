# trn21 G1 retained-checkpoint calibration — pending, not dispatchable

This package prepares **only** a fresh-process, single-process public-20M evaluator
calibration on trn21. It contains no new model, training proposal, run result, or
launch authorization. The exact request is in request.pending.json. Its
expected_commit is deliberately null, and launch_authorized is false. A separate
Operator must stage and reauthenticate it; a separate Verifier and Supervisor must
review and authorize a finalized one-run request. The Implementer did not access
either host.

## Fixed references

| Item | Exact reference | Evidence |
| --- | --- | --- |
| Planned label | fs-20260925-g1-public20m-calibration-trn21-r1 | Absent from this checkout's run/dispatch records before drafting; host uniqueness still unknown |
| Pinned source commit | 347f81a8572d6af8876067cd7c777a341e019929 | Clean isolated local worktree |
| Trained candidate | submissions/dense6x1024_direct_review_matrixlr015/train.py; SHA-256 116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d | Local hash and historical trn21 Operator receipt |
| Trusted evaluator | prepare.py; SHA-256 5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb | Local hash and G1 source inventory |
| Retained checkpoint | /home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/train/out/final.pt; 469,831,009 bytes; SHA-256 5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9 | Historical Operator receipt and 24 September read-only disk packet; **must be freshly checked** |
| Known score | 0.9918353324459732 BPB, 20,971,520 tokens, 83,462,186 bytes, 57,379,240 nats | Historical full20m.metrics.json; calibration reference only |

The original trn21 training checkout's root train.py was a different file.
The request therefore names the **nested trained 116188 file** by absolute
path, with the clean staged repository root in PYTHONPATH. Substituting root
train.py, S0, c5c, a newly ported file, another checkpoint, a distributed
evaluation, or default 2M evaluation invalidates this calibration.

## Stop conditions before any evaluator process

1. Human volume expansion and guest filesystem growth must be complete. The
   most recent archived trn21 free space was 6,901,800,960 bytes: below the
   immutable 6.5-GiB operational floor of 6,979,321,856 bytes. This plan
   reserves another 2 GiB for evaluation, requiring **9,126,805,504 bytes
   free after staging**. If fresh df is below that, stop. Do not delete
   anything, lower the gate, or infer that a historical cache is still present.
2. Operator must stage a **new clean checkout** at the pinned source commit,
   without copying secrets or altering the retained training checkout.
   Before any launch, fill expected_commit with the actual clean staged HEAD,
   verify that it equals the pinned commit, and hash the exact nested
   train.py and prepare.py. Both must match the table above. Confirm the
   hostname is ip-172-31-34-85 and the checkout is fully clean, including
   untracked paths.
3. Reauthenticate the retained checkpoint as a regular unheld file with the
   exact byte count and SHA-256 above. The six warm eval cache directories
   named in request.pending.json came from the historical
   eval-rehearsal.json; their **present existence is unknown**. Require all
   six to exist as directories. If not, stop and prepare a new reviewed
   cold-cache/disk plan. Do not clear or recreate these old caches under this
   request. Confirm no inherited rank/distributed variables, alternate
   data/tokenizer paths, or scorer step-log override; capture the effective
   environment without printing secrets.
4. One-job gate: no torchrun, train.py, prepare.py eval, CPU G2, other Neuron
   work, device holder, or conflicting controller lease on trn21. Both the
   host-specific and global breakers must be closed. Confirm this label is
   absent on the host and from controller dispatch/run logs. A consumed label
   must never be launched again.
5. An independent Verifier must review the finalized request, hashes,
   headroom projection, guard/quiescence path, and host preflight. The
   Supervisor then issues a **separate, one-label authorization**. This
   pending plan itself conveys no launch authority.

## Exact evaluation shape after those gates

The command and environment are recorded as argv and named variables in
request.pending.json. The evaluator is a fresh Python process, not torchrun.
It invokes the unchanged prepare.py eval-public with:

    --train-py /home/ubuntu/trainium_freshstart_20260925_g1_public20m_calibration_trn21_r1/submissions/dense6x1024_direct_review_matrixlr015/train.py
    --checkpoint /home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/train/out/final.pt
    --device neuron --batch-size 1 --seq-len 1024 --eval-tokens 20971520 --timeout-seconds 1200

The 1,800-second outer timeout bounds the guarded process; it does not waive
the inner 1,200-second evaluator timeout. Keep the historical optlevel-1
compiler flags, the six verified warm-cache paths, PYTHONPATH at the staged
repository root, and continuous disk monitoring. If free space approaches or
crosses the immutable floor, stop the owned guarded process according to
the approved containment protocol and record a failed calibration. Do not
silently retry or reuse the label.

## Completion and evidence

The Operator must retain the launch, preflight, process, full evaluator JSON,
log, postflight, and two stable quiescence/hash checks. Success requires
exit code zero; no leftover process/device holder; source and checkpoint
hashes unchanged; exactly 20,971,520 evaluated tokens and 83,462,186 bytes;
finite total_nats and val_bpb; eval_over_timeout false; and independent
recomputation of total_nats / (num_bytes × ln(2)) within absolute 0.0002
of the historical 0.9918353324459732 BPB. A separate Verifier reviews the
Operator receipt before research/v2/g1.json can mark trn21 PASS.

This is a **CLI and environment calibration only**. It is not a new
candidate score, a same-host A0 control, a table result, or evidence that
the official private score improved. Do not append it to the model-result
ledger, delete the checkpoint, reset a breaker, authorize another job, or
submit anything.

## Static preparation scope

The Implementer checked the local source SHA-256s at the pinned commit and
the historical receipt, G1, and disk packet. The pending request parses as
JSON and hashes to b39f3e5573793f5b9eb9d559f0d977c17d1f3f2dde304edf27c806f2222e9a4a.
No host was contacted. Fresh
disk, checkout, checkpoint, cache, label-uniqueness and process facts remain
to be established by the future Operator and independent Verifier.
