K2 minimal native index repair: isolated independent review
Authority: FINAL SPRINT; one bounded review, no extra review loop.
Default REFUSE until evidence complete. No SSH, native test, or launch.
Only candidate.diff, this lane spec, and attached gate receipts are allowed;
do not read implementer notes, other worktrees, source branch, or shared chat.
This directory is not the implementer's branch/worktree.

Old frozen scientific source SHA63925519ac6c0aab1a1d598e96fef3f4413fbd083d9b6d177901a2e972060db5.
New scientific source SHAa3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71.
Full source diff contains exactly two line changes in the dtype guard/error:
allow two-dimensional torch.int32 or torch.int64, reject other dtypes/dims.
No added cast, altered metadata operation, gather, backward or OFF-path.

Real unmodified prepare.py uses dtype=torch.int32 for cpu_inputs at479,
device inputs at481, and row copies at514. Native first attempt on all4ranks
failed beforestep0 at the int64 guard; dtype itself was not printed.
Full native failure archive SHA52c017298afa972fb5731e479f1c3fa4eb72ee5ae7ed8eb39c067c41de4976de;
exit1,0steps,26.57568083005026s, no checkpoint. Complete quiescent postflight.
Original one full source review CPU/OFF3/ON3 and25 finite cases PASS SHA
2ddb87ac3c38897abf306e323d06c747b0db41423e2f6126ede94e98cadcf502.
This is evidence for unchanged code only, not new native support.

Independently execute your OWN small CPU reference test for at least one
exactness claim. Suggested: derive stable per-row positions/counts from an
ordinary Python reference and compare the diff-context metadata function for
the same valid duplicate IDs represented int32 vs int64. Check boundaries
0 and8191, tensor dimensions, invalid float/bool types, snapshot isolation;
no validation data. Check gather and sequential FP32 summed gradients from
the returned positions match a straightforward reference.
Report CPU outputs/command/hashes and scope limits. Native lowering/timing
remains unknown; no whole-step pass from CPU.
Use /home/poojith-devan/venv/bin/python -B. Put your own code and receipts only
in this isolated directory. Deliver REVIEW.md with PASS_FOR_ONE_NATIVE_RETRY
or REFUSE and actual evidence. <=5min target; no fabricated range-check claim.
Author CPU gate receipt will be added to this directory when complete.
