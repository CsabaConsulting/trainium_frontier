# A2 r1 terminal evaluation and research-only r2 launch

## r1 archived, ineligible for table-quality decisions

The original detached driver recorded fresh evaluation finishing with exit0
at **2026-09-25 16:01:51 UTC**. Trusted public evaluation returned:

- BPB **2.850923606746586**, context1024, exactly20,971,520 tokens.
- 924.4022069479106 evaluation seconds; timeout1200; `eval_over_timeout: false`.
- Harness exit0 and wall937.6386117920047 seconds.

This is the measured BPB of the **five-step startup-truncated checkpoint**.
It is not a valid full-budget table-quality result, must not be compared as
an A2 loss, and cannot select the no-table branch. The training source,
timer and evaluator were not altered. The terminal eval/harness/driver logs
are archived exactly under `a2-r1-final/`, with hashes in
`a2-handoff-retrieval.json`. The checkpoint was neither downloaded nor deleted.

## Automatic common-lock handoff to r2

At **16:01:54.160978 UTC / 21:31:54 IST**, existing queue PID3050509 created
the new r2 attempt, with fresh preflight at **16:01:54.168078 UTC**;
at **16:01:54.172316 UTC**, it started
training timeout-wrapper PID3051234. This follows the prior driver/eval
completion by approximately three seconds. No second launcher, overlap,
process interruption, SDK switch or timer change was involved.

At **16:03:47 UTC**, live process evidence showed torchrun **3051235** and
all four native workers **3051250–3051253**. Prior evaluation PID3048593 and
driver PID3043794 were absent. The new run had initialized the exact A2
model and four optimizer ranks; no completed training boundary was present
in that initial snapshot. This is **RESEARCH_ONLY**, not submission eligible.

The source is exact c5c, the immutable proposal and resolution hashes match
the activation receipt, and launch argv are archived verbatim under
`a2-r2-launch/`. Native full-chip LNC2/four ranks, seed42 and1800-second
budget remain unchanged. Every required existing train/eval cache root is
reused without cleanup. Disk at the live observation was50,682,077,184 bytes.

At **16:04:21 UTC**, kernel lock evidence identified queue PID3050509 as
the exclusive FLOCK holder of device0.lock inode832589. The timeout wrapper
and torchrun inherited its open lock descriptor, as designed. The separate
`device-lock.json` retains exact evidence; the four worker processes themselves
need not inherit it for the torchrun parent to keep the phase protected.

No BPB or cache-speedup claim is made for r2 until it finishes training and
its separate fresh-process public20M/context1024 evaluation. Any resulting
record needs independent result review before scientific use.
