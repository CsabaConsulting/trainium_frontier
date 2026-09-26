# Actual retained-checkpoint independent review

ACCEPT_ACTUAL_CPU_CHECKPOINT_SYNTHETIC_ONLY. My independent script passes 188 checks, exit0. This closes the declared CPU retained-checkpoint synthetic qualification only. A clean cold-five remains separately gated; this report is not launch, native, BPB, adoption or submission authorization.

Exact command:

```
/home/poojith-devan/venv/bin/python /home/poojith-devan/checkpoint-fresh-review-vgtBCW/review_actual_helper.py
```

Code: review_actual_helper.py. Final raw output: actual-helper-final-raw.json. The supplied bundle and postflight hashes match c6243eac08765729e39976dd6294ec847753c372f185159997e621bd1f72f9d7 and a4b28ec8656563a659b432756bc63267f6ef6a0c71f1febcb9c7a0483992d27a. All three added live snapshots were authenticated by their supplied SHA256, recorded in my raw output.

I independently compared all 65 full parameter records across the two ordinary loaders: exact keys, shapes, dtypes and full SHA256 match; element products sum to113256724. Their digest prefixes also match the accepted G2 pair. All per-shape initial parameter/buffer records remain identical within each arm; cos/sin records agree across arms. The reviewed host helper enforces loaded/saved bytes and all parameter/persistent/nonpersistent-buffer/input checks around every forward. The receipt contains before-state and successful enforcement markers, not separate raw post-state tensors; I do not claim to have independently rehashed checkpoint/model tensors locally.

For each B1/B2/B3 x T256/T257 case I regenerated only the tiny seed7 token inputs, row slices and reversed suffixes. Every recorded input hash matches, and suffix changes are nondegenerate per row. Every original/repeat/row/suffix output record agrees across the two arms (30 paired call records); the repeat, concatenated rowwise and full cross-source comparison records agree byte-for-byte with zero maximum/relative difference. All twelve causal-prefix receipts have maximum difference0 and exact prefix equality. This checks complete recorded output hashes and reviewed helper arithmetic; raw logits were not supplied and no full-size model was loaded by me.

I recomputed both entire8192-square expected causal masks independently, streaming scalar-formula little-endian float row bytes with only small scratch memory. Both full SHA256 match the actual mask receipts. Full mask has33558528 zeros and33550336 negative infinities; short4096-window mask has25167872 zeros and41940992 negative infinities. NaN/+infinity counts are zero. The candidate is its exact ordinary GPT with cos/sin buffers only, reference its existing evaluation subclass; neither report uses a dtype override. Host runtime was Python3.12.3/Torch2.11.0+cpu, one thread.

The exact17 file byte-count/SHA256 pins match frozen INPUTS, before, after, prior independently accepted G2 retained identities and final retrieval. The staged helper SHA matches the previously reviewed65e7eb1a version. The actual command binds the exact manifest SHA,600s timeout and60s kill grace. Retrieved helper/outer/terminal exit codes are0. RAM/disk footprint preflight meets the declared floor.

All five dispatcher/flock/driver/timeout/helper identities are now verified against two raw live snapshots at03:53:58 and03:55:15. PID, PPID, PGID, SID, starttime, command and working directory are unchanged; they form the exact recorded parent chain in session3177118. The03:59:37 snapshot is empty. Independently filtering the final254-unique-PID census against all five PID/PPID/PGID/SID values finds no remaining related rows; explicit PID checks report all five absent. Device/fuser results are idle and the original device66305/inode596823 lock is available. Under-lock before/launch/post receipts identify that same original lock and driver.

The initial reviewer script's empty-stderr assertion failed. I inspected and preserved that failure: stderr is a single Torch/Neuron aten::gather.out registration warning for PrivateUse1, with no traceback. The revised reviewer script records this warning and checks actual normal exit/numerical evidence; the candidate/helper was never edited. See actual-helper-initial-failure.txt. Initial live evidence was prose-only and acceptance held pending raw rows; actual-helper-initial-raw.json preserves that intermediate limitation. Raw snapshots subsequently resolved it.

No host/network/AWS/model/data operations, production edits or implementer-branch access were performed by this reviewer. The accepted outcome is CPU synthetic exactness/purity/causal-prefix evidence for these retained checkpoints and shapes only. It cannot establish native kernel behavior, long-context/general causality or official BPB. Earlier failures and other pending gates remain preserved.
