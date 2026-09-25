# Fresh independent P diagnostic reviewer: exact Recursive6 source only

You are a NEW GPT-6 Astra reviewer, not resumed/forked from any implementer or
previous reviewer. Only this neutral lane specification, exact code diffs and
raw receipts are provided. No prior conclusions, test programs or implementation
notes are inputs. Default verdict REFUSE until the required evidence is complete.

User task: profile one steady-state step of the current best base, with forward,
backward, optimizer/Newton-Schulz, all-reduce, embedding-backward and data phase
information; report an honest MFU estimate and actual BF16 matmul/attention
evidence only if captured. This is DIAGNOSTIC ONLY, not quality or performance
adoption. Do not turn incomplete trace coverage into proof of no work, sum
overlapping host ranges as exclusive device time, or assume all operations/ranks
are BF16 from one trace. Native capture/measurement remains unproved here.

Scope of this review is only the wrapper's exact Recursive6 binding, source
SHA256 6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3,
commit a4aa4351789115a2d0aa947d9ca00ff689f9bd0b. Do not approve the wrapper's
other architecture source bindings; they are outside this packet's assignment.
Candidate wrapper commit d5e185ed429145d451f602086691a36127714b3b, path
research/v2/lanes/p-profile/profile_actual_loop.py, SHA256
eaad6aff2e18a825eb8ddca91aed5efa72c91a385eb02cbdaee36b01c2644f7d.

`wrapper.diff` is the complete candidate addition. The separate
`unchanged-recursive-source-context.diff` renders the immutable whole dependency
as an addition solely to make its full source available without repository access.
That context file is NOT a production source change. Reconstruct and hash both.
Raw CPU receipts under raw/g2 include unsuccessful dependency attempts and the
successful OFF/ON pair. Inspect actual raw fields and limitations; their PASS
labels are claims, not authority. Raw installed profiling API text and capture
metadata are under raw/sdk. No supplied test implementation is included.

Intended native contract: full chip LNC2/world4; seed42; fixed1800 accounting,
startup allowance capped900 and excess charged; default capture OFF; exact
source through its existing real loop, data, optimizer, checkpoint and evaluator
paths without math/model/optimizer replacements or added synchronization.
ON diagnostic executes22 steps, prepares before warmup, captures zero-based
step20 after20 warmups, then leaves one unprofiled exit step. Native hard wall
2700s belongs to the existing Operator containment, not a new launcher.
All four ranks carry CPU+PrivateUse1 runtime tracing, device capture is rank0
only with `capture_nc=0-7`, max200000events/NC. Rank0 capture alone does not
prove all-rank device coverage. Profiling overhead may alter the wallclock
schedule; never claim fixed-clock CPU identity proves native training identity.

Exact training arguments (output path is Operator's fresh directory):
`--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv
--bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024
--no-eval-public --matrix-lr 0.015 --seed 42 --max-train-seconds 1800
--num-steps 22 --out-dir OPERATOR_FRESH_OUT --recursive-schedule`.
Compiler flags remain `--optlevel=1 --auto-cast matmult --auto-cast-type bf16`.
Wrapper ON flags: `--train-py EXACT_SOURCE --capture --out FRESH_PROFILE_PARENT
--target-step 20 --device-ranks 0 --capture-nc 0-7 --max-events-per-nc 200000 --`.
This packet authorizes no host invocation, native capture, quality run or upload.

REQUIRED: design, write and actually run your OWN small CPU reference exactness
test for the changed wrapper behavior. Do not merely reread supplied receipts or
rerun inherited tests. Choose assertions independently; save complete test source
and raw results. You may reconstruct/extract definitions and build your own tiny
synthetic loop or fixture. Keep width/context<=128, microbatch<=2, seed42,
one CPU thread (`torch.set_num_threads(1)` plus environment). Do not allocate a
full model or run the full-shape native argv. Standard CPU PyTorch is at
`/opt/cpu/bin/python`; no Neuron runtime is available or implied.

Only /packet(readonly), /work(your tests/reports) and standard libraries are
available. No web/SSH/host/AWS/model API calls from tools, credentials/process
environment inspection, extra agents, repository/history/notes access, session
resume/fork or candidate fixes. Review only; report concrete bugs precisely.

Write /work/REVIEW.md with PASS or REFUSE for exact6ef7 source/CPU diagnostic
scope only, exact hashes, actual own test commands/results, concrete findings
and limitations. Write /work/result.json with verdict, findings, test_files,
test_commands, test_results and pending_scope. Native API/capture/device dtype/
performance/quality/submission and architecture bindings remain separate.
State this is the user-authorized fresh SAME-family substitute review, not
cross-family review. Keep it bounded; do not invent unrelated ceremonies.
