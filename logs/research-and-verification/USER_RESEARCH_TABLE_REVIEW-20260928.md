# Supplied jump research — checked against our code and logs

Checked 2026-09-28, approximately 16:50 IST. Research assessment only: no
candidate edit, host execution, queue change, submission, or new BPB result.
Attachment SHA256: `508671df16c244e527e63b564d0b81e9179715a044742b109fd5d430dd495ee4`.
Current comparison mean: **0.9766558936725349**, CLEAN+CONV b585; shared-path
cleanup source **6497d163d930275dfbc0bb494fbb546456478e3751119dcc1baf918f5638f9d3**.

## Verdict

There is a plausible, differentiated hypothesis: **one last-layer bigram value
table, using FP32 accumulation, on the current eight-layer stack**. Its quality
gain and native compatibility are unmeasured. The attachment does not establish
a likely 0.005 gain, and most of its suggested repairs were already in A1.
The table lane is not reopened or a full run authorized by this review.

I located the quoted source and logs locally after Poojith requested that I
resolve provenance myself. No further file request to Poojith is necessary.
The named `poojithdevan4D/trainium-research-context` is our context repository;
the matching measurements are our historical experiments, not independent
competitor replication. Detailed exact-source/receipt cross-check:
[table evidence](20260928-r49-user-table-crosscheck.md).

| Attachment claim | What our evidence actually establishes | Disposition |
|---|---|---|
| A1 may have zeroed both table and projection | Exact c5c source zeroed tables but randomly initialized projections. Gates started neutral, not branch-disabling. | Not an identified bug; no checkpoint RMS measurement was made. |
| Switching from Embedding to `table[idx]` fixes A1 | A1 log405 explicitly reports `index_gather=True`; source251/2110 executes the Parameter's direct indexing, despite the Embedding container. | Already present; do not repeat as a new speed fix. |
| RMSProp, flat LR and separate table clipping are missing | All were enabled in A1's actual launch. Separate clipping did not mean no table clipping. | Mostly existing recipe, not new corrections. |
| Tables gained about 0.012 on a competitor's Trn2 | Our Sep12 six-layer640 logs contain the measurement. At context1024: 1.049507 → 1.036938, delta −0.012569; training context2048, about288M tokens. The quoted −0.011994 used evaluation context2048. | Real local historical evidence, but an older model and evaluation protocol, not current fresh-process20M proof. |
| Our old A1 loss can be dismissed | A1 fresh public20M/context1024 was0.9986312057 vs0.9914739940, delta+0.0071572117. It used two tables and reached pass2. | Valid negative result; neither initialization failure nor memorization is demonstrated as its cause. |
| Scorer-shaped long rows are an untried gain | Current source uses `PACK_FACTOR=32`, packs32768-token rows and reshapes into32 context1024 chunks. | Already implemented; proposed P4 is not a new alignment correction. |
| Residual placement caused NCC_IMGN901 | Actual R42 failures injected into residuals; R41 value-path probe was superseded before launch. A1 value-path compiled with NKI conv enabled. | Different placement is a testable compiler hypothesis, not an established root cause or guarantee. |
| BF16 table gradients are acceptable | The standing FP32 decision applies to table master parameters, accumulated gradients and RMS state; cast only gathered activations to BF16. | Do not import the report's BF16 Parameter/reduction recipe. |

## What remains genuinely different

The current eight-layer base has no hashed table. Adding just one bigram table
would differ from A1's two-table six-layer experiment in both backbone and table
count. FP32 accumulation is also a real change from historical A1 BF16 arithmetic.
These differences are sufficient to describe a distinct research hypothesis;
they are not enough to forecast its effect or erase the prior negative result.

If selected under the standing R49 proposal/review rules, the clean first test
would be one **524288×128 FP32** table, zero-initialized, random128→1024
projection, neutral learned gate into layer7 V, direct indexing, and the common
causal train/eval forward. Keep current gate-optimizer policy, ordinary trained
weights, flat table RMSProp and separate clipping explicit. No K2, residual
injection, trigram, custom backward, data or evaluator change. Preserve common
parameter initialization and verify collision/microbatch FP32 gradients on the
instance CPU before a native compile/timing probe. A timing pass is not a BPB win.

One table is **67,108,864 parameters**:256MiB each for FP32 weights, dense gradient
and second moment, **768MiB/rank before temporaries**. The attachment's134MB BF16
gradient payload becomes268MB in FP32; its5–15ms estimate cannot be retained
unchanged. Actual whole-step cost, cold startup<=900s and memory must be measured.
No production CPU numerical tests were run on the laptop for this review.

Do **not** stack epoch2 freeze into the first diagnostic. It is an untested
quality intervention, not a proven repair. The packer's epoch field marks a
rank-local document-buffer refill; buffered tokens can cross passes, and ranks
do not share an exact first-repeat boundary. A later freeze experiment needs a
defined coordinated boundary and evidence from the unfrozen arm. Fixed step1770
is not justified by the current ramp. Likewise, early training-loss superiority
must not replace final fixed-budget public20M selection.

## Packing and external-source checks

Root read current `train.py:1375–1390` and the unchanged `prepare.py` loader code.
Training's32 long-row subdivisions already include mid-document starts;
evaluation concatenates BOS-prefixed documents into a contiguous stream. This
does not assert an empirically measured BOS fraction or identical distributions.
The prior actual packer audit measured0.0978341% cropping; that is distinct from
BOS alignment. No validation tokens were read or evaluator changed.

[Recursive's public solution](https://raw.githubusercontent.com/recursive-org/first-steps-toward-automated-ai-research/main/nanochat_autoresearch/solutions/optimized_from_karpathy.py)
does contain value-path hashed tables and RMSProp, but its0.9109 header describes
a GPU solution with many simultaneous recipe changes. Its model defaults include
context2048; runtime context is supplied by its imported library. This is not an
isolated table ablation or an attainable-score proof for our1024-context Trn2 run.

[Nanochat's own log](https://raw.githubusercontent.com/karpathy/nanochat/master/dev/LOG.md)
records removal of bigram embeddings at larger scale after their small quality
gain failed to survive wall-clock comparison. This is relevant counterevidence,
not proof that every small-model table experiment must lose.

[AWS repository issue1381](https://github.com/aws-neuron/aws-neuron-sdk/issues/1381)
is a user's reproducible report about other indexing/compiler defects. It does
not report NCC_IMGN901 or identify our failing operation. It motivates numerical
checking; it does not validate the attachment's conv/norm diagnosis.

## Decision and result boundary

Keep one-table FP32 value injection as the specific research lead, not a promised
jump or launch-ready arm. Reject the redundant packing/init/lookup fixes and the
BF16-gradient regression. Do not reopen closed scatter kernels or the exhausted
R46 sort repair. New source still needs a completed proposal and independent
checks before a scored run. No automatic queue change follows this attachment.
Best official remains human-reported **0.9768**; no new result was produced here.
A0.005 improvement from the base mean would be0.9716558937 public, which is not
itself a guarantee of beating the human-reported0.9700 official top10 line.
