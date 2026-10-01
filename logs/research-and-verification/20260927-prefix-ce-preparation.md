# Round13 C5 — prefix-token CE preparation (UNVERIFIED; not launch authority)
Timestamp: 2026-09-27 09:03 UTC / 14:33 IST; bounded out-of-cycle source review, not a cadence cycle.
Priority remains below every ready Round13 arm; no queue, cursor, brief, model, tokenizer, evaluator, or data change.
Local parent inspected: `research/v2/proposals/20260927-r13-gate-adamw-c239/train.py`.
Parent SHA256: `500f80fec9eac2a42d6f0eae646b0e91643d46c4524ee3722e332434751fd368`.

PR337 maps the immediate next-token target to its longest proper byte-prefix that is itself a vocabulary token; absent prefixes are ignored, not replaced by another label.
It adds weighted CE on the SAME logits, without another head; this is not probability summed over a prefix class.
Its nonpersistent vocabulary table uses GPT-2; a legal local adaptation must derive a read-only table from the unchanged official 8192-token vocabulary, never import GPT-2.
Upstream weight is .25 for the first third, linearly .25→0 over the second third, then zero; masked positions contribute no prefix loss/gradient.
The kernel sums active weighted objectives, without local MTP's division by total weights. Source: [PR337 diff](https://github.com/KellerJordan/modded-nanogpt/pull/337/files).

Local lines 615–639 instead compute (CE(k1)+w2·boundary-corrected CE(k2))/(1+w2), with w2=.5→0 linearly over the first two-thirds.
k2 predicts the next-next token; prefix CE supervises a shorter spelling of k1. Therefore it is not redundant, but its competing target changes the same softmax gradient (inference).
PR author reports stacking with MTP and worse results when replacing MTP; compatibility on this parent remains UNVERIFIED. [PR337 discussion](https://github.com/KellerJordan/modded-nanogpt/pull/337)

Shape implication (local inference): retain [32,1024,8192] logits and existing parameters; add fixed [8192] lookup plus [32,1024] labels/mask and scalar weight, not a new projection.
A straightforward third local CE adds forward/backward loss work; upstream CUDA fusion is not a portable Neuron implementation.
Compile time, memory, step cost, full-runtime cost, and BPB gain: UNKNOWN. Fixed-shape runtime weights could avoid schedule recompiles; native proof is absent.
Mask invalid targets only after substituting a safe in-range CE index; do not pass -1 into the current NKI target path.
Keep local softcap15 and targetless forward (lines 959–960) unchanged; prefix supervision is training-only, with no prefix remapping or auxiliary CE in scoring.

External evidence only: upstream 8×H100 N=10 reports baseline1390 steps 79.273s/loss3.27874 versus prefix1375 steps 78.489s/loss3.27848.
Those unequal-step CUDA/GPT-2 loss/time results are not comparable Trainium BPB measurements and do not justify a local expected gain. [PR337 results](https://github.com/KellerJordan/modded-nanogpt/pull/337)
STOP before implementation: C5 still needs an exact authorized local normalization choice (add to normalized MTP versus normalize the combined objective), progress-clock binding, and missing-prefix averaging convention.
These choices alter gradient scale; silently copying the upstream weight would not define one unambiguous local arm. No implementation or operational recommendation is made.
