# Fresh independent P0 review — PASS within bounded source/CPU scope

Completed 2026-09-26 07:02 UTC. Reviewer role only; no candidate edits, host contact, training, evaluator calls, data access, release creation, cleanup, submission, or approval-state changes.

Frozen input: `/home/poojith-devan/p0-sprint-fresh-review-8kEym4`.
Its `MANIFEST.json` SHA256 is `c28e772438afd2d8c46be423700dc6d8cdeea085e3465c901558c762cce356a3`, matching the assigned identity. All 68 listed file digests were independently verified. This is a single review of the supplied spec, source diffs, source, receipts and six disabled bindings; no implementer branch or private implementation notes were used.

## Verdicts

| Surface | Verdict | Bound identity |
|---|---|---|
| Recursive source port | PASS for source/CPU gate | `50fa6d144f19758dcdf9980e03f2302453174b956f0fa1a3e0845c4c0b72e6fb` |
| Batch/LR source port | PASS for source/CPU gate | `8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee` |
| Lifetime guardian change and host finish wiring | PASS for bounded source/local mechanics gate | `5df6897deeaea3435a963f79b912ce49f320d3a65624f60a89762249c67ecb6e` |
| Six frozen disabled host variants | PASS for internal binding/source consistency | Per-file identities in frozen manifest |

No concrete blocking failure was found. These verdicts do not establish native timing, numerical behavior on Neuron, completed experiment success, host idleness, launch authority, or adoption. Actual native initial-step timing remains to be observed by the separately authorized Operator. Generic G2 wrapper timeouts are not PASS evidence; the supplied receipt correctly says so. No new prospective scientific gate is added by this review.

## Independent checks

Executed with `/home/poojith-devan/venv/bin/python` (PyTorch `2.13.0+cpu`) and one CPU thread:

```
python -B research/v2/verification/20260926-p0-sprint-fresh-independent/check.py
Manifest: all 68 file digests match
Independent scalar reference exact: 40 cases
Changed scalar-to-tensor beta2 lerp: exact
Actual tiny model+optimizer 3-step base/recursive-OFF/batch-default: bit exact
Guardian reporting failure: independent idle called, failed result, original lock held through finish
PASS

python -B research/v2/verification/20260926-p0-sprint-fresh-independent/mechanics.py
Six frozen bindings: internal command/environment/source/budget consistency PASS
Guardian missing receipt + live owned child: child reaped, independent idle called, failed return, held lock PASS
Guardian busy/unknown independent idle: park path reached with original lock held PASS
```

Both commands exited 0. Test programs are retained alongside this report. Guardian test receipt/lock files belong solely to this review; receipt creation uses exclusive creation, so reruns need fresh review-only receipt names.

The independent scalar reference covers 40 step/progress cases including endpoints and out-of-range elapsed fractions, separate Muon/AdamW cooldown, momentum and beta2. The beta2 lerp check independently verifies the changed FP32 tensor scalar against the plain scalar operation. The actual-model identity check constructs an eight-layer, width-128, vocabulary-256, sequence-32 model from each source, seed 42, and runs real backward/optimizer updates for three identical token batches at progress 0, 0.5, 0.98. Losses and every model state tensor are bit-identical (zero tolerance). This is bounded CPU evidence, not the full 113M model or distributed/native evidence.

AST comparison confirms all model/forward/eval definitions are unchanged from the supplied audited `e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43` base. Recursive changes only `MuonAdamW`, schedule functions and `main`, adding `muon_beta2`; batch changes only `main`. Recursive defaults OFF, preserving the original scalar beta2 path; batch defaults remain 131072 and 0.015. Intended batch arms are 262144/0.015 and 262144/0.021. Their microbatch arithmetic is valid at four ranks, pack 32 and context 1024 (accumulation increases from 1 to 2).

Guardian review confirms that reporting errors do not authorize release by themselves: owned descendants are quiesced/reaped, the original held inode is checked, the host callback performs a fresh ps/neuron-ls/fuser check before attempted reporting, helper descendants must be absent, and the original lock is checked again. A report error returns failure. Busy/unknown idle evidence reaches the terminal park path while the original lock remains held. The local live-child test verifies cleanup/reaping despite a receipt-construction failure; busy/unknown tests replace the infinite park with a test exception to inspect the retained lock. They do not claim real-host device evidence.

All six bindings have matching host-driver command/environment values, package source hashes, 1800-second science budget, 900-second startup cap, seed 42 and fresh-process public-20971520-token evaluation. Launch argv preserves context 1024, LNC2/four ranks and each exact requested science arm. All remain disabled with no RELEASE.json and require actual host facts/normal Operator checks before any separate release. Supplied protected evaluator/harness hashes and existing cache paths remain bound; their remote bytes were not inspected in this source-only review.
