# Independent successor validator-repair reassessment

Verdict: **PASS — limited source/input-only validator repair suitability.**

This verdict applies only to the exported successor identified as `0009d9de8e37317952c7ff4753b74f2607f57ed4`, with validator SHA256 `b6b73db9c5da424a9b8d3cb38d1666cea2dc08dae82acec41e3c6467d370afb5`. The original `54fd9a10d0beee22d2d8209a3dce9d3f200da2b0` verdict remains **REFUSE**, preserved without edits in the original independent review archive, commit `672b8a0c9c711d68294461d486c6eeb34d250faf`.

There is no candidate-native success, source adoption, launch authorization, full-run approval, performance result, or BPB claim here. No host, device, native runtime, install, network, training, or production action occurred. Candidate and input bytes were not edited.

## Authenticated scope

The review used only the fresh export and this reviewer's prior original packet/results. The supplied manifest SHA256 is `fcb2c40adbe5a33a93b4ded6edb544afdff9a42c1f17f86a84d7519a346fd6d2`. All 23 supplied listed files authenticate. Five deliberately absent files are excluded, not silently assumed reviewed: implementer README, repair notes, implementer tests, replay test source, and the duplicate original-reviewer log. The latter is already authenticated among the prior own outputs. No implementer conclusions were needed for the independent verdict. Repair diff SHA256: `2b77543fea36483f768824fca1b260c06dc0f42d23c6ad0616a76b976862ca0f`.

The original input manifest (`4c2248de3a8dd8fb91954d34ab7655808e7b0452d82b0bf48b9f09c1cfb1c3b4`) and all supplied original files authenticate again. The prior own output manifest (`14fa1b666709863d52a87123107fba05a47ed77ba0927a9a89c8ab18b24e4511`) and all its 41 entries authenticate again, including the independently generated baseline, counterexamples, source, and raw failures.

Exact byte comparisons prove unchanged `check_native.py`, `adapter.py`, `consumer_nki.py`, `reference.py`, `rank_entry.py`, `idle_guard.py`, `dispatch_driver.py`, proposal, and bound command. The driver is exactly its original text with one validator-hash replacement. The binding changes only its driver hash and validator support hash; it remains launch-disabled. No unrelated scientific or launcher review was rebuilt.

## Independent execution

Both completed runs, normal Python and `-O`, exited 0 and passed all 57 tests: 12 positive/legitimate-old-failure controls and 45 expected rejections. Raw logs are `normal-attempt2.log` and `optimized-attempt1.log`. Runtime is local Python 3.14.4, stdlib only. No Torch, Torch-Neuron, or NKI module was imported.

The four original frozen counterexamples all reject in both modes:

| Original counterexample | Successor rejection |
| --- | --- |
| Contradictory same-input candidate forward records | `SAME_INPUT_FORWARD_RECORD_DIFFER` |
| Contradictory upstream nonzero count | `FIXTURE_INPUT_RECORD` |
| Missing runtime identities | fail-closed `KeyError: 'torch'` |
| Malformed old mismatch actual record | `TENSOR_SCHEMA` |

The original own positive full archive still accepts: all four synthetic ranks, the complete 20-case set, nonzero FP32 sentinel, and two changed-tangent replays. This is explicitly synthetic validator evidence, not a native receipt.

The reviewer independently reimplemented scalar FP32 accumulation, integer/FP32/BF16 byte encoding, full input records, and nonzero counts for 23 fixture contracts (20 cases, two changed tangents, sentinel). The frozen reference supplied fixture definitions only; its dense accumulator and byte encoder were not used as the oracle. Every reconstructed contract matched. An independent synthetic weight lookup and BF16 nearest-even conversion also verified the positive forward records. Maximum numeric buffer size was 32,768 elements; no tensors were allocated.

The new strict field checks reject altered forward or upstream shape, dtype, finiteness, nonzero count, and SHA; blank, missing, wrong-typed, or wrong runtime versions; malformed old records and inconsistent exact/status/count/error claims. Missing rank/case/replay, sentinel-only, missing or zero candidate gradients, nonfinite candidate records, and old routes substituted for required candidate routes remain rejected.

The validator still retains separately recorded old-route all-zero gradients, NaN, positive and negative infinity, missing gradients, execution errors, early missing input observations, post-observation errors, dtype/shape mismatch, and signed-zero byte mismatch. None is promoted into candidate exactness, and none is required to become numerically correct before the candidate archive can be inspected. In particular, strict byte equality remains distinct from zero numerical difference for signed zero.

The first normal attempt failed in the reviewer's own test selector (`zero` instead of the frozen fixture name `signed_zero`), not in candidate code. Exact source `reassess_attempt1.py`, raw traceback `normal-attempt1.log`, and explanation `FAILED_ATTEMPTS.md` are retained. Only that selector was corrected; completed normal and optimized runs independently reran all checks.

## Runtime identity and remaining gates

The new runtime pins are Torch `2.11.0+cpu` and NKI `0.5.0+28631259367.ga768afa6`. Their values and receipt hashes were checked against the four previously supplied and authenticated raw FP32-first native receipts, not accepted solely from the new implementer summary. These are historical failure receipts, not registered-candidate success. The four raw receipt hashes are recorded in the authenticated runtime-authentication input. No claim is made here about independently redecoding an absent outer bundle.

Earlier adapter CPU evidence used local Torch 2.13 and remains source/CPU-only; it does not establish behavior of the required native Torch 2.11/NKI 0.5 environment. This successor review imports neither runtime.

All pre-existing independent gates remain: separately authorized Operator preflight, fresh process/device/STOP/disk checks, original lock identity, one active launcher, world-4/LNC-2 binding, 600-second TERM plus 60-second KILL lifecycle, all-outcome descendant/device/lock cleanup evidence, retrieval and independent review of actual sentinel/full20/all4 results, and later performance and comparable BPB gates. No automatic next action is authorized. The prior archived source/mocked mechanics evidence remains applicable by exact byte identity; it was never a substitute for a real machine preflight or lifecycle result.

The repair closes the four demonstrated validator defects without changing the scientific checker or suppressing the tested legitimate original failures. Within this narrowly assigned repair gate, no remaining blocker was found.
