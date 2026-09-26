# Fresh independent review

Verdict: ACCEPT SOURCE/TINY SUITABILITY ONLY for helper SHA256 `65e7eb1a21b0e880ee2362461ff3664c4dbd52933816e5ec0684645c92cd75e8`. REFUSE real retained-checkpoint qualification, host/native execution, adoption, BPB or submission claims. Actual frozen INPUTS and full-size results are absent by design. No implementation defect was reproduced within the reviewed bounded scope.

I independently reconstructed the helper from the isolated full-context diff, authenticated the exact diff/helper hashes, read the immutable completed spec and baseline helper, and subsequently authenticated the two pinned source bodies supplied in the packet. I did not inspect implementer notes, other review verdicts, branches or broad history. No scientific source was imported, no full-size model/checkpoint was allocated or loaded, and no host, AWS, network, training or data operation occurred. My test files are independent reviewer artifacts, not candidate edits.

The 96-check independent suite passed under ordinary and optimized Python (3.14.4, Torch 2.13.0+cpu, one thread, about 245 MiB maximum RSS). It includes all six B1/B2/B3 and T256/T257 shapes on a 16-token tiny oracle; streamed masks independently constructed by scalar loops at lengths 4,7,65; masks with wrong zeros, negative zero, NaN, +infinity, wrong dimensions/dtype; parameter, nonpersistent-buffer and input mutation; submodule mode/hook changes; noncausal final-token dependence; batch/rowwise mismatch; mutable output aliasing; output shape/nonfinite failures; saved/loaded state identity; malformed config, state count and total count; exact cross-source bytes/dtype/shape; legal-token and nondegenerate-suffix requirements; and source/checkpoint/private-path/prepare/manifest/G2 authentication negatives. Tiny state was explicitly rejected as full113M.

An additional 23 source/boundary checks passed. The reference safe loader wrapper and candidate direct load both set weights_only=True. Both loaders use strict state loading, eval and freezing. The actual candidate constructs ordinary GPT with cos/sin buffers; reference retains its existing evaluation subclass and mask buffers. The six designated baseline functions are AST-identical. The new helper directly calls each source's loader and has no model/module monkeypatch assignments. An isolated synthetic pair-failure test verifies the real file reauthentication runs in finally and preserves the pair error. That test mocks only the reviewer process's manifest/pair seam to avoid production paths; it is not a full run or source-loader test.

Separate real small-file pre/post checks authenticated a reviewer-owned 22-byte fixture, then rejected a same-length change as FILE_IDENTITY. The original fixture content was `original tiny fixture` plus newline; its original SHA256 was `45470f257bbc1ca7e2baed4762b644739490b6c81a9341dce42fc699a4cb7618`. The current fixture is deliberately changed to `modified tiny fixture` plus newline, so the post rejection remains reproducible. No protected artifact was changed.

The helper's run sequence pins/authenticates the manifest, exact arm private checkpoint locations, source/request/G2/protected prepare artifacts, and real file SHA/byte lengths before loading. It requires accepted identical G2 metadata and standard comparison, compares saved state records and strict-loaded parameters, tests repeat/rowwise/cross-source bytes, and reauthenticates all files in finally after successful or failed pair/model checks. Full model shape safety also depends on the authenticated source's strict loader; malformed tiny shapes/counts are refused by the tests. CPU loader success, timing/RAM/disk preflight and the actual 65-tensor/113256724-parameter checkpoints were not tested here.

Both supplied initial failure logs remain part of the evidence: an initial future-use test returned FALSE_ACCEPT and an initial static test assumed a direct reference torch.load. I do not erase or reinterpret them as passes. My independent final-token future-dependence negative is rejected; my source review follows the actual reference safe-load wrapper. The raw supplied successful receipts are supporting evidence only, not my independent verdict.

The declared organizer-style prefix criterion remains <=1e-2, with exact prefix equality separately reported. A suffix-reversal probe is a bounded synthetic check, not proof of causality for every input. Any legitimate real numeric mismatch must still refuse exact equality; no tolerance was introduced. Real checkpoint identity, native loader/causality, cold1800+fresh20M, the required separate audit and human adoption acknowledgement remain open.

## Exact independent commands

```
/home/poojith-devan/venv/bin/python /home/poojith-devan/checkpoint-fresh-review-vgtBCW/review_candidate.py
/home/poojith-devan/venv/bin/python -O /home/poojith-devan/checkpoint-fresh-review-vgtBCW/review_candidate.py
/home/poojith-devan/venv/bin/python /home/poojith-devan/checkpoint-fresh-review-vgtBCW/boundary_review.py
/home/poojith-devan/venv/bin/python /home/poojith-devan/checkpoint-fresh-review-vgtBCW/check_immutability.py pre
# Reviewer fixture changed with apply_patch; see exact contents above.
/home/poojith-devan/venv/bin/python /home/poojith-devan/checkpoint-fresh-review-vgtBCW/check_immutability.py post
```

Raw outputs are normal-raw.json, optimized-raw.json, boundary-raw.json and immutability-raw.json in this directory. Every command exited 0; rejection cases are required successful negative tests. Source/diff authentication and file hashes are recorded in MANIFEST.json. The pre fixture command will now reject until the documented original reviewer fixture content is restored; the post command is directly repeatable. This is no launch authorization.
