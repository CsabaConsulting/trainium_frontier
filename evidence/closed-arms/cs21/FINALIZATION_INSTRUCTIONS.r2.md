# e1bd-F1 audit handoff stages — additive revision 2

This document supersedes only the readiness/finalization wording in the frozen draft at `research/v2/package-prep/20260930-r162-e1bd-f1-export-draft` (EXPORT_MANIFEST SHA `27b4dd6ff877f01b399cbda2dcbce7ea1b9e672a1212d69f80ac8f16954cda8f`). It does not modify that draft, source25c5, receipts, checklist wording, launch gates, live state or any Downloads package. It performs no export, host action, audit acceptance or submission.

## Separate stages

| Stage | Meaning | Claude verdict needed? |
| --- | --- | --- |
| EXPORT_DRAFT_NOT_READY | Preparation; the actual F1 FULL receipt or independent integrity check is still pending. | No |
| READY_FOR_AUDIT | Exact F1 FULL cold/budget/score evidence has been collected and authenticated under the existing protocol; final package bytes and receipt references have an independent integrity check; outstanding compliance questions are explicitly listed. The package can be handed to Claude promptly. | **No** — seeking that verdict is the purpose of this stage. |
| ACCEPTED_FOR_UPLOAD | Claude has issued an explicit verdict for the exact F1 source/package and any conditions are resolved. Upload remains Poojith's separate action/authority. | **Yes** |

No stage transition is automated by these instructions. `READY_FOR_AUDIT` does not mean model/source/compliance acceptance, an official score, permission to upload or permission to launch another experiment. A source/CPU review, timing result or launch receipt alone does not complete the required actual F1 FULL evidence. An actual failed experiment remains failed under the original protocol.

## READY_FOR_AUDIT completion, using existing evidence only

1. Collect the actual exact25c5 F1 FULL receipt from the already-launched packet04d5. Authenticate the existing full-run proof: exact source/command/environment/seed, original binding/Guardian, six fresh training caches and measured cold<=900 s, charged-through-exit<=1800 s, finite/rank checks, final stable checkpoint, successful fresh public20971520-token/T1024 evaluation, protected files, terminal and descendant release. Record the actual score rather than a forecast. This is the original required FULL evidence, not a new native or source gate.
2. Carry the already-real independent source/CPU reviewe18, exact LNC1/unset eval receipts7c2f/02c7 (each0.9617131518505532), native60 raw6178/comparison589e, and separate Root within-noise dispositiona960. Keep their scopes distinct: timing executed derivative65c5, not exact25c5; public2M diagnostics used parent checkpoint3a0004, not the future F1 checkpoint.
3. Preserve R158's explicit historical second-receipt rationale. The original e1bd receipts may be cited as covering the second receipt because explicit-LNC2 training is unchanged under that instruction; label them as **historical e1bd runs**, not a second actual F1 run or an F1-only mean. One actual new F1 FULL remains required by R158.
4. Assemble a new final handoff revision, without overwriting frozen exports. Use exact source25c5, its exact diff/parent/command, actual raw evidence and an updated brief. Have an independent reviewer check final package bytes, manifest hashes, raw-member integrity and receipt/reference consistency. Do not equate this byte/receipt integrity review with Claude's compliance verdict.
5. Include the unchanged eight-item human rejection checklist and honest item-by-item evidence. Preserve unresolved or unmeasured items as explicit questions/FAIL-pending-adjudication where the human wording calls for PASS/FAIL. Those unresolved compliance questions do **not** prevent handing a valid, integrity-checked evidence package to Claude as READY_FOR_AUDIT. Do not manufacture PASS findings to make the label available.

When these existing evidence/integrity steps are complete, the package can state **READY_FOR_AUDIT — CLAUDE VERDICT PENDING**, with `claude_package_verdict: null` and `accepted_for_upload: false`. It need not wait for Claude to accept it before Claude can receive it.

## Report unknowns; do not add launch or handoff gates

R143 explicitly says Codex's earlier clean-box standard was rendered moot by the organisers' acceptance of A3. The exact relevant historical clause is preserved in `R143_CLEAN_BOX_CLAUSE.txt`. Consequently this revision does not require a new pristine-AMI trial, dynamic network trace, empty-eval-cache experiment, or cumulative-I/O instrumentation before launch or audit handoff. No such work is authorized here.

Report existing observations accurately: research environment identity, existing-cache versus empty-cache scope, immutable input hashes, checkpoint bytes, available output accounting and disk snapshots. If cumulative/transient/overwritten checkpoint/log/cache bytes, backup state or final residual inventory were not measured, say **UNMEASURED/UNKNOWN**. A final checkpoint size or free-space snapshot must not be mislabeled cumulative writes. Unknown telemetry remains visible to Claude; it does not silently become a new experiment, launch or handoff prerequisite.

R143's historical LaneB acceptance is not an acceptance of this F1 package. Nor does citing its clean-box-standard clause revive its later-disproved LaneB fallback statement. Preserve the correction that trusted logits evaluation bypasses participant CE2; missing LaneB native handles are not a proven SDPA fallback. The official crash cause remains unknown beyond the actual local diagnostic evidence.

## ACCEPTED_FOR_UPLOAD is a separate later state

Record Claude's exact source/package-specific verdict and conditions as a new artifact, preserving the reviewed package identity. Until that actual verdict exists, the upload stage remains false/null. Even an accepted package does not cause this agent to upload or submit; Poojith retains that action. This revision changes no runtime safety check, resource floor, stopping rule, scientific criterion or authority boundary.

Current state of this revision: **documentation-only preparation**. Actual F1 FULL completion and independent final-package integrity review are not supplied here; READY_FOR_AUDIT and ACCEPTED_FOR_UPLOAD are both false.
