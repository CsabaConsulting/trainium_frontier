# Scoped independent review: PASS source/schedule checks; REFUSE completed CPU qualification

Reviewed only the supplied review-input and raw directories. Candidate reconstructed from the full-context diff has SHA256 `4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea`; reconstructed parent has `36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb`. No source edits, host access, external data, native operation or launch authority.

## Findings

The source change is confined to the proposed default-False toggle, CLI/save registration, three schedule functions, compiled Muon complementary-beta2 selection, and group LR routing. The three function AST hashes exactly match the neutral proposal's donor hashes. Both LR calls consume schedule_elapsed; every non-Muon kind receives the non-Muon route; no_lr_decay retains precedence. Recursive ON precedes and bypasses legacy LR quantization; OFF retains the original schedule branches. The startup helper is byte-identical. The full diff introduces no timer, reset, budget, stopping, step-boundary, model, loader, evaluator or checkpoint-body changes.

Executed independent_reference.py with OMP/MKL/OPENBLAS threads set to one: 2,304 LR comparisons plus momentum/beta2 comparisons, and 448 actual extracted routing executions across all four toggle combinations. Cases include specified progress boundaries, warmups, step boundaries, process ages and offsets, all/non-Muon/flat groups. The post-fifth selected elapsed uses the existing rounded-monotonic reset formula. Five required wrong outcomes are distinguished: raw elapsed only for non-Muon LR, zeroing charged startup, adjusting again after step five, ignoring flat precedence, and scalar beta2 when Recursive ON. These are focused semantic rejection controls; they are not a claim of end-to-end mutation testing of a native launcher.

Executed independent_tiny_update.py using torch 2.13.0+cpu, one thread, seed 42, only a 1x3x4 parameter tensor. This runs AST-extracted unchanged production Muon math with a differentiable matrix loss for six updates crossing startup exclusion five. Exact loss, gradients, weights, momentum and second moments agree OFF versus base and ON versus independently supplied formulas in both clock modes. It does not run the full model/main/AdamW or prove native graph capture.

Independently compared supplied raw receipts, without reading their test implementations: default and explicit OFF match parent in both clock modes; ON matches supplied reference arms in both clock modes. Six arm pairs x four ranks x six updates have exactly equal recorded loss and per-step gradient, parameter and optimizer-state digest structures. Receipts explicitly record startup exclusion five, and ON captures device-tensor beta2. Rank-zero checkpoint checks report successful deterministic reload; final_digests and checkpoint_check are rank-zero-only fields. Raw hashes establish receipt integrity, not independent execution of those full-model tests.

The dedicated checkpoint raw receipts report equal model-state/output hashes for base, old missing-key load, OFF and ON, with the expected absent/False/True metadata distinction. They use a synthetic trained tiny model and test-only EVAL_MIN_SEQ_LEN=16; serialized-container-byte equality is correctly not claimed. These receipts explicitly set organizer_causality_or_native_claim=false. Deterministic repeated logits do not establish prefix causality or evaluation state invariance.

## Remaining gates and verdict limits

PASS for the exact-source and independently executed bounded CPU scope is sufficient evidence to advance this source to the separately bound full-size CPU G2 review/testing stage, provided the Supervisor supplies its own applicable CPU-only assignment and machine checks. The missing causal/state-invariance/loading qualifications must be executed there and accepted independently; this statement does not mark those checks complete, waive them, or authorize a host/native launch.

REFUSE unconditional completed CPU qualification under this proposal's minimum requirement 4: supplied evidence does not establish the trained-checkpoint causal/prefix tests and evaluation state-invariance rules. No such experiment was independently executed here. A distinct bounded trained-model causality/state-invariance receipt is required before marking that minimum complete. This is an evidence gap, not a discovered candidate source defect.

Exact-size 113256724-parameter CPU qualification, independent mechanically bound cold packet, true child cold-five <=900s (target870), checkpoint/through-exit accounting, native capture, subsequent full quality/fresh20M and research adoption all remain unpassed. Parent cold results do not transfer. No submission authority.

## Provenance and preserved failures

Reviewer session/task: `/root/clock36_recursive_fresh_reviewer`, independent child of `/root`; role Verifier. Review environment date: 2026-09-26. Python 3.14.4; torch 2.13.0+cpu. Exact serving model identifier is not introspectable through the available child tools/context and is not independently attested. The Supervisor reports that its controller tool receipt binds this spawn to `fork_turns=none` and `model=gpt-6-astra`; that is controller-reported provenance, not a runtime introspection result.

Preserved all supplied failed receipts through the input hash manifest. Preserved three own failed reference attempts: guessed entrypoint `train` instead of actual `main`; missing nonzero-rank final_digests; missing nonzero-rank checkpoint_check. These were test/schema errors corrected without source changes. The successful raw artifacts retain the narrower executed scopes and unsupported claims.

The independent-reference-raw.json contains SHA256 hashes for every supplied input and raw receipt. OUTPUT_SHA256.txt binds reviewer-created artifacts.
