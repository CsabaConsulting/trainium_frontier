# Recursive architecture source and reduced-receipt audit

Technical disposition: **PASS for the inspected source and reduced numerical scope; fresh independent source review remains pending.** Candidatefe0969fd/SHA75b944f7 is research-only, not the clean submission file. No native, full113M CPU, cold, quality/adoption or submission gate is passed here. No source/host/tensor mutation or model execution was used for this audit.

Independently inspected the entire35-add/3-remove source diff against exact3e79/b4d83b5d and exact6ef7/a4aa4351 schedule reference. GPT, GPTConfig, loader, `_SubmissionEvalGPT`, checkpoint writer and submission training adapters remain AST-identical to the architecture baseline. The three schedule functions are AST-identical to the adopted reference. The loop's only optimizer changes are schedule selection by group kind plus the already-existing beta2 tensor-scalar path under ON. No LR base value, batch, architecture, data, timing/reset, loss or weight-update mathematics was changed by this port.

`audit_source.py` passed **2,835 independent checks** normally and under `-O`. It executes actual extracted schedule functions and actual training-loop group LR assignments, not supplied Implementer tests. The numerical grid covers step0/1/2/3/299/300/600, warmup0/3/300 and eleven elapsed points including negative time, both sides of5% onset,35%, midpoint, endpoint and over-budget time. OFF matches baseline Python floats exactly; ON matches adopted-reference outputs exactly. An independently written high-precision Decimal piecewise LR reference has462 cases, largest error1.1102230246251565e-16, below2e-15 reassociation allowance. Independent momentum/beta2 algebra matches to2e-15. This tolerance is for the independent differently-associated scalar reference, not a relaxation of the exact ON/adopted or OFF/baseline comparisons.

The audit exercises actual group-kind LR assignment and flat-table precedence, compiled beta2 tensor-vs-Python-scalar selection, old checkpoint missing-toggle defaultFalse and saved OFF/ON restoration. The scalar fill helper itself is unchanged. Native compiler specialization/recompile behaviour is **unmeasured**, not proved by selecting a tensor branch. Source numeric examples and exact source/raw hashes are in SOURCE_RESULT.json.

Independently parsed rawd1ceed83 `tiny-g2-{reference,default,off,on}.json`. All four runs are three-step, four-rank8x128, synthetic1x16/rank, CPU/no-compile with a prepare stub. The reference/default/explicit-OFF twelve losses,65 retained rank0 parameter digests and ownership plans match exactly. Other ranks do not include individual final digest maps; their raw receipts report no cross-rank mismatches and65 changed parameters. This distinction is preserved: no additional per-rank digest or intermediate-gradient evidence is invented. ON is finite/rank-consistent, reloads deterministically atB1/T16, and all65 final rank0 digests intentionally differ from OFF. These are reduced synthetic64-total-token/update observations, not131072 production-token updates or full113,256,724-parameter evidence. T16 is not native eligible-shape correctness.

Preparation failure preserved: the first verifier script run incorrectly assumed that each rank's JSON contained `final_digests`; it failed with `KeyError: 'final_digests'` before completion. Inspection showed those maps are retained only for rank0, while other ranks expose consistency summaries. The verifier was corrected to check the actual available scope rather than fabricate stronger raw evidence; subsequent normal/optimized runs pass. Candidate and raw receipts were not changed.

Default/OFF model-state equality does not imply serialized checkpoint-file equality: the candidate adds savedFalse schedule metadata. ON changes schedule/update outputs by design. Existing reference schedule adoption is not new-architecture quality evidence. The unchanged source loader remains the research per-row wrapper, distinct from the clean-source route question; no clean-file source refusal is addressed by this port review.

FullCPU packet57511815 is originally bound totrn22. A root request fortrn21 therefore required a separately frozen mechanical rebind; this mismatch was reported immediately, not silently relocated. The later65b0d5ee successor is subject to its own binding audit. Neither packet is released by this source note.

Reproduce, no Torch import:

```text
python3 research/v2/verification/20260926-architecture-recursive-independent/audit_source.py
python3 -O research/v2/verification/20260926-architecture-recursive-independent/audit_source.py
```
