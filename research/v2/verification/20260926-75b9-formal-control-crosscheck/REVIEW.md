# 75b9 raw score and genuine control cross-check

Disposition: **DISCARD is justified by the unchanged selector**, not a training/correctness failure.

I independently parsed retrieved evaluator output and recomputed BPB. Candidate75b9:57,253,516 nats / (83,462,186 bytes × ln2) = **0.9896621160468637**. Genuine same-host6ef7 control-r2:57,270,336 nats / the same byte denominator = **0.9899528600562257**. Both actually evaluated20,971,520 tokens at context1024/B1 and exited0 without eval timeout.

Exact delta is **-0.0002907440093620206** (selector rounds to -0.0002907). It does not reach the single-host nomination threshold -0.0006. architecture_schedule_stack_arm is not HIGH_PRIOR (only table_arm/data_order_arm are), so the ambiguous band yields DISCARD. This is not the two-host -0.0003 rule, a step-window gate, or evidence that the measured score is worse. No immediate replicate/adoption follows.

The genuine control was prospectively proposed as same_config_control_repeat, retained kind=control in raw terminal and ledger, source6ef7/hosttrn21/base6ef7, seed42, exact1800 declaration and source charge estimate1785.1. It is the selector's only eligible same-host control for this base. Earlier schedule-arm0.9900560897366009 remains a distinct historical comparison; it is not silently relabeled. The older3e79 raw architecture parent is not the formal control.

## Own evidence

audit.py does not run or reuse audit_final.py. It hashes and checks the eight control retrieval-manifest files against immutable e78c9e7a, all26 candidate bundle entries against their hashes/sizes and immutable f7db0821, source Git blobs, proposal/source/prepare pins, launch/evaluation settings, exit chronology, budget evidence and retained checkpoint receipt. It constructs its own selector row from raw candidate receipts, then queries the unchanged selector and compares the supplied final-audit result only after its own computation. **77 checks pass normally and under -O**. Detailed file hashes are in normal.json/optimized.json, including:
- Control fresh20M log:37cf2f4d0d9ed52ebe6baa0234da6114167abbb12566d3f287893a3f378f70df.
- Candidate fresh20M bundled log:871833f3202cacaec778d7aa9bac4ceaa779a81aea48cbf1f218e77f0fcf4345.
- Candidate final raw bundle:33d202098b13a847c1e4349a751fdf978098032b41777f3df7d52330e9bb631f.
- Selector:fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac.

## Scope and limits

I previously implemented the75b9 port: this is an independent raw-evidence calculation, **not** a fresh independent source review or approval of my own code. No host/tensor/checkpoint-content/evaluator execution occurred. No source, ledger or selector was changed; before/after byte identity was checked.

Control actual uncapped startup and measured charged-through-exit remain unknown; source estimates and the authenticated1800 declaration satisfy the existing reviewed selector, not a new cold/submission gate. Candidate actual charged-through-exit1789.1628205840243 is below1800. Different historical eval environment/PYTHONPATH contexts exist; prescribed formal eligibility does not establish bitwise reproducibility or perfect operational equivalence. No new causality/quality adoption/submission authority is asserted.
