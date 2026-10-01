# Exact6497 fallback: audit handoff addendum
Date: 2026-09-28 17:33 IST. Read-only Root check; not Claude acceptance or upload authority.

## Payload and requested review
Existing five-file folder:
`/home/poojith-devan/Downloads/Trainium-CLEAN-CONV-R37-FALLBACK-CLAUDE-AUDIT-20260928-6497d163/`.
Upload payload, if the human later approves, is train.py only:
`6497d163d930275dfbc0bb494fbb546456478e3751119dcc1baf918f5638f9d3`, 84,679 bytes.
Command unchanged:
`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`.
Nothing in the folder was edited for this addendum; no sixth package file added.

## Evidence verified now
- All five files remain present and all four embedded manifest entries match actual SHA256/length.
- Manifest itself: `8632d70f6cce2342f336a995bb4bf5001af51768826c04a85db8ee227d35f033`.
- Parent exact b585 source is unchanged. AST imports match exactly. After removing only the authorized
  `not self.training` disjunct from the parent's MLP.forward condition and stripping docstrings,
  complete parent/candidate ASTs are identical. Stdlib check executed successfully; no Torch/model run.
- Existing independent source/symbolic check: 26,040 parent/candidate pairs, training routes identical;
  receipt `research/v2/verification/20260928-r37-eval-path-independent/RECEIPT.json`.
- Actual native saved-checkpoint re-evaluation: original and candidate both 0.9766242696436925,
  delta0 <1e-5, 20,971,520 tokens/context1024/batch1. Exact checkpoint7ba52e1d unchanged.
  Verdict `research/v2/verification/20260928-r37-eval-path-independent/NATIVE_IDENTITY_VERDICT.json`,
  SHA `9fc25ab1aec025a9f711a58a64434200345990ae9ad85ab746e7558ec2be7d86`.
- Existing Root package-scope check is PASS for audit handoff only:
  `research/v2/verification/20260928-r48-fallback-root-scope.json`.
  The frozen brief's older “Root pending” text predates that separate receipt.

## Reconcile the checklist without inventing a run
Parent Claude audit:
`/home/poojith-devan/Downloads/Trainium-CLEAN-CONV-CLAUDE-AUDIT-20260928-0800-b585d0a6/CLAUDE_AUDIT_RESULT.md`,
SHA `b37601233f9ce326273f7ee34c226ff3e8cb25f02123deac7c3bcae520d13d22`.
It accepts b585, recommends precisely this eval-branch fix with saved-checkpoint identity,
explicitly says no retraining is needed for that recommendation, and calls unknown cumulative
write bytes non-blocking. Those are the prior auditor's instructions, not a new6497 acceptance.
The package honestly preserves parent-only cold provenance and unknown exact-source full-run,
clean-runtime-environment and cumulative-write evidence; no unknown row is silently made PASS.
Eight b585 cold runs: mean0.9766558936725349; maximum startup684.242494167s.
New supplementary evidence: timing derivative26c (only seven reversible timing annotations on6497)
completed60steps, six empty caches, raw cold655.561928727s, 292609.879tok/s, archive73ce2120.
This is NOT a full cold training run or public20M measurement of exact6497 and does not replace one.

## What remains human-only
Claude must audit the exact6497 source and disposition the disclosed gaps; Poojith decides any upload.
No new quality improvement is claimed. Official b585 remains human-reported0.9768.
R49 table/native result and MUDD are parked for cost; neither is in this fallback.
