# Separate L1 FP8 startup predictor port

This isolated worktree branches from L1 commit
`8b6d6cd9cf3079ab098794abc44864bf6d513b21`; the original FP8 builder worktree is
untouched. No architecture changes from L2 are included.

- Original FP8 SHA-256: `dea47910547572c704d8d5cc0e3dd058282fc8d862c913939bfc5e40830713b4`.
- Corrected FP8 SHA-256: `921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858`.
- Source: `workflow/proposals/research-v2/train.py`.
- Reviewed timer source: `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b`.

The only two executable edits clear `recent_step_times` at the existing fifth-step
startup-exclusion boundary and add `default=0.0` to the existing next-step maximum.
No FP8/model/kernel/optimizer/schedule change is added. Startup cap, launch margin,
charged excess, training budget, checkpoint reserve and step limit remain unchanged.

The nine exact reviewed stdlib timer tests plus three port-specific tests passed:
12 tests in 1.315 seconds, no failures/errors. The inverse restores the original
dea479 source **byte-for-byte and AST-for-AST**. The corrected timer AST matches
a4aa4351. The synthetic A2 trace still charges 1092.09 seconds while avoiding the old
premature fifth-step stop. `startup-predictor-reset/timer-port-self-check.json`
contains the source-bound result. Reproduce with:

```text
python3 research/v2/l1-fp8-startup-reset/startup-predictor-reset/test_l1_port.py
```

Tests never import train.py, allocate tensors, run training, contact hosts or call
model APIs. `prospective-116188.patch` is copied solely for the original suite's
in-memory test; the scored submission remains untouched. `git diff --check` passes.

This is Implementer evidence, not independent approval or a native result. All L1
feasibility/numerical gates remain. The user's performance acceptance rule is at
least **3% faster whole-step time on Trn2** versus a matching control; component-only
gains cannot satisfy it. No timing or BPB improvement is claimed from this timer port.
