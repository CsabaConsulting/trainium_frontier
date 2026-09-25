# Exact-score architecture on the proven 116188 vehicle

Status: **SELF-CHECKED CANDIDATE, NOT UPLOAD-READY.** Independent source/fresh
review, full-parameter CPU comparison and native cold qualification/replay
remain pending. No host was contacted, no launch was performed, and no existing
submission, Downloads file, evaluator, data path, queue or ledger was changed.

Source SHA-256:
`3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f`.
Path: `workflow/candidates/architecture3e79-116188/train.py`.
Branch: `codex/20260926-architecture-scored-clean`, from canonical7132a648.

The research reference is exact3e79, commit
`b4d83b5db66971f3c348837f04f53cba65aa6d33`, SHA
`3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`.
Its reported fresh public20M score is0.9863403052402583, not an official
leaderboard score and **not a measured score of this clean source**. Its
matched trn22 c5c reference is0.991513958330512. Independent result review
and cross-host research replication remain Supervisor-owned.

## What changed

`vs-116188.patch` is the complete zero-context diff against immutable
`submissions/dense6x1024_direct_review_matrixlr015/train.py`, source SHA
`116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`.

- Bake eight layers and attention indices0,2,4,6,7. Other blocks contain only
  the existing input/residual mix and MLP; they have no unused attention,
  convolution or value-embedding parameters. Only layer7 has a value table.
- Bake four zero-initialized FP32 scalar U-Net gates, encoder outputs feeding
  decoder inputs in reverse order. Preserve exact `x + gate.to(x.dtype) * skip`
  operation order, initialization order, names and existing scalar AdamW group.
- Add the differentiable first cache refresh after counted `t0` and before
  zero-grad/first forward, in ordinary grad mode. Retain every post-update
  refresh. There is no external prewarming or work moved outside accounting.
- Carry the already-reviewed startup predictor reset: clear excluded startup
  samples at the fifth boundary; maximum of empty recent history is0. The
  charged startup excess,900-second cap,30-second margin,15-second reserve,
  official1800 budget and weight-decay progress are unchanged.
- Correct only the FLOP estimate's attention-block count and add attribution.

The old scored schedule is deliberately retained. **No Recursive package**, new
optimizer, LR change, batch change, FP8, n-gram table, kernel, research installer
or `_SubmissionEvalGPT` is introduced. All116188 optimizer classes, kernels,
attention/MLP/Linear components and other unchanged helpers are AST-identical.
The ordinary `GPT` loader, `weights_only=True`, strict checkpoint loading and
evaluation function body remain116188 exactly. There is no scorer-dependent
forward, row wrapper, test-time update or runtime import shim in production.

The clean file intentionally has baked architecture/cache behavior, not an
experiment flag. OFF-equivalence is inapplicable; comparison is to exact3e79
ON with the original old schedule, plus unchanged116188 unaffected paths.

## Launch contract

`launch-command.txt` is byte-identical to the human-proven116188 package:

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```

No `--unet`, `--initial-cache-refresh` or schedule argument is needed/accepted.
The long launch is required; the earlier4083 bare specialization is NOT reused.
During preparation, an untested draft briefly specialized width/context/pack/
eval defaults for bare invocation; those edits were removed when Supervisor
clarified the preserved long-launch requirement, before all tensor tests.
All17 parsed arguments match116188 under that command. The effective profile
is8x1024, heads8, head-dim128, MLP ratio4, context1024, pack32, batch131072,
world4/one accumulation, seed42, matrixLR.015, max1800, old cooldown/momentum.
Original environment overrides remain and must be checked; this packet grants
no authority to change them. The actual architecture has113,256,724 parameters,
not the dense6x117M count. The count is derived from the model shapes, not a
local full-size allocation: `84*d*d + 24586*d + (d/128)*32 + 20`, d=1024.

## Implementer evidence

- `test_port.py`: **17 tests PASS normal and -O**. Exact source pins, unchanged
  complete parser/long command, unaffected helpers/kernels/optimizer/loader,
  actual schedule-group assignments, >1000 progress points and neighboring
  floats, precise main-loop delta and inherited timer regression cases.
- Unchanged G2 harness SHA
  `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`:
  sequential **8x128**,4 CPU ranks,3 steps, synthetic1x16 inputs, BF16 CPU,
  deterministic1ms fake clock. Research and clean runs PASS6.9s/5.5s.
  All12 loss values, all65 final state digests and ownership plan
  `83b06c658067902fe9e255779ce792e5117006563c932b16bc1f4bba6f6595d5`
  match. No cross-rank mismatch, missing parameter update or bad reload.
- `check_tiny_checkpoints.py` compares the actual retained CPU checkpoint
  tensors, not only short digests: all65 match exactly. Ordinary clean loader
  returns `GPT`; B1 logits at lengths16/64/256 match research exactly. B2 versus
  independent reference rows has max error0, as do causal-prefix mutations.
- A separate synthetic **1x256x128** training probe crosses the actual custom
  MLP path selection boundary (the1x16 G2 input does not). First loss and all65
  parameter gradients match exactly with no missing gradient. This uses the
  existing research native-forward/custom-backward installer and clean direct
  class on CPU; it proves no Neuron compiler behavior.

Tensor runtime: `/home/poojith-devan/venv/bin/python`, Torch2.13.0+cpu;
OMP/MKL/OpenBLAS1. Local test-only `--stub-prepare` and import-only PyArrow
support reject real data reads. G2 records every synthetic/fake-clock/no-compile
shim. No package was installed and no real data was read. **Do not deploy local
support stubs to a host or final submission.** No failed tensor attempt occurred
for this packet. The initial reference lookup used a source SHA as a Git ref
and failed before any model/test action; the correct Git/source distinction
was then used. All source artifacts remain unchanged after the passing runs.

## Pending gates and honest limits

`FULL_CPU_PLAN.disabled.json` contains exact source/harness/prepare pins and
prospective commands. Full8x1024 four-rank CPU gate runs only on an independently
bound idle large-memory host, never this RAM-limited laptop. Synthetic2x256
inputs and fake clock are disclosed: this is full-parameter, not a full native
32x1024 update. The standard harness exposes loss, final parameter/ownership
digests and checkpoint reload, not every full-size gradient/optimizer state.
Do not broaden that claim from the separate tiny gradient test.

Next: independent review/fresh own-reference check, full-parameter CPU identity,
then separately assigned genuine-cold ~5step test (max1800/seed42/full chip).
Architecture target<=870s and mandatory<=900s before a separately selected full
run remain. Measure uncapped launch→fifth boundary and charged-through-exit;
do not substitute capped startup allowance or compiler compile_us0 for proof.
Only after those gates: native full1800, checkpoint/causality and unchanged fresh
public20M/context1024 evaluation, final compliance/human upload approval.
No next run, staging, native launch or upload is authorized by this packet.

Cold graph/fusion behavior may differ when research wrappers become direct
classes. Faster startup also changes elapsed-based LR/WD in a real run; fixed
synthetic-clock equality does not promise identical native first50 updates.
The current score source used a per-row eval subclass; this port intentionally
uses the proven ordinary116188 loader. Tiny B1/B2 equivalence is evidence, not
a waiver of native batch-independence/causality verification.

## Reversible decision log

2026-09-26 approximately02:35–02:42IST: Supervisor's exact-score package request
preempted the Recursive research arm. Checkpointed that isolated arm as
`fe0969fd311d654e143dc6113614416c8970a424`, tests still pending, and created this
new clean worktree. Reason: deliver the measured old-schedule architecture,
not an unmeasured schedule stack. Undo: discard this new candidate only; prior
116188,3e79,4083,Recursive branch and Downloads are untouched. K2 binding8555
remains disabled after its separate checker nonfinite-reporting review failure;
that repair is deferred, not hidden or mixed into this package.
