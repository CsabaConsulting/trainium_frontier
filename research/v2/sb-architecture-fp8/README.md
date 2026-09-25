# S-B architecture + FP8: offline integration only

Candidate SHA-256:
`a2f6b188db8418f4bccf4f2704db1a68bd9c3588bcc69d1fc7819701db39a238`.
The only production change is `workflow/proposals/research-v2/train.py`.
This branch starts at architecture commit `1e21c3d7` (93ae) and transplants
FP8 from `50a90137` (9212). Both component branches and canonical6ef7 remain
untouched. This is Implementer evidence, not independent approval.

**Not launchable:** exact full-size G2, native compilation/numerics/causality,
whole-step performance and public BPB are all pending. The architecture-alone
run and FP8-alone >=3% whole-step gate take priority. No host access, hardware
run, launcher or new gating framework was used or created.

## Integration and conflict resolutions

- Relative to93ae:215 added lines,13 removed. FP8 helpers and custom autograd
  implementation are copied unchanged; their definitions sit directly after
  the FP8 feature switch. `Linear`, `CausalSelfAttention`, `MLP`, and the
  submission training installer match9212 by AST.
- Kept architecture's scalar skips and attention/table filtering unchanged.
  FP8 marks the MLP projections in **every** surviving block, including
  MLP-only blocks, and Q/K/V/O only where attention exists:36 marked linears
  for8 layers with5 attention blocks. Embeddings, head, VE/small gates and
  learned skip scalars retain the ordinary path.
- The saved-toggle registry contains both architecture settings and
  `USE_FP8_PROJ`; neither registration replaces the other. FP8 is an arithmetic
  setting, while the inherited architecture fields remain shape-critical.
- Retained FP8's conditional training MLP installer: ON must not be overwritten
  by the old BF16 adapter; OFF follows exactly the architecture component's
  training path. No new adapter or evaluation-only forward was introduced.
- The identical shared timer reset was already present in93ae, so its incoming
  duplicate hunk was omitted. The entire training-loop AST matches both exact
  timer-fixed components. Loader, schedules, optimizer, checkpoint helpers,
  `load_for_eval` and `_SubmissionEvalGPT` are unchanged from93ae.
- `--unet-8x1024` and `--fp8-proj` stay independent/default-OFF. The combined
  preset is `--unet-8x1024 --attention-layers 0,2,4,6,7 --fp8-proj`.
  No Recursive schedule; total batch remains131072. A later262144 proposal is
  a separate option, not this candidate's default.

No semantic conflict required a workaround. Existing FP8 incompatibility
checks for transposed/coalesced weights and loss scaling are retained.

## Evidence

All tests use synthetic tokens, seed42, one OMP/MKL/OpenBLAS thread and the
existing `/home/poojith-devan/venv/bin/python` (Torch2.13.0+cpu).
Seven serial runs of the **unchanged** `workflow/tools/preflight_dryrun.py`
pass3 real optimizer steps on4 CPU ranks. Dense shape is6x128; architecture
shape is8x128. Vocab8192 and context1024 remain, but each training microbatch
is sliced to1x16. The actual8x1024 preset is parser-tested before runtime,
never allocated. The prior test-only PyArrow stub rejects all data reads;
G2's existing synthetic prepare, BF16, clock, and NKI-CE shims are recorded.
No test shim is deployable production code.

| Combined mode | Exact reference | Three-step comparison |
|---|---|---|
| Both OFF | c5c dense |12 losses,62 parameter digests and ownership plan identical|
| Architecture ON, FP8 OFF |93ae architecture ON|12 losses,65 digests and ownership plan identical|
| Architecture OFF, FP8 ON |9212 FP8 ON|12 losses,62 digests and ownership plan identical|
| Both ON |New combined mode|Four-rank finite/consistent updates and deterministic strict checkpoint reload PASS|

`identity-*.json` are outputs from the unchanged G2 `--compare` tool.
`g2-*.json` retain all rank reports and the exact argv/source hashes.
Temporary G2 checkpoints/logs were retained with `--keep`; their paths appear
in the receipts. These are reduced checks, **not** exact-size G2 receipts.

`check_interactions.py` additionally passes:

- 3-step direct Muon/AdamW isolation with exact losses, gradients and weights;
- combined finite gradients/FP32 masters, learned scalar skips, no unused
  attention tables, and no stale FP8 weight cache;
- exact prefix causality, batch independence, determinism and input sensitivity;
- saved combined toggles restored after intentionally changing live globals;
- weights-only strict checkpoint reload with equal weights/logits and frozen
  parameters; missing skip weights rejected; legacy absent flags restore OFF;
- source AST invariants and all4 CLI mode combinations before allocation.

The direct test is at most8x128/vocab64/context32. Its BF16 compute shim and
eval-mask32 bound affect only the test. The first self-check invocation had a
test-discovery assertion because it looked for a `for step` loop; production
uses `while step`. The test was corrected to select that actual loop and
passed; no production change was made for this test failure.

Reproduce the focused check from this worktree:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=research/v2/l2-unet8x1024/test_support:workflow/tools/stub_prepare /home/poojith-devan/venv/bin/python -B research/v2/sb-architecture-fp8/check_interactions.py --train-py workflow/proposals/research-v2/train.py --out /tmp/sb-self-check-new.json
```

Use each G2 receipt's argv with `--steps 3 --nproc 4 --micro-rows 1
--micro-tokens 16 --timeout 180 --keep --stub-prepare`. Do not substitute the
full-size architecture preset on this laptop.

## Attribution, assumptions and risks

Architecture attribution and assumptions are inherited from L2 and the public
[Trainium of Thought write-up](https://ritwikareddykancharla.github.io/projects/trainium-of-thought.html):
8 layers, width1024, partial attention and U-Net skip idea. Exact attention
indices0,2,4,6,7, zero-init scalar gates, encoder-output to reversed decoder-input
pairing, optimizer group and table filtering are the L2 task's explicit choices,
not details claimed to be specified by that article. Combining this particular
architecture with this FP8 implementation is our unmeasured integration.

FP8 code/attribution is copied from reviewed L1: repository row-quantizer and
byte-safe transpose work, AWS NKI Library commit92d11f63 (Apache-2.0,
Copyright Amazon.com, Inc. or its affiliates), and the
[AWS Trainium2 NKI guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html).
No external data, weights, vendor runtime files or unreviewed FP8 algorithm
were imported. CPU FP8 is only the explicit numerical reference; it does not
execute the NKI forward.

Joint risks include quantization/skip-learning interactions, losing the old
fused MLP schedule, preparation/write overhead, compiler region boundaries,
SBUF spills, effective-operand HBM lifetime and startup cost. Fewer attention
blocks do not prove a faster whole step or better final BPB. Do not infer native
or quality success from these CPU equivalence checks. The independent component
results and a new reviewed exact-source stack proposal must precede any combined
native work; no adoption or leaderboard submission is authorized here.
