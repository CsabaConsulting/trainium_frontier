# CLEAN audit amendment — 2026-09-28 03:36 IST

This adds current evidence and human Round31 to the immutable five-file folder
`Trainium-CLEAN-GATEFIX-CLAUDE-AUDIT-20260928-881a4f49`. It does not replace
its historical receipts, approve submission, or change any candidate bytes.

Source SHA256: `881a4f491d256fb89af33246a3b568fc3ffead9616070eea1860d7d8f7f9f749`.
Public20M/context1024: **0.9772633533246753**, 2682 steps; empty-cache cold
startup **662.9083368468564 s**, charged through exit **1789.0335643270519 s**.
Cold archive SHA256: `9d24309ebea890b878c82a3ec16f1e2cf9dd0aba6c8850f4cd1aa3f01cb8dd49`.

Round27 CPU three-batch identity **PASS** on actual instance CPU. Parent/CLEAN
losses are exactly `9.010618209838867`, `8.975030899047852`,
`8.866786003112793`; all checked gradients and weights also match. This is
reduced-shape CPU fallback evidence, not production Neuron bit identity.
Receipt SHA256: `1a851945f7a51a29d88d7c801eef80834a043df575cd34cc9fccf76c15a5e0b2`.

Human Round31 explicitly accepts the better score when this identity check
passes. Therefore the old absolute ±0.0005 rejection is superseded; its original
FAIL record is retained for provenance. Claude compliance acceptance is pending.

## New eight-item checklist — evidence status, not additional invented gates

| Item | Status | Current evidence / unresolved point |
| --- | --- | --- |
| 1. Single file, exact launch, provided imports | FAIL — evidence incomplete | Isolated exact source and bare four-rank launch succeeded. Imports use provided prepare/Torch/Neuron/NKI; organizer-version compatibility still needs audit confirmation, not a new SDK install. |
| 2. Trained saved model, causal shared forward | FAIL — concrete source question | Strict weights-only load into the same GPT. `MLP.forward` line772 selects ordinary Linear inference versus `_FusedMLP.apply` for training. Intended algebra agrees, but the literal same-kernel/no-mode-branch instruction requires Claude's resolution. |
| 3. No prohibited data/network/cache reliance | FAIL — portability evidence incomplete | Source trains only from supplied train loader; no direct network client or external weights. Six training caches were empty. Eval used a shared research cache; necessity of that cache is not established and must not be assumed. |
| 4. Cold startup and charged budget | PASS — measured run | 662.908337 s startup; 1789.033564 s charged through exit, retained checkpoint included. |
| 5. Disk footprint | FAIL — inventory pending | Checkpoint453,044,690 bytes. Total cache/log footprint and transient write volume have not been measured; archive bytes are not a substitute. No protected artifact deletion is authorized. |
| 6. Defaults ON, no hidden required environment | FAIL — evidence incomplete | Fixed recipe defaults and optional compiler/cache defaults are present. Actual launch environment is recorded; optional defaults are not themselves forbidden external reliance. |
| 7. Fixed seed and allowed timing behavior | PASS — scoped source audit | Seed42; time-based recipe schedules and required budget accounting. No claim of bitwise replay determinism. |
| 8. Readability and no disabled experiments | FAIL — same-path question pending | Top docstring already explains rotary RMSNorm, convolution, CE and PyTorch MLP. Separate per-function docstrings are not a new requirement. Remaining mode-specific MLP behavior is disclosed under item2. |

Missing evidence is distinguished from an observed defect. This checklist does
not impose an unrequested network trace, pristine instance, SDK upgrade or extra
full run. Full supporting review is in the repository:
`research/v2/submission-prep/20260928-rejection-checklist/CHECKLIST.md`.
### Actual footprint / installed-package observation at03:38:23

Read-only `du --apparent-size` of the completed CLEAN root measured
**580,548,028 bytes retained**, including126,409,285bytes private cold caches
and453,383,127bytes in the training directory (453,044,690byte checkpoint).
Shared eval caches are excluded and cumulative/transient writes are unknown;
this does not retroactively turn checklist5 into an unconditional PASS.
Nothing was deleted.

Installed host package metadata: torch-neuronx `2.11.3.0.0+unknown.dev`
(editable source `/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx`),
neuronx-cc `2.26.6360.0+6f180f47`, NKI `0.5.0+28631259367.ga768afa6`.
These are observed host facts, not a claim about the organizer image.
Raw receipt: `research/v2/operator/20260928-round30-facts/audit-metadata-trn22-20260927T220817.json`.

No file in the frozen audit folder was modified. No upload was performed.
