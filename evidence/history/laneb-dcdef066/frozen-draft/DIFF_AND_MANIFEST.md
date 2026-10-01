# Exact-source diff and manifest

No source transformation was made for packaging. `train.py` is exactly the scored
dcdef066d8d0827f4ecdf0a80f8517e5348af28eea7acc06e03df4db49d4e90d, copied via apply_patch.
The command remains exactly the scored explicit `--tbl_freeze_tokens 0` command.
No header/default cleanup, seed change, inferred upload derivative, checkpoint,
external weight, prepare/evaluator change or new model helper is included.

## Closest exact research-proven ancestor

Direct scientific parent:
`/home/poojith-devan/Desktop/Trainium/research/v2/proposals/20260929-r109-ce2-flat-zbundle/train.py`
SHA256 e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310.
Its actual FULL21 archive088637f6 / packet28d29a6e is authenticated under the same
original raw FULL gates in ancestor-evidence.json: public0.9672630039667548,
cold678.5742937349714, charged1793.4091268959455. This establishes a research-proven
direct ancestor, NOT a last Claude-audited ancestor. The latter is UNKNOWN.
The parent is not included in LaneB's two-receipt mean, nor claimed a matched
two-receipt causal-effect estimate.

`SOURCE.diff` is the exact existing parent-to-LaneB diff, SHA256
894a76cc54f31fa870af864cdd239c6ecdae69d01492848eaa4818f414e0e910.
The checker recomputes it from the exact source pair and replays the six existing
literal EDITS.json replacements to reconstruct the candidate byte-for-byte.

| Changed region | Exact scientific delta |
| --- | --- |
| constants / GPTConfig | LANE_B=True; saved config lane_b defaultFalse for old checkpoint compatibility |
| wrapped attention insertion | Installed bundled attention_cte/attention_bwd imports; wrap_nki grid2; one autograd Function with current Q/K/V/O/LSE |
| layout / LSE / backward | Explicit BHTD/channel-major copies; LSE sign/token tiling; causal BF16 native forward/backward |
| attention dispatch | LaneB Neuron branch invokes the same native wrapper in train and targets=None eval; CPU/CUDA retain parent SDPA |
| build_config | New training config explicitly binds lane_b=LANE_B |

No change in the exact parent diff to CE2, FLATCACHE, table/hash/update schedule,
optimizer/ramp, main compiler block, checkpoint loader or scoring functions.
This statement is about the literal diff, not proof that the new attention has
identical native numerics to the parent in every production/evaluation shape.

## Legacy receipt schema adapter

Pinned ace0b4db assumes `R98_ROOT_SELECTION.json`. The actual first LaneB packet
contains `R113_ROOT_SELECTION.json`; direct parent contains `R100_ROOT_SELECTION.json`.
`LEGACY_FULL21_ADAPTER.diff` and `LEGACY_PARENT_ADAPTER.diff` change only that
lookup string in memory, after the pinned original verifier is authenticated.
The selected real member is also byte-compared between exact packet and raw archive.
The original mismatch remains in LEGACY_SCHEMA_DIAGNOSTIC.json. There is no new
member, synthesized selection, modified archive, weakened check or rewritten receipt.
Repeat22 uses the pinned helper unchanged.

MANIFEST.json pins every package file except itself and records launch/export/
submission authorityfalse. REFERENCES.json pins absolute canonical external files
and sizes; large raw archives remain in place. cold-evidence.json indexes all raw
members so a reviewer can locate the precise source/command/terminal/cache/budget/
checkpoint/evaluation witnesses. This is a local draft, not a self-contained export.

The source, original source-review/CPU/native/FULL receipts and original helpers
remain unchanged. `verify_local.py` replays the read-only metadata proofs and
checks every local/external pin; its success is not a submission verdict.
