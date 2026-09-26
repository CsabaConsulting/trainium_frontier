# One framework A1 lottery — user authorized26 September

Exactly one full1800s run, seed42, on a free host after final candidate work
has priority. Eight layers, width1024, batch262144, matrixLR.021, ordinary
warmdown.75, momentum warmup300. Only scientific addition: last-layer bigram
and trigram value tables,524288x128 each, as the frozen A1 layout.
FP32 masters/gradient accumulation/RMSProp states; BF16 gathered activations.
Use F.embedding and its STANDARD framework backward. K2 CLI is rejected;
no K2 registry, custom backward or metadata preparation executes.
Old inactive K2 definitions remain byte-identical, never evidence of a native
success. This is not a submission vehicle; only train.py is a model change.

Source derivation: frozen4b74 implementation, two base constants and the CLI
restriction inverted to reject K2. All non-main function/class ASTs unchanged.
Local tests must prove table-OFF identity against clean cd0c, FP32 collision
reference, ordinary table ON training, causality and checkpoint roundtrip.
Fresh-process public20M/context1024 after full budget; report cold startup,
startup excess above900s, charged runtime, steps, steady tok/s, BPB and actual
same-host batch/LR delta. No speed gate and no step-count adoption gate.
One trial only; no new lane, retry or automatic custom-table resurrection.
Cold cache for this lottery measures its real submission-relevant cost.
Human-only submission, unchanged evaluator/data and original lifetime lock.
CPU checks are author evidence, not a new independent review or native PASS.
