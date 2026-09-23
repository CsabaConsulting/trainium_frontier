# c595 reviewer audit

The candidate keeps the same GPT and strict `weights_only` checkpoint load. With `targets=None`, forward returns softcapped logits. The fused MLP/autograd path and NKI backward wrappers are training-only and have documented justification. `setdefault` respects inherited overrides, which is why an uncontrolled parent environment is an operational risk; this cohort pins the training compiler string and scrubs the evaluation environment. Rotary extension remains nonpersistent.

Review found no runtime patching or hooks, evaluator changes, network dependency, external training data, pretrained weights, or arbitrary-pickle load. Unused private imports are review noise, not a semantic claim.

The earlier component result and discriminator-r3 top-level results are terminal **FAIL** evidence. They remain failures: the component's CE/optimizer gates failed, and the discriminator's predeclared optimizer comparison plus postflight failed even though source and direct agreed on the exercised production-shaped fixtures. This control proposal neither relabels nor rescues those results; it is a separately human-approved cold-training measurement.

The frozen source logger prints steps 0–4 and then multiples of ten. Therefore exact comparisons cover 0–4, 10, 20, 30, and 40, with 50 reported separately. It cannot provide every loss at steps 1–49.
