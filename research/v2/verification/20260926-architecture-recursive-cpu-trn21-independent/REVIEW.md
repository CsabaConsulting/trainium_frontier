# Recursive architecture trn21 CPU binding — mechanical PASS only

Independently reviewed frozen `65b0d5eec9a5ae3623730bfcec4772e16db9c687`.
Plan SHA256 `8d690b0a5e441155a5b90e3ed7b929728002da4c0b5b2d5b5d0a6e020bc113c4`;
driver `af37bcb63c4a61a0666313a484bb66a64d3678a677ff84b2cafccecde41b0370`.
The original trn22 packet `57511815` is preserved. This disposition applies only
to the explicit trn21 successor, not a free-form relocation.

Own stdlib audit passes **99 checks**, identically with normal Python and `-O`.
No Torch import, tensors, full model, checkpoint deserialization, host action or
provider request was needed. Reproduce from this repository with
`python3 research/v2/verification/20260926-architecture-recursive-cpu-trn21-independent/audit.py`
(and repeat with `python3 -O`). `RESULT.json` retains the complete checks.

All 14 manifest entries authenticate. Reference `3e79a7c1…`, candidate
`75b944f7…`, unchanged prepare `5807c46e…`, standard G2 `6a545891…`, shared
guard `adafb54a…` and dispatch `c71474bc…` match their pins. Driver differences
from the original packet are exactly host/root/prepare literals and request
hashes; receipt/checkpoint helper bytes and every scientific argv are unchanged.
Reference/default have the scored architecture argv; explicit OFF and ON add
only their respective Recursive flag.

The single original common flock spans four sequential arms and comparisons.
Both child sites inherit the authenticated FD, and each phase repeats the same
fresh in-lock ps/neuron-ls guard. The repaired secondary process scan excludes
only self and the authenticated outer parent. Own exact-function fixtures
refuse a changed parent, unrelated host driver, torchrun, wrong host, insufficient
RAM/disk and fuser busy/error. No unlock or automatic native/next-job path exists.
Existing deeper-child FD-closure limitations are not erased: the sole Operator
must establish actual quiescence before any later reuse.

The 64 GiB RAM / 6.5 GiB disk floors, synthetic CPU/no-compile G2, four ranks,
three steps, 2×256 micro-batch, per-arm 3600-second timeout and bounded 600+60
checkpoint comparison are explicit. This is a full **113,256,724-parameter**
shape, but only 2,048 global tokens/update, not the 131,072-token production
update. It is not a native throughput, cold-start or quality test.

Own malformed-receipt tests reject wrong source/argv, stub prepare, tiny shape,
wrong ownership, missing rank/digest, unchanged parameter, rank mismatch,
nonfinite loss, reload failure and any reference/default/OFF numeric mismatch.
ON loss/parameter changes are intentionally permitted, while its individual
finite/update/rank/reload requirements remain enforced. All 65 rank-0 digests
and all 12 losses are the identity scope; digest maps on other ranks are not
invented.

Independent tiny stdlib doubles exercise the exact checkpoint comparator's
byte, signed-zero, dtype, finite, key/config and logit refusal branches. This is
control-flow evidence, not an actual Torch/checkpoint run. The future comparison
uses `weights_only=True` on CPU and a **disclosed CPU BF16 dtype shim**; it checks
reference/default/OFF model-state bytes and B1×256 logits. ON strict reload is
the standard G2 check, not equality to OFF. Neither ordinary unshimmed CPU eval,
B2 behavior, optimizer-state/gradient identity nor native correctness is claimed.

**No blocking binding defect found.** This permits the CPU packet to be
considered after the separate fresh source gate and sole-Operator assignment;
it does not itself dispatch or authorize a host job. Full-size CPU results are
absent; native acceptance, quality, clean-source submission and adoption remain
outside this review. Existing clean-source REFUSE is unchanged.
