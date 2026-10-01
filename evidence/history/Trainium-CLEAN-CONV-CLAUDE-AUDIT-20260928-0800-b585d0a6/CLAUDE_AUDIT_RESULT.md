# Claude source audit: CLEAN+CONV b585

## Verdict: SOURCE-AUDIT ACCEPT (upload allowed), with one recommended fix before the FINAL package

- Audited file: train.py SHA256 b585d0a647962009ae13068fd0f823e17fd19392b5d0dbf1087e6e5f40e48742 (84085 bytes)
- Parent: audited_parent.py (CLEAN) SHA256 881a4f491d256fb89af33246a3b568fc3ffead9616070eea1860d7d8f7f9f749
- Launch command (as used by all six cold runs):
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py
- Audit date: 2026-09-28 ~09:05 IST

## What was checked
1. Diff vs CLEAN: only `_conv2_dw_partial` changes (lines 510-524): the same FP32 per-channel reductions are
   transposed on TensorE and stored as one contiguous partial per tap instead of eight 128-element stores.
   No arithmetic change; output partial layout, later sum/cast, forward, eval, optimizer and loader untouched.
   NKI simulator byte-equality receipts (8 cases) support this. OK.
2. Data: only `make_dataloader(..., 'train', ...)` from the provided prepare module; tokenizer via
   ensure_tokenizer(build_if_missing=False). No validation data, no network, no file reads beyond these. OK.
3. Imports: stdlib, torch, torch_neuronx (wrap_nki), nki, prepare. No extra packages. OK.
4. load_for_eval: torch.load(weights_only=True), GPT(saved config), load_state_dict(strict=True), eval(),
   requires_grad_(False). Rotary tables extended to 8192 positions; values at positions < 1024 are
   identical (computed elementwise from the same base), so 1024-token scoring is unaffected. OK.
5. Eval forward: GPT.forward with targets=None returns 15*tanh(logits/15), the same softcap applied inside
   the training loss kernel. Pure causal (SDPA is_causal=True, causal conv). OK.
6. Budget: six cold runs, startup 644-678 s, charged 1788.8-1789.6 s. OK.

## Finding (inherited, not introduced by CONV): MLP train/eval branch
`MLP.forward` (line 778) uses `_FusedMLP` (custom autograd) when training and plain `c_fc -> relu^2 -> c_proj`
when `not self.training`. The FORWARD math is the same aten ops (F.linear with the weight cast to the
activation dtype, relu, square, F.linear; see reference_forward lines 734-736), so scores are unaffected.
The custom autograd only changes what the backward saves. RAMP (already uploaded, official 0.9775) has the
same branch. Risk: a static/LLM reviewer may read `if not self.training` as "different eval path".
Recommended for the FINAL package (math-identical, training bytes unaffected, no retraining needed):
drop `not self.training or` from the condition so eval on Neuron BF16 also calls `_FusedMLP.apply`, then
re-evaluate one saved checkpoint with the patched load_for_eval and confirm public20M is unchanged
(|delta| < 1e-5). Add a one-line comment that the fallback branch only serves non-BF16/odd shapes.

## Not blocking
- Checklist #5 (total bytes written) unknown: retained checkpoint ~453 MB is normal.
- Missing docstrings on conv helpers: cosmetic; add in the FINAL package if touched.
