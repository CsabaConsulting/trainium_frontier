# Claude audit result: R37 fallback package (train.py 6497d163)

Date: 2026-09-29 00:20 IST. Auditor: Claude (session with Poojith). Scope: source audit of the upload payload
train.py, SHA256 6497d163d930275dfbc0bb494fbb546456478e3751119dcc1baf918f5638f9d3, 84,679 bytes, against the
accepted parent b585d0a647962009ae13068fd0f823e17fd19392b5d0dbf1087e6e5f40e48742 (official 0.9768).

## Verdict: SOURCE-AUDIT ACCEPT (upload allowed as the FALLBACK package)

This file is NOT a quality improvement. It is the accepted b585 recipe with the eval-branch fix that the
b585 audit recommended. Expected official score: the same as b585 within run noise (about 0.9762 to 0.9772).
Upload it only as the planned fallback slot (it costs one of the 15 weekly submissions); do not delay an
improved package for it.

## What changed versus b585 (verified by byte diff on the laptop copies)
1. MLP.forward line 784: the condition `if not self.training or x.dtype != torch.bfloat16 or ...` became
   `if x.dtype != torch.bfloat16 or ...`. Effect: eval now takes the same _FusedMLP path as training instead
   of the eager fallback. The training route is unchanged (self.training was True there anyway).
2. Seven docstrings and one comment added inside the conv NKI helpers and MLP.forward. Docstrings are inert
   string constants; they change no instruction. The timing derivative 26c (this source plus reversible timing
   annotations) ran 60 native steps at 292,610 tok/s, raw cold 655.56 s, so the docstring'd kernels compile and run.
3. Nothing else: no other lines differ. `self.training` no longer appears anywhere in the file (0 occurrences),
   so the model has no train/eval code branch at all.

## Evidence checked
- Native saved-checkpoint re-evaluation of the exact candidate source: 0.9766242696436925, identical to the
  parent's evaluation of the same checkpoint (delta 0, 20,971,520 tokens, context 1024). This exercises the
  changed eval path natively.
- Independent symbolic check: 26,040 parent/candidate pairs, training routes identical.
- AST parse OK; one torch.load call, weights_only=True, strict=True state_dict load; load_for_eval unchanged.
- record_step_boundary(step) is called once per optimizer step after synchronize(device) (line 1443).
- No prepare.* attribute access or monkey-patching; prepare is imported only for the documented names
  (make_dataloader, record_step_boundary, constants, print0, get_dist_info, ensure_tokenizer).
- No evaluate_bpb reimplementation; no eval-only or train-only classes; no external data or weights;
  tokenizer untouched; launch command unchanged:
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py

## Eight-item rejection checklist (source scope)
1. Only train.py changed: PASS.
2. No prepare monkey-patching or scoring reimplementation: PASS.
3. record_step_boundary after every optimizer step: PASS.
4. Same model code path for train and eval, load_for_eval unchanged: PASS (improved versus b585).
5. Checkpoint loads with weights_only=True, strict=True: PASS.
6. Cold startup under 900 s: PASS by evidence (parent eight cold runs max 684.2 s; derivative 26c 655.6 s;
   no full cold run of the exact file, which is acceptable because the training route is byte-identical).
7. No external data, weights, tokenizer or SDK change: PASS.
8. Disk and runtime hygiene (cumulative writes unknown): non-blocking, same as the parent audit.

## Notes for Poojith
- Upload payload is train.py only. Keep this folder unchanged as the record.
- If an improved package (D1-vi, D2 or a stack) clears the two-run gate before the fallback slot is needed,
  upload the improved package instead and keep this one in reserve.
