# Disabled startup-adjusted schedule full-parameter CPU gate

Preparation only, for trn21. No staging, host access, launch, reservation, native binding, adoption or submission authorization. Source36a86274 is frozen at7419927c on the separately published codex/20260926-startup-adjusted-schedule branch; this packet does not copy it into the canonical research source.

## Exact test and comparison

Four sequential arms: exact3e79 reference; new source with default flag; explicit --no-startup-adjusted-schedule; explicit --startup-adjusted-schedule. All other scored architecture arguments remain identical: 8x1024, attention0,2,4,6,7, UNet, initial-cache-refresh, LR0.015, global batch131072, context1024, pack32, seed42, source MAX1800. No Recursive package is present.

The full model has **113,256,724 parameters in65 tensors**. Unchanged standard G2 (6a545891) runs four CPU/gloo ranks for three updates using synthetic2x256 inputs/rank. Every model parameter retains production shape. This is2048 diagnostic global tokens/update, not the production131072-token hardware update. There is no real-data read or tensor execution during packet preparation.

The unchanged real prepare.py is pinned5807c46e and imported from the known trn21 project. Its startup count is **five**. No stub prepare, tiny observer, startup-count override or new clock instrumentation is deployed. Standard G2 uses its existing deterministic1ms-per-call clock, CPU BF16 dtype shim, no-compile override, disabled native CE installer, synthetic loader/tokenizer and training micro-batch slicing. The unchanged checkpoint comparator also uses the already-reviewed CPU BF16 dtype shim. Neither ordinary unshimmed CPU eval nor native behavior is established.

All arms must pass finite three-step losses, all65 parameter updates, four-rank consistency, exact ownership plan and strict deterministic checkpoint reload. Reference/default/OFF additionally require all12 losses, all65 rank0 final parameter digests, actual checkpoint model-state bytes and B1x256 logits to match exactly. Other-rank consistency uses G2's own checks; full-size per-step gradients/optimizer state are not emitted and are not claimed. Checkpoint file hashes are recorded, but toggle metadata can differ; payload/logit equality is the strict comparison.

ON changes the scientific schedule, so it must pass individual full-size checks but is not compared numerically to OFF. This three-update run **does not reach the fifth boundary** or simulate long compile time. Separate tiny225f27d1 evidence includes corrected six-step fixed-clock runs plus preserved failed two-boundary attempts. A fresh independent source/tiny review is pending. Do not relabel any of it as a full-size boundary test, performance result, or explanation of the cross-host BPB spread.

## Mechanical derivation and future assignment

This is a source/argv/label/root-only successor of9cb0d53e, independently reviewed9969a7f7. Complete driver bytes equal that parent after replacing the stage-root and request hashes. Validator/checkpoint helper bytes, G2, shared guard/dispatcher, trn21 hostname/prepare, clean environment, ordering, original lifetime flock, authenticated-parent scan and inheritedFD are unchanged. Every arm and comparison repeats fresh in-lock ps+neuron-ls checks; all host/STOP/hash/fuser/resource gates remain.

New absent root: /home/ubuntu/trainium_arch3e79_startup_adjusted_cpu_20260926_trn21_r1. Minimum64GiB availableRAM and6.5GiB free disk. The RAM-limited laptop is explicitly prohibited. The inherited3600s per-G2 timeout and600s+60s checkpoint timeout are safety ceilings, not predicted time or a change to the official1800s budget. Timeout may leave descendants; stop the sequence, retain evidence, and require the sole Operator to establish actual quiescence before later work.

Only after separate source/fresh review, independent packet review and exact Operator assignment: stage the absent root and four absent arm/tmp directories, copy exact helpers/requests/current guard/dispatch/G2, and extract source Git blobs to each arm/train.py using CPU_PLAN.disabled.json pins. Do not overwrite a root. Protected prepare stays in its existing project. Verify every manifest/pin before using the exact detached_dispatch_command; expected_outer_command is the original common-lock invocation. No queue or active binding is changed by this packet. No automatic full, native, evaluation, retry or follow-on exists.

## Local evidence and undo

Only stdlib tests are permitted here. Rebinding tests prove exact driver/helper identity and one-factor argv, pins, real-prepare count and disabled gates. Validator tests use a genuine archived full3e79 receipt **as a mutated fixture only**, not candidate evidence; loss/digest/nonfinite/shape/ownership/stub/parameter/rank/exit/checkpoint negatives must refuse. Normal and optimized Python receipts are retained. The future checkpoint comparator is not executed locally.

2026-09-26 preparation decision: make the selected diagnostic CPU gate reviewable without consuming a native slot or the fresh reviewer's tensor slot. Undo: never stage or assign this separate packet. Original Recursive/clean/K2/cache packets and canonical train.py remain untouched.
