# dense6x1024_lean

Same recipe, same launch command and same training arithmetic as `dense6x1024_public0992851` (public
val_bpb 0.9928513 over 20,971,520 tokens; that package was rejected by the organisers' LLM review
as "large and sophisticated"). This package is that file with everything the launch command does
not use removed: 4779 -> 2041 lines, 136 -> 17 command-line
options, 177 unused top-level definitions and 11 methods gone, every configuration
global frozen to its effective literal.

What is in it: the dense 6-layer x 1024-wide GPT (RoPE, QK-norm, ReLU^2 MLP, value embeddings on
alternating layers, learned residual/x0 mix, causal two-tap local conv), Muon + AdamW with the
compiled Muon body, the fused rope+norm, softcap+cross-entropy and conv NKI kernels, the BF16
branch-input precision policy, the fused-MLP training adapter, the two-core NKI CE adapter and
optimizer ownership across the four LNC2 ranks. NKI kernel bodies are byte-for-byte the verified
originals (`nki_kernels_changed: []` in build-report.json).

How it was made: `python research/build_lean_submission.py --write` (mechanical AST transforms:
argparse pruning, constant freezing, branch/expression folding, dead-definition pruning, single-
call-site and method-parameter specialisation; three asserted hand edits listed in
build-report.json `hand_edits`). `review/*.diff` holds the per-definition diff against the source
for every kept definition whose AST changed.

Verification so far (no Trainium involved):
- static gates: compiles; organiser `check_submission` clean; no undefined names; re-simulated
  effective configuration equals the source's; no removed name still referenced; kernels unchanged
- CPU equivalence (`cpu-equivalence.json`, research/scratch/lean_submission_cpu_equiv.py): both
  files trained on CPU from the same synthetic data with a frozen clock (the LR/WD/momentum schedules
  are wall-clock functions); every tensor in final.pt bit-identical and the logged losses identical;
  load_for_eval logits identical for the lean file on its own and on the original file's checkpoint;
  fused-MLP adapter outputs/gradients identical in bf16; optimizer-ownership adapter identical over a
  2-rank gloo group; Neuron-only glue AST-identical

Still required before submitting (hardware, not run here): `python prepare.py check`, the CPU and
Neuron contract checks used for the source package, a 20M-token public replay (expect 0.99285 within
run-to-run noise, since the arithmetic is unchanged) and a cold dress rehearsal with the exact
launch command below.

Launch command (byte-identical to the source package):

    NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
