# Round13 independent gate-routing verification

Clock read: 2026-09-27 08:25:34 UTC / 13:55:34 IST.
Scope: independent exact-source finding and a conservative group recommendation;
no candidate implementation, training, hosts, queues, ledger or state changes.

**CONFIRMED CONTRACT-ROUTING MISMATCH:** exact c239 sends the FP32
`transformer.h.7.attn.ve_gate.weight`, shape[8,32], to Muon. This conflicts
with Round13's explicit instruction that gate matrices must not use Muon.
It does not establish that the previously measured c239 BPB is invalid or
that the correction will improve BPB.

Parent source:
`research/v2/operator/20260927-c239-qualification/stack-ramp-bare/snapshot/train.py`,
SHA `c23935404ac44314e333ddf3608a6c671f4526263a3348eb901bb3f8d4ca05b0`.
At line706 the gate is a Linear32→8; line904 recursively includes all
attention parameters except `head_gate_bias` in matrix_params. Consequently
`ve_gate.weight` is not excluded. Shapes then determine Muon groups. The
gate is used at line718 in2*sigmoid, not a scalar mixing coefficient.

The independent check executed the actual extracted model classes and
setup_optimizer at production topology on the Torch meta device. It found65
parameter tensors in9 groups, every tensor exactly once. The only matrix gate
is the256-element layer7 VE gate in group5: Muon, LR.021, initial WD.2. The
four1-D U-Net skip gates correctly remain in scalar AdamW group2, LR.2, WD0.
Do not move those scalar gates as part of the matrix-gate correction.

## Recommended isolated routing correction

Use a dedicated gate-only AdamW group with the repository's existing gate
policy: `lr=MATRIX_LR` (currently.021), `betas=(.9,.95)`, `eps=1e-10`,
`weight_decay=0`. Preserve FP32 masters/moments, initialization, cache handling,
ownership framework and the existing shared LR schedule. Do not use
SCALAR_LR.2, embedding width scaling, a new LR search, parameter banking,
compiled-Adam fusion, or any forward change in this correction.

These constants already exist in `train.py:3322` (SHA
`52ac9e6560a40592255e3499f4d42ac6e7073ea82da1b428e03dc984e6d4baa8`) and
`workflow/proposals/research-v2/train.py:2187` (SHA
`6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`).
They are an inherited local policy, not a claim of optimal gate LR. The gate
changes from Muon's scheduled cautious decay to that AdamW group's zero
decay; record this explicitly as part of the group reassignment, not as an
unmentioned numerical equivalence. All other parameters' groups stay fixed.

The independently opened [modded-nanogpt PR187](https://github.com/KellerJordan/modded-nanogpt/pull/187)
supports investigating gate→Adam with FP32 moments. Its reported combined
improvement also includes parameter banks and compiled Adam; the author says
the gate move alone did not clearly improve loss. Neither its combined GPU
result nor its hyperparameters prove a Trainium gain. This recommendation
uses the existing local policy instead of importing that package.

## Executed CPU reference and remaining gates

Command:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/research_cycles/20260927-round13-gate-reference-verification.py`.
Script SHA `30e102e7ebfd2e2dbb86cb2cb1dceec921b5ae68afaff14595b614784dfa0e69`.
Exit0 in1.45s; Torch2.13.0+cpu. Meta grouping confirms the actual source
assignment. Four synthetic8x32 updates using the extracted existing AdamW
implementation match an independent FP64 recurrence within8.74294e-9;
both moment tensors areFP32 and their step counter advances exactly once.
This is a tiny update reference, not training or host2.11/native evidence.

Candidate still needs its own exact diff and independent review: all matrix
gate identities absent from Muon and present once in gate AdamW; no missing
or duplicate parameters; non-gate group settings unchanged; ownership/cache
and checkpoint invariants; appropriate OFF/control evidence, production CPU
checks and original native/cold/budget/fresh20M gates. Bind to the actual
qualifying shape parent after current runs; never silently apply c239 bytes
to another architecture. Root controls the confirmed-fix priority.

No NorMuon orientation bug is claimed here. Canonical MLP c_fc[4096,1024]
and c_proj[1024,4096] are different orientations; the independently opened
[PR168](https://github.com/KellerJordan/modded-nanogpt/pull/168) discusses layout
changes that cannot be judged from dimension numbers alone. The separate
canonical-axis audit must establish any mismatch before changing it.
