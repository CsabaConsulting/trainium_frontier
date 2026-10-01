# Independent Round13 gate-only candidate verification

Recorded: 2026-09-27 08:30:34 UTC. Reviewer: separate standing source/legality verifier. Author receipts were inputs, not independent evidence. No candidate, queue, ledger, state, launcher, evaluator, data, or host was changed.

## Verdict

**PASS_SOURCE_CPU_ONLY** for candidate SHA256 `500f80fec9eac2a42d6f0eae646b0e91643d46c4524ee3722e332434751fd368`, relative to exact c239 parent `c23935404ac44314e333ddf3608a6c671f4526263a3348eb901bb3f8d4ca05b0`. No source/CPU blocker found. This is the independent implementation verification, not launch authorization, native qualification, a BPB win, or an adoption verdict.

The Round13 directive explicitly excludes gate matrices from Muon. This candidate implements only that requested routing/policy correction. It does not touch evaluator, validation data, scoring, launch command, model architecture, forward path, or the existing optimizer implementations. No new Claude legality decision is needed for the reviewed diff; any later expansion beyond this hash must be reviewed separately. Normal approved machine/proposal/native/artifact gates remain the Operator's responsibility; this review adds no separate native experiment requirement and must not block already-ready host jobs.

## Exact diff and policy

Independently removed the two intended added blocks (five source lines total) and recovered the parent byte-for-byte. All additions are in `GPT.setup_optimizer`: collect named two-dimensional gate parameters, remove their identities from Muon matrices, and add the inherited dedicated AdamW group.

Actual production-parameter meta inventory: **65 unique parameter tensors, 113,256,724 elements, 9 optimizer groups**, every parameter present exactly once. Width 1024, 8 layers, 8 heads, vocabulary 8192; the meta inventory used an 8-position rotary buffer, not a native context-1024 execution. There is exactly one matrix gate: `transformer.h.7.attn.ve_gate.weight`, shape `[8,32]`, FP32. Only its policy differs from c239:

| Property | c239 | Candidate |
|---|---|---|
| Optimizer | Muon | AdamW |
| Base LR | 0.021 | 0.021 |
| Moments | momentum 0.95, beta2 0.9 | betas `(0.9,0.95)` |
| Other math | NS steps 5 | epsilon `1e-10` |
| Weight decay | initial 0.2, existing scheduled cautious Muon decay | 0 |

The move therefore includes cessation of the gate's Muon weight decay, not merely an optimizer-name change. The candidate retains the existing global LR multiplier through `initial_lr`; it does not receive embedding width scaling or scalar LR 0.2. All 64 non-gate parameter policies are identical. The four one-dimensional U-Net gates remain in their original scalar AdamW group at LR 0.2.

A synthetic named `[128,128]` extra gate was routed to AdamW while a same-shaped attention query matrix stayed in Muon; all 66 parameters remained unique. This verifies identity/name selection rather than a shape-based exclusion.

## Independently executed checks

Verifier script: `research/v2/research_cycles/20260927-r13-gate-candidate-reference-verification.py`, SHA256 `de23f3db139604a64fea84bbbdeade7a7fbbb716de0a50946143a572a084f426`.

Command:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/research_cycles/20260927-r13-gate-candidate-reference-verification.py
```

Exit 0; local Torch `2.13.0+cpu`; reported check-body time 0.9818 seconds. It hash-binds and AST-extracts actual candidate and parent definitions, without importing their runtime or `prepare.py`, reading datasets, compiling for Neuron, or accessing a host.

- Exact source reverse-diff, complete parameter coverage, FP32 parameter dtype, and unchanged non-gate policies: PASS.
- Actual unchanged `make_plan` with four owners: all 65 tensors covered once; owner tensor counts `[10,24,20,11]`; element counts `[27262976,28322058,27262986,30408704]`; packed offsets contiguous/nonoverlapping within each owner; shard size 30,408,704 elements. Gate is AdamW group 3, index 22, owner 1, offset 28,321,792. No real distributed collective was executed.
- Tiny synthetic actual-model comparison (`B=1,T=7,D=128,H=8,L=8,V=128`, seed 42, MTP weights `[1,0.5]`): loss `4.84840726852417`; candidate and c239 forward output and all 65 parameter gradients bitwise equal and finite before optimizer changes. This is a CPU logic check, not a performance or quality measurement.
- Altering suffix tokens leaves the first four outputs bitwise unchanged: PASS. In-memory `weights_only=True`, strict state-dict load and output parity: PASS.
- The actual candidate gate parameter and actual candidate-created AdamW group were isolated in the unchanged optimizer. Four varying-LR steps `[0.021,0.0105,0,0.00105]` against an independently written FP64 Adam recurrence: maximum absolute error `1.6875307179420318e-08` (threshold `5e-8`); exactly one state-counter increment per update; both moments FP32; all other model parameters unchanged by these gate-only steps. A fresh nonzero gate with zero gradient is unchanged, confirming no multiplicative weight decay.

An initial harness invocation mistakenly called the full model optimizer while deliberately excluding the Muon function/cache definitions. It stopped at `NameError: _MUON_COMPILED` after the gate update. The verifier harness was corrected to isolate the actual gate group as intended, then rerun cleanly. This was a verifier-harness failure, not a candidate/native failure; the passing run does not claim a full-model optimizer step.

## Limits and handoff

- Source SHA and author input receipt hashes were independently checked. Author CPU and static receipts were not counted as verifier execution. `CPU_RECEIPT.json`: `b6d93132df3c1ac1440ecafbf6f9f24419624dea38f4f737214b4d8d3b8ade1d`; `STATIC_RECEIPT.json`: `8f44278c89771b3d8fe79b5eb1b4cc6464a151536240bd48e432d3d732b91197`; `SPEC.md`: `b734524b86224d8730033ac4dced932e5340d79cef8d72c1724aa2e5f5edc2d5`; proposal stub: `b60bcb30c1d65f3946523d1f2253bb4168653a17539575c958e587013b153828`.
- The author proposal is an implementation stub, not a completed native run packet. The Supervisor must bind the exact hash into its normal unique-label proposal and unchanged approved launch command. This receipt does not independently approve a command/environment/host not supplied here.
- Changing groups changes owner assignment and optimizer-state layout. **Fresh start only** relative to c239 optimizer state; do not reuse an old owned-optimizer checkpoint. Model weights/architecture are unchanged, and the tested strict model-state load is separate from optimizer-state compatibility.
- No Neuron/installed-host-SDK execution, full-model optimizer update, distributed communication, native timing, local BPB result, or gain estimate is asserted. Local CPU Torch differs from the previously retrieved host Torch `2.11.0+cpu`. Finite full-chip behavior and wall-clock quality remain empirical.
- [modded-nanogpt PR187](https://github.com/KellerJordan/modded-nanogpt/pull/187), independently opened in the preceding routing audit, supports the direction of moving gates to Adam; it does not establish a local quality gain or uniquely validate these constants. The constants come from the existing repository's dedicated gate policy, documented in `20260927-round13-gate-verification.md`.
