# Independent gate-fixed C2 / C3 / C4 review

Recorded 2026-09-27 08:50:39 UTC. **Each arm PASS_SOURCE_CPU_ONLY**, no source/CPU blockers, against reviewed gate-fix parent `500f80fec9eac2a42d6f0eae646b0e91643d46c4524ee3722e332434751fd368`.

| Arm | New source SHA256 |
|---|---|
| C2 | `dc468dabc77f1a3e0e0af0ce6834c4ade8f553ecc0177bb0d2fe2d6728f8a564` |
| C3 | `6e3bd7f53b201067bdbde3561f326eb5ba753220a882cc5fa9f717b5960c95a9` |
| C4 | `04443101f58208c475539f6793178ea5940500cd10aa009dc8ef101ca89f7b21` |

**All comparisons require the fresh 500f control.** Old c239-based sources are archived isolated drafts, not current launch candidates. This review binds the new sources, not merely the old science receipts. No native/launch/adoption approval or BPB gain is implied; normal completed proposal and Operator checks remain. No extra native experiment is requested.

## Union and gate checks

For each new file, independently removing only the exact five-line gate-routing fix recovers its hash-bound archived science source byte-for-byte. Removing only the original science delta recovers **500f byte-for-byte**. No additional model, evaluator, data, command, clock or optimizer-policy change is hidden in a rebase.

Actual meta models confirm all matrix gates in AdamW. In every arm, `transformer.h.7.attn.ve_gate.weight` `[8,32]` has base LR 0.021, betas `(0.9,0.95)`, epsilon `1e-10`, WD 0. All four scalar U-Net gates retain AdamW LR 0.2. C2 has 63 tensors; C3/C4 have 65. Every parameter is covered once. C4's complete policies and four-owner plan are identical to 500f before the declared schedule changes.

## Independent rerun

Ran one local process, in order **C2 → C3 → C4**, first invocation exit 0, total body 2.4688 seconds, Torch `2.13.0+cpu`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/research_cycles/20260927-r13-gatefixed-c2-c3-c4-reference-verification.py
```

Wrapper SHA `b0c131245a4f46454a5f083c6c3e0bf95e5f6889c8ed10b7889ec5c971ec44c2`. It hash-pins and re-executes the verifier's own previous reference programs on the new source hashes/500f parent; it does not replay old outputs or run author tests. Adaptations are explicit, in memory: source bindings, the inherited-gate report label, and C4's added gate AdamW template count. Original reference files and all candidate files remain unchanged. Reference SHAs: C2/C3 `136e0c94d6e1c8fa91e79c4c9fd0279b9ccb518d99089f19e9fd16b00e7253ae`; C4 `4a55100b1a540fe7aa671d6c43c67edeaca1e1a22cb57a67f782119ebb574f39`.

- **C2:** 63 tensors / 104,868,116 elements; exactly seven MLPs; layer-0 attention/convolution/scalars and skip `0→7` preserved. All 63 gradients finite and nonzero in the tiny synthetic connectivity case; explicit layer-0/full-forward oracle, all four skip pairs, prefix causality, sequence-one output and actual `load_for_eval` strict/weights-only round trip pass. Old-parent model strict load appropriately rejects the two removed MLP keys. Same-seed shared initialization changes 30 tiny-model tensors; this documented RNG consequence is not a blocker or paired-initialization claim.
- **C3:** 14,784 actual schedule-group cases pass; maximum independent LR error `5.551115123125783e-17`. All Muon fields are exactly 500f-equal. Only LR changes for AdamW; fixed decay/betas, original warmup, final fraction 0.05 and `no_lr_decay` override remain intact. **The matrix gate now also follows the 40%-length AdamW cooldown**, as required by the correct composition; it is no longer a Muon exception. Parent model strict-load and actual eval round-trip pass.
- **C4:** 9,792 actual schedule-group cases and 1,001-point Muon curves pass. Seven distinct scalar policy templates now include six AdamW templates, hence more cases than the archived draft's test. Maximum LR/momentum/WD errors are `5.551115123125783e-17`, `2.220446049250313e-16`, `2.7755575615628914e-17`. The matrix gate receives the requested 40-step LR warmup but retains AdamW WD 0; cosine decay affects Muon only. The original C4 overlap semantics remain: momentum blends its currently warmed value toward 0.90 over trailing 75% charged time, with no guaranteed 0.97 plateau. Step-0–39 LR warmup takes precedence over charged cooldown; step 40 returns to the original schedule. Model/eval/optimizer implementations and clock/stop/data AST remain unchanged apart from the declared bundle.

No native compilation, real collective, full-model optimizer update, host, training/validation dataset or evaluator was exercised. Synthetic CPU losses are logic-check values, not BPB. Local CPU Torch differs from the previously retrieved host Torch; native performance remains empirical. The JSON records the exact author input hashes but does not count author receipts as independent execution. No candidate, queue, ledger, state or launcher was edited.
