# Independent review: remaining gatefixed500f R12 arms

Verdict: **PASS_SOURCE_CPU_ONLY** for clip0, QK1.2, warmdown40, and bigram8192-direct. Bigram remains **COMPILE60 ONLY**, with no full-run authority. This receipt neither launches nor adopts an arm. Expected Trainium gain is UNKNOWN; compare fresh authentic 500f, not a combined delta against historical c239.

Reviewed 2026-09-27 09:43:28 UTC. Independent verifier did not implement or alter the candidate sources, use hosts/devices, or change active machinery. Full source/artifact bindings and executable results are in the adjacent JSON.

| Arm | Source SHA256 | Own source/CPU evidence |
| --- | --- | --- |
| Clip-off | `31595afc34617e401977edca06749adf7aa98307d10b740ddb0f21cd94b3faf9` | Only GRAD_CLIP 1→0. Actual sync path averages every nonmissing gradient, including parameters outside the clip list; three synthetic four-rank cases preserve zero/small/large averaged gradients without clipping. |
| QK1.2 | `0c02018753cba1947c3ed7f75e7871feab1e4fa15ab62258522e06b44289fe63` | Actual post-normalization SDPA scale 1.2/√128 = 0.10606601717798211, not two 1.2 factors. Independent masked-softmax/BF16 integration error ≤0.0001220703125 forward and 0.00048828125 input gradient; two FP64 formula/gradient cases ≤1.3877787807814457e-15. |
| Warmdown40 | `43511a00ea3b0e306b87904ce3ef2779ed0b129d47d4c710b4eb3318f75f924e` | Only .75→.40. Sixty schedule boundaries plus five warmup-branch cases; decay begins at 60% charged progress (1080/1800s), final fraction .05, default warmup 0. Actual main group loop changes every existing group's LR; AdamW WD/betas/eps and Muon WD/momentum/beta2 unchanged. |
| Bigram | `584f38a7f047e415113ff56000e05ac9771ca643b90a32e1200780050542ee0f` | 256 Python-integer hash pairs; row reset, sequence1 and future exclusion; 1025 colliding BF16 contributions sum to FP32 1.25. Flag-off state/output/gradients/optimizer policy and actual checkpoint reload match parent. Zero table preserves initial output/common gradients; scale gradients correctly start at zero, table gradients nonzero. One lookup in train/eval; independent per-use forward oracle bitwise. |

## Common independent checks

Exact original science EDITS applied to 500f reproduce each source byte-for-byte; reversing them restores 500f, and removing only gate routing restores the original c239 science source. main/save/load/sync/clip AST is unchanged. All matrix gates remain dedicated AdamW lr .021, betas (.9,.95), eps 1e-10, WD0; scalar U-Net gates retain lr .2. Actual production-width meta model/group/owner inventories cover every parameter exactly once: 65 tensors / 113,256,724 elements for the three scalar/scale arms; 67 / 121,645,340 for bigram. Existing parameter policies match 500f. Bigram table follows embedding AdamW (production lr .2598076211353316, betas .8/.95, WD .001); its 1-D scales follow scalar AdamW .2, .9/.95, WD0.

All four actual tiny eight-layer models pass finite nonzero gradients under a nonzero fixture, two BF16 CE boundaries, future-prefix causality, batch-row independence, sequence1, and the actual strict weights_only load_for_eval path with same-dtype output parity, eval mode and frozen parameters. The bigram zero-table case separately expects zero scale gradients. These are synthetic logic checks, not training results.

## Caveats and limits

The specs contain two nonblocking template inaccuracies: irrelevant bigram prose in the other three arms, and 'defaults unchanged' despite the named clip/warmdown default change. QK unconditionally overrides even a positive config.attn_scale; only the intended fixed-scale setting is reviewed. Bigram intentionally adds embedding/scalar optimizer entries, while existing entries retain 500f policy. No candidate source correction was needed.

The reference initially supplied an incompatible FP32 direct-attention fixture; correcting the fixture to the actual BF16 input contract resolved that harness failure. The final complete independent run passed in 1.197 seconds on Torch 2.13.0+cpu, one CPU thread. No native/compiler call, real distributed collective, training/evaluation data, full-model optimizer update, or operational edit occurred. No new native prerequisites are imposed; ordinary existing proposal/role/preflight guards still apply. Bigram native indexing/compiler viability remains UNKNOWN, and prior failed compiler evidence is not reversed by this CPU pass.

Own reference: `research/v2/research_cycles/20260927-r12-remaining-gatefixed-reference-verification.py`, SHA256 `d8a6e742c49bd5e68a9dae6e406f6a7d211f34f0862f2989d40616095146371f`.
