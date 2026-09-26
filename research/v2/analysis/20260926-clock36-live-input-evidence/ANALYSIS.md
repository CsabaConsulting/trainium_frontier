# Full36 retained-cache startup: actual live inputs, not yet cache causality

Analyst-only, local archived evidence; no host access, native job, cache scan/delete, source change, SDK import or compiler-input rewriting. Exact source **36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb**, Git7419927c, workflow/proposals/research-v2/train.py.

**Bottom line:** real compiler pipeline work occurred for two distinct binary inputs while using the retained cold36 cache roots. The repeated ~13min first step is not merely slow loading of already-compiled artifacts. But the archive does not yet establish the complete principal-graph mapping, persistent-cache lookup outcome, or why exact prior NEFFs were not reused. It does **not** show the old second-step cache-state specialization returning.

## Measured facts

- Early snapshot59bdcace, research/v2/operator/20260926-trn22-startup-clock-full/early-training-compact-snapshot.json, SHAe0d702df2ee68509d949717d88501b919b1135ca7febb2007a422fc2ef7ae1ad: printed step0 **783.64s**; steps1–4 **0.49s** each. This is source step duration, **not** exact launch-to-fifth, compilation subtotal or charged-through-exit; final harness receipt is pending. Cold36 had printed step0 816.95s and exact fifth845.627337750s (e08afdec bundle SHA6c45190b3c1db2242073fdb04a1a5ca997ec2eb51c7abf655e70488b0b0f7e18).
- The full launch and all four live worker environments use exactly the six roots from that cold receipt, including NEFF/HLO/local/compiler/debug, URL=compiler. Retained directory names establish intended locations, not entry existence/readability or a graph-specific hit.
- Captures belong to compiler PIDs3191465 and3191631, each descended from authenticated current rank0 worker3191056. They entered HLOToTensorizer/Frontend/StaticIOTranspose/WalrusDriver/NeffWrapper; logs start23:31:23 and23:34:57UTC. Captures were23:33:55 and23:36:17. Log prefixes contain actual partitioning/Tensorizer activity, not merely a stale cache filename.
- Both commands explicitly include **-O2**, **--auto-cast=none**, followed by **--optlevel=1 --auto-cast matmult --auto-cast-type bf16**, targettrn2/LNC2. Do not silently report a lone -O2 or claim a new flag mismatch: the earlier3e79 evidence records the same compound command. Compiler version2.26.6360.0+6f180f47; captured Tensorizer options include --fp32-cast=matmult-bf16. The archive does not provide a separately parsed final optimization-level field.
- Each complete binary input begins with MLIR bytecode/MLIR22.0.0git. Only printable strings and embedded JSON were inspected; no semantic MLIR decoder/normalizer was used. All before/open/after/path identities and SHA checks agree. Each compiler log is only the first65,536bytes, from full sizes3,569,926 and715,045bytes—not its completion or full timing record.

| Capture | Complete module bytes / SHA256 | Additional exact identifiers |
| --- | --- | --- |
| A, bdde21c9 observation-1.json |17463 /5a0c39fda4d4962afd222db766c1b2eda6ad6ed7d0b0fd1889afc42edc9a4e3c|directory5818832e…; compiler log module IR signature61720c5b…|
| B,9b525f5f observation-2.json |20784 /f21636e78487960b13e9682fbfd8cc987854ffc2da4d7df7ad314474ec467848|directory9cb2b39a…; compiler log module IR signaturee59fc5a2…|

These three identifier types differ. **Neither directory suffix nor compiler IR signature is the raw module SHA or an authenticated persistent NEFF key.** No captured prefix contains Persistent cache hit/miss or execution_cache_key messages; missing debug messages mean unknown.

A contains metadata named _rope_norm_fwd, _conv2_forward, attention_cte and build_kernels.forward; B contains _rope_norm_bwd, attention_bwd, _conv2_backward and build_kernels.backward. These are strong direction cues, **not yet an exact join to full-run model_default forward/backward rows**. The compiler HLO instruction counts702/1303 likewise must not be equated to FX node counts.

## What source and prior evidence support

Both modules embed the full run's absolute …/fs-20260926-arch36a8-startup-clock-trn22-r1-full-packet/train/train.py. Cold source uses a different unique sandbox. Archived SDK compile.py:111–143 strips only torch.debug_dump_path then hashes serialized bytecode (plus version hash under the archived default route). CompilationCacheKey.cpp:76–105 additionally derives a persistent key using compiler configuration. Thus location/backend metadata embedded in bytecode is **credible changing key material**, even when source bytes and six roots match. No cold36 binary input was captured for a semantic field-by-field comparison; source-path-only causality remains unproved. NKI backend_config variability and other key/config material remain alternatives.

Already closed cold36 CSVs label one666-node model forward and one1169-node model backward, shared across all four ranks, with prefixes9836e351be15c436667378517bb45d19d032facf0d2a9c5e283c3e63ac0ad6c2 and74f4bcee2eb7b52306f5b86a3ce3bcf9d6175e7b6707674531e9bd2dc005d03f. Positive compile times are202.864s(rank3) and470.027s(rank0). Neither current captured SHA equals any cold graph prefix. That is an exact comparison of these identifiers; do not yet assert a complete matched-graph key change or reuse outcome for full36 until the closed full CSV join.

The source still refreshes differentiable cached weights before first forward inside t0 (4045–4047) and after every update (4110–4111); Linear cache branch is1486–1490. Model compile is3932. New startup-adjusted schedule function3302–3312 feeds optimizer LR/WD at4083–4098 **after forward/backward**; it does not move compilation outside official accounting or prewarm a model. Fifth-boundary accounting/reset4118–4125 is unchanged. Muon2963–2977 caches compiled bodies by shape/NS layout with tensor LR scalars; the full log reports three shapes. No second-step model or schedule-driven graph explosion is established by this snapshot.

Historical0e783999/3e79 evidence already demonstrated different cold/full principal keys despite retained roots and real positive compile times; its raw binary backward hash matched a CSV prefix. It did not isolate source paths or map persistent NEFF reuse. Stable S1 later failed capture, and its three named directories were absent in0c28f643; do not resurrect S2 or treat absence as proof of earlier cleanup. Zero compile_us remains ambiguous under the archived SDK metrics implementation.

## Smallest NEXT diagnostic—no new run

Use **normal final artifact retrieval from the current full36**, once it naturally exits; no extra live capture or host slot now. Locally:
1. Authenticate the four closed graph_stats CSVs and final rehearsal/log.
2. Join only exact A/B complete module SHA values to graph_id prefixes. Record all matching graph names/nodes/ranks/suffixes and positive compile times; if unmatched, leave identity unresolved. Compare those matched rows with the already archived cold36 principal rows.
3. Check the normally retrieved logs for an explicit key-bound persistent hit/miss/read/write message. None in these prefixes is not proof that none exists in a later raw log.

This is a tiny offline join of existing scheduled artifacts, not another cold experiment. It distinguishes matched new compiler input keys from unsupported graph labels and yields actual startup/compile attribution. If it confirms changed principal keys, the **remaining** missing causal evidence is a matched cold/full binary pair (or an explicit named NEFF lookup record), not another timing-only repeat. Do not propose cache deletion, flags, new logging, normalization, SDK modification or broad NEFF scans from this note.

## Reproducibility

extract_facts.py independently authenticates immutable captures/cold bundle/early snapshot, decodes only archived bytes, verifies all six roots for all workers and exact source SHA. FACTS.json retains38 explicit checks, kernel metadata, numbered log excerpts, all cold graph rows and SHA pointers. No full-run quality result or native speed win is asserted.
