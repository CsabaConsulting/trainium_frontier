# K1: two distinct, unselected throughput hypotheses

Researcher-only note, 26 September 2026, approximately00:09UTC. Exact adopted source is `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b:workflow/proposals/research-v2/train.py`, SHA256 `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`. No production code, tensors, hosts, launcher, evaluator, source gate or ledger was changed. This note is **not a completed native experiment proposal or implementation assignment**. No pending decoded P output is assumed.

## Recommendation

**Insufficient evidence to select a >=5% whole-step optimizer arm now.** There are two distinct testable ideas below, but neither has a defensible measured speed forecast. The smaller first idea targets a different copy from the parked owner-copy omission; the second makes the already-mentioned shape-aware-owner possibility concrete. Neither re-proposes literal batching, existing ownership, singleton-unpadding, or the final owner writeback omission. Do not spend a native slot merely because a redundant operation or an algebra-count reduction exists.

The exact source already batches same-shape matrices, compiles the whole Muon body, stages five scalars together **per group**, computes only on owners, and performs one padded all-gather. P observed three compiled Muon calls per rank. `_muon_math` lines2302–2373, `_muon_step`2861–2921, `make_plan`4558–4584 and `OwnedOptimizer`4586–4664 were read directly from the pinned source.

## What the evidence actually says

Existing evidence: `research/v2/lanes/k1-muon-batching/{STATUS.md,AUDIT.md,plan.json}`, `research/v2/analysis/20260926-p-runtime-interpretation/{ANALYSIS.md,findings.json}`, and `research/v2/analysis/20260926-k1-writeback-runtime/{ANALYSIS.md,result.json}`. `EVIDENCE.json` pins the relevant files plus all four original runtime JSON hashes and records a new bounded, read-only correlation of Muon's input-weight copies.

| Rank | Real square1024² / tall4096×1024 / wide1024×4096 | Sum of Muon graph intervals, ms | Sum of input-weight-restage intervals, ms | Owner gather interval, ms |
|---|---|---:|---:|---:|
|0|6 /1+dummy /0 (plus three8×32 gates)|7.574627|2.978120|6.925985|
|1|5 /2 /2|10.988568|3.834161|8.983221|
|2|5 /2 /2|11.611239|4.348668|8.811717|
|3|8 /1+dummy /2|21.297725|5.367100|19.006368|

These are correlated **runtime device-labeled intervals**, not decoded instruction-active or exclusive costs. Columns and nested spans must not be added. Rank0 starts its forward annotation about1.24s late during profiling; the other ranks'1.25s clip/reduce host ranges reflect waiting. The actual steady critical rank is therefore **unresolved**. Especially, rank3's batch2 tall graph spans13.741601ms while rank0's same batch2 tall shape spans3.818947ms: static shape alone cannot explain or assign that difference. No MFU or unmeasured dtype claims follow.

The restage mapping is an inference from the exact source order: each Muon host group contains exactly three foreach copies (gradient-in, weight-in, output-to-live). Select the second, then correlate its External id to exactly one `privateuse1_driver` event. All four input files were size/SHA-checked again. Rank3's selected External ids2129/2210/2279 map to event49029/49066/49103, with0.203850/2.992547/2.170703ms. Their host calls are separately0.472267/0.381160/0.375822ms, **not extra device time**. No candidate or numerical test ran.

## H1 — retain the previous Muon output stack between ordinary updates

Small reversible candidate, if selected later: default-OFF, compiled FP32-owned-Muon-only. Skip only `_muon_step`'s steady `stacked_params <- live params` input copy after a successful initialized update. Keep gradient staging, compiled math and scalar arithmetic, live-parameter output copy, send packing, full final writeback and all-gather unchanged. Do not alias or replace any live Parameter storage.

Reason: lines2917–2920 retain the new `w` in `state['stacked_params']` and copy precisely it into the live owner parameters. The next ordinary step recopies these same values back. The owner exchange returns those identical owner values; cache refresh reads/casts weights without updating FP32 masters. This is a value-equivalence hypothesis for the fixed training path, **not** a general guarantee under external weight mutation, state loading, EMA weight swaps, device/dtype migration or alternate optimizer paths.

An implementation must refill on first use, state/model restore and invalidated ownership/state; unsupported mutation paths must retain baseline behavior or refuse the opt-in path. Do not introduce an unvalidated `_version` heuristic as proof. The present optimizer state_dict omits scratch stacks; resume must refill, and an in-memory load must not retain stale scratch. Default/OFF must retain exact existing statements. The source's None-gradient-to-zero behavior and padding must remain unchanged.

**Own CPU test plan if assigned:** independent extracted reference, single thread, <=64MiB tensors; small square/tall/wide/gate batches including dummy batch2. Run6 fixed-schedule updates with varying LR/WD/momentum/beta2, None and explicit-zero gradients, a skipped full step, skipped AdamW, state save/load into fresh and already-used optimizers, and explicit out-of-band parameter mutation as a negative/invalidation case. Compare every FP32 master, momentum/second state, scalar operand, BF16 cache and checkpoint/reload logits byte-for-byte against baseline; verify live Parameter identities remain unchanged. Then three-step tiny actual-loop OFF/default identity and ON equality. No tolerance widening is justified for merely omitting a redundant copy.

**Expected bound:** under an unchanged-schedule counterfactual, removable time cannot exceed the actual exposed restage work; P's2.98–5.37ms summed intervals are not even that exposed time. They do not support a >=5% whole-step gain. This is the lowest implementation-risk experiment of the two, but presently **low expected payoff and not selected**. Do not combine it with the rejected owner-copy omission to manufacture one result.

## H2 — two whole-matrix owner exchanges to remove tall dummy work

Distinct compute/placement hypothesis; this refines the prior unselected pairing sketch, not a newly discovered absence of ownership. For this exact six-layer inventory only: move the rank3 tall matrix (group position2) to rank0, and move rank0 square position2 to rank3. Tall real counts become `[2,2,2,0]`, squares `[5,5,5,9]`; wide counts stay `[0,2,2,2]`. No tall singleton is executed or unpadded. Two dummy tall matrices disappear,39 real matrices stay, local Muon calls fall from12 to11, and rank3 loses its entire tall group while gaining one square matrix.

Arithmetic metadata gives owner element counts `[30,411,524,30,414,851,30,412,805,26,214,400]`. Maximum owner still rank1, so padded stride30,414,976 and received486,639,616bytes stay unchanged; it is not a communication-volume saving. Offsets and ownership metadata must be regenerated deterministically, with all62 parameters assigned exactly once. Existing optimizer checkpoints with a different plan must remain rejected, never silently remapped.

This removes193.27G five-iteration matmul **algebra operations** for two dummy tall rows, not measured hardware FLOPs or wall time. Rank3's nine-square specialization is new. P's13.74ms tall span may be mostly waiting, and the apparent heavy rank may change. Both numerical behavior under changed batch geometry and the whole-step benefit are unproved. Historical singleton failures prohibit inferring native equivalence from CPU algebra.

**Own CPU plan if assigned:** separately reconstruct both owner plans from metadata; assert unique ownership, exact element/offset coverage, no tall batch1, unchanged gather bytes. Independently run small four-owner Muon references for6 updates, comparing each real parameter and its moments by original identity, with None/zero gradients and exact dynamic scalars. Preserve zero dummy state and state_dict mismatch rejection. Default/OFF actual-loop identity is mandatory. Native batch9 versus baseline and per-matrix/reference checks would still require the existing independent/native gates. This is broader and riskier than H1; no selected implementation or speed prediction.

## Whole-step rule, only after normal qualification and separate assignment

Keep the existing **>=5% faster whole-step** requirement: precommit `median_off / median_on - 1 >=0.05` on a same-host matched pair with identical exact source, data/seed42, microshape, global batch, flags, fixed warmup/measured window and complete synchronized update work. A possible minimal window is20 warmup +80 measured updates per arm, but it must be bound in the selected experiment before launch. Report raw distributions/all-rank progress, compile/recompile status, startup/capped charge and checkpoint/finite outcomes; profiling-skewed P windows and kernel-only timings cannot pass. Add a replicate only if uncertainty or drift makes the threshold inconclusive; otherwise discard a subthreshold arm. Correctness, CPU/default-OFF, fresh independent review, native checks and official900/1800 accounting remain intact. A speed gate is not BPB adoption; any full1800+fresh20M work needs a separate selected proposal and comparable control.

Other tempting restatements were excluded: final owner-copy omission remains unselected; aggregate scalar batching is already present per group; flat gradient bucketing already exists and historically hung on native (results lines1814–1838); NS-only splitting already regressed. No broad fusion/alias framework or new profiler run is proposed here.
