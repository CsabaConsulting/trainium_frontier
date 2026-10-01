# Trainium Frontier Phase 1: full context brief (Wed 2026-09-30, 01:40 IST)

## 1. The competition

- Task: train a language model from scratch in 30 minutes of charged training time on one trn2.3xlarge (one Trainium2 chip, LNC=2, so four data-parallel ranks), with torch.compile(backend='neuron'); NKI kernels are allowed inside train.py.
- Data: a fixed ClimbMix subset, fixed 8192-token BPE tokenizer, pre-tokenized by the organisers' prepare.py. Metric: validation bits per byte (val_bpb) on about 20M held-out tokens. The evaluator slices a contiguous stream of full documents into 1024-token windows (Codex verified this in prepare.py tonight; documents are not cropped and windows do not need to start at BOS). Zero-byte tokens (BOS) are not scored.
- Only train.py may be changed. prepare.py, the evaluator, the tokenizer and the data are fixed. Compilation is excluded from the 30 minutes but cold start must be under 900 s. Two warm-up steps are excluded from the charged clock.
- 15 submissions per week. A human reviewer rejects files that look like tampering (monkey-patching prepare functions, eval-only branches, train/eval mismatch) and has rejected one large kernel-heavy file without a stated reason.
- Deadline: Thursday Oct 1, 12:29 IST. Our hosts (trn21, trn22) are live until about Thursday 16:00 IST.

## 2. People, tools and the working protocol

- Poojith: team member, makes every upload himself, only after Claude's audit. Uploads only something leaderboard-worthy (not 9th or 10th if avoidable).
- Csaba: team head, owns the AWS account. Has stopped his seed sweep, has no research of his own to report, offered to look up conceptual questions. A short reply to him was drafted (status, the eval-packing question, the ask for a third instance).
- Codex: an agent on Poojith's laptop that runs the two hosts. It builds packets, reviews sources, launches runs, watches them, records results in $HOME/mnt/Trainium-STATE-LIVE/STATE.md (newest entry at the top), and asks questions as "QUESTION FOR CLAUDE Qnn".
- Claude (me): director. Writes numbered Rounds into $HOME/mnt/Trainium-STATE-LIVE/CLAUDE_ROUNDS.md (Codex reads it every 10 minutes, acknowledges with a SHA), answers Codex's questions within 10 minutes, drafts code as tarballs into /home/poojith-devan/Downloads/Trainium-STATE-LIVE/ (CLAUDE_DRAFTS_hhmm.tgz), runs research agents on the side, and checks STATE.md on a 45-minute loop (loop check 27 is due 02:12 IST).
- Codex's harness has strict guards (locks, disk floors, provenance SHAs, symlink checks). Many runs were killed by guards earlier; the rules below were relaxed round by round to stop that.

## 3. Standing rules (must stay in force)

- Only train.py changes. No evaluator or prepare.py edits, no reading validation data, no external data or weights, no tokenizer or SDK change.
- No AWS spend, no new instances (Poojith's rule; a third instance is Csaba's call).
- No uploads by Codex. Poojith uploads after Claude's audit (8-item checklist: default 0 plus header like the scored file, all receipts present, same code path for train and eval, load_for_eval unchanged, no eval-only branch, no monkey-patching, record_step_boundary() after every step with the existing synchronize, cold start under 900 s).
- Never delete best/candidate/package artifacts. Idle-only deletion of CLOSED runs' checkpoints/caches and unreferenced /tmp/nki_* scratch older than 3 hours is approved (up to 60 GiB per host between runs).
- Never write AWS keys to any file or repo. Never copy other teams' code (ideas only).
- Disk floors: runtime 22 GiB (20 for timing runs), FULL entry 26 GiB. No CPU tests on a host during a cold compile. Device gap target 10 minutes between runs; packets are staged during runs.
- Do not exploit the startup-allowance gray area (training harder in the excluded warm-up steps) without the organisers' written yes.
- Writing style for Poojith: short, plain, no em dashes, exact ETAs and expected scores in 12-hour maps, honest odds, never tell him what to settle for.

## 4. Leaderboard (Sep 29, 23:40 IST) and our position

1 Them Bones 0.9356, 2 Team Awesome 0.9507, 3 shiv 0.9538, 4 msamuel 0.9553, 5 Loka 0.9558, 6 Next Token 0.9568, 7 Trainium of Thought 0.9571, 8 Eldrun 0.9586, 9 Quanta 0.9607, 10 DLYog 0.9651. Scores are moving down quickly.

Us: official 0.9713 (file A3, SHA 29e5cd2c, uploaded). Local best 0.967263 public with one receipt (stack CE2 + FLATCACHE + ZBUNDLE, run id e1bd). Official is about 0.0004 better than public. So today we are about 0.9669 official, rank 11.

Gap to #10: 0.002. Gap to #5: 0.011. Gap to #2: 0.016. Gap to #1: 0.031.

## 5. Our model and recipe (the e1bd base)

- 113M parameters: 8 layers x 1024 wide, 8 heads x 128, ReLU-squared MLP 4096 (PyTorch reference forward, not a kernel), attention in layers 0, 2, 4, 6, 7 via compiled F.scaled_dot_product_attention (is_causal, default 1/sqrt(128) scale), U-Net style scalar skips (resid_lambda, x0_lambda), 2-tap causal depthwise conv (NKI kernel) before attention, fused RoPE plus RMSNorm (NKI kernel, bare jit call), softcap-15 cross entropy as an NKI kernel wrapped in torch.autograd.Function via torch_neuronx.wrap_nki (now a two-target kernel for the parameter-free multi-token prediction loss), untied embeddings, gated unigram value embedding at layer 7, one hashed bigram table (262,144 x 256) injected at the embedding and at the layer-7 value path with its own RMSprop update.
- Optimizer: Muon (Polar Express 5 steps, NorMuon, cautious weight decay 0.2 decaying to 0) for matrices; AdamW for embeddings and scalars. Owner-sharded optimizer step with flat send/receive buffers (OwnedOptimizer, all_gather_into_tensor). Flat bf16 weight cache and eager flat tail (FLATCACHE).
- Constants: MATRIX_LR 0.021, EMBEDDING_LR 0.3, UNEMBEDDING_LR 0.006, SCALAR_LR 0.2, WEIGHT_DECAY 0.2, WARMDOWN_RATIO 0.75 (linear to FINAL_LR_FRAC 0.05), Muon momentum 0.85 to 0.95 over 300 steps then cooldown to 0.85 over the last 4.5%, LOGIT_SOFTCAP 15, INIT_SCALE 0.5, ATTN_CPROJ_INIT_SCALE 0.1, MTP auxiliary weight 0.5 decaying to 0 at 2/3 of time, seed 42.
- Batch: micro batch is [32, 1024] per rank (PACK_FACTOR 32: the organiser loader is called with seq_len 32768 and the row is reshaped into 32 rows of 1024), 131,072 tokens per step for the first third of the time, then 262,144 (grad accumulation 2). About 2,849 steps, 564M tokens, about two passes over the data. Throughput about 325k tok/s in the 262k phase.
- Eval and training use the same forward. Tail (optimizer step and cache refresh) about 59 ms per step.

## 6. What is measured to work (public val_bpb on 20M tokens unless noted)

- Hashed bigram table: 0.0054.
- ZBUNDLE (norm computed once per block instead of four times, one gate slice, skip vocab slice, table rebind, persistent scalars, RoPE wrap, full-graph core): +4.9% steps, 0.0022 (0.969524).
- CE2 (fused two-target cross-entropy kernel): +2.99% steps. FLATCACHE (flat master weights, one flat bf16 cast, eager flat tail): +1.4% steps. Together the base went 0.969524 to 0.967263.
- The batch ramp (131k then 262k) was worth 0.0038 over fixed 262k earlier.
- NKI attention kernel pair (nki-library attention_cte forward plus attention_bwd) in a clean harness: 8.84 ms per call versus compiled SDPA 11.65 ms, parity max relRMS 0.003, no recompiles. In-model timing is running now; expected about +3% tok/s if the layout glue does not eat it.
- Exchange rate late in training: about 0.0004 bpb per 1% more steps.

## 7. What failed today and earlier (do not retry without a new causal argument)

- Fused RMSNorm kernel: compiler already at 1.03x of it. Fused MLP epilogue kernel: -1.2%. Gated-add kernels at the value site (VSITE, -0.6%) and the embedding site (EMB, -0.24%): the compiler already fuses these.
- Block-causal split of attention: cold compile over 900 s. bf16-attention compiler flag: +0.03% (nothing). FA2-style attention written in PyTorch (store P): 12% slower. One-graph compiled optimizer tail: compiler instruction limit NCC_EXTP003 (300k). --optlevel 3: cold-start cap. --model-type transformer is already in the compiler argv.
- @nki_op and torch.library wrappers fragment the Dynamo graph (-63%); the only working integration pattern is torch.autograd.Function calling torch_neuronx.wrap_nki(kernel)[lanes](...).
- Schedule arms from the speedrun record: phase-2 LR scaling +0.0015 worse, table-LR warmdown +0.0018 worse. Asymmetric softcap +0.0003 (noise). Byte-weighted CE +0.048 worse. Attention score bucketing -8% speed.
- Multi-table (two hashed tables) net worse. Width 1152 parked after a profile; head dim 64 worse; MLP ratio 3 and 6 worse at an older config; SwiGLU worse; MQA worse; epoch reshuffle within noise.
- Compiler fp8 auto-cast: training diverges (bpb 3.2). bf16 auto-cast of everything: faster but worse quality (Muon math needs fp32).
- Red-team verdict tonight: attention backward is the game; document masking, head gates and the ANVIL optimizer rule were dropped as not worth slots; the record's endgame (LR floor 0.30, warmdown 0.80, tail EMA of embeddings, tail averaging of matrices with decontraction) is being run as a FULL now.
- Wild-ideas round: EVALPACK (pass-2 eval-identical packing) was dropped after Codex read the real evaluator (our rows already match it); the n-gram oracle was dropped (it would read validation data); progressive stacking, layer dropping, mixture of depths, Rho-1, token dropping, MoE, byte heads, retrieval were rejected on evidence.
- Screening: 15-minute screens with a 5M-token eval cannot resolve differences under 0.001, so every science arm is now a full 55-minute run.
- Prefetching the loader before backward: the organiser loader reuses its input buffers in place, so the draft was unsafe; a background thread version measured -1.6% earlier. Parked behind a 60-step timer of fetch() cost.

## 8. The wall, in numbers

Per rank per micro-step (32,768 tokens) the model needs about 21 TFLOP. The step runs at about 370 ms per micro plus the tail, which is about 57 TFLOP/s per logical core, roughly 34% of the bf16 peak of a two-core LNC=2 logical core (about 167 TFLOP/s). The MLP side runs near 50% MFU; the five attention layers cost about 40 ms each, of which the attention core (SDPA forward plus backward) is only 11.7 ms and the projections should be about 9 ms; the remaining 19 ms per layer is layout glue (transposes between [B,T,H,D] and [B,H,T,D], RoPE-norm boundaries, contiguity copies). That glue, about 95 ms per micro (25%), is the largest identifiable target, and the NKI attention pair with a matching layout is the only lever aimed at it. HBM traffic and raw FLOPs do not explain the time; the compiler's per-op overhead and layouts do. Everything else on the list is worth 1 to 3% each.

## 9. Adoption rules and noise

- Engineering (math-identical) arms: timing gate >= +1.5% tok/s over a matched 60-step control (two items at >= +1.0% may share a FULL); adopt at a FULL score <= base - 0.0004.
- Numerics-changing arms: adopt at <= base - 0.0008 on a FULL; 0.0005 to 0.0008 earns a second receipt; above base - 0.0005 closes the arm.
- Same-seed repeat noise is about 0.0003 to 0.0005. Two receipts per packaged stack.
- Package rule: the first two-receipt stack at or under 0.9650 public gets the audit; Wednesday 23:00 IST is the safety package (tier 1); the final stack is frozen Thursday 05:00 IST, run with seeds 42, 43, 44 (best seed ships; the official run replays the seed), audited by 09:00, uploaded by 10:00 IST (tier 2).

## 10. What is running and queued (as of 01:40 IST)

- trn22: ENDGAME + attention temperature 0.12 FULL, score 02:05 to 02:15 (adopt at <= 0.96646). Then the exact e1bd second receipt (about 03:15). Then CS arms.
- trn21: exact e1bd 60-step control finished about 01:29; the LaneB ON60 in-model timing of the NKI attention pair follows (result about 02:00; gate >= +1.5% tok/s; if it passes, a FULL on the e1bd base, adopt at <= 0.96686). Then CS1 and the other constant arms.
- Constant-sweep program (Round 116), one constant per FULL on the exact base, seed 42, in order: CS1 three-stage batch ramp (accum 1, 2, 3 at 1/3 and 2/3 of time; patch delivered), CS2 MATRIX_LR 0.025, CS3 WARMDOWN 0.65, CS4 MATRIX_LR 0.017, CS5 MTP weight 0.3, CS6 UNEMBEDDING_LR 0.009, CS7 WEIGHT_DECAY 0.1, CS8 momentum cooldown 0.15, CS9 EMBEDDING_LR 0.45, CS10 MTP weight ending at 1/2.
- Floor-raisers still open: host-sync trimming (math-identical, CPU test pending), fetch() timer, table update every 4 steps (numerics-changing, rides along), --disable-mixed-precision-accumulation flag as a 60-step timing only when a host is idle.
- Package hygiene for the next build: mask BOS targets and the cross-document MTP second target in the training mean (free, about 0.0002).
- About 60 FULL slots remain across the two hosts until Thursday 09:00 IST; the engineering queue uses fewer than ten.

## 11. Honest expectation (12-hour map from 01:30 IST)

- 02:15: kernel-pair timing (50/50 to pass; worth about 0.001) and the endgame score (35% to adopt; worth 0.001 to 0.002).
- 03:15: base second receipt; first legal package at 0.9673 public (about 0.9669 official, rank 11).
- 09:00: five or six constant-arm results; expected stack 0.9655 to 0.9665 public (rank 10 or 11).
- 13:30: six to eight more results; with two or three hits, 0.9640 to 0.9655 public (rank 9 or 10).
- Odds for 13:30 official: 60% at 0.963 to 0.966, 25% at 0.960 to 0.963, 15% no hits (0.9669).
- Top 5 (0.9558) needs 0.011 more than anything measured or queued can give.

## 12. Open questions and unknowns

- Does the NKI attention pair keep its 2.8 ms per-call saving inside the full model, or does layout glue eat it? (Answer at about 02:00.)
- Is the endgame family (LR floor plus averaging) worth anything on top of a 75% warmdown? (Answer at about 02:10.)
- What does fetch() actually cost per micro? Unknown until the timer runs. If it is 30 ms, a safe prefetch is worth 7%.
- Are our learning-rate and schedule constants still optimal after the 10% step gain and the two-target CE? The CS arms answer this one constant at a time.
- Could Csaba get a third instance? Not asked yet. Would the organisers rule on the startup gray area? Not asked; Poojith's choice.
- What do the top three teams do? Unknown; the gap (0.016 to 0.031) is larger than any measured lever here, which suggests a different throughput path (full NKI stack or a different compiler route) or a much bigger effective token budget.

## 13. What a better prompt for me should state explicitly

- The decision rights: what I may decide alone (queue order, gates, drafts, Codex answers) versus what needs Poojith (uploads, spend, organiser questions, package choice).
- The priorities in one line: lowest official val_bpb by Thursday 10:00 IST, with a safety package by Wednesday 23:00 IST.
- The cadence: check STATE.md every 45 minutes, answer Codex within 10 minutes, message Poojith only when a number changes or a decision is needed, one 12-hour map on request.
- The reporting format Poojith wants: 3 to 5 lines, numbers with times, odds, no hedging words, no em dashes.
- The research rule: new ideas must state expected bpb, probability, host slots, compile-shape risk and reviewer risk before they enter the queue, and must check the tested list above first.
- The things I should never do again: propose arms already tested, spawn report-writing agents when a short answer is wanted, quote a number Codex has not recorded in STATE.md, or promise a rank.
