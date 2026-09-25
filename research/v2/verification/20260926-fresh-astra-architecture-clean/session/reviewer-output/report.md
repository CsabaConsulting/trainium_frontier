**Decision: REFUSE.** The candidate does not preserve the effective research checkpoint evaluation path. This is a concrete source discrepancy, independently observed through actual tiny CPU model calls. The numerical CPU equivalence checks passed; they do not remove the discrepancy.

Review scope: the supplied packet, reconstructed frozen sources, and independently written sequential CPU tests only. No candidate/production edits, external sources, network access, package installation, native compilation, distributed workers, extra agents, or submission actions were used.

`native_executed:false`, `full_size_cpu_executed:false`, `launch_authorized:false`, `submission_authorized:false`, `candidate_quality:unmeasured`.

**Blocking finding — EVAL_BATCH_WRAPPER_OMITTED**

The reference's [load_for_eval](/work/scored_reference.py:3290) constructs `_SubmissionEvalGPT`. Its [effective forward](/work/scored_reference.py:4908) validates a nonempty two-dimensional input and, for batches larger than one, invokes the underlying GPT separately on each row and concatenates the results. This explicitly fixes the arithmetic batch dimension at one. The candidate's [load_for_eval](/work/candidate.py:1290) constructs plain `GPT`; its forward executes the whole batch together. The vehicle also lacked this wrapper; leaving the vehicle path unchanged does not preserve the specified effective reference semantics.

Using separately trained, state-identical tiny checkpoints, forward hooks observed the following actual first-block inputs for a `[2,32]` token batch:

| Path | Executed block inputs | CPU output error |
|---|---|---|
| Reference checkpoint loader | `[1,32,128]`, then `[1,32,128]` | Baseline |
| Candidate checkpoint loader | `[2,32,128]` | max absolute = 0, max relative = 0 |

This route difference occurred at all five tested context lengths in both compute-dtype configurations. The source wrapper promises per-row arithmetic regardless of scorer batch size. The candidate removes that guarantee and changes the sizes passed to attention, projections and convolution. At native-eligible sequence lengths, the convolution source also selects lanes based on batch parity, so omitting the wrapper can change the selected launch grid. Native numerical consequences were **not** measured, and no claim of a demonstrated native error or CPU logit mismatch is made. REFUSE rests on the confirmed omission of the required effective evaluation behavior, not on speculation that these small CPU outputs should differ.

The concrete correction needed before reconsideration is to preserve the reference's evaluation-only row wrapper when loading the trained checkpoint, followed by a bounded verification of that effective loader path. No correction was made during this independent review.

**Input authentication and source audit**

[authentication.json](/work/authentication.json) records byte counts and SHA256 verification for all seven manifested packet files. [reconstruct.py](/work/reconstruct.py) reconstructed each full-context diff without changing its bytes:

| Source | Lines | SHA256 |
|---|---:|---|
| Vehicle | 1602 | `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d` |
| Candidate, commit `bd9e33eee671ee489a1201485faf748bf0a35dea` | 1628 | `3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f` |
| Research reference, commit `b4d83b5db66971f3c348837f04f53cba65aa6d33` | 4948 | `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c` |

The full vehicle diff changes depth/attention placement, U-Net skip parameters and routing, initialization guards, optimizer membership, FLOP estimation, initial differentiable cache refresh, and the startup timing predictor reset. Its remaining model, optimizer, kernel, loader and checkpoint code was checked against the reference's effective enabled paths, including the appended training adapters and evaluation subclass. A complete definition inventory and 20 exact AST comparisons, ignoring docstrings and the root function name, are in [source_audit_results.json](/work/source_audit_results.json). Those 20 comparisons include the RoPE/norm kernels and autograd class, convolution helpers/kernels/autograd class, the wrapper-installed adaptive CE kernels, MLP reference forward, Muon arithmetic/scalar helpers, and optimizer checkpoint helpers. These are source comparisons, not NKI execution.

The exact required LONG command remains:

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```

Both actual argument parsers and global-setting paths were executed through the call immediately before runtime initialization, then stopped deliberately. No model was allocated at the resulting full configuration. The reference received the specified additional science flags, including `--unet-8x1024` and `--initial-cache-refresh`. Captured arguments and effective globals are in the two `*_effective_config.json` artifacts. This checks the actual command, not bare defaults. No additional competition environment overrides were supplied.

| Required behavior | Audit result and evidence |
|---|---|
| Command/configuration | Candidate resolves seed 42, matrix LR 0.015, source budget 1800 s, step cap 1,000,000, width 1024, depth 8, heads 8, head dimension 128, context 1024, MLP ratio 4, dense KV heads, pack factor 32. Four ranks and LNC2 are specified by the unchanged launcher; both training paths require an initialized four-rank group. No launcher was executed. |
| Architecture | Attention and causal convolution only at layers 0,2,4,6,7; MLP-only 1,3,5. Scalar FP32 U-Net gates start at zero. Encoder outputs feed decoder inputs in reversed order: 3→4, 2→5, 1→6, 0→7. Value embedding and its 8×32 gate exist only at layer 7. |
| Precise U-Net pairing | `unet_skips.pop()` gives **3→4, 2→5, 1→6, 0→7**. Gate indices 0,1,2,3 correspond to decoder layers 4,5,6,7. There is no value embedding at removed attention layers 1,3,5. |
| Initialization | Token embedding normal std 0.8, head normal std 0.001; `s=0.5*sqrt(3)/sqrt(width)` for projections; attention output scale 0.1; MLP input ±0.4s and output zero; residual/input scalars 1/0.05; convolution, VE gates and U-Net gates zero. Tiny initialized state is exactly identical. |
| Dtypes/normalization | FP32 master parameters, BF16 cached linear weights and branch inputs on Neuron, FP32 RMS statistics with epsilon `2^-23`, BF16 norm outputs, FP32 residual mixing and logits, softcap 15. No transposed master layout, FP8, or alternate schedule enabled. |
| Effective MLP training | Reference installation uses `forward_backend='native'`, custom transposed-ReLU backward, ordinary input-gradient ordering, tile limit 512, cache enabled. Candidate embeds that implementation. Tiny models take the fallback because ≤64 tokens; a separate 256-token width-128 MLP exercised the eligible custom backward exactly. |
| Attention/loss/native dispatch | Causal direct SDPA is enabled, long-window causal fast path is selected, QK shift and document masking are off. The adaptive CE lane selection and installed kernel bodies match; ordinary loss reduction is preserved. Native kernels were statically inspected only. |
| Optimizer | Same grouping and order: embeddings/head/VE/scalars/convolution use AdamW; attention matrices, VE gate and MLP matrices use Muon. U-Net gates join scalar AdamW. Same width LR scaling, moment betas, epsilon, decay, five-step Polar Express math, FP32 scalar transport, BF16 Muon iterate and cautious Muon decay. Reference optional branches are disabled under these flags. |
| Ownership/update | Same cost-based four-owner planning, offsets, padded shard size, singleton Muon padding, local group update and all-gather/copy semantics. The reference installs an optimizer adapter; candidate calls the equivalent adapter explicitly. Shape-only full-model planning agrees without allocating tensors: owner counts `[27262976,28315914,27269130,30408704]`, shard elements 30,408,704. All-reduce/all-gather were not rerun independently. |
| Clipping | Gradients average across ranks before global norm clipping at 1.0; all parameters are included because n-gram tables are absent. Loss scale 1, no dynamic scaling, Adam update every step. Independently tested clipping and optimizer arithmetic are single-process; collective correctness remains receipt/source evidence. |
| Old wall-clock schedule | LR stays flat through 25% of elapsed budget, then linearly decays to 0.05 at 1800 s. Muon weight decay decays from 0.2 to zero; momentum warms 0.85→0.95 over 300 steps and cools to 0.85 over final 4.5%. Five schedule probes match exactly. Demon, changing beta2, depth multipliers, ramped batch size and EMA are off. |
| Cache/autograd | Both paths refresh casts with grad mode enabled inside the first timed step and after each update. Candidate refreshes before zero_grad; reference after zero_grad, with no intervening parameter update. The cast remains attached to the FP32 master weight. The independent detached-cache control catches lost gradients. |
| Timing/budget | Both clear recent step durations after excluded startup and use `max(..., default=0.0)` for the predictor. Both retain 15 s checkpoint reserve, 30 s launch margin, organizer startup cap and step-boundary hook. The startup constants imported from prepare.py are not supplied and were not independently validated or timed. |
| Loader | Both select the imported organizer `make_dataloader`, with no epoch shuffle/fresh-tail/requeue path under this command. Fetch size is 1×32768 tokens per rank, reshaped to 32×1024; four ranks give 131072 tokens and accumulation 1. Real data packing/tokenization was not executed; prepare.py is absent. |
| Checkpoint/evaluation | Both save trained FP32 model state/config, strictly reload it, rebuild nonpersistent rotary tables, disable gradients and enter eval mode. Fixed candidate architecture replaces the reference's saved-toggle restoration for this command; EMA is off. Both have source eval table floor 8192. **The evaluation row wrapper is missing in candidate, as detailed above.** |

The static parameter calculation is:

```text
3 × 8192 × 1024                token embedding + untied head + layer-7 VE
+ 5 × 4 × 1024²              attention matrices
+ 8 × 8 × 1024²              two MLP matrices per layer, expansion 4
+ 5 × 2 × 1024               two-tap depthwise convolutions
+ 8 × 32                    layer-7 value gate
+ 16 + 4                    residual/input scalars and U-Net gates
= 113,256,724
```

No full-size model or full-size tensor was allocated. Full ownership planning used shape-only objects and the source's extracted pure planner; it is not a full-size CPU gate.

**Supplied receipt assessment**

The receipt files are authenticated against the packet manifest, not independently signed host attestations. Their source hash fields match the reconstructed sources. I independently compared the two G2 receipts rather than accepting the `tiny-lockstep.json` verdict: per-rank losses, plan digests/summaries and all 65 rank-0 final parameter digests agree. The final digests are only 16 hexadecimal characters each; full tensor bytes, test source and the prepare shim source are not supplied.

Their actual scope is width 128, depth 8, vocab 8192, 4,523,316 parameters, four recorded CPU ranks, three updates, and synthetic per-rank token streams. Although argv/config retains sequence length 1024 and pack factor 32, the forward shim slices training to **1×16** tokens. BF16 compute is forced, module time is a deterministic 1 ms fake clock, prepare is a stub, compilation is disabled, and the reference's NKI CE installer is replaced with a no-op. The logged word “COMPILED” for Muon does not establish compilation: the CPU branch calls the ordinary Python math function. These receipts do not measure real throughput, budget compliance or quality.

The checkpoint receipt additionally reports context lengths 16/64/256, batch comparisons, causal prefixes, 65 tensors, and an eligible `[1,256,128]` MLP input. It does not provide checkpoint bytes, exact model configuration or test implementation, so those assertions cannot be replayed literally. It claims width-128 BF16 synthetic CPU scope. Neither its passing batch checks nor my passing numerical batch checks establish retention of the evaluation wrapper; the independently observed route differs.

**Independent executed test and raw evidence**

Source: [independent_test.py](/work/independent_test.py). Command:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /opt/cpu/bin/python /work/independent_test.py > /work/test_attempt1.stdout 2> /work/test_attempt1.stderr
```

One attempt, exit code 0; no retries, changed thresholds, or changed assumptions. Torch `2.13.0+cpu`, one intra-op and one inter-op thread, one process, sequential models. [results.json](/work/test_artifacts/results.json) contains 596 records with per-tensor absolute/relative errors, equality decisions and diagnostic metadata. Exact comparisons use `atol=rtol=0`; relative error denominator is `max(abs(reference),1e-30)`. A missing gradient has null numerical error, explicitly distinguished from a zero gradient.

- Models: width 128, depth 8, vocab 256, context at most 32, batch at most 2; 1,475,892 parameters each. Only one live training model at a time.
- Independently seeded construction and initialization: all 65 state tensors exactly match.
- Three synthetic steps with differentiable cache refresh: both loss outputs, all 65 gradients each step, and all 65 post-update weights each step match exactly. Steps probe elapsed 0/900/1764 s through the original schedule functions; this is prescribed scalar input, not wall-clock training.
- Actual `MuonAdamW` groups and updates execute on CPU. Ownership plans compare exactly at both tiny and full static shapes. The independent test does not emulate distributed collectives or claim a distributed training-loop run.
- Tiny checkpoints are saved by each source's `save_checkpoint` after the three updates, then loaded through its actual `load_for_eval`. Context lengths 1,7,16,31,32 are checked in normal CPU FP32 compute selection and forced BF16 selection. All logits, batch-versus-independent-row outputs and changed-suffix causal prefixes match exactly in this CPU environment.
- An isolated `[1,256,128]` BF16 MLP tests the wrapper-installed custom-autograd branch, with identical input, output, input gradient, weights and weight gradients.
- Negative control 1 detaches the candidate's BF16 cache: forward outputs remain exactly equal, but all 16,384 formerly nonzero master-weight gradient elements become a missing gradient. The checker detects this.
- Negative control 2 changes only a test-extracted forward's skip-stack operation from LIFO to FIFO. Unequal nonzero gate values prevent zero initialization from masking the error. It produces max absolute logit error **0.016597986221313477**, max relative error **56.75460122699386**, and is detected. Both outputs, altered test assumptions, model state and inputs are saved.

The 596 passing numerical/diagnostic records are **not** the final review verdict. The independent [source audit](/work/source_audit.py) compares the saved actual hook traces and records ten failed route-equivalence comparisons, all with identical CPU numerical outputs. Its structured blocking finding is in [source_audit_results.json](/work/source_audit_results.json).

Tensor bounds were enforced by tracking live tensor handles, including saved-for-backward tensors, and deduplicating their storage. Observed peak was **43,590,044 bytes (41.57 MiB)**. A separate retained-state inventory plus a conservative 20 MiB workspace allowance gave at most 52,968,792 bytes (50.51 MiB), below 64 MiB. These are tensor-storage figures, not process RSS.

Exact test-only substitutions and omissions:

- `prepare` and `pyarrow.parquet` are import stubs; unused external operations fail closed. Vocab 8192 for parser/config capture comes from the packet's recorded config; startup/budget import placeholders are identified in the test and were not used to validate organizer behavior.
- `init_runtime` is temporarily replaced with a deliberate exception after capturing real parsed arguments. No distributed initialization, data loading or main training loop runs.
- BF16 compute is selected explicitly for training on CPU. The reference MLP training installer is called explicitly with its effective `forward_backend='native'`. The outer training installer, NKI CE installation and owned distributed optimizer installation do not run.
- The test changes only the module's **nonpersistent eval table floor** to 32 to meet the lane's context and memory bounds. The frozen source still says 8192. Widening to 8192 is inspected statically, not allocated/tested. Model math and weights are not replaced.
- No NKI/Neuron, CUDA, torch.compile backend, distributed collective, real tokenizer/data reader, official scorer, full-budget loop or quality measurement executes. “Native” in the reference MLP option denotes its Torch matmul forward, not native Neuron execution.
- Raw stderr preserves Torch cpuinfo/NNPACK initialization warnings. The CPU test completed successfully through the available CPU paths; warnings were neither deleted nor converted into a claim about native legality.

[attempts.json](/work/attempts.json) records all script invocations and their status. `/work/test_artifacts/*.pt` preserves the input tensors, initial states, loss/gradient outputs, post-update states, tiny checkpoints, evaluation outputs and negative controls. [test_attempt1.stdout](/work/test_attempt1.stdout) and [test_attempt1.stderr](/work/test_attempt1.stderr) are unedited raw streams. [artifact_manifest.json](/work/artifact_manifest.json) provides SHA256s and byte counts for the review artifacts, excluding itself by definition.

The explicitly later gates remain unexecuted: full113M CPU G2, native legality/compilation/cold startup, full-budget quality replay, final compliance review and human submission approval. Their absence is **not** the reason for this REFUSE. The recorded public20M BPB `0.9863403052402583` belongs only to the research reference; the candidate's quality remains unmeasured.
