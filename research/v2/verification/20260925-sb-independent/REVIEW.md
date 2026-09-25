# Independent S-B integration review — source and reduced CPU PASS

Date: 2026-09-25. Role: independent Verifier; no candidate edits, host access,
native compilation, real data, training launch, or launcher/state changes.

**Verdict: PASS_SOURCE_AND_REDUCED_CPU_ONLY.** No concrete integration defect
found in the complete production diff or the bounded checks below. This accepts
the offline integration evidence, not hardware eligibility, performance, quality,
or an accepted stack. Architecture-alone work and the FP8-alone **>=3% matched
whole-step** gate retain priority; this review cannot preempt or waive either.

## Exact identity and review boundary

- Candidate commit: `ab7a007ed2f9ae8104b83632e772b3921639fb84`.
- Production file: `workflow/proposals/research-v2/train.py`.
- Combined SHA-256: `a2f6b188db8418f4bccf4f2704db1a68bd9c3588bcc69d1fc7819701db39a238`.
- Architecture reference: `1e21c3d797eed764f108c41546b270065ad5650c`, source
  `93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5`.
- FP8 reference: `50a901374ae6f76b35b8556d32fd72b1ece28fc6`, source
  `921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858`.
- Dense reference: `b18fb3586494c21f10008a9c63fd2056d1e6c00a`, source
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.

The complete diff against 93ae is 215 insertions/13 deletions in the production
file; other changes are confined to `research/v2/sb-architecture-fp8/` evidence.
Protected evaluator/tokenizer/scoring/data preparation files are unchanged.
Source identities were independently checked against Git objects. Component
review lineage is the architecture independent review, original FP8 receipt
`20ec99c2dafac893011d5a921a2d18b63fe1cb66`, and exact timer-port review
`e9109dec72e41f0d481bd55df49aaedf7655bb6a`.

## Source findings

- The FP8 quantizer, NKI kernels, autograd function, Linear/attention/MLP classes,
  and conditional training installer are AST-identical to 9212. Their earlier
  definition placement does not eagerly build/import the native kernels.
  Native forward remains row-scaled legacy E4M3 double-row NKI, with BF16
  effective operands saved for BF16 STE products and gradients returned to
  FP32 master weights. Current weights are prepared each forward; FP8 linears
  clear/avoid the old BF16 cache. CPU tests execute only the explicit reference.
- Architecture Block/GPT, skip pairing, scalar optimizer group, attention and
  value-table filtering remain unchanged from 93ae. At the selected eight-layer
  configuration, all 16 MLP linears and 20 Q/K/V/O linears in attention layers
  0,2,4,6,7 are marked: 36 total. Head, embeddings, small gates and four scalar
  skips are not FP8. MLP-only blocks retain FP8 MLPs without unused attention
  modules or value tables. The old fused BF16 MLP installer is bypassed only
  when FP8 is ON; OFF retains the reviewed architecture installer.
- Independent/default-OFF architecture and FP8 flags coexist. The preset is
  parsed before construction; existing shape/parallel-block restrictions and
  FP8 transposed/coalesced-weight and loss-scaling rejection are retained.
  The intended width-1024 projection shapes meet inherited K%256/N%128 limits;
  this static observation is not compiler/SBUF feasibility evidence.
- Both architecture settings and `USE_FP8_PROJ` are registered before evaluation
  defaults are captured. Saved toggles are restored before model construction,
  with strict weights-only load and frozen evaluation. Missing legacy flags
  revert to OFF; missing skip parameters are rejected. No alternate combined
  evaluation arithmetic was introduced.
- The whole training-loop AST matches both exact timer-fixed components, so the
  already reviewed startup-history reset is retained once. Loader, schedules,
  optimizer bodies and checkpoint/evaluation helpers remain the architecture
  versions. No Recursive schedule or batch-default change was added:
  `TOTAL_BATCH_SIZE=131072`.

## Independent execution and receipt audit

1. Inspected then reran the candidate's `check_interactions.py` unchanged.
   `interactions-rerun.json` records all ten checks passing. Each isolated mode
   matches its exact reference's initialization and all losses, gradients and
   weights over three optimizer updates. Both-ON passes finite gradients/master
   weights, learned scalar skips, cleared caches, prefix causality, batch
   independence, determinism, input sensitivity, and trained strict checkpoint
   restoration after deliberately changing the live flags/attention indices.
   Combined losses: `4.160888671875, 3.0385212898254395, 1.4280568361282349`.
   The raw output retains the script's hard-coded Implementer/self-check labels;
   it was independently executed here and is not presented as native approval.
2. Reran the unchanged four-rank preflight harness for both-ON, three actual
   optimizer steps, width128/depth8, synthetic vocab8192, microbatch1x16.
   `g2-combined-rerun.json` is PASS: all 12 losses finite, all 65 parameters
   updated, no cross-rank mismatch, matching ownership plan, and deterministic
   saved-checkpoint reload. The independent rerun exactly reproduces the
   builder's 12 losses and all 65 final parameter digests.
3. Ran the tensor-free `audit_receipts.py`. It checks all seven supplied raw G2
   receipts plus the independent rerun: exact sources, four rank reports,
   three finite steps, updates, cross-rank consistency, plan, and checkpoint.
   It reruns the unchanged receipt comparator, rather than trusting the three
   supplied identity summaries. Both-OFF versus c5c, architecture-only versus
   93ae, and FP8-only versus 9212 all have identical losses/plans/digests
   (62/65/62 parameter tensors respectively). Results: `receipt-audit.json`.

Reproduction from this review worktree (all commands completed with exit 0):

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=research/v2/l2-unet8x1024/test_support:workflow/tools/stub_prepare /home/poojith-devan/venv/bin/python -B research/v2/sb-architecture-fp8/check_interactions.py --train-py workflow/proposals/research-v2/train.py --out research/v2/verification/20260925-sb-independent/interactions-rerun.json
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=research/v2/l2-unet8x1024/test_support /home/poojith-devan/venv/bin/python -B workflow/tools/preflight_dryrun.py --train-py workflow/proposals/research-v2/train.py --stub-prepare --steps 3 --nproc 4 --micro-rows 1 --micro-tokens 16 --timeout 180 --keep --out research/v2/verification/20260925-sb-independent/g2-combined-rerun.json -- --n-embd 128 --depth 6 --no-ngram-ve --no-qk-shift --bf16-norm-output --compile-sdpa-direct --nki-local-conv --pack-factor 32 --clip-after-reduce --seq-len 1024 --matrix-lr 0.015 --seed 42 --no-eval-public --depth 8 --unet-skip --partial-attn-8 --attention-layers 0,2,4,6,7 --fp8-proj
/home/poojith-devan/venv/bin/python -B research/v2/verification/20260925-sb-independent/audit_receipts.py
```

The repeated `--depth` in G2 is the builder's exact argv; the final value is 8.
The public 8x1024 preset was parser-tested only, stopped before runtime/model
allocation. Direct numerical checks use at most two 8x128/vocab64/context32
models, seed42 and one CPU thread. G2 retains configured context1024 but slices
training to1x16. The test-only BF16, synthetic prepare, fake clock, NKI-CE bypass
and evaluation-mask bounds are explicit in the raw receipts/script. Test elapsed
times and fake-clock budget output have **no native timing/budget meaning**.

## Still unestablished

No exact-size combined G2, NKI execution, native finite/backward/checkpoint/
causality evidence, native compiler/memory feasibility, whole-step improvement,
or public BPB was produced. Width128 CPU execution specifically cannot validate
the native K%256 geometry. Those pending items retain their existing workflow
scope; this receipt adds no new hardware ceremony or launcher. A later combined
hardware experiment remains subject to component results and its separately
reviewed exact-source proposal. No launch or adoption is authorized here.
