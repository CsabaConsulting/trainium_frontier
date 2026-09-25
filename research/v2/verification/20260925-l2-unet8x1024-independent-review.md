# L2 8x1024 architecture: independent source and reduced-size review

Verdict: **PASS_DIFF_AND_REDUCED_SIZE_NUMERIC_ONLY**.

The prior architecture-draft findings are resolved. No new source-level blocker was
found for the proposed standalone, no-table L2 arm. This is **not** exact-size G2,
Neuron/NKI validation, a throughput/BPB result, full-run authorization or submission
approval. Exact 8x1024 four-rank G2 and its matching default-OFF/base comparison
remain pending; do not promote the smaller evidence below to those gates.

Reviewer: independent Verifier `/root/architecture_readonly_verifier`.
Review worktree: `/home/poojith-devan/Desktop/Trainium-l2-independent-verifier`.
No host was accessed, no candidate source was edited and no full-size model was
allocated during this review. The largest independently rerun model was 8x128.

## Exact identities and instruction precedence

- Candidate commit: `99261e750ec8712f6ccb4a2d961d6134d3f87b12`.
- Reviewed combined source diff: `b18fb358..99261e75`, only
  `workflow/proposals/research-v2/train.py` changes model behavior.
- Exact c5c base SHA-256:
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
- Candidate SHA-256:
  `52e234e90f33fc2ccfcc22edb7afe101be25089020a58c352afda4470ac5a7c1`.
- Unchanged `prepare.py` SHA-256:
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Unchanged `preflight_dryrun.py` SHA-256:
  `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`.

The canonical controller's FINAL PUSH, verbatim Lane 2 prompt and
`workflow/RESUME_AND_TABLE_CORRECTION_2026-09-25.md` control. The latter explicitly
resumes development and assigns L2 to local Codex; this review does not invoke
Fable or reinterpret the provider refusal. Historical stand-alone U-Net losses
are negative evidence, not an override of the newly selected combined architecture.

## Source findings

1. `--unet-8x1024` defaults OFF and, when enabled, assigns depth 8, width 1024,
   U-Net skips and partial attention before runtime/model construction
   (`train.py:3604`, `3652`). The preset's override of explicit shape arguments
   is documented and exercised by the bounded parser test.
2. `ATTENTION_LAYERS` defaults to `0,2,4,6,7`; parsing rejects empty, duplicated,
   negative, non-integer and out-of-range lists (`train.py:284`). The configurable
   list and both architecture toggles are checkpoint-persisted and shape-critical.
3. Four FP32 one-element zero-initialized parameters are scalar skip gates
   (`train.py:1992`), not the old per-channel vectors. They join the existing
   scalar AdamW group exactly once (`train.py:2218`), retaining its LR/betas/WD.
   The LIFO encoder-output additions implement pairs 0->7, 1->6, 2->5, 3->4.
4. Odd/too-small skip depths are rejected in both CLI and GPT construction
   (`train.py:1936`, `3659`), closing the old out-of-range decoder access.
5. MLP-only blocks omit attention, local convolution and QK-shift. `has_ve`
   additionally requires active attention, so default depth-8 VE allocation is
   only layer 7; n-gram eligibility inherits that filter. Explicit table choices
   on removed, otherwise eligible VE sites fail instead of silently allocating
   unused parameters (`train.py:1940`).
6. The new skips are positionwise combinations of causal earlier-block outputs;
   no forward collective, validation-data access, persistent inference state or
   different eval architecture was introduced. Strict `load_for_eval` restores
   the new toggles and attention list before construction. Existing training-only
   MLP/optimizer adapters and the inherited eval wrapper are unchanged.
7. The default-OFF source path preserves model creation/RNG ordering, original
   optimizer routing and attention availability. This source conclusion is
   supported by the three-step reduced-size comparison below, not asserted as
   an exact-size or native numerical proof.

The [author's public write-up](https://ritwikareddykancharla.github.io/projects/trainium-of-thought.html)
was checked directly: it describes roughly width 1024, eight layers, selective
attention and U-Net-style skips. The exact indices, scalar gates and pairing
are the human's task choices, not documented details of that recipe. Candidate
attribution and README make this distinction explicit. No external code or weights
are present in the diff.

## Evidence independently checked

`git diff --check b18fb358..99261e75` passed. The supplied tiny G2 JSON records
were inspected and independently recomputed rather than accepting their top-level
PASS labels alone:

| Receipt | Source/shape | Result checked |
| --- | --- | --- |
| `g2-base-small.json` | c5c, 6x128 | Four ranks, three finite losses each, 62/62 state tensors changed, zero rank mismatches, strict checkpoint check |
| `g2-off-small.json` | exact candidate, 6x128, new flags OFF | Same argv, runtime metadata, synthetic micro-batch, ownership plan, all 12 rank-step losses and all 62 final digests as base |
| `g2-on-small-r2.json` | exact candidate, 8x128, skips+partial attention ON | Four ranks, three finite losses each, 65/65 state tensors changed, zero rank mismatches, strict deterministic checkpoint check |

All three report torchrun exit 0, no timeout, four Gloo ranks, three steps and
1x16 synthetic training micro-batches. ON uses the ablation flags to avoid
allocating the preset's 8x1024 shape; the actual preset is tested separately up
to the runtime boundary. These CPU receipts use documented BF16/fake-clock/data/
CE shims; they are not native-kernel tests. The original missing-PyArrow failure
remains preserved as `g2-on-small.json`, not erased or mislabeled a model failure.

I independently reran `check_small.py` with OMP/MKL/OpenBLAS threads limited to 1,
the existing CPU Python environment, a 60-second timeout and the documented
import-only PyArrow/test-prepare stubs. It exited 0. The output is preserved as
`research/v2/l2-unet8x1024/independent-self-check-small.json` and is byte-identical
to the Implementer's self-check receipt:
`f4b362d6d58e1f6647c59bb1052db9093b3df23b06f56062e49cc4de2cd306ab`.

That rerun confirms the preset parser, scalar routing, absence of unused VE,
zero-init skip identity against the same partial-attention architecture without
skips, three finite updates and learned skip values, strict checkpoint/logit
parity, determinism, row-batch independence, prefix causality, missing-key
rejection, malformed list/depth rejection and tiny-table/custom-pattern behavior.
It sets `EVAL_MIN_SEQ_LEN=32` only in the unit-test process; production source is
unchanged, and the separate G2 records use the original eval helper.

## Scope limits and next gate

- Exact 8x1024 CPU G2/flag-OFF lockstep is **not complete**. A sufficiently
  provisioned authorized machine and distinct Operator are required; this review
  deliberately avoids heavy laptop tests.
- No native compilation, full-step timing, optimizer memory profile or BPB exists
  for this candidate. Scalar LR suitability and removal of three VE sites remain
  empirical risks, not correctness failures.
- Default attention excludes A2's layer 1. Combining this source with A2 `1,-1`
  must fail under the default list; a different list and a declared stack proposal
  are required. Do not call default L2 an A2-preserving S-C stack.
- The proposal's run label/control host are intentionally unassigned, and
  `launch_authorized` remains false. The completed source review clears no
  machine preflight or scientific selection threshold.

Minimum next step: exact-source, exact-shape four-rank CPU G2 and OFF/base
comparison, then the Supervisor/Operator can apply current run gates to a uniquely
labelled architecture-only experiment. No implementation repair is requested by
this source/reduced-size review.
