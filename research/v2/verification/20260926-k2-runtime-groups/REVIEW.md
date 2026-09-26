# Independent narrow review: K2 runtime row groups

Verdict: **PASS_CHANGED_SOURCE_CORRECTNESS_ONLY**. No defect was found in the
seven-added/eight-removed-line outer-loop and HBM-selection delta. This accepts
only the supplied diff's address, order and ownership equivalence at the stated
geometry, supported by the supplied compile/simulator receipts and this
reviewer's independently executed reference. It is not launch or adoption
approval. Native performance and full-model cold startup remain unmeasured by
this review; the required **net tokens gain >=3% after charging cold startup**
remains pending native evidence.

## Scope and independence

Reviewed source pin:
`f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e`.
Parent pin:
`662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00`.

Correctness inputs were exclusively the seven files named in
`OUTPUT.json:supplied_evidence.files`, read from
`/home/poojith-devan/Desktop/Trainium/research/v2/lanes/k2-runtime-groups/`,
plus the six referenced compiler artifacts, read only for SHA-256/size validation.
No implementer REPORT, STATUS, test source, full candidate/base source, branch or
history was read. No existing acceptance conclusion was used. The reference
script and cases were independently authored for this review; they do not import
Torch, NKI, the candidate or an implementer harness.

The user-provided AGENTS contract required reading, in order, local `README.md`,
`research/agentic/CURRENT_STATE.md`, `research/experiment-protocol.md` and
`research/agentic/DECISION_LOG.md`. These were read in the Documents/ChatGPT
checkout for contract compliance. Scientific claims and verdicts from those files
were not used as evidence for this correctness decision. No linked science
documents were followed. The Supervisor explicitly confirmed this separation.
Local `pwd`, `git status --short` and `git rev-parse --show-toplevel` established
the output workspace. A file-name listing was restricted to the compile receipt's
artifact directory to resolve manifest aliases; no unlisted artifact content was
inspected.

## Reasoning for the changed expressions

For contiguous counts/starts, reshaping `[8192,1]` to `[64,128,1]` and selecting
axis 0 at group `g` gives offset `128*g`, shape `[128,1]`, and the same row stride
as the removed slice. Maxima `[64,1]` -> `[64,1,1]` gives offset `g` and shape
`[1,1]`. Output `[8192,1024]` -> `[64,128,1024]` gives offset `g*128*1024`,
shape `[128,1024]`, and unchanged feature stride. Its unchanged owner slice
therefore writes `(g*128+p)*1024 + owner*512 + feature`.

The runtime group induction visits 0 through 63 in order. Counts, pointers and
maximum are reloaded, and the accumulator is zeroed, inside every group body.
The diff changes neither the inner occurrence loop nor its arithmetic/masks.
Thus a zero-maximum group still executes its zeroing and dense store, while
repeated encounters within each row retain their original accumulation order.
Owner 0 covers features 0..511 and owner 1 covers 512..1023 without overlap.

## Executed independent proof

Command, from `/home/poojith-devan/Documents/ChatGPT/Trainium`:

```text
python3 research/v2/verification/20260926-k2-runtime-groups/reference_check.py
```

The single execution returned exit code 0 on 2026-09-26, from
10:21:41.990446 UTC to 10:21:43.385436 UTC. It used `/usr/bin/python3`, Python
3.14.4, standard library only. Exact environment description, source/case
hashes and results are in `OUTPUT.json`. The persisted JSON preserves the
stdout values with whitespace compacted when added using `apply_patch`.

- Exhaustive production address test: all 8,388,608 dense output elements matched
  the original slice formula and were written exactly once across both owners.
  All 8,192 counts/starts addresses and all 64 maximum addresses per owner
  matched; the final output address was 8,388,607.
- Seven explicit reduced-geometry cases and 64 deterministic seed-42 cases
  matched both static and runtime group models bitwise against a separate
  direct-token-order FP32 oracle. The oracle has no row-group loop or metadata.
- Cases include empty middle/trailing groups, mixed zero/nonzero counts,
  unequal occurrence counts, the final row, duplicate IDs and owner halves
  with distinct values. All untouched rows stored positive-zero bits.
- The duplicate-order discriminator uses BF16-exact values: `[2^24,1,-2^24]`
  yields FP32 zero, whereas `[2^24,-2^24,1]` yields one. Both original orders
  were preserved. This would detect a reordered sum, unlike integer-only sums.

The all-empty synthetic case has zero tokens and checks loop-body semantics
only; it does not assert that zero tokens are a supported production input.
The small arithmetic model represents ordered FP32 addition and active-row
encounters. It does not emulate the unchanged NKI mask/sentinel instructions
or hardware scheduling. Real runtime-register legality is supported by the
supplied production compile and official-simulator receipts, not by treating
Python integer indices as native execution.

## Supplied evidence authentication

All seven evidence files were SHA-256 hashed. The artifact manifest's exact
compile-receipt hash matched. All six referenced artifact byte counts and
hashes matched, including the 61,338-byte NEFF with SHA-256
`e64314f25fbcbab62d6bcc41bdc1408e830821c296b8898e2c9a8569ed5441ac`.
`OUTPUT.json` records the alias-to-original-path mapping and every hash.

The production compile receipt declares `NEFF_WRITTEN`, BIR PASS, Trn2/LNC2,
production default options, unchanged source and the exact requested geometry.
Both official simulator receipts pin the same source and backend harness and
report exact FP32 `[8192,1024]` equality: uniform four collisions per row, and
sparse eight collisions per even row with 4,096 untouched rows. These are
authenticated supplied results, not gates re-executed by this reviewer.

The CPU receipt reports 12 finite exact cases for each of INT32 and INT64 IDs;
all recorded actual/expected bit arrays agree. Its four nonfinite cases per
dtype remain explicitly failed nonfinite diagnostics, not accepted finite
training evidence. The reduced-model receipt was inspected but no new
framework-forward gate was run or required by this narrow review.

## Limits and disposition

The whitelist excludes the full source and harness bytes. Consequently the
candidate and parent source hashes are consistent supplied declarations, not
independent full-file rehashes by this reviewer. This review does not establish
the completeness of the delivered source against an uninspected base; it
accepts the exact diff with SHA-256
`4ed25ae91e40bb07ed6fbeb5eb4a80ecc69c994463de9a8803ecd82ce8bc4ade`
under the specified previously reviewed base assumption. It does not reopen
that base or introduce another full-source review cycle.

No host, data, AWS, training, launcher or native call occurred. No compiler,
simulator or framework-forward rerun occurred. No production source changed.
The passing local address/ordering evidence and compact NEFF do not establish
whole-model startup, native speed, fixed-budget tokens, BPB, launch eligibility
or adoption. The Supervisor retains prospective proposal/control/command
binding, and the sole Operator retains any separately authorized native action.
