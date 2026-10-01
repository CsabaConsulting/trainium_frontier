# Fresh independent D2 scored-source review

ACCEPT — RESEARCH FULL ONLY

Machine-readable verdict: `ACCEPT_RESEARCH_FULL_ONLY`.

Accepted candidate SHA-256:
`b758450b599d4acc5b1fc17d37cccb87421d22bb99c213f6d8ed29fdede13fa8`.
Exact D1 parent SHA-256:
`acfb3aefab0e6503498aca8387db63fff583930848fb16fcc9650887a57129ea`.

This is an independent Verifier receipt for both Round62 research FULL arms on
the identical frozen source: CONTINUE `--tbl_freeze_tokens 0` and FREEZE
`--tbl_freeze_tokens 286000000`, seed42, table every2, FP32 duplicate sink.
It supplies the requested scored-source review item. It does not operate a host,
release a packet, change the queue, or replace Root's existing exact-source,
predecessor-result, original-lock, resource and machine protocol checks.

Review session: `/root/r50_timing_preparer/r62_d2_fresh_scored_review`.
Runtime identity exposed to this session: Codex / GPT-6; an exact backend model
identifier is not exposed. The reviewer neither proposed nor implemented nor
operated this candidate. No earlier reviewer verdicts, implementation notes,
provided static-test code/results, or earlier source-review conclusions were
consumed. Inputs were current contracts, exact parent/candidate bytes, SPEC,
SOURCE.diff, and the actual CPU14 archive and bound worker/driver/collector code.

## Independently executed evidence

Command, exit 0:

```text
python3 -I -B research/v2/verification/20260929-r62-d2-fresh-scored-review/audit_static.py
```

`AST_RESULT.json` records the output as JSON with whitespace compressed. The
audit is my own stdlib program. It imports no model, Torch, worker or operator
module; executes no compiler; and makes no host calls. Its tiny AST interpreter
evaluates only source integer/boolean control and records update events, without
tensor, gradient, optimizer or floating-point model computation.

The audit regenerated the complete unified diff from the pinned sources and
matched SOURCE.diff exactly. All module-level imports, constants, kernel bodies
and budget constants remain AST-identical. Exactly 17 declared definitions
change; 68 stay identical. Explicit unchanged checks include GPT.forward,
Block.forward, norm, d1_rms_step, save_checkpoint, load_for_eval, main gradient
sync/clip, optimizer ownership planning, and synchronized batch ramp. The shared
forward has no `self.training`-dependent model math.

The actual finish_step AST passes an independent 32-case truth table over
frozen state, pending step count, force, abort and freeze. An agreed abort raises
before any state mutation/update; freeze flushes one pending group before setting
frozen; a previously frozen updater makes no update. Actual main-loop
completed_tokens/local_freeze AST expressions pass both exact runtime thresholds
across four synthetic ramp positions and 6,000 steps each. Zero never freezes;
286,000,000 triggers once at the first completed global-token crossing. These
are control exactness proofs, not model numerical evidence.

## CPU14 archive authentication

Collected archive:
`research/v2/operator/20260929-r62-d2-concurrent-cpu-trn21-preparation/collect-20260928T193318670491Z.json`
SHA-256 `af2fec9279cc9d8ece735cfbe5dc30d453e54350cfc8e44bb5ce486a19a15e29`.

Worker SHA-256:
`83d8552e49c27820721eb4950b2920bf6043e1e1d58e2823da10d27cac60f3da`.
Driver SHA-256:
`3a733b1c08344fe805de82066215601eef8b3a77999af58ee24909b246a2f99d`.
Result member SHA-256:
`4c9cbe5aa714523ea97ababe85306c08106915add95ea6c4806ad16ee5cc9d06`.

All seven member text hashes, full-file hashes, byte lengths and the binding,
attempt, dispatch, terminal, worker/source/parent/prepare pins match independently.
The collector request wire was reconstructed from pinned local file bytes and
its literal BODY; its SHA exactly matches
`842374a8188fb7d03009a6c754918bccf0e3b5005005779c98a92d4a2d3f9afc`.
The archive and worker terminal agree on PASS, exit0 and all 14 ordered checks.
Protected inputs remained unchanged, the worker child was reaped, the CPU lock
was released, and no training tensors were retrieved. The record says no device
lock or training signals were used.

I read the worker assertions underlying every group. Actual runtime was
11.803303 seconds with 2,601,054,208 bytes peak RSS on one instance CPU. The table
geometry was real 262144x256; the transformer was reduced B2/T16/L8/C128/H1/vocab128.
Coverage includes three-seed all-OFF/D2-OFF D1 identity with nonzero tables,
causal row-local hash, detached shared-core leaf, independent duplicate scatter
and historical BF16 RMS references, token-weighted every2/partial groups,
four-rank control mock, parameter routing, learning at both slices, independent
L7 attention reference, cached-BF16 residual dtype, learned causality/batch
independence, strict nonzero checkpoint roundtrips, freeze/continue and continued
projection/gate learning after table freeze. The four-rank check is a sequential
mock, not actual distributed transport. The log text `COMPILED body` is emitted
by the inherited wrapper even for its CPU eager branch; the worker forbids
torch.compile and the pinned source selects the eager function on CPU.

## Source findings and both flags

One persistent zero-initialized BF16 table changes shape from 524288x128 to
262144x256, preserving 67,108,864 elements. There is one row-local int32 bigram
hash and one model-owned lookup. Self-at-t0 uses the current token as previous;
later positions use only the preceding/current tokens. The rows-minus-one
modulus is 262143; the unused final row follows the inherited hash convention.
No trigram, new data source, external indices or pretrained weights are added.

Slice0 [:128] feeds the inherited embedding-before-norm projection/scalar gate
and therefore both initial x and persistent x0. Slice1 [128:256] feeds the new
random projection and L7-only value addition before head reshape. Its zeroed
FP32 [8,32] gate reads the existing normalized pre-convolution residual channels
32:64, giving initial effective gain1. The explicit cast keeps the gate product
in the projected activation dtype. Both projections enter Muon; the scalar and
L7 gate enter AdamW. The table is a buffer outside main parameter ownership,
gradient AVG and clipping. Construction/initialization of new random weights
uses saved/restored CPU RNG contexts. No added math touches protected evaluation
or preparation code.

The FP32 duplicate sink remains duplicate-safe and token-weighted, with one cast
at the inherited BF16 table AVG/RMS boundary. The BF16 table/RMS state, beta2,
bias correction, epsilon, width-scaled LR and literal ramp factor1 remain as
specified; this is not an FP32-master table. Dense-table peak memory/native
compiler behavior is not measured by the persistent-storage arithmetic.

The runtime CLI is parsed as int and passed directly into build_config, then
saved in model_config. Both arms keep D2 enabled and every2. For D2, loader
epoch2 does not zero LR: CONTINUE0 keeps normal table updates. FREEZE286000000
counts completed global training tokens, including startup batches, flushes the
whole pending group once at normal LR on the boundary, and then skips scatter,
table AVG and RMS while preserving the shared leaf/backward graph and ordinary
projection/gate updates. The MAX-reduced freeze bit travels with epoch/force/
expired before table update. The inherited pending-tail refusal and budget
checks are unchanged; no post-loop table update is introduced.

Legacy saved config defaults D2 OFF. D2 OFF retains D1 geometry and epoch policy.
The unchanged strict load_for_eval restores model_state after initialization,
including trained nonzero tables and both projections, and returns the same
causal model. Neither flag changes the scoring forward or loader.

## Scope and nonblocking observations

Round62 explicitly supersedes SPEC's older wait/automatic-A-B queue sentences;
this receipt follows Round62 and adds no scientific gate. No D2 native production
parity, cold startup, full-budget success, fresh20M score, speed gain or submission
readiness is established here. Native results remain future run evidence under
the existing workflow. No source, packet, evaluator, prepare file or host was
changed; only this new verification folder was written.

The inherited module docstring still says output-directory is the only CLI
option although the freeze option is now present. This is documentation drift,
not a runtime/source gate for these explicitly flagged research arms. Frozen
candidate bytes were preserved.

The first local audit attempt failed an auditor assumption: it treated archive
payload_sha256 as a stdout hash. Collector code shows this is the request-wire
hash. The verifier was corrected to reconstruct that exact wire; it then passed
without altering any source, receipt or assertion threshold. `AUDIT_ATTEMPT1.txt`
preserves the failed attempt, and the final executable/output pins bind the
corrected independent audit.
