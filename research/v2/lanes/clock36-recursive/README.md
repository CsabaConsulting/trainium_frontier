# Clock36 + Recursive: Implementer self-check packet

Selected proposal628b8933/correction68eab362, Supervisor selectiona18b754a.
Source-only commit44926a6f35b7c221bb6cb6652eda3c77e9f10efd is based on
7419927c7417b5666a984ec9c2b849feadabaaa9. Candidate SHA256:
`4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea`.
No canonical source, K2 worktree, prepare/evaluator/data, host or queue changed.

## Exact scientific change

35 added/3 replaced lines port only the existing default-OFF Recursive package:
Boolean CLI/global, saved toggle, three donor-exact functions, device beta2
selection and per-group LR routing. Both LR calls use `schedule_elapsed`.
Every non-Muon group uses the donor Adam-style schedule; flat LR takes precedence.
The original36 startup helper, model, optimizer arithmetic outside the beta2
selector, loader, budget/stop/fifth-boundary reset, checkpoint and launch bodies
are untouched. Reversing precisely these changes reconstructs exact36 bytes.
Recursive formulas are AST-identical to75b9 (and its adopted6ef7 donor).
No LR/batch/seed retuning or source adoption is implied.

## Completed self-checks (not independent or native acceptance)

Seven stdlib test groups pass normal/-O: source reconstruction/pins, exact donor
helpers, default/Boolean toggle and old missing-key restore, four-toggle scalar
grid, progress/warmup/flat-kind/clamping/quantization precedence, real fifth
boundary with fake ages779/790/869.999/870/870.001/900/1000 and shifted process
starts, and all five specified negative scientific alternatives. The first
offset expectation failed because `(start+age)-start` rounds differently from
the requested age; only the reference was corrected. The failure is retained.

Ten sequential G2 arms pass: for each clock OFF/ON, exact36 baseline, child
Recursive omitted, explicit OFF, ON, and ON with independently written reference
schedule functions installed only in the test module. Each runs six actual
updates on four CPU/gloo ranks, model8x128 (4,523,316 parameters,65 tensors),
synthetic2x128 input microshape. This is NOT the113,256,724-parameter native
8x1024 model or full hardware microbatch. One thread per rank; initial available
RAM7,623,020KiB exceeded the6GiB guard. No peak memory measurement is claimed.

Default/OFF losses, every parameter gradient including missing status, clipped
gradients, parameter updates, optimizer state and checkpoint model contents match
base36 exactly in both clock states. ON updates match independent donor-formula
reference arms in both states. The real CPU `_muon_math` receives tensor beta2
when ON and the original scalar when OFF. This is CPU dispatch evidence, not
Neuron graph-capture proof. All65 parameter records are checked at every update.
Six receipt corruption/omission controls refuse normal/-O. Container checkpoint
file-byte identity is NOT claimed: the new saved toggle is expected metadata.

Standard G2 is unchanged and hash-bound. Its CPU-only shims are recorded in raw
receipts: synthetic tokenizer/data, BF16 compute, no native compilation/CE,
import-only PyArrow stub rejecting real reads, sliced microshape and fake1ms
clock. Test observer restores imported startup count from stub2 to real prepare5
and advances only the fake clock1000 seconds after the first update. This crosses
the actual unchanged fifth-boundary reset and retains capped excess on both sides.
Reference arms replace only test module schedule functions with independently
written formulas on the same selected clock; no source file is rewritten.
The ordinary candidate arms have no formula substitution. The helper test varies
below/at/above-cap ages separately. Faster real startup can change actual LR;
these fixed-clock results never promise native first50 identity.

Separate normal/-O tiny loader probe uses8x32,92,724 parameters,syntheticB2T16,
and one diagnostic SGD update (not a training recipe). It saves/loads through the
unchanged source loader and tests base/child-old-missing/child-OFF/child-ON logits,
toggle restore, repeat, independent rows, permutation, prefix causality and
parameter/buffer state invariance with strict equality. Test-only CPU float32
and EVAL_MIN_SEQ_LEN16 prevent large masks. No organizer causality/data/scorer
routine is executed. The source loader itself is unchanged.

## Reproduction and preserved failures

Run `test_static.py` and `compare_tiny.py` with python3 and python3 -O.
The fixed local tensor runtime is `/home/poojith-devan/venv/bin/python`,
Torch2.13.0+cpu, one-thread OMP/MKL/OpenBLAS. `run_tiny.py` refuses existing
receipt overwrite; reserve the shared laptop CPU slot before any fresh run.
`test_checkpoint.py` is the separate normal/-O tiny loader probe. Exact G2 argv,
source hashes and environments are `raw/commands.json`; raw receipts and logs
are under `raw/`. Standard support hashes are embedded in every G2 receipt.

Preserved failures: initial scalar offset reference (isolated and full-run tool
transcripts); first comparator's incorrect assumption that every rank exposes
G2's rank0-only `final_digests`; first loader probe supplied a tuple where the
existing attention parser requires a comma-separated string. The comparator
now uses all-rank observer digests and optional standard rank0 digest, and the
fixture uses the correct string. None required a candidate/source change.
Initial missing documentation-path lookups were harmless read-only errors;
the actual selected proposal/spec were read in full before implementation.

## Fresh review boundary and pending gates

`review-input/` contains only full-context source diff and the original neutral
proposal/spec. Supply that plus the immutable raw receipts/commands/failure
artifacts to the fresh reviewer, NOT this README, implementation tests or summary
conclusions. Raw input size is bounded but includes all parameter/gradient
digests. Root owns the fresh-review export and source selection.

Fresh independent review, exact full-size CPU qualification, new child-specific
cold-five<=900seconds(target870), native correctness and separate Operator
assignment all remain pending. Parent source/cold results are not transferable.
No native packet, launch, BPB/speed claim, adoption or upload authority exists.
