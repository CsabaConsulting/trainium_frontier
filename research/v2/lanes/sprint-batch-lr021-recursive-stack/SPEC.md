# Prospective batch262144/LR.021 + Recursive schedule source

Implementer package for `research/v2/proposals/20260926-batch-lr021-recursive-stack/EXPERIMENT_PROPOSAL.json`.
This is speculative off-host preparation, not an adoption, result, source approval,
or host launch release. The Supervisor expressly assigned preparation while actual
trn22 confirmation runs, narrowly superseding the proposal's implementation timing
condition. Actual confirmation, the one scoped independent review/composition
acceptance, selected-host comparator binding and normal machine admission still
control any launch. K2 and tables retain priority.

## Frozen inputs and source

- Parent: `research/v2/proposals/20260926-sprint-p0-clean-batch/train.py`,
  SHA256 `8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee`.
- Donor: `research/v2/proposals/20260926-sprint-p0-clean-recursive/train.py`,
  SHA256 `50fa6d144f19758dcdf9980e03f2302453174b956f0fa1a3e0845c4c0b72e6fb`.
- Donor diff: `research/v2/proposals/20260926-sprint-p0-clean-recursive/source.diff`,
  SHA256 `a71addfd36ab78cfdd30fa47ecc87401f6359eda2a7f5eb95586bff304fe130d`.
- New `train.py` SHA256:
  `c2e27ccde59e619d74102429e340716dbec16a6ba538ea9beca9c714d0d33f87`.
- `source.diff` is the complete diff against exact8c6. `MANIFEST.json` binds
  the final source, full diff, specification, author test and receipts.

The source is exact50fa plus the original six batch-port AST statements / seven
physical lines already present in exact8c6. Removing those exact batch statements
reproduces the donor byte for byte. No Recursive arithmetic was rewritten.

## Exact change

The new BooleanOptionalAction `--recursive-schedule` defaults OFF. ON uses
the donor's distinct Muon/AdamW warmdown, momentum and beta2 recurrences, its
nested group-LR routing (including `no_lr_decay`), and its persistent tensor
`one_b2_t` in the FP32 Muon update path. OFF preserves the original scalar
`1 - group['beta2']` path and original schedule results.

Existing `--total-batch-size` / `--matrix-lr` parser, validation and defaults
remain 131072 / .015. The intended arm passes 262144 / .021. MATRIX_LR retains
its existing use for both local-convolution AdamW and matrix Muon groups.
All other hyperparameters, model/data/forward definitions, packing, ownership,
reduction/clipping, clock/predictor/startup/reserve/checkpoint/eval behavior are
unchanged. No K2, tables, model-size change, launcher or new runtime dependency.

The proposal's full argv is parsed in the author check; no host command is
executed. The intended context1024/pack32/four-rank batch arithmetic remains
131072 tokens per microstep and accumulation2 for batch262144. The official
1800-second charged budget, seed42 and fresh public20M/context1024 remain
proposal requirements. No result can be inferred from preparing the source.

## Bounded author evidence

`author_cpu.py` uses the actual unchanged GPT and MuonAdamW classes with
synthetic inputs, eight layers, width128, vocabulary256, sequence32, BF16
compute and FP32 parameters. It uses one CPU thread, a 4 GiB address-space
cap and a 180-second outer test timeout. Data/evaluator imports are removed
only in the test's in-memory module load; scientific definitions execute
unchanged. No dataset or checkpoint is read, no host is contacted, and no
distributed ownership execution is claimed.

Completed receipt scope (see exact `CPU_RECEIPT.json`):

- Exact input hashes and literal byte composition; unchanged clock helper,
  model, checkpoint/eval loader, Muon math and ownership definitions.
- Actual CLI prefix parses all proposed flags, defaults OFF, and accepts
  explicit `--no-recursive-schedule`; production accumulation arithmetic.
- Donor-exact ON recurrences over the existing 40 cases plus adjacent-float
  cooldown-boundary witnesses, and actual extracted group routing for Muon,
  AdamW and other kinds with both `no_lr_decay` settings.
- Three-step parent/candidate OFF bit identity at default .015 with reduced
  accumulation1 and target .021 with reduced accumulation2. Losses, synthetic
  input order, clipped gradients, model and optimizer states, RNG and config
  are compared. The true 131072/262144-token global batches are not materialized.
- A small donor/candidate ON three-step witness at .021/accumulation2, including
  observation of the actual `one_beta2` tensor transport in the CPU fallback
  of the compiled Muon body. No Neuron compiler/native claim.

`test-attempt-1.log` preserves an author-harness failure: its observer read
positional argument13 (`to_bf16`) instead of argument12 (`one_beta2`). The
observer was corrected to bind the original function signature by name.
Candidate bytes did not change. That attempt supplies no PASS receipt.
The final `CPU_RECEIPT.json` / `test-attempt-2.log` record the complete rerun.

These are Implementer checks, not independent approval. No full113M model,
distributed execution, actual native throughput/startup, BPB, checkpoint I/O,
machine admission, scientific confirmation or submission result is established.
The existing donor review never constituted approval of an absent combined
file; this exact composition is delivered for the single scoped review.

## Handoff requirements

Supervisor binds the delivered source/commit and exact full diff only after
accepting the package. The original proposal remains untouched here. No
queue, state, ledger, selector, authority gate, host binding, run label or
launch permission is changed by this package. Source preparation alone does
not change the proposal's actual-confirmation condition or enable dispatch.
Raw same-host delta and unchanged-selector limitations in the proposal remain.
Only the existing sole Operator may perform any subsequently authorized run.
