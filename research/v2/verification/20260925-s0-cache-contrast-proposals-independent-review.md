# Independent proposal review: S0 warm and 116188 cold cache diagnostics

Review scope: Researcher commit `43dcc53f` only. This Verifier did not author either
proposal, operate a host, change `train.py`, or approve a leaderboard submission.
The later direct human instruction authorizes these diagnostics subject to a new
unique-label Supervisor assignment and fresh Operator preflight. This receipt is
**proposal review, not launch authority or a native result**.

## Bound source and CPU evidence

- S0 `workflow/candidates/S0/train.py`: SHA-256
  `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb`.
  Its bare four-rank, three-step CPU G2 is `PASS` with the real trusted
  `prepare.py` SHA-256 `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`:
  `research/v2/g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/remote-run/receipts/fs-20260925-g2-s0-bare-trn22-r1.json`
  (receipt SHA-256 `70c779f38f6768d1f78e41fa00e4000918a3767ece3ad8ca84423a81a31cdd58`).
- Known-good `submissions/dense6x1024_direct_review_matrixlr015/train.py`: SHA-256
  `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`.
  Its real-`prepare.py`, four-rank, three-step CPU G2 `PASS` has the *same ordered
  argv* as the cold-contrast proposal:
  `research/v2/g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/remote-run/receipts/fs-20260925-g2-s0-116188-trn22-r1.json`
  (receipt SHA-256 `654a0db296e60b34551c2ee0a48324796f87c1f9f8ea76e8aee4a7d75761d1f3`).
  The argv also matches the completed September 24 116188 launch in
  `research/submission-validation/20260924-c595-matrixlr015-trn21/EXPERIMENT_PROPOSAL.json`.
  The source bakes matrix LR 0.015, so absence of a `--matrix-lr` flag is correct.
- The CPU `116188-vs-s0.json` comparison reports matching plans, all three
  losses and 62 parameter digests. This is **CPU evidence only** and cannot
  replace S0's cold native replay.
- Ledger/decision/stop-list alias search found only a September 10 cold-cache
  run on a different, dirty 55.7M-parameter source. Neither exact diagnostic
  label was previously run. The S0 r1 failure is a different consumed label.

## S0 warm-cache diagnostic on trn21: PASS for proposal; dispatch still HOLD

`20260925-s0-warm-cache-diagnostic-trn21-r2.json` changes the cache-state
family only: exact S0 bytes, bare argv, trn21, seed and budget stay fixed.
It is a useful binary discriminator for whether S0 reaches native steps, **not**
proof of official cold-start safety or a scored quality improvement. Its
postmortem honestly does not prove the cause of r1's missing NEFF.

Before dispatch, the one-use assignment/Operator receipt must pin the actual
populated cache path(s), demonstrate those paths contain pre-existing relevant
objects, show the effective cache URL and compiler flags, and exclude any other
Neuron job or concurrent writer. The Operator must recheck source/prepare
hashes, seed/budget, data/tokenizer, ≥6.5 GiB free, label absence and idle
device immediately before launch; no old cache or artifact may be modified by
cleanup. If the compiled graph is not proven to hit a populated cache, a
successful run is an S0 native-start result but **not** evidence for the
warm-versus-cold mechanism. Full checkpoint and fresh public-20M evaluation
are still required before its BPB can be used.

## Known-good 116188 cold contrast on trn22: HOLD as written

`20260925-116188-cold-cache-contrast-trn22-r1.json` has the right source,
historical argv, G2, unique label, full-chip topology, disk/idle checks and
diagnostic interpretation. Its described use of
`NEURON_COMPILE_CACHE_URL` is consistent with the [AWS Neuron cache
contract](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/frameworks/torch/torch-neuronx/api-reference-guide/training/torch-neuron-envvars.html):
a non-`s3://` value is a local path, defaulting otherwise to
`/var/tmp/neuron-compile-cache`; `NEURON_CC_FLAGS --cache_dir` would override
the URL. However, the proposal does not bind a **specific** new, empty
persistent-cache path, and `candidate.expected_diff` says to copy the trusted
`prepare.py` into the sandbox. A mutable extra evaluator copy is unnecessary
and weakens the trusted-import proof. As written, do not call this a true-cold
run or dispatch it on the proposal alone.

A narrow Researcher amendment **or an explicit, immutable Supervisor
assignment controlling the same test** can clear this HOLD without changing
the science: copy only the exact 116188 `train.py`; use `PYTHONPATH` to the
existing immutable trusted `prepare.py`, verify its SHA on the host before
launch, and record the actually imported `prepare.__file__` and SHA in a new
process. Pin absolute, unique, precreated *empty* directories for
`NEURON_COMPILE_CACHE_URL`, the other named native/inductor caches and the
debug output; forbid `NEURON_CC_FLAGS --cache_dir` or other override. The
Operator must record prelaunch emptiness and post-first-compile files plus
effective cache-path evidence, not infer coldness merely from an empty
`NEURON_COMPILE_CACHE_DIR`. Verify no inherited `NEURON_COMPETITION_R1_*`
environment value changes the baked 42 seed, 1800-second budget, output or
training/eval configuration. Keep the exact historical launch argv and
compiler-default effective flags, plus disk/idle/STOP/source checks.

If those assignment-level conditions are met and checked before launch, this
Verifier's disposition is **CONDITIONAL PASS for the diagnostic design**;
otherwise HOLD. A trn22 contrast can only support a cache/setup hypothesis,
not establish its root cause across hosts. S0 remains **NOT SUBMITTABLE**
until its own new-label, true-cold full replay and fresh-process public-20M
evaluation finish cleanly, regardless of either diagnostic outcome.
