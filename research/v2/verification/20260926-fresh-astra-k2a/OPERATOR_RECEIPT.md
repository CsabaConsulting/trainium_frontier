# Fresh Astra K2-A review operation

This is the session operator's evidence archive, **not that operator's independent
source review**. The independently authored report and tests are archived unchanged
under `session/reviewer-output/`. This operation cannot authorize source adoption,
a merge, native execution, speed or quality claims, or submission.

## Fresh input and session

- Candidate commit: `25b731f7638672aa8d03a7cd15b6a02b8f5776ea`.
- Candidate source SHA256:
  `e4d93babb4cec8ce53ee1ef9f9d4cc4238c40b93dff0245b66fbcf6bae158b03`.
- Base commit: `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b`.
- Base source SHA256:
  `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.
- Packet manifest SHA256:
  `9dd4f4814a2d4d7e06f7f74538d0495c49792d9108369f9f7650121348aacb2f`.
- NEW session: `01a0da15-9827-7cc0-8845-a39138c52543`, started
  `2026-09-25T19:40:43.266494+00:00`, CLI `0.155.0-alpha.16.4`, explicitly selected
  model `gpt-6-astra`. JSON events do not independently echo the provider model.
  This is a fresh same-family substitute, not a cross-family claim.

Only the complete source diff, neutral user lane specification and eight raw CPU
receipt JSON files were supplied. No implementation test source, README,
SELF_CHECK, implementation/review notes or conclusions, previous session history,
branch or repository was exposed. No mid-review feedback was sent. Raw supplied
PASS fields were evidence to audit, not the operator's acceptance.

The previously reviewed outer bubblewrap operation mechanism was reused, **not a
model session**: fresh non-repository directory, empty home, ephemeral CLI, no
resume/fork or shared branch/history. Only read-only OS/CPU libraries and `/packet`,
plus writable `/work`, were exposed. No original home, repository, credentials,
SSH files or process filesystem was mounted. A fixed `/proc/self/exe` symlink
supports the CLI. Provider credentials existed only in the CLI runtime environment;
model shell tools received an explicit credential-free safe environment. API-key
material was neither copied nor printed. Provider network was allowed; tool
network was forbidden by the specification rather than a network-layer filter.
CLI built-in instructions are not inherited implementation context.

The exact flags, preflight and executed-command events are archived. Outer
filesystem isolation remained active while the redundant inner CLI sandbox was
disabled, as previously authorized. This is not a retry of any provider refusal,
and no Opus or Fable call was made. The reviewer was required to preserve failures
and keep REFUSE until its own bounded CPU evidence was complete.

## Result

Fresh reviewer verdict: **REFUSE for the requested source/CPU exactness gate**.
The operation completed at `2026-09-25T19:47:21.610657+00:00`, elapsed
`398.3441591990013 s`, operation exit `0`. Its independently authored numerical
test exited `1`: 708/714 checks passed, with six native-algorithm CPU-emulation
exactness failures and no harness exception. All 696 non-emulation checks passed,
including the actual candidate CPU backward, repeated FP32 accumulation, default/
OFF/ON tiny three-step optimizer identity, gradient aliasing and evaluation bypass.

The blocking finding is per-128-token-tile regrouping of duplicate-index FP32
additions, source lines 1945/1970/1977 versus the CPU `index_add_` path at 2008.
For IDs `[2,128]`, width 8, vocabulary 256, one row receives exactly-BF16 upstream
values `[2**24, 1, -2**24]` at flattened positions `[0,128,129]`. Actual dense and
candidate CPU backward produce zero; both reviewer-written grouped emulations
produce one. The unit-scaled case `[1,2**-24,-1]` also differs. This is a source
semantic/CPU counterexample, **not measured native kernel failure**.

The promised exactness requirement was zero tolerance / identical bytes against
the dense CPU accumulation path. The supplied component receipts instead permit
nonzero absolute errors; their PASS labels do not establish that exactness.
Neither this operator nor the reviewer changed that contract or any native gate.
The fresh report requires a correction preserving agreed semantics; an approximate
contract would require a separate explicit policy decision, not reviewer fiat.

The test used Python 3.14.4 / CPU PyTorch 2.13.0, seed42, all CPU threads set to 1,
maximum width128/vocabulary512/tokens256/context128/microbatch2. It executed exact
AST-extracted CPU definitions and tiny call sites/AdamW methods plus its own dense
oracle and grouping emulations, not full-model/native/distributed training.
Raw stdout, failures, warnings, test source and tensor outputs are retained.

## Requested raw failure supplement

The reviewer preserved outputs for all six failures and inputs for both explicit
cross-tile cases in `reference_tensors.pt`. That unmodified 1,132,367-byte diagnostic
file is retained outside Git at the hash/path in `BINARY_RETENTION.json`. It is not
a trained checkpoint. The complete seeded reviewer test reproduces the remaining
input; no input from an implementer was fed into the review.

After the session ended, the operator added a separately labelled descriptive
supplement under `operator-derived/`, at the parent's request. It exports every
failing input and actual/reference output losslessly as sparse JSON, and computes
absolute/relative errors from the retained reviewer tensors. Recovery of seeded
inputs is checked against both saved output hashes. The first recovery attempt
refused because host AVX2 dispatch differed from the reviewer's recorded DEFAULT
dispatch; that failure is preserved. Explicit DEFAULT dispatch then reproduced
all twelve saved actual/reference hashes. This is not a reviewer retry, new model
request, candidate modification, tolerance change or independent acceptance.

Each row below represents two failing cases (matrix-multiply and sequential
within-tile emulations). Full per-case metrics and lossless values are archived.

| Input case | Max absolute error | Relative L2 / relative RMS |
| --- | ---: | ---: |
| Seeded all-same ID | 4.76837158203125e-7 | 2.957298025315422e-9 |
| Cross-tile unit scale | 5.960464477539063e-8 | Undefined: zero reference norm, nonzero error |
| Cross-tile large scale | 1 | Undefined: zero reference norm, nonzero error |

No epsilon denominator or new tolerance turns any case into a pass. The tiny CPU
slot was released after the session and this short supplementary computation;
no worker processes remain. Subsequent archive validation is stdlib-only.

## Scope and immutable support

The source candidate was not edited. No native/compiler/production-shape test,
whole-step performance, cold startup, quality, full-model load, merge or hardware
launch is established or authorized. The Operator alone owns any separately
assigned native work and the common per-host lock/idle protocol. No launcher,
queue, ledger or acceptance rule was introduced by this archive.

Original reviewer `/work/` links map to `session/reviewer-output/`. All reviewer
text/JSON and session events are copied unchanged. The operator-derived files are
outside that directory and are not counted in the reviewer's 714 checks.

| Unmodified reviewer artifact | SHA256 |
| --- | --- |
| `REVIEW.md` | `1cc7d41568e1bef6a047de1c4e3f062e9b8439fc3dc1ab29ead62901100c72c8` |
| `result.json` | `9f1dd5e379298db1f67884d1bf9f4ce27f28c482969218fd7ba4ae062b1fb608` |
| `reference_exactness_test.py` | `24422d083a090a3d92759bbda935a24a13dc73b909f003d91350f4ac4f9d6d84` |
| `reference_results.json` | `c24b3b77c3654c210010ade7581a85b89d49fe23500ceec347aaa104606ef91c` |
| `reference_tensors.pt` | `3b5be8e027de4897891a2983157a987a2e4331f7bfb3753ae4fc9cfb91d690c4` |

Returned CLI counters: input839793, cached-input744222, cache-write-input95529,
output20635, reasoning-output4262 tokens. These are observed usage counters, not
measured dollar cost or a provider-echoed model identity. No refusal was retried.
