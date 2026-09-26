# K2 symmetric native outcome: authenticated exactness failure

Disposition: **FAILED native correctness gate; PARKED.** This audit accepts the
integrity and interpretation of the failed experiment, not its kernel, speed,
production use or quality. No retry, source edit, host call or native execution
was performed. This reviewer did not implement the symmetric consumer.

## Evidence and replay

Frozen raw commit `c087cbb7e3de7c6f2306f480d71c587c13eb61e2`,
`research/v2/operator/20260926-trn21-k2-symmetric-native/completed/`:

- Rehashed all **31 files / 556,014 bytes** from `results-bundle.json`.
  Bundle SHA256 `16dc0f7d84eed33eaaa821cb6afd21d00fc5b6d60b4ad1b9f93e55f3e07b9c7b`.
- Source `9ce2e6a037212aaa76bd85b9b4ab7d311a481757`; kernel
  `2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae`;
  checker `89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850`;
  reference `14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f`.
- Binding `710ea6772d5525d97d990ecdf7f835f7f65d8380` matches actual staged
  driver/support/command hashes, all four worker environments/argv, LNC2 and six
  recorded initially-empty cache roots. Those records do not prove every SDK
  cache lookup's behavior.
- `audit.py`: **252 checks**, normal and `-O`, identical result SHA256
  `481c8b046c93fe297015e1a49c72b13807119650746fab15ffe123f74669b6fc`.
  It uses explicit exceptions, not removable `assert`s.
- `cpu_wrapper_reference.py`: exact checker `make_wrapper` AST, but explicitly
  substituted 8×4 CPU gather and `aot_eager` backend. Twelve eager/compiled
  BF16/FP32 zero→nonzero→different-nonzero cases pass forward and gradient-byte
  comparisons, normal and `-O`. Torch **2.13.0+cpu**, not native Torch **2.11.0**;
  no NKI imported, no tensor payload above 1 MiB. Result SHA256
  `1128c9b16f5765de931b03b537705e9e69e9b592d3244b84ccd052e44c09d40f`.

Replay from repository root (the audit reads the immutable shared Git objects):

```sh
python3 -B research/v2/verification/20260926-k2-symmetric-native-outcome/audit.py
python3 -O -B research/v2/verification/20260926-k2-symmetric-native-outcome/audit.py
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/verification/20260926-k2-symmetric-native-outcome/cpu_wrapper_reference.py
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -O -B research/v2/verification/20260926-k2-symmetric-native-outcome/cpu_wrapper_reference.py
```

## What failed

All four ranks record **13/20 cases**: ten BF16 cases and the first two FP32
cases pass; `unique/fp32` fails only
`compiled_enclosing_autograd_backward`. Seven FP32 cases are unexecuted.
Each failing-case record has:

| Observation | Exact result |
| --- | --- |
| Upstream roundtrip, direct consumer, repeated direct, dense CPU oracle, eager custom backward | 32,709 nonzeros, expected gradient bytes, zero error |
| Eager and compiled forward | Expected bytes, 32,768 nonzeros |
| Compiled enclosing custom backward | 32,768 positive FP32 zeros; expected 32,709 nonzeros; max absolute error 8 |

An independent stdlib reconstruction uses the specified seed-42 integers divided
by 32 and the one-occurrence-per-row identity. Its 131,072 output bytes hash to
`dee95c1da97debf6a901e00260bd60a74e7be1992eca64b832c4f6633cccc671`, exactly
the recorded expected and direct/eager gradient hashes. The recorded failed hash
`fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471` is exactly
131,072 zero bytes. Derived relative L2 error is **1.0**. This is a real failed
gradient, not a tolerance or finite-only issue.

Crucially, **BF16 `unique` compiled backward passes with the same expected FP32
gradient hash**. Conversely, the first two FP32 cancellation fixtures expect
all-zero gradients; their passes cannot demonstrate a functioning nonzero FP32
compiled backward. The retained BF16 direct-component timing samples are not a
whole-step speedup or accepted performance result.

## Strongest supported localization; remaining uncertainty

The evidence localizes the discrepancy to the **FP32 compiled enclosing
custom-autograd execution path for this fixture/history**. It does not establish
that the arithmetic is generally wrong: the exact native consumer produces the
right bytes directly and in eager backward for this case. Nor does it exonerate
the kernel's integration/lowering in a compiled backward graph.

The checker saves counts/starts/positions/maximums and calls the kernel on
`gradient.contiguous()` in backward (`check_native.py:89–106`), then supplies the
explicit nonzero upstream (`:146–151`). Each mode receives a fresh device leaf;
the returned gradient is checked after synchronization. No obvious Python-side
omission of backward or zeroing instruction explains the failure. The CPU
substitute did not reproduce a generic wrapper/dtype-history defect, but cannot
test Neuron/NKI capture, alias handling, lowering or caching.

Possible explanations remain compiled saved-input/VJP capture, native NKI
integration/lowering, or state/specialization/cache behavior. The FP32 forward
cast is a dtype-preserving operation, unlike the BF16 cast; that is a difference
to investigate, **not a demonstrated cause or proposed fix**. Four graph-stat
entries per rank show two forward/backward specializations (node counts 9/7 and
8/8). Rank 3 reports nonzero compile times, other ranks zero; without graph bodies
or cache-key traces this does not prove stale cache reuse or wrong graph selection.

Current receipts contain hashes/scalars, not full native tensors. They observe
upstream before the calls, not its value at the compiled backward boundary; saved
metadata contents and their pre/post-call hashes are absent. No input mutation,
stale metadata, alias defect or backend bug can be concluded from this archive.

## Smallest next discriminator — proposal only, not authorization

If selected separately, use unchanged kernel/reference/wrapper bytes in a separate
diagnostic, **FP32 `unique` as the first compiled fixture in a fresh process**,
with direct/eager controls and one ordinary compiled `F.embedding` autograd control
on identical nonzero inputs. Record upstream and all metadata pre/post hashes,
actual gradient hashes/nonzero counts and the compiled graph identifiers. No full
model or entire 20-case replay is needed to answer this first question.

- If the custom compiled route still fails but ordinary compiled embedding passes,
  the issue is narrower than generic embedding backward, and does not require
  prior BF16/cancellation-case history.
- If it passes, history/order/state dependence becomes plausible, not proven
  cache causation; a subsequent explicitly selected history replay would be needed.
- Only if further localization is selected, a compiled explicit consumer call
  taking upstream/metadata directly (no autograd) separates nested native-kernel
  lowering from saved-VJP plumbing. Observing saved tensors or inserting hooks
  changes the graph and must be marked as an observer-effect diagnostic.

CPU references can check the generic Python/autograd and fixture algebra. They
cannot adjudicate the installed Neuron/NKI 0.5 path. None of these possible tests
waives the unchanged strict 20-case/four-rank gate or grants a retry.

## Terminal and containment

Child exit 1 after **30.136715991 s**, terminal
`DIAGNOSTIC_CHILD_FAILED` at **00:09:18.565542 UTC**. All four wrapper receipts
record return 1; torchrun subsequently signaled peers, which is distinct from
claiming every final OS worker exit was 1. No evaluation or automatic next job.

External postflight at **00:10:44.858629 UTC** records all eight known
dispatch/flock/driver/torchrun/worker PIDs absent, no related PID/PPID/PGID/SID in
the 261-row census, device idle and original lock **66305:596823** free. Timeout
and short-lived compiler PIDs were not observed before terminal; no identities
are invented. The full census and compiler/worker checks support quiescence at
that observation time. `candidate-ledger-row.json` is a proposed failed diagnostic
record only; this audit did not append or edit the ledger.
