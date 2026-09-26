# Fresh independent repair review

Literal scoped verdict: **ACCEPT_SOURCE_CPU_RECEIPT_REPAIR_ONLY**.
Separate host/native verdict: **HOLD — NOT EXECUTED, NOT AUTHORIZED**.

Date: 2026-09-26. Fresh independent Verifier session, requested as Astra by
Supervisor. This runtime does not provide independently attested model identity.
Only the frozen 16-file packet at
`/home/poojith-devan/clean36-native-repair-fresh-IrAY3D` was used as input.
No author test code, implementer branch or additional implementer context was
read. The selected proposal/specification, exact patch, baseline, scientific
snapshots, foundation and factual prior refusal/repair receipt were inspected.
No scientific source edits, native execution/import, full113M allocation,
full-width output/mask allocation, host access, data/evaluator execution,
training, launch or authorization change occurred.

## Findings and closure

The two prior source/CPU refusal findings are closed for this pinned repair.
The latest partial receipt is published before work and at call/case/arm
boundaries. Independent subprocess tests execute the real helper main with a
tiny CPU-only run surrogate. Actual SIGTERM during the second forward preserves
the first complete call, its independently calculated output hash, active case
and PARTIAL/UNRESOLVED markers. Actual SIGTERM just before replacement likewise
leaves parseable prior evidence. This proves tested process-signal behavior,
not a real native timeout or host retrieval workflow.

Independent injected write, file fsync, atomic replace, directory fsync and cap
failures refuse publication. Before replacement, the previous partial remains
byte-identical; directory-sync failure can leave the newer intact partial.
Final link/fsync/directory-sync/unlink faults retain the partial. Existing final
files are never replaced; a completed writer rejects further publication.
Source inspection confirms streamed serialization checks the newline-inclusive
CAP//3 limit before each write. A partial, temporary and final name can coexist;
their combined bound is 16,777,215 bytes, below the 16 MiB per-rank cap. A process
killed mid-write may leave a temporary file; retrieval must preserve it. Directory
cleanup unlink is not fsynced, so resurrection can conservatively cause refusal.
No power-loss or actual storage-device durability claim is made.

The offline validator rejects PARTIAL markers renamed to final filenames and
rejects extra partial filenames through its exact file set. Source logic keeps
the partial until final publication and temporary cleanup succeed. LOCAL_COMPLETE
is assigned only after successful work and process-group cleanup. Tiny mocked
main tests verify run failure -> REFUSE, cleanup failure -> UNRESOLVED, and success
-> LOCAL_COMPLETE. These are lifecycle controls, not native initialization tests.

An independently implemented future-token leak now persists the complete failing
case, all six completed calls, current reversed_suffix call, exact observed
magnitude, expected/actual prefix hashes, original/altered input hashes, cut128,
prefix range [0,127), maximum-error location and failure details. Test compares
these against independently calculated CPU tensors. A small future-dependent
perturbation below0.01 still passes the original causal criterion and reports
unequal prefix bytes. Repeat/row/output exactness remains byte-exact.

## Independent regression evidence

New reviewer-owned tests use a scalar Python cumulative-sum oracle and packed
float bytes for every B1/B2/B3 x T256/T257 shape. All output/input hashes agree;
30 calls per arm are checked. Negative controls reject parameter, nonpersistent
buffer, direct tensor attribute, mode, requires_grad, input and hook mutation;
NaN/+infinity, repeat divergence, batch dependence, future leakage, scratch
alias corruption and degenerate suffixes. Exact comparison rejects signed-zero,
one-ULP, dtype and shape differences; zero-reference RMS is explicitly checked.
Tiny independent causal/window masks accept the legitimate -infinity formula
and reject wrong values, NaN and +infinity.

Metadata-only rank/case fixtures reject missing/duplicate ranks, missing shapes
or calls, incomplete case states and cross-source hashes. Their8192 output-shape
fields are synthetic validator metadata; no full-width arrays were allocated.
Input tests authenticate the frozen17-file manifest structurally and reject
changed/symlink files and malformed payload/config/count without full allocation.
Absent synchronization is rejected in a tiny snapshot call.

Independent patch reconstruction exactly reproduces the repaired probe from the
frozen baseline. AST-only execution of actual scientific predicates confirms
T256 native-convolution eligibility, T257 fallback, and B2 two-lane versus B1/B3
one-lane selection. Science files, ordinary-loader path, source pins, strict
loaded bytes, frozen foundation and unchanged FP32 causal threshold are outside
the executable repair diff. The current review does not re-establish every
earlier ordinary-loader/route test as a new independent native result.

Normal and optimized Python each passed14 reviewer tests:

`/home/poojith-devan/venv/bin/python -B reviewer_tests.py`

`/home/poojith-devan/venv/bin/python -O -B reviewer_tests.py`

An initial reviewer run had a test-harness exception-class identity mismatch
because foundation() reimports its class. It correctly raised TERMINAL_RECEIPTS;
the harness was corrected to catch RuntimeError, without changing the helper.

## Remaining gates

Native models and four-rank numerical behavior, actual checkpoint loading,
restored native dtype/configuration, compiler graphs, resource/time sufficiency,
exact environment/cache binding, guardian acceptance, original lifetime-lock
identity, machine preflight, partial artifact retrieval and descendant quiescence
remain unproven here. This acceptance releases no host gate or Operator authority.
The disabled binding continues to prohibit stage/launch/submission. No BPB,
quality, adoption, public evaluation or submission readiness is claimed.

## Exact identities

- Frozen SHA256SUMS:119cb04cdcddcf7de4496bb86b55fb928ec03221692197060a4470d8ea5cd849; all16 entries verified.
- Probe:937e2f239ad90b5aa61e0b40ccf067b19ac02febe246492735e948a35c6b86af.
- Exact patch:0d92373d0af6fd51029c88b9d62b6664c08509c70e80185c9a933bad09c7b032.
- Specification:1b5a311e04904872dbdf614037a2def9ef45a5362c173d555c058e66eb7f599d.
- Proposal:76f90b5db0cbf9d8d903e71ad6ce7d3bf05e484056a10f3a6011579f2fec5cf8.
- Foundation:65e7eb1a21b0e880ee2362461ff3664c4dbd52933816e5ec0684645c92cd75e8.
- Reference science:36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb.
- Candidate science:e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43.
- reviewer_tests.py:dad0a797c19403e5275167325232267beae98b7112813fed9d5ae7081bb04aa7.
- normal.log:d2c3a0eebc5543c45f541d717c4ef33c0999fb339442db5b5d6e340a1bae660e.
- optimized.log:431460d3b13261382ce3121e7978fc732cf9fb04d13cd116969c2a2c212aa588.
