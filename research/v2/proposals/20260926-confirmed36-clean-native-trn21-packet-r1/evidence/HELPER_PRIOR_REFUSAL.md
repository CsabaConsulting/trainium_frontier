# Fresh independent review: REFUSE

Reviewed only the frozen `/home/poojith-devan/clean36-native-fresh-WIP7Cm`
packet, selected specification/proposal, pinned foundation/native pattern,
scientific source snapshots and factual author receipt. No author tests were
imported or executed. Session role: fresh independent Verifier, parent-requested
Astra session (the runtime does not expose an independently attested model ID).
Date: 2026-09-26. No source edits, host access, network calls, native imports,
full model allocation, full-size masks, training, or authorization changes.

## Decisive findings

1. **P1 — completed partial native evidence is not durable before timeout.**
   `probe.py:295–313` invokes the entire diagnostic before its sole rank-receipt
   write. `run:230–231` stores completed shape records only in memory. A normal
   TERM/KILL deadline does not execute Python's finally. The specification
   explicitly requires timeout to preserve partial receipts, including per-rank
   complete or partial evidence. Independent CPU subprocess reproduction entered
   the real `main`, populated a completed partial shape, then delivered SIGTERM
   to itself: exit -15 and zero rank JSON files. This is a helper persistence
   defect, separate from the missing guardian/host binding. Fix by durably
   checkpointing bounded scalar/hash/token progress during execution; retain
   immutable final receipt semantics and distinguish partial from completion.

2. **P2 — causal failure loses its numerical evidence and current call records.**
   `probe.py:116` raises a detail-free CheckFailure before returning the case;
   `run:230–231` therefore never attaches its calls. `main:298` persists
   `error=CAUSAL_PREFIX, details=null`, with no measured magnitude or failing
   case's hashes. An independently implemented future-token leak reproduced
   exactly this. Preserve the observed error, cut/prefix range, input/output
   hashes and completed call records before raising, without changing tolerance.

These findings prevent complete source/specification acceptance. They do not
identify an always-failing production numerical route: I found no such defect
in the pinned ordinary-loader/configuration/byte-check path. Fixing reporting
does not authorize native execution or imply numerical success.

## Evidence that passed within the CPU/source scope

`reviewer_tests.py` is a new reviewer-owned implementation. Its scalar Python
oracle independently computes tiny causal outputs; all B1/B2/B3 × T256/T257
cases match exact bytes and input hashes, with 30 calls per arm. Normal Python
and optimized Python each passed 13 tests. Two named tests intentionally
reproduce the defects above; test-suite success is not helper acceptance.

Negative controls reject parameter/nonpersistent-buffer/direct-attribute
mutation, requires_grad changes, mode changes, device-input mutation on CPU,
hooks, NaN/+infinity, batch dependence, future dependence, repeat divergence,
aliased scratch corruption, signed zero, one-ULP bytes, dtype/shape mismatch,
degenerate suffixes, bad full/window masks and changed validated-mask bytes.
Tiny masks retain legitimate causal negative infinities; direct attributes
receive no infinity exemption. Zero-reference relative RMS is explicit.

Structural tests reject missing/duplicate ranks, shapes/calls and cross-source
hash disagreement. Production-width fields in these metadata-only validator
fixtures are explicitly synthetic; no full-width outputs were allocated.
File tests reject changed and symlink inputs and changed scientific pins;
the frozen 17-input manifest validates. Wrong config/count/payloads reject
without allocating a full model. CPU stand-ins test absent saved toggles,
dtype/NKI/floor failures and restored short-window configuration. They do not
prove a real native loader ran.

Source review and AST-only predicate execution confirm native convolution
conditions, T256 activation/T257 fallback, odd B1/B3 single-lane versus B2
two-lane dispatch, research's singleton wrapper, and the unchanged RoPE length
condition. Both loader bodies use ordinary compute_dtype_for, strict loading,
device transfer, eval/frozen parameters and the 8192 table floor; pinned source
uses weights_only. Source does not inject hooks, dtype shims or forward edits.
Serial model deletion/gc/synchronization occurs before the next loader. Main's
exception/finally boundary is independently CPU-mocked: initialized process
group destruction is attempted; destruction failure produces UNRESOLVED.
Missing synchronization raises in tiny tests. Real initialization and teardown
were not exercised and are not claimed.

Actual native models, restored device state, four-rank receipts, runtime/graph
stats, exact environment/cache/guardian binding, available memory/disk,
partial-timeout retrieval and descendant quiescence remain unproven. The
disabled binding correctly leaves staging/launch/submission unauthorized.
No source/numeric/host acceptance may be inferred from this report.

## Exact identities

- probe.py: `73c84e13ce8f9b92b1c74f4b4b3f5063be03cf73919cfc8e738174042c23c96d`
- REVIEW_SPEC.md: `1b5a311e04904872dbdf614037a2def9ef45a5362c173d555c058e66eb7f599d`
- EXPERIMENT_PROPOSAL.json: `76f90b5db0cbf9d8d903e71ad6ce7d3bf05e484056a10f3a6011579f2fec5cf8`
- implementation.patch: `d7a869d522fd38c37ec9f825d400d140509f129e25c09547889d5f6ce61b4a4d`
- BINDING.disabled.json: `fb9806e681aa79d554aee3a914391c088c4e34207955819c895967307c7842cd`
- CPU foundation: `65e7eb1a21b0e880ee2362461ff3664c4dbd52933816e5ec0684645c92cd75e8`
- Native pattern: `d21d35c68440e6a2b62abf5338603f0c8a4891a454bb61d3df0fdc420d7682a1`
- Reference science: `36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb`
- Candidate science: `e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43`
- reviewer_tests.py: `9f34bda7bb72b81b6ccb1f490c10f7be19b8652515cddbe31156cc0f6c54263c`
- normal.log: `6e44ed10d0cdc1993556c23749bff9a254a83c9d20642744504458cdf5d4eb6e`
- optimized.log: `d0af615604122aac3ef6da93549993b7f2cf1f99430e61698fdbcbfaf52e12e7`

Commands: `/home/poojith-devan/venv/bin/python -B reviewer_tests.py` and
`/home/poojith-devan/venv/bin/python -O -B reviewer_tests.py`, with artifacts in
`/home/poojith-devan/clean36-independent-review-m8vrKn`.
