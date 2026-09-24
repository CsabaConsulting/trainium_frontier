# G0 comparator r4 independent verification

Verdict: **PASS** for local evidence comparison only.

I independently reviewed `994d302b0036c56aaab63797f0561797bc4cf431` against parent
`4a32f9b3`. The diff is limited to the six comparator package, documentation, request-state, and
test files listed in the adjacent JSON receipt. The production candidate, `prepare.py`, observer,
hybrid and ordinary gate sources, model checker, and exact A1 ownership manifest are unchanged.

The checked-in package suite passed 60/60 tests, the hybrid-ownership suite passed 13/13 tests,
`py_compile` and `git diff --check` passed, and an independent publication check bound the returned
digest and full inode identity to the exact 0600 single-link result bytes. An injected publication
failure left no accepted or temporary result.

Replaying the immutable G0 archive produced exact `PASS` result SHA-256
`7198d39b9c21f1eb7e47f84e52398ca85d239ad564a36e6d524674d83355ab4a`: 66 owned entries,
owner element counts `[27405059, 30540035, 30408963, 29362179]`, shard stride `30540160`, total
padded elements `122160640`, and five owner exchanges per rank. Archived G0 is therefore valid
under the corrected comparator.

This receipt authorizes no run. G1 and G2 remain locked and all request state remains
`executable=false`. The Supervisor may draft a separate non-executable G1 request, but it must be
independently bound and authorized before any Operator action. This receipt does not establish
G0/G1 non-perturbation or G1/G2 equivalence.
