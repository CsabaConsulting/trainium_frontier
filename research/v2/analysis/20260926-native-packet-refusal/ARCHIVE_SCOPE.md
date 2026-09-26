# Independent native-packet refusal: archive scope

The exact fresh report and all test sources, normal/optimized results, raw
process events, logs and original manifest are preserved in
`research/v2/verification/20260926-clean-native-host-packet-fresh/`.
Verdict is REFUSE of the **new disabled native packet**, not the running
clean full cold replay. Failed directory scans were silently counted as zero.

Only six reconstructible, zero-filled synthetic fault fixtures are omitted
from Git: `artifact/oversize.log` (268,435,457 bytes) and
`hidden/receipts/.rank0.interrupted.tmp` (16,777,217 bytes), each under
`initial-normal/`, `normal/` and `optimized/`. These are test stimuli, not
operational receipts. Their original files remain unchanged in
`/home/poojith-devan/clean36-host-review-OqqNW1/`; their complete hashes remain
in the original manifest. No actual error output or receipt is truncated.
The test source reconstructs these sparse fixtures. The archive audit streams
and checks their original bytes too, without loading them all into memory.

This explicit selection avoids committing 855,638,022 apparent bytes of
synthetic sparse padding. No file was deleted. Hash checks authenticate the
evidence; they do not constitute new numerical experiments or native proof.

Six negative-test symbolic-link fixtures (`sizes/bad` and `inputlink` in each
mode directory) are preserved literally, never followed. Their exact targets
are allowlisted by the audit and checked against the original fixture links.
An initial archive audit rejected these previously unenumerated fixture links
with `RuntimeError: no special archive members` in both Python modes. The
archive-policy clarification is not a change to the reviewed packet or its
REFUSE verdict. No arbitrary symlink or other special member is accepted.

The separate Implementer is assigned only fail-closed root/nested traversal
repair. Science, sources, checkpoints, limits, guardian and disabled authority
remain unchanged. A fresh independent repair review is still required.
