# trn22 final-r3 post-label timing diagnostic — DISPATCH HOLD

This is a fixed, read-only feasibility study for the independent final-r3
[HOLD](../../verification/20260925-trn22-fifo-final-r3-independent-HOLD.md).
It cannot clear that HOLD by itself. The checked-in
`GATE_CONTRACT.HOLD.json` has `dispatch_authorized: false`; the enabled
request, independent timing review, and Supervisor one-attempt assignment
named below do not exist. **No Operator may run the collector on trn22 from
this disabled package.** The Implementer has not contacted the host or run
the synthetic tests. A separate Verifier must review exact bytes and run the
offline tests before a Supervisor considers a read-only assignment.

## Frozen subject and scope

| Fixed item | Exact value |
| --- | --- |
| Held transfer source commit | `d571ea61b9fd2c3c7a21f254e044338ff873edba` |
| Frozen final-r3 remote SHA-256 | `fcf6f59e19e168a986de155338457d7b209a7bc2131674eea0841f7112ea0d5a` |
| Frozen final-r3 transport SHA-256 | `a12b8b202e18a8e809355c1fe0499ac7b2b68a94c5766ddb383f331f07b0c6c1` |
| Independent HOLD receipt SHA-256 | `34377be826b5cad75759082f94dcb7bd1539f5f93c388b4a22cedac529d87fdd` |
| Host/endpoint | `trn22`, `ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com` |
| Pinned ED25519 fingerprint | `SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U` |
| Pinned local known-hosts SHA-256 | `e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3` |
| Frozen remote target, observed by metadata only | `/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz` |
| Diagnostic remote SHA-256 | `452113ab31188bf84211f0d59c9ae78b8c27f96b8e080a673b349051a79750b9` |
| Collector SHA-256 | `e15469f7d7cb71e247977603cf7c79d7004d57e98cd2d799fa45f7e2dc3bb43c` |
| Supplied synthetic tests SHA-256 | `21fdfab28010f796b26eb9ca504b16fcca22fcf91789c413eb07dfe75f04fe19` |
| Fixed count | 20 trials, one Operator attempt, no retry |

The local collector is `operator_timing_readonly.py`. It authenticates the
unchanged frozen remote and transport hashes, the diagnostic hash, the HOLD
receipt, the local STOP/breaker state, and the pinned known-hosts file and
ED25519 fingerprint before any SSH. It never reads private-key contents.
The only remote argv shape is `exec python3 -B -c <pinned in-memory source>`
with operations `probe`, `timing`, `probe`; SSH uses `/dev/null` config,
strict pinned host-key options, `-T`, and `/dev/null` stdin. There is no
`write` or `verify` operation in its dispatcher and no transfer stream.

## What the study does

1. A full exact final-r3 `probe()` runs in its own SSH process. The frozen
   final-r3 transport validator requires idle processes, empty STOP and
   science-label scans, no checkpoint/Neuron/lock holders, protected
   checkpoint/cache pins, old-path inventory, absent target/future paths,
   sufficient disk/inodes, and correct host/boot/mount/lock identity.
2. The diagnostic SSH process rechecks pinned host/boot/mount, protected
   files, old inventory, future paths, and existing lock. It runs the exact
   frozen `_live_prewrite_safety()` once before the trials. For each of 20
   fixed trials it opens only the existing cooperative lock read-only and
   no-follow, acquires `LOCK_EX|LOCK_NB`, checks STOP/process state, calls the
   exact frozen `_final_label_observation()`, then calls the exact frozen
   `_post_final_label_safety()`. The lock is released before the next trial.
   There is **no archive/target open**: the recorded `would_open` timestamp
   is a clock reading only. A busy/replaced lock, activity, metadata drift,
   unknown helper failure, or target appearance aborts further trials.
   Only the exact helper's two one-second expiry errors count as timing misses
   and permit the next read-only trial.
3. The diagnostic records raw monotonic starts, completions, and durations
   for the final-label call/source scan; checkpoint `fuser`, `/dev/neuron0`
   `fuser`, `neuron-ls`, lock/ACL, first STOP sweep, `/proc` inventory, and
   final STOP sweep; plus the label-completion-to-post-end and hypothetical
   label-completion-to-`would_open` intervals. In-memory wrappers only time
   calls into the frozen helper. Their overhead makes the sample conservative
   relative to the uninstrumented writer and needs explicit review.
4. A second full exact `probe()` runs in a distinct SSH process after a
   timing attempt, including after a timing SSH failure. The collector
   compares both probes and the timing session for clean host/boot/mount,
   protected assets, old inventory, future paths, lock and STOP continuity.

The collector creates only the **local** fixed
`OPERATOR_RAW_TIMING_RESULT.json` evidence file with exclusive creation
before host contact. The file consumes the one attempt even on an error and
is retained, never overwritten or retried. It contains exact raw base64
stdout/stderr, byte counts, SHA-256 and return codes for all reached SSH
phases, redacted argv plus exact command hashes, all parsed raw timing
records, failures, and the clean before/after comparison. The remote
process uses `python3 -B` and has no marker, journal, archive create/open,
target open, checkout, training, evaluation, cleanup, breaker reset, AWS,
or submission operation. It acquires only a nonblocking advisory lock on
the existing read-only lock descriptor, as in the prior read-only lock
feasibility check.

## Separate gates before any Operator execution

An enabled `TIMING_REQUEST.ENABLED.json` must bind the exact hashes of the
disabled contract, frozen source, transport, diagnostic, and collector,
the 20-trial count, pinned host key, `/dev/null` stdin and fixed local
receipt path. A separate Verifier must provide
`INDEPENDENT_TIMING_REVIEW.PASS.json` binding the actual enabled request
and code hashes. Only then may the Supervisor provide
`SUPERVISOR_TIMING_ASSIGNMENT.json`, bound to the exact review and request,
for one read-only attempt. The collector checks the complete fixed JSON
schemas before creating its local receipt or contacting trn22. Without
all three files it exits locally. The enabled request must use the exact
field set implemented in `_authorize()` and canonical UTF-8 JSON (sorted
keys, compact separators, one final newline); the existing HOLD contract
must remain untouched. Local STOP/breakers and a fresh full host preflight
remain gates at execution time.

The independent Verifier can run the supplied offline tests with
`python3 -B -m unittest test_timing_readonly.py` from this directory.
That command has no SSH path. Once separately enabled and assigned, the
Operator's sole command is `python3 -B operator_timing_readonly.py` from
this directory, with **no arguments and no retry**. The local result file
must be retrieved, reviewed independently, and archived even if it reports
failure or a timing miss. A conservative worst-observed-time margin must
be set by the Supervisor before any feasibility interpretation; an
all-under-one-second receipt is not automatically a transfer PASS. Transfer,
checkout, CPU G2, native work and submission still require their own gates.
