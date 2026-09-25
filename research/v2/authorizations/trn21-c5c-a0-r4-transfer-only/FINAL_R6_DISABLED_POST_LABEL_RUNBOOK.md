# trn21 r4/446b final-r6 source-transfer repair — disabled HOLD

This is an Implementer-only, **disabled** successor to the held r5 enabled
request. It grants no source transfer, checkout, CPU G2, native launch,
evaluation, cleanup, breaker reset, AWS action, or submission. Both r6 request
authorization flags are false, the assignment ID is `...r6-not-issued`, and
the r6 wrapper stops before review/assignment reads, one-shot marker, journal,
or SSH. Neither a final r6 review nor a Supervisor r6 assignment is supplied.
The r1–r5 subjects, reviews, and HOLDs remain unchanged. The frozen 446b
native launcher has its own separate HOLD; this package addresses only an
intermediate source-staging transfer path and does not confer native authority.

## Exact local bytes for independent review

All SHA-256 values below are of file bytes. New-file paths are relative to this
directory. The runbook deliberately has no self-referential hash; an
independent Verifier must compute and pin it in a separate review.

| New r6 file | SHA-256 |
| --- | --- |
| `TRANSFER_REQUEST.FINAL_R6_HOLD.json` | `7cae6fba6cbd6e763ede016d4fec7dfa5aa8be78fa0fd7dfa0f7ae7b5b8f3b71` |
| `operator_transfer_final_r6_once.py` | `0520350e7aa18a04e70bfc54ee71dddd97b7e16c34c524cffc6bccbfca420787` |
| `stage_r4_transport_final_r6.py` | `c93f432132d3e23d6fc73ba541029149942acf6a41c2852cb83fa54b33c754bc` |
| `stage_r4_remote_final_r6.py` | `bee6c660e18e084c0d55a66473b1418772ac58adcad294c0775aee0acaee71b6` |
| `test_operator_transfer_final_r6_once.py` | `1055860408a8bf90ec751e05727ecaee75a2d2976fcb8ba05f510d73f0bfb85f` |

The held r5 enabled subject is commit
`fb3f52af4f522146e41b45314f2a9b356107dece`; its independent HOLD is
origin commit `169902beb1563d0a0f2fd81fca04ae763e57b9ed`, receipt SHA-256
`21a68fa2aa8e9ae3d0fa88d3f01366daee6e569c445ef3af4ea0a2337e1a40d3`,
and adversarial STOP-during-scan test SHA-256
`9a391ee9798aee778815be1c7dd6966958d9731927d741eded2f3a3b871f6135`.
The held r5 request, wrapper, tests and runbook remain pinned in the r6 wrapper
at their reviewed hashes. The old r5 inode-floor static PASS, r4 private-lock
HOLD, r3 disabled PASS, and older r1/r2 HOLD bytes are also pinned and never
edited. The r6 wrapper checks the actual local predecessor bytes, its own
request hash, and the new remote/transport source hashes before any authority
or I/O. A later enabled request would need new, separate exact review and
Supervisor assignment; changing either false flag in this r6 request fails
its hash and exact-schema gates.

## Repair and safety boundary

The held r5 writer scanned for the science label *after* its last live
STOP/process/Neuron check. A persistent STOP introduced during that up-to-45s
`find` could therefore precede target `O_EXCL`. In r6, the writer acquires the
existing launcher-compatible exclusive flock, checks full live activity,
performs a first clean label scan, checks full live activity again, performs a
second clean label scan, rechecks named/opened lock identity, then checks
target, STOP, process and Neuron state after that last scan. A second cheap
target/process/STOP pass follows the potentially slow Neuron commands, with
STOP sampled last. Any observed activity, unknown data, changed lock, or
unclean label holds before target creation. The write receipt carries both
scans, both postscan activity gates, lock identity rechecks, and their order;
the distinct transport validator rejects missing, dirty, or reordered proof.

Monotonic evidence must show no more than **1 second** from final label-scan
completion to the beginning of target `O_EXCL`. An expired scan holds *before*
target creation. Target-open duration must also be no more than 1 second; if
the open itself was slow, it may already have created the target, which is
retained for investigation rather than deleted or retried. The transport
receipt validator checks these exact intervals and the ordered monotonic
samples. These are fail-closed timing gates, **not a demonstrated live timing
result**. The two `find` calls can each take up to 45 seconds, and `fuser` and
`neuron-ls` each have up-to-20-second command timeouts; no read-only trn21
measurement shows that the last scan, final activity gate, and `O_EXCL` can
reliably meet the one-second bound. Therefore disposition remains **HOLD**
even if offline static tests pass. A separately authorized read-only live
timing assessment and independent review, or a simpler reviewed transfer
design, would be required before considering any enabled successor.

The prior r5 terminal free-inode repair remains: both `identity_before` and
`identity_after` of every full preflight, writer prewrite and postflight probe
must have at least 10,000 free root inodes. All prior source/host/boot/mount,
protected checkpoint/cache, STOP and local breaker, label, fresh full probe,
readback, unconditional postflight, one-shot and no-overwrite gates remain.
The exact read-only `O_NONBLOCK|O_NOFOLLOW` host-lock open and named/opened
identity checks remain. Only the observed private `.cache` mode `0700`
ancestry permits the `0775` inner directory and one-link `0664` lock, with
the pinned owner/inodes/modes, absent POSIX ACLs and no observed relevant
mount alias in this namespace. Unknown or changed ACL, path, mount, or lock
evidence is HOLD. The lock stays held through target write and receipt.

These observations are bounded to ordinary DAC and cooperative flock in the
observed namespace. Same-UID or privileged actors, already-open descriptors,
other mount namespaces and uncooperative writers can race the final sample;
neither the one-second bound nor repeated checks provide atomic exclusion.
No target deletion, rename, replacement, or retry is authorized on failure.

## Offline checks and next decision

The new local tests use disposable `/dev/shm` fixtures and no real capsule or
SSH. They exercise persistent STOP during **either** label scan, STOP during
the postscan Neuron command, late process/Neuron and science-label appearance,
scan expiry before `O_EXCL`, clean zero-byte write, mutated receipt timing and
STOP/lock evidence, and inherited inode/ACL/mount/readback negatives. The
existing package tests and archived independent negatives remain part of the
offline regression suite; they are not a substitute for a new independent
Verifier. No host, AWS, transfer, training, or assignment was contacted or
performed by this Implementer. No dispatch command is provided here.

At freeze, `python3 -B -m unittest discover -s
research/v2/authorizations/trn21-c5c-a0-r4-transfer-only -p 'test_*.py' -q`
passed **76/76**; the same suite under `python3 -O -B` passed **76/76**.
The three archived r4/r5 independent-negative modules passed **14/14** in
both normal and optimized Python. These are local Implementer regression
reruns, not independent r6 approval.
