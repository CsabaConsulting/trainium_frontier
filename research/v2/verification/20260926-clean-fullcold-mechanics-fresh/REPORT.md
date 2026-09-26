# Fresh independent mechanics review: REFUSE

Verdict: **REFUSE acceptance of the proposed full-cold lifetime-lock/timeout-containment guarantee.** The declared four-literal driver delta is exact, and the tested normal train → fresh-public20M flow matches the fixed-purpose proposal. An independent CPU fault reproduction shows that a harness timeout can return control and release the outer lock while the harness's child remains alive. This inherited control-plane defect is sufficient to withhold mechanics acceptance and any full-run release. It is not a scientific finding about either source.

This review used only `/home/poojith-devan/clean-fullcold-review-DrcoWi` and new local CPU fixtures under this report's directory. It did not browse the repository, contact a machine or network, execute training/prepare/evaluator code, modify the supplied packet, or implement a fix. No implementer test or note was used. The actual preceding cold-five result is pending and is explicitly outside any acceptance claim here. No all-goals qualification or submission PASS is given.

## Exact inputs and literal delta

Independently verified hashes:

| Input | SHA256 |
| --- | --- |
| host_driver.py | ee5bc9f7a984c77b2c489eb5f361be37949de29f7b48637c58eb3a384c27dd90 |
| baseline_host_driver.py | 0b0f34cba9757f9371ab9c0b6e2b33adc6294023f5ea7f61464eaf96f2b9a511 |
| package/train.py | e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43 |
| submission_rehearsal.py | 7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b |
| idle_guard.py | adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737 |
| dispatch_driver.py | c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a |
| stage-dispatch.py (later supplied) | adcb67146b1f542289b7cc21ac552c2bcd558e25d40d624c461a2d8ddc5127ec |

The candidate driver equals the baseline byte-for-byte after exactly the four ordered replacements declared in the binding: old packet root → new contained packet root (10 occurrences), remaining old label → new label (3 occurrences), old source SHA → e1e2 source SHA (1), and old provenance SHA → dcec405e… provenance SHA (1). No function, numerical behavior, argv flag, timeout, guard, or evaluator flow is added by that delta. The old 3f5035da source's scientific refusal remains unchanged and confers no approval on e1e2. E1e2's separate scientific review was not repeated or inferred here.

The packaged launch command hashes to b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd, matches the proposal and binding, and resolves to the specified venv torchrun through controlled PATH. It has four ranks, context 1024, the declared unchanged flags, and no five-step diagnostic override. The binding and proposal retain false stage/launch/submission authorization.

All supplied manifest members matched their expected hashes. README.md and test_binding.py were deliberately omitted from this fresh-review packet; therefore this is not independent validation of the complete original manifest. The first harness hash actually observed by this reviewer was the correct 7a55cb87…; no earlier extra-blank-line copy was observed.

## Own CPU evidence

Commands run from this report directory:

```text
python3 -B verify.py > normal.raw.txt 2>&1
python3 -O -B verify.py > optimized.raw.txt 2>&1
```

Both exited 0 with 110 unconditional verifier checks. The raw `all_checks_passed: true` means the assertions about the observed evidence succeeded, including successful reproduction of the containment defect; it is **not** an experiment or launcher PASS. The verifier's own checks do not use Python `assert`. Runtime AST fixtures are explicitly compiled with optimization 0 to model the fixed normal-Python launch. A separate optimized-driver fixture verifies `OPTIMIZED_DRIVER_FORBIDDEN`. Running the verifier itself under -O confirms its checks are not discarded; it does not claim that an optimized harness invocation is authorized.

The actual driver body was evaluated with local synthetic files and intercepted process/guard calls. It performs exactly train then eval, passes the same FD, rechecks the guard before eval, and has no successor. Occupied phase paths, altered source hash, first or second guard refusal, training nonzero exit, bad budget receipt, missing checkpoint, changed FD, eval preflight failure, and eval nonzero exit produce the expected refusal/terminal transitions. Hostname, STOP, disk, URL, and optimized-driver faults refuse.

The pinned actual harness was executed in local contained fixtures with prepare and child execution replaced by CPU stubs. Its actual predicate rejects an outside sandbox before creation; contained train and eval fixtures pass. It creates six empty private cache directories per phase, strips PYTHONPATH, uses matching compiler URL/DIR, preserves the training argv, and builds the separate public20M/context1024/B1/1200-second eval argv. Its tested synthetic accounting is wall 100 minus startup allowance 10 = charged 90. This establishes construction and arithmetic, not actual compiler routing or native performance.

The exact idle guard was additionally used under real local `/usr/bin/flock`. Two harmless child phases received the same FD; four competing acquisitions returned 1; the lock became available after the driver exited. Native-job, occupied-device, unknown-device-schema, missing-parent and ps-command faults were also tested using fake census responses.

The later supplied transport's actual remote pre-staging gate statements were tested with local files and intercepted lock/process/device calls. Wrong hostname, consumed root, wrong original inode, busy lock, 11 GiB free disk, STOP, changed prepare hash, outside sandbox, shared compiler URL, busy process/lineage/device/fuser all refuse. Its local repository/authority/archive/network path was read but never run; no nonexistent cold archive was synthesized as acceptance evidence.

## Blocking timeout fault and exact lifecycle

The driver uses `timeout --foreground` in both phase commands (host_driver.py lines 19–20) and passes its lock FD to that subprocess (line 50). The pinned harness starts the train/eval child with `subprocess.Popen(command, cwd=sandbox, env=env)` (submission_rehearsal.py line 88). That call uses default `close_fds=True`, supplies no lock FD, and does not create or supervise a contained process group.

The independent reproduction uses the exact harness Popen AST statement and exact idle guard, with an explicitly harmless CPU command and shortened 0.3-second timeout / 0.1-second kill grace. Its child sleeps for 1.5 seconds and then finishes. It does not execute a replacement trainer or claim an observed native orphan.

The observed lifecycle, identical in normal and -O raw outputs, is:

1. Outer flock owns the local original inode; the driver authenticates the inherited FD.
2. Driver starts foreground timeout → harness fixture, passing that FD.
3. Exact harness Popen statement starts the CPU child, which reports that it does not possess the original lock FD.
4. Timeout terminates the harness and returns 124. The driver and outer flock exit.
5. The child is still alive, while a new competing flock returns 0. The fixture child subsequently completes normally; no test child was deliberately left running.

Raw result fields: `outer_returncode=124`, `child_alive_after_outer_exit=true`, `child_has_original_lock_fd=false`, `new_competitor_returncode=0`. Thus the control-plane structure cannot guarantee that the lifetime lock covers live descendants on its timeout path. `no_automatic_next_job` is correctly true but does not repair that guarantee.

## Additional composition limits

Stage-dispatch.py checks device/inode 66305/832589 and 13.5 GiB under its staging lock, then explicitly unlocks at line 152 before detached dispatch at line 154. The runtime guard authenticates whichever inode currently occupies the path; it does not compare the pinned original identity. The driver independently checks only 10.5 GiB. Tests demonstrate that 11 GiB passes the driver and fails staging. Normal train/eval lock continuity is proven locally, but staging-to-runtime continuity and original-inode identity at the final acquisition are not unconditional executable guarantees.

Checkpoint immutability after evaluation, all-rank finite native completion, effective compiler routing, real cold startup, source over-budget status, actual descendants/device/lock postflight, score, cold-five acceptance, and supervisor/operator authority require their separate actual evidence. None is supplied or claimed by this CPU review. The unchanged prepare source itself was not supplied or executed; only its binding and command usage were checked.

## Minimum repair invariants for a fixed-job guardian

A new narrow control-plane revision must retain the authenticated original lock until all descendants of its one train/eval job have exited, including timeout, signal, harness crash, and evaluator failure. It must pin the original device/inode at the lifetime acquisition, identify only its own process lineage with stable identities, terminate and reap that owned lineage when required, and remain failed/locked while descendant quiescence cannot be established. Train failure must prevent eval; either phase failure must prevent any successor. Receipts must preserve the timeout, signals, observed identities, exits, and lock-held-through-quiescence evidence.

This is a requirement for repair and fresh fault verification, not a proposed general launcher or authorization to modify the protected harness/science. Merely removing `--foreground` has not been tested here and must not be assumed sufficient. Existing live jobs must remain untouched.

## Evidence hashes

| Artifact | SHA256 |
| --- | --- |
| verify.py | 39975c5ff12df7d90e89a15626311615508d177c211c6bd3d0ad4c12f7e3d3c9 |
| lock_probe.py | 9f5698bf4764e56c941ac2f87312d145c269aee6492767d66d122e638430e65f |
| phase_probe.py | e0ba6f6fdeeba986109eba1eddcc4e42b683b8d68d0cf5b7b9327a0c2ddd246b |
| timeout_driver_probe.py | d4f241b5abaf8e34b03df1875af0d54458398c3c2a95204af4595856c917575c |
| timeout_harness_probe.py | ad516eb150ca8c77653fc59917504b87ea7063175f8dc946d6144180e7ca8455 |
| timeout_child_probe.py | 271cb0b8ab2f467ebc5139f8461b96576831d031ebcd0e4df765b94f260078e9 |
| normal.raw.txt | c1b83d6a9a91313385bf2bd95d804f0ff2cd2228a80f5063d626e8870776024c |
| optimized.raw.txt | be8baf9d9c592f548fdbca181844d564927d93cdad4144ae02adefeb96cf57f1 |

Driver, harness, and scientific source hashes remained unchanged after both test runs. Review ends at REFUSE; no fix, stage, launch, native result acceptance, submission action, or all-goals qualification was performed.
