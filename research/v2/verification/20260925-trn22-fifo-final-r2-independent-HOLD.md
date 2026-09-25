# Independent trn22 FIFO final-r2 transfer review — HOLD

Reviewed on 2026-09-25 by a Verifier separate from the Implementer. Exact
subject: `d49558fdf63ffa5f537a08cc00c011fb8cf3a806`, pushed at
`origin/codex/20260925-trn22-r4-transfer-lock-final-request-r2`. The released
Implementer checkout was clean at that commit; this receipt and its test live
only on a separate review branch. The repository contract was read in its
required order; the five required instruction files are unchanged from the
prior final-r1 review.

**Verdict: HOLD for the exact final-r2 transfer request. No fixed-path final
PASS, Supervisor assignment, marker, journal, SSH, transfer, checkout, native
launch, evaluation, cleanup, AWS action, or host contact was made.** A PASS on
the disabled `5740a0e0` FIFO repair is static-only and cannot authorize this
enabled request. The final-r1 independent HOLD remains preserved, not authority.

## Blocking host-side timing gap

The local r1 STOP gaps are repaired: final-r2 checks local STOP/breakers after
slow second authorization and directly before marker `O_EXCL`, and again
after SSH host-key setup, source-FD rehash and journal fsync directly before
write `Popen`. Independent synthetic tests confirm a persistent local STOP
in either slow interval prevents the marker or write spawn.

The unchanged reviewed remote writer still checks host STOP/processes at
`stage_r4_remote_lock_hold.py:768`, **before** its final label observation at
line 775. That observation runs `find` with a 45-second timeout (lines 587–609).
The next gate only bounds time **after the scan** to target `O_EXCL` at
lines 776–780; it does not re-read host STOP/processes. Thus a persistent host
STOP that appears during this potentially slow scan is present before archive
creation but is never seen. The local last-minute STOP check cannot detect a
new host-side STOP after SSH begins. This is a slow check-to-side-effect window,
not the unavoidable tiny race after a final STOP observation.

The independent host-free witness
`test_remote_stop_during_final_label_scan_reaches_exclusive_create` injects a
persistent synthetic STOP *inside* the final label scan against the exact
remote source with only temporary synthetic paths. It observes one host STOP
check, then actual synthetic target `O_EXCL` creation and retained payload.
The test deliberately supplies a wrong synthetic payload, causing a later
source-hash failure; that later failure does not undo the prohibited creation.
No real host, archive target, marker, journal, or SSH process is touched.

This does not claim that a STOP was present on trn22. It shows that the
requested full live host STOP safety cannot be assured for this one-shot
write. A distinct repaired remote/request candidate must close the slow
interval while retaining fresh final-label and other gates; it then needs a
new independent final review. No source modification is made by this Verifier.

## Exact bytes and bounded checks

| Evidence | SHA-256 |
| --- | --- |
| Final-r2 request | `b8bcfbb5d898bfb6c3b6308600b9c2e91d474fdccee66620bd6ff055f30f2ec2` |
| Final-r2 core | `42d21f959ca2c7bab156bb5bdc320203c1278971092fd40e95da27e24e0846b8` |
| Final-r2 wrapper | `f761ce825691a2ac911adb65f3e6dbdc9c90ec613e1de11b8f9e46f1690e66ae` |
| Final-r2 transport | `fde41bbe3ab5367840d1b225316b2d961a562a2dee4e6ca9509120e83b060d84` |
| Final-r2 runbook | `592f8c435e2f78789c300c6ba35f7698eef8691e20b6585fff30dc415db8c29f` |
| Supplied final-r2 test | `375d3257752d842b8f6519b17c9657ccb1f2b28848254707944ab8507d2c1ad1` |
| Disabled remote, unchanged from `5740a0e0` | `cdf06f700722e144dcb304e517a7e000fe0c6b2bcb210e4d2bb38f92757d5e27` |
| Disabled static-only independent PASS | `67c3e430bd14623b4f11a305f9a445ad6d3276e6e63833a693d004aa94988f6d` |
| Historical final-r1 independent HOLD | `8d1bf8caa6fa0643b067c9242cbeed1e8adb756592bbedc274fefcc328ca04ad` |
| Source-stage contract / independent review | `79878b16f459d4c12b32cb0ebb85d205dad86d584ae09654e0e8161920deab69` / `d49e8c03567210283074c44a2f2937594ada39f43e399f97d176c373c1a32c1e` |
| 104,154,875-byte local capsule | `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a` |
| Local known-hosts, without private-key contents | `e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3` |
| Verifier-owned nine-test witness suite | `5b752b8ebb12649736bcdf0577d00957ee6ebf1b7ea84f05050224632b28cfb0` |

The r2 transport is byte-identical to final-r1 transport. The disabled
successor files are unchanged from `5740a0e0`, and final-r1 package files are
unchanged from `06ecfd8b`. Exact wrapper/request/core/transport, source-stage,
review and HOLD bindings were checked. The capsule passed the production
same-FD local-source checker; local ED25519 known-hosts/fingerprint verification
passed without host contact or reading the private key. The fixed final-r2
review, Supervisor assignment, one-shot marker, and journal were absent.

Supplied final-r2 tests passed **11/11** normally and **11/11** under
`python -O`. Verifier-owned adversarial tests passed **9/9** normally and
**9/9** under `python -O`: exact role/hash-chain mutations, absent/nofollow
authority files, both local STOP intervals, pre-marker source FD inode/hash
changes, optimized dispatch refusal, private-cache ACL/mode, lock, mount,
disk, old-inventory, late-label, process, Neuron and STOP receipt mutations,
and the blocking host-side STOP witness. Earlier disabled/archived suites
passed 14/14, 4/4, 13/13, 9/9 and 3/3. Optimized standalone core/transport and
alternate-operation entrypoints refused dispatch. `git diff --check` passed.

The proof is static and synthetic. It is not a fresh trn22 preflight or a
claim about the current host. The current request remains **DISPATCH HOLD**.
