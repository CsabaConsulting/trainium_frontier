# trn21 e65 → r10 r5 source-stage request — FINAL REVIEW PENDING / HOLD

The exact disabled r4 source at `826d1dc5` received independent **static-only**
PASS at `b8c9c36b4fa5b72a225d8d55aa223b8e2b341bec`, receipt SHA-256
`dd251ab06b59fe01136773d1c4f178869bf9f98771ffb58ee2cd81a9fe648d32`.
The later enabled r4 request `49f2941f` received independent final **HOLD**
at `06451f16d48026b9a167edf803ce57fdec787ce4`, receipt SHA-256
`0b9200787ed6a1d79b9b056d7f3068329fd89cf15fbbc0bd4df96aa1f654a515`:
a STOP/breaker arriving during the local SSH identity check could permit a
read-only remote timeout probe before the next local gate. The distinct r5
Implementer repair `2db60693` added that gate and removed the held r4 enabled
request. It received independent **static-only** PASS at
`739c5dac9e04dae56700e4d3c3d5a6493150cf89`, receipt SHA-256
`99e1fd9770171467c3679a396dff63d648f858513c0dcf1176eb0ad9257b7b2a`.
This successor adds one canonical r5
`REQUEST.ENABLED_PENDING_FINAL_REVIEW.json` for **SOURCE STAGE ONLY**. Its
true dispatch/source-stage bits express request intent, not launch authority:
the separate request-bound independent final **stage** PASS and Supervisor
one-attempt assignment files are absent. The executable entrypoint exits
HOLD before marker, journal, or SSH. No host contact, transfer, new checkout,
native run, AWS action, cleanup, breaker change, evaluation, or submission
occurred. This package remains distinct from the held
`c03e3ac3` trn21 r10 stage; that frozen r1 package and its independent
`31e022d5` HOLD remain untouched. Historical transfer packages and the
accepted e65 checkout remain untouched. Do not substitute held full-capsule
writers or treat the r10 source PASS as native authority.

The r4 source descends from r3 `7205abf5`, whose active Desktop
controller-root correction received independent **static-only** PASS. The
one-shot read-only trn21 timing result `c238fe24` and independent raw review
`228abbb5` are HOLD: the first trial spent `1.1614667139947414` seconds
in the pre-open helper and rejected two sequential whole-home label scans
before any actual `O_EXCL`. Its fixed result and timing attempt are consumed;
neither is a source-stage attempt, and neither may be rerun.

The inherited r4 remote source and local final receipt validator remain
unchanged. One whole-home label scan
remains immediately before `O_EXCL`, after the full write-phase `_probe`
scan, and the existing whole-home post-open scan remains mandatory **before
payload**. This r5 changes only the local operator STOP boundary: after each
bounded `_ssh_identity()`, it rechecks local active Desktop STOP/breakers
**before** the first possible remote timeout-probe SSH. The phase wrapper
constructs its command before identity so a STOP arriving during that work
is also seen at the new gate. Existing post-probe STOP checks remain.
All full activity/Neuron, STOP, root byte/inode, source/Git,
checkpoint/cache, protected-stage-file, mount/lock/ACL/group,
distinct-session, one-shot and local active Desktop
controller checks remain. The stage label, target, bundle, science bytes and
one-attempt marker are unchanged. The previous r4 static PASS did **not**
review this changed r5 operator; the independent r5 static PASS above does.
It does not review this new enabled request or establish live timing.
The frozen operator comments describing an absent enabled request predate
this final-pending request; its executable gate still requires exact final
review and assignment.

## Frozen local subject

| Item | Exact value |
| --- | --- |
| Accepted private source | `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`, commit `e65a3097a5b00f6178188a935bb2acb9a5042c6c` |
| New target from r10 profile | `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10`, commit `22bfbab5a77d6feb670d2d737504f8b70b08ddc1` |
| Separate stage label | `fs-20260925-trn21-c5c-a0-r10-incremental-stage-r2` (not the native science label) |
| Remote bundle path | `/home/ubuntu/trainium-c5c-e65-to-r10-20260925.trn21-stage-r2.bundle` (new, exclusive) |
| Local one-shot marker | `/home/poojith-devan/Desktop/Trainium-source-artifacts/trn21-c5c-a0-r10-incremental-stage-r2.one-attempt.jsonl` (absent in this package) |
| Git bundle | `e65-to-r10.bundle`, 109,800 B, SHA-256 `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e` |
| Remote source | `remote_incremental.py`, SHA-256 `6f0da61a76595a6fea18662d07d7311c504ab3a4b641ac972c440438bda08bf6` |
| Local entry | `operator_once.py`, SHA-256 `f48a35c460498a81a56cc73f68f5130a4d91f319a88be7c0ad6a1668f80de70b` |
| Disabled request | `REQUEST.HOLD.json`, SHA-256 `2f2451cdcc03828fea1f9d7ee85726604ef7cc99daba1b7ecb9dc1e85f8686be` |
| Exact enabled r5 stage-only request | `REQUEST.ENABLED_PENDING_FINAL_REVIEW.json`, SHA-256 `ad7a4dd6d4b93b705e936d781c0817bfdec7cb12560567aa3128e3f2b879c36a` |
| Held r4 request (not present here) | `49f2941f` request SHA-256 `cc2bddbdda631c79f6fc8b6a7555504a4d994c3c3a5665d203c74e1fdfd72a76`; independent final HOLD `06451f16`, receipt SHA-256 `0b9200787ed6a1d79b9b056d7f3068329fd89cf15fbbc0bd4df96aa1f654a515` |
| Independent r4 static receipt | `research/v2/verification/20260925-trn21-r10-stage-r4-single-final-scan-independent-static-PASS.md`, SHA-256 `dd251ab06b59fe01136773d1c4f178869bf9f98771ffb58ee2cd81a9fe648d32` at `b8c9c36b` |
| Independent r5 static receipt | `research/v2/verification/20260925-trn21-r10-stage-r5-late-stop-independent-static-PASS.md`, SHA-256 `99e1fd9770171467c3679a396dff63d648f858513c0dcf1176eb0ad9257b7b2a` at `739c5dac` |
| Offline tests | `test_incremental_stage.py`, SHA-256 `d993e860f0d9f22a8512986491fc44081b38a4eafba922ff5e512e012ac49083`; `test_final_boundary_r2.py`, SHA-256 `a3df70766254e7deb73ff9712763ce2f7ac8e75b1920cca735408a824ddf3204`; `test_r5_stop_boundary.py`, SHA-256 `e914317840ff77e218285fca5c3014ed356a7bbbc86fb6328d465ec53506d506`; `test_r5_final_pending_request.py`, SHA-256 `7ce34930b555355c82d9858ed2a860b9ff893668c468d53b9cbf305d43ddeb86` |

The bundle verifies with e65 as its only prerequisite and r10 as its only
HEAD. The `e65..r10` inventory is 141 Git objects: 15 commits, 61 trees,
65 blobs, and 126 named paths. SHA-256 of sorted path names joined by newline
is `f185620b6af55681c778bff3a2c4807facae15f7de523ba25f64396b5bef1c0f`.
A bounded offline path scan found zero `.env`/key/credential-like filenames;
the 65 new blobs had zero PEM-private-key, AWS access-key-ID, GitHub,
Hugging Face or OpenAI token-signature hits. The scan does not prove that
arbitrary natural-language content has no sensitive value; independent
review must check the exact bundle too. No credential bytes were printed.

This exact enabled request machine-binds the new r5 operator, unchanged r4
remote source/bundle, e65/r10 commits, r10 source-review receipt, trn21
paths, label and marker. The frozen authority schema does not accept an
extra static-review field. The independent final reviewer must explicitly
cite the `739c5dac` r5 static receipt and the inherited
empty-provisional/noncooperating-actor risk; adding an unrecognized request
key would fail the exact authority gate. The held r4 request hash cannot
serve as a request hash for this changed operator.

`train.py` is still SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
`prepare.py` is still
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The new checkout checker pins all eleven production source fingerprints
used by the r10 launcher, including its new launcher hash
`1be19bc31d2bdcde8abe9fe90cc1ac884d6efd5f7be197d7bdbfe64c257dcff7`,
requires exact r10 HEAD, 15-commit e65 ancestry, clean status, no remotes or
alternates, and strict full Git fsck.

## Evidence is point-in-time, not authority

- The independently source-PASSed **disabled** trn21 r10 subject is commit
  `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`; its independent receipt
  `research/v2/launcher-safety/INDEPENDENT_SOURCE_VERIFIER_PASS_R10_TRN21.md`
  is at commit `ee9a4b14cc30ab372162fd434692e59b13ac44ab`, SHA-256
  `5e6fbfe6fd46608d0b220b99402185c78c809523026df7153882d3637de20c96`.
  This review confirms a trn21-only A0 launcher profile and the exact new
  r10 target, not a source-stage or native run. The future enabled stage
  request must bind this receipt and r10 commit.
- The independently reviewed dual-host read-only Operator snapshot is
  `research/v2/observations/20260925-dual-host-readonly-operator/OPERATOR_READONLY_DUAL_HOST_SNAPSHOT_20260925T0733Z.md`
  at `47e64a32b37ec72f10bf387601bf3a69fb63e9ef`, SHA-256
  `3665847467803f05f1a61a040f0b38e7a88e486bc7af52b2fbfcf24472343777`;
  trn21 raw SHA-256
  `e573fa14da5743285b7d349b65660d89afde84e0dc9584dd1d8cd6dca8e192b5`.
  Its independent bounded-facts PASS is
  `research/v2/verification/20260925-dual-host-readonly-final-pair-independent-PASS.md`
  at `9868d9a6`, SHA-256
  `915724781f316ee82866605cf0b8290d2ce40a5f10b0524b477fbc1928dfd652`.
  The trn21 e65 checkout was clean/private `0700`, device `66305`, inode
  `3168319`, link count 11; its `.git` is a `0700` directory, inode `3168320`.
  It is a shallow one-commit source, **not** the trn22 linked worktree.
- The trn21 existing-lock read-only receipt
  `research/v2/authorizations/trn21-c5c-a0-r4-source-feasibility/OPERATOR_LOCK_READONLY_FEASIBILITY_20260925T053029Z.md`
  pins `.cache` `0700` inode `295736`, inner lock dir `0775` inode `596822`,
  existing zero-byte lock `0664` inode `596823`; raw SHA-256
  `687b4d7b57c4de041adad37352a385e15ead7cfba0b83c4d3a3df5bd12058d14`.
  The separate ACL/group follow-up raw SHA-256 is
  `484ea4e76b6a59d5e5ec6757db6936b0bc3e7fc3ad89f69aa304a84da483443c`.
  It observed no POSIX ACL in the private lock ancestry and no other local
  GID-1000 account. This is bounded ordinary-DAC evidence, not proof against
  external NSS, same-UID, privileged access or other namespaces. Source
  ancestry ACLs were **not** established by this snapshot; the runtime must
  see absent ACLs and exact private source/Git identity or HOLD.
- The accepted e65 trn21 four-rank G2 result is
  `research/v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-final/INDEPENDENT_CPU_G2_RESULT_REVIEW.PASS.json`,
  SHA-256 `57c5bf42682feac334a2c75d6f5ce003190764b8260914f2a6e8d4df8a4e757d`.
  It is **CPU numeric only**; r10 source-identity carry-forward and native
  authority are separate later decisions.

## Future guarded stage (not authorized now)

1. A separate Verifier has reviewed the exact disabled r5 source **statically**,
   not this enabled request or live timing. Only an actual independent final
   **stage** PASS bound to this request SHA-256
   `ad7a4dd6d4b93b705e936d781c0817bfdec7cb12560567aa3128e3f2b879c36a`
   and a Supervisor one-shot assignment can complete `stage_once` authority.
   Both fixed future files are absent here; do not synthesize or infer them.
   The checked-in HOLD JSON cannot be edited into authority, and the r4
   final HOLD cannot be converted into PASS by assigning its old request.
2. After local STOP/breaker and pinned trn21 SSH identity checks, the future
   Operator uses bounded, read-only pinned SSH to check **remote**
   `/usr/bin/timeout --version` for GNU coreutils, but first rechecks local
   STOP and active Desktop breakers immediately **after** local identity.
   It rechecks again after the remote probe before exclusively creating a one-attempt
   marker. This SSH probe is reached only after the exact enabled request,
   independent final stage PASS and Supervisor assignment; the last two are
   absent here.
   Each subsequent phase rechecks STOP/breakers after local identity,
   repeats the remote timeout probe, then rechecks again before its pinned
   SSH command. Local
   `/usr/bin/timeout` is uutils on this controller and is not the binary
   wrapping the remote phase; the r3 local-GNU check would have falsely
   blocked the marker. A remote non-GNU version, failed pinned transport,
   stderr or oversized probe response remains HOLD. Distinct SSH processes
   perform full fresh preflight, exclusive bundle/new-checkout creation,
   cross-session Git/hash readback, and an **unconditional attempt** at full
   read-only postflight. A local/remote STOP or transport failure can prevent
   that last SSH/probe from completing; this records `POSTFLIGHT_HOLD`, never
   a successful stage. Only an actual completed full postflight receipt can
   support success.
   Neither the accepted source nor protected checkpoints/caches or prior
   stage artifacts are patched, deleted, cleaned or re-used as a target.
3. Every remote phase takes the existing launcher-compatible no-follow,
   nonblocking `flock`; checks host/boot/ext4 mount/UID, exact private
   source/Git HEAD/config/shallow bytes and clean e65 commit, source and
   lock ancestry/ACL/NSS, two checkpoint hashes, six cache identities,
   three preserved e65-stage files, STOP, process/Neuron holder, and broad
   stage-label absence. Both before and after space snapshots must meet
   13,177,973,911 B for preflight/write or 10,737,418,240 B for
   readback/postflight, and **always** the 6.5-GiB and 10,000-inode floors.
   The remote phase uses absolute GNU timeout and both entrypoints refuse
   optimized Python before any host write.
4. Immediately before the first bundle `O_EXCL`, under that lock, the writer
   runs full activity, **one** broad `/home/ubuntu` label scan, full
   post-label activity, the root byte/inode floor, then direct STOP and
   target-absence checks. Its scan duration and all activity/label/space ages
   must be finite and at most one second at the last no-write sample **and at
   the return of the actual exclusive open**; the open-call duration itself
   must also be at most one second. This does not relax the original limit.
   After open but before payload, another whole-home label scan, full
   process/Neuron/STOP check, root byte/inode sample, finite ≤1-second ages,
   and same-empty-inode check remain mandatory. A persistent label injected
   during the final pre-open scan after its cursor passed is caught by that
   post-open scan: at most an **empty provisional bundle** exists; no payload
   or checkout is created. The bundle is retained, and the one-shot marker
   remains consumed. There is no unlink, cleanup or automatic retry.
5. **Risk/semantic limit:** r4 is **not equivalent** to r3's two pre-open
   scans for the first `O_EXCL`. The late-label case above may create an
   empty provisional inode where r3's second pre-open scan might have
   refused before creation. The guaranteed property in the tested bounded
   cooperative-flock/ordinary-DAC model is no *successful* stage or payload
   when a persistent duplicate or STOP arrives during the final scan or at
   open and remains through the post-open check. A noncooperating same-UID or
   privileged actor can still create a label **after the last post-open
   scan**, between userspace checks and payload; r3 already had this
   documented residual, which an adversarial test reproduces here. No finite
   `find` sequence makes global uniqueness atomic. The independent final
   reviewer must explicitly decide whether the changed empty-provisional
   exposure is acceptable; this disabled source makes no such authority
   decision. The r4 full activity and single-scan live duration is
   unmeasured, so the unchanged ≤1-second actual-open gate may still refuse.
6. Only a new `0700` detached r10 checkout and `0600` bundle may result.
   Failure preserves the consumed marker and any partial target/bundle for
   inspection: no retry, overwrite, cleanup or native run. Even a successful
   stage is only evidence for independent stage-result and G2 carry-forward
   review, followed by distinct native A0 request, paired monitors/observer,
   fresh host checks and one-shot authority.

## Exact future entrypoint — DO NOT RUN in this final-pending state

Only after request-bound independent final stage PASS, Supervisor one-attempt
assignment, exact clean source hashes, fresh host checks and active Desktop
STOP/breaker gates all hold may the designated Operator use this one absolute
command from this clean r5 worktree:

`/usr/bin/python3 -B /home/poojith-devan/Desktop/Trainium-trn21-r10-stage-r5-enabled-final-pending/research/v2/authorizations/trn21-c5c-a0-r10-incremental-stage-r2-hold/operator_once.py`

Do not run it now. Do not substitute the historical Documents checkout, the
active Desktop controller checkout, a copied package, `-O`, a different
endpoint or marker, or an ad-hoc SSH command. The active Desktop
`/home/poojith-devan/Desktop/Trainium/research/v2/state.json` and STOP files
in that checkout, the Documents checkout, and this package remain controlling.
A remote GNU-timeout probe is an SSH contact and is not allowed before final
authority. No r4/r5 live one-second timing has been established; any later
authorized stage may HOLD at the unchanged freshness gate, retaining its
one-attempt marker and any partial without cleanup or retry.

## Cross-host science comparability remains a separate gate

`research/experiment-protocol.md` §1 requires the same Git HEAD on local,
trn21 and trn22 before comparable scored runs. This package pins only a
**trn21 r10** target. The separate trn22 e65→r9 stage draft would leave
different HEADs; neither stage review may silently waive the protocol.
If paired scored work is intended, a separately reviewed common-r10 source
stage on trn22 (r10 descends from r9) or an explicit human protocol decision
is needed, then fresh exact source and launcher checks on both hosts.

Implementer-local checks:

`TMPDIR=/dev/shm python3 -B -m unittest discover -s research/v2/authorizations/trn21-c5c-a0-r10-incremental-stage-r2-hold -p 'test_*.py' -q`

`TMPDIR=/dev/shm python3 -O -B -m unittest discover -s research/v2/authorizations/trn21-c5c-a0-r10-incremental-stage-r2-hold -p 'test_*.py' -q`

Both returned 41/41 PASS offline; `git bundle verify` passed again and the
bundle bytes are unchanged. These are
Implementer regression checks, not independent verification. No host was
contacted.
