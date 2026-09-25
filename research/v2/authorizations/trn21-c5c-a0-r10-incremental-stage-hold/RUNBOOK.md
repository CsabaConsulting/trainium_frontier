# trn21 e65 → r10 small-bundle source stage — DISABLED / HOLD

This is an Implementer-only, stage-only candidate. No enabled request,
independent final **stage** PASS, one-attempt Supervisor assignment, host
contact, transfer, new checkout, native run, AWS action, cleanup, breaker
change, evaluation, or submission occurred. The exact entrypoint
`operator_once.py` exits HOLD before marker, journal, or SSH because its
enabled request is absent. Historical r1–r9 transfer packages and the
accepted e65 checkout remain untouched. Do not substitute held full-capsule
writers or treat the r10 source PASS as native authority.

## Frozen local subject

| Item | Exact value |
| --- | --- |
| Accepted private source | `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`, commit `e65a3097a5b00f6178188a935bb2acb9a5042c6c` |
| New target from r10 profile | `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10`, commit `22bfbab5a77d6feb670d2d737504f8b70b08ddc1` |
| Separate stage label | `fs-20260925-trn21-c5c-a0-r10-incremental-stage-r1` (not the native science label) |
| Git bundle | `e65-to-r10.bundle`, 109,800 B, SHA-256 `b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e` |
| Remote source | `remote_incremental.py`, SHA-256 `eb1b1d9150dc35d6c96c00ad49c321bc9944c0cbaeef0b1494272346eb78c9f0` |
| Local entry | `operator_once.py`, SHA-256 `8ed2266c20774c5f4f1f750bbe9f94c8d68a8e8459670308ef689c8e714901c1` |
| Disabled request | `REQUEST.HOLD.json`, SHA-256 `9012dfdab9887e0ea9c030e3d508b42b95c97c1367dba4697f0392a6d1473716` |
| Offline tests | `test_incremental_stage.py`, SHA-256 `cedf30eb20d80cfa64a8f1b103cccedb23d839e988efeba3b4547ebadbdb7bc0` |

The bundle verifies with e65 as its only prerequisite and r10 as its only
HEAD. The `e65..r10` inventory is 141 Git objects: 15 commits, 61 trees,
65 blobs, and 126 named paths. SHA-256 of sorted path names joined by newline
is `f185620b6af55681c778bff3a2c4807facae15f7de523ba25f64396b5bef1c0f`.
A bounded offline path scan found zero `.env`/key/credential-like filenames;
the 65 new blobs had zero PEM-private-key, AWS access-key-ID, GitHub,
Hugging Face or OpenAI token-signature hits. The scan does not prove that
arbitrary natural-language content has no sensitive value; independent
review must check the exact bundle too. No credential bytes were printed.

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

1. A separate Verifier reviews these exact bytes. Only a later exact enabled
   request, actual independent final **stage** PASS, and Supervisor one-shot
   assignment can pass `stage_once`. The three fixed future files are absent
   here. The checked-in HOLD JSON cannot be edited into authority.
2. After local STOP/breaker and pinned trn21 SSH identity checks, the future
   Operator exclusively creates a one-attempt marker. Distinct SSH processes
   perform full fresh preflight, exclusive bundle/new-checkout creation,
   cross-session Git/hash readback, and **unconditional full postflight**.
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
   The remote and local commands use absolute GNU timeout and refuse
   optimized Python before any host write.
4. Immediately before the first bundle O_EXCL, under that lock, the writer
   repeats the slow activity check, performs **two** broad label scans,
   directly rechecks STOP and target absence, and requires both activity and
   final label proofs to be at most one second old. Synthetic STOP or label
   insertion during the slow checks and a 1.1-second stale proof all prevent
   bundle and target creation. Actual trn21 timing is **unmeasured**; this
   strict gate may fail live. It must not be relaxed or described as
   dispatch-ready without separate feasibility and review. Noncooperating
   same-UID/privileged races after the final check cannot be absolutely
   excluded; the design's claim is bounded freshness and ordinary DAC plus
   cooperative lock, not absolute non-malleability.
5. Only a new `0700` detached r10 checkout and `0600` bundle may result.
   Failure preserves the consumed marker and any partial target/bundle for
   inspection: no retry, overwrite, cleanup or native run. Even a successful
   stage is only evidence for independent stage-result and G2 carry-forward
   review, followed by distinct native A0 request, paired monitors/observer,
   fresh host checks and one-shot authority.

Implementer-local checks:

`python3 -B -m unittest discover -s research/v2/authorizations/trn21-c5c-a0-r10-incremental-stage-hold -p 'test_*.py' -q`

`python3 -O -B -m unittest discover -s research/v2/authorizations/trn21-c5c-a0-r10-incremental-stage-hold -p 'test_*.py' -q`

Both returned 17/17 PASS offline; `git bundle verify` passed. These are
Implementer regression checks, not independent verification. No host was
contacted.
