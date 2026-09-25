# trn21 c5c A0 r2 source checkout — review-only HOLD

The currently assigned Operator may transfer only the three exact new ext4
files in the [r2 transfer-only assignment](../fs-20260925-v2-c5c-a0-control-trn21-r1.source-stage-r2-transfer-assignment.json).
That assignment explicitly excludes checkout creation and extraction. This
Implementer package supplies the later extraction and verification logic for
independent review. It provides **no SSH command or dispatch entry point**.
Running `stage_r2_checkout.py` directly exits `DISPATCH HOLD`. No one should
run its `stage_fixed_checkout` function on trn21 under the transfer assignment.

## Frozen inputs and target

The only eligible capsule is the cross-session-verified regular `0600` ext4
file `/home/ubuntu/trainium-c5c-e65-one-commit-20260925.trn21-stage-r2.tar.gz`:
104,005,787 bytes, SHA-256
`32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264`.
Its source commit is `e65a3097a5b00f6178188a935bb2acb9a5042c6c`;
the root `train.py` and `prepare.py` hashes must be
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The one new checkout path is
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`.
It must be absent before its sole creation. The old checkout path, old
`/dev/shm` transfer target, two retained checkpoints and six historical warm
caches must not be altered.

The exact predecessor transfer contract SHA-256 is
`631f3e7c1a2c4ec962b58be3b3a5399d323bae22d41add1aaca7884cd4efe069`;
the transfer-only Supervisor assignment SHA-256 is
`0127358fed6e97787b8a7d60208ede3cadb0e870791baece2f12830300c09fd5`.
The extraction code includes both pins. `GATE_CONTRACT.json` pins the code
and synthetic-test file hashes. A future exact invocation request must pin
the same bytes and its full argv hash before any host execution.

## Gates before any future checkout attempt

1. The Operator must archive **all nine raw writer receipts**: a write, a
   distinct-session first verification and a distinct-session final readback
   for each of capsule, preflight and containment tool. The final readback
   pass must occur after all three first verifications. Preserve each SSH
   argv/hash, exit, stdout and stderr. The in-process write digest by itself
   is insufficient. A separate Verifier must review the complete transfer
   result and confirm unchanged host, boot, mount, parent, target inode,
   mode, byte count and SHA. Any absent/replaced file is HOLD.
2. A separate Verifier must review this exact code, tests, archive scan and
   contract. Then a Supervisor must issue a **new, checkout-only** assignment
   and a separately reviewed exact invocation request. The request must bind
   the complete transfer receipt hash, exact code hash and full argv. This
   package deliberately contains no such executable request.
3. Immediately before that one attempt, the Operator must freshly perform
   the complete trn21 read-only preflight: pinned SSH identity/host key,
   hostname `ip-172-31-34-85`, UID 1000, boot/mount namespace, root ext4 and
   safe `/`, `/home`, `/home/ubuntu` ancestry; root free at least
   10,737,418,240 bytes; no global/trn21 STOP or breaker, matched
   train/eval/G2/guard process, Neuron/device holder or shared lock; science
   label/output and exact new checkout absent; both checkpoint size/SHA and
   all six cache device/inode identities unchanged. Do not clean or replace
   any asset to make a gate pass. The code repeats the checks that can be
   verified locally during extraction, but it is not the complete machine
   preflight.

## Exact behavior of the proposed stage logic

`stage_fixed_checkout` first validates the three-artifact receipt triplets,
including distinct process identities and a final read-only pass after all
first verifications. It binds the live host, boot ID, mount namespace, root
ext4 mount and parent inode to those receipts. It snapshots the two pinned
checkpoint hashes and seven warm-cache directory identities before any new
checkout is created. It reads the ext4 capsule by a no-follow file descriptor,
requires the final transfer receipt's device/inode/mode/size, and checks the
entire compressed SHA-256. It also rehashes the two persistently transferred
CPU tools against their final-session inodes before and after checkout work.

Before creation, a raw tar-header scan and a second `tarfile` pass agree on
**6,338 actual entries**: 6,283 regular files, 55 directories, 188,636,254
unpacked file bytes and four GNU long-name control headers. These four are
part of the pinned archive and are matched to safe effective names. PAX
headers/overrides, links, special/unknown entries, setuid/setgid/sticky
bits, absolute or traversal paths, duplicates, ambiguous names, missing
parent directories, active hooks, unsafe Git config and credential/data or
checkpoint payload paths fail before checkout creation. The member manifest
SHA-256 must be
`a9eff72e760f314717e712a5e3d23d300232db4feeb361896c57b16f7bdd1bf4`.
The bounded marker scan excludes opaque Git pack payloads; exact capsule SHA,
Git fsck and clean status remain separate requirements.

After all precreation checks, the code makes the absent checkout with
`mkdir` and mode `0700`. It walks only no-follow directory descriptors,
creates every member with `O_EXCL|O_NOFOLLOW`, writes it in bounded chunks,
fsyncs and reads back each file, and fsyncs directories. It never applies
archive ownership or group/other permissions; regular files become `0600`,
or `0700` when Git expects the executable bit. It then rehashes the same
capsule descriptor. A failure at any point leaves the partial checkout in
place and emits no success receipt. There is no deletion or same-path retry.

The read-only checkout verifier requires detached HEAD exactly `e65a3097…`,
the exact `.git/shallow` and minimal `.git/config` hashes, a shallow repository
with one reachable commit across HEAD and all refs, no remotes or alternates,
`git fsck --full --strict --no-reflogs --connectivity-only`, clean porcelain
status plus an independent all-untracked check, and the pinned root
`train.py`/`prepare.py` hashes. Git runs with a minimal environment that
disables inherited Git configuration and optional locks. Science output must
be absent. A final host identity, capsule hash and protected-asset snapshot
must still match. The returned JSON-compatible receipt records the source,
scan, extraction, Git checks, host state and protected assets for retrieval.

A distinct Verifier must review the complete host stage receipt before a
separate CPU G2 request is even considered. CPU G2, native A0, fresh
evaluation, adoption and submission remain HOLD under this package.

## Local Implementer checks

`python3 -B research/v2/authorizations/trn21-c5c-a0-stage-r2-extraction/test_stage_r2_checkout.py -v`
passes nine tests: contract and predecessor/hash binding, the pinned 6,338-member
scan; malicious synthetic tar paths/types/PAX/config; owner-private exclusive
extraction, no overwrite and preserved partial output; symlink collision;
three-artifact persistence receipt rejection; transferred-tool drift;
protected-asset drift; and a direct-execution HOLD. A separate local
temporary-directory extraction of
the exact capsule also produced the e65 HEAD, shallow `true`, empty status
and untracked output, and `git fsck` rc 0. These checks are local only; they
do not establish trn21 persistence or authorize a host command.
