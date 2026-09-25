# trn21 c5c A0 persistent source stage r2 — HOLD

This is an **Implementer-prepared, non-dispatchable** correction to the failed
tmpfs transfer. A separate Verifier must review it, a Supervisor must assign a
new stage-only attempt, and an Operator must pass fresh full preflight before
any host write. This package does not authorize SSH, CPU G2, native training,
fresh evaluation, cleanup, AWS changes or submission.

## Why r2 exists

The first transfer process said it wrote and SHA-verified the pinned e65 Git
capsule under `/dev/shm`, but a distinct SSH process could not find that exact
file seconds later. Follow-up sessions agreed on hostname, boot ID and tmpfs
mount namespace, but the transfer session's mount identity and exact writer
source were not retained. The root cause is **unknown**. Read the immutable
[HOLD receipt](../fs-20260925-v2-c5c-a0-control-trn21-r1.source-stage-hold.md)
and [forensic addendum](../fs-20260925-v2-c5c-a0-control-trn21-r1.source-stage-forensic-addendum.md).
Do not retry its `/dev/shm` host target or the consumed assignment. It did
not create a checkout or start CPU G2/native work; the A0 science label itself
remains unused.

R2 uses three new **persistent ext4** file targets directly under
`/home/ubuntu`:

| Artifact | Local pinned source | New host target | Bytes | SHA-256 |
|---|---|---|---:|---|
| e65 one-commit capsule | `/dev/shm/trainium-c5c-e65-one-commit-20260925.tar.gz` | `/home/ubuntu/trainium-c5c-e65-one-commit-20260925.trn21-stage-r2.tar.gz` | 104,005,787 | `32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264` |
| CPU preflight | `workflow/tools/preflight_dryrun.py` | `/home/ubuntu/trainium-v2-g2-preflight-dryrun-20260925-trn21-stage-r2.py` | 23,550 | `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8` |
| CPU containment | `workflow/tools/contained.py` | `/home/ubuntu/trainium-v2-g2-contained-20260925-trn21-stage-r2.py` | 9,185 | `0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d` |

These filenames are not the old target. Every exact new target must be absent
on the live host before its first and only write. A failed write leaves a
partial file in place; **never** delete, overwrite, rename, or retry it under
the same path. An r3 plan would need new paths and fresh independent review.

## Exact reviewed transfer code

The static [remote writer](stage_r2_remote_writer.py) SHA and
[local command builder](stage_r2_transport.py) SHA are pinned in
[GATE_CONTRACT.json](GATE_CONTRACT.json). The builder checks the literal
writer-file hash, uses only enumerated artifact/operation tokens, and produces
one `ssh -T` argv with the already-pinned host key, strict host-key checking,
`BatchMode`, and a reversibly quoted `exec python3 -c <the reviewed file's
literal bytes> write|verify <artifact>` remote command. No code is generated
from host output, shell interpolation, or a chat response. The builder has no
dispatch entry point: running it directly exits HOLD. Its full argv, SHA-256,
writer hash and local source hash must be saved in the later immutable
Operator request **before** any execution. Do not print key contents.

For `write`, the remote code first binds hostname `ip-172-31-34-85`, UID 1000,
boot ID, mount namespace, `/home/ubuntu` parent device/inode and owner, the
covering ext4 mount ID/source, and root free bytes. It opens the fixed basename
relative to a no-follow `/home/ubuntu` directory FD with
`O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW`, writes at most the fixed byte count in
4-MiB chunks, checks streamed SHA, fsyncs file and parent, reads the entire
new file back from its open FD, rechecks SHA/size/fstat/lstat/mode `0600`, and
rebinds host/boot/mount/root-free identity. It emits one JSON success receipt
only after every in-process check passes. Any error exits nonzero and leaves
the target untouched thereafter; the Operator archives stdout/stderr/exit
and stops. This protects against the prior missing-writer-source forensic gap,
but **one process's success is not persistence proof**.

For each artifact, the Operator must open a **different, new pinned SSH
session** with the same literal writer bytes in read-only `verify` mode. That
session opens the exact file no-follow, hashes all bytes and records its own
host/boot/mount/parent/device/inode/mode/UID/free-space receipt. The local
`compare_cross_session` helper rejects a missing/replaced file, changed
boot/mount/parent/target identity, wrong digest or count, same process-start
identity, or a verifier timestamp not later than the write. After all three
transfers, use three additional new read-only sessions, one per artifact, to
verify all three again;
archive each independent transport exit and receipt. A write receipt alone,
even with the correct hash, is **HOLD**. The Verifier later checks the raw
receipts and full argv rather than accepting a helper's PASS uncritically.

## Fresh preflight, source scan and clean checkout

The historical trn21 audit showed about 58.7 GB free, intact protected
checkpoints/caches and an idle host, but is not a present-time preflight.
Before **each** write and after the stage, the Operator must freshly record:

1. Exact hostname/UID/SSH host key; boot ID and mount namespace; `/home/ubuntu`
   on ext4 with stable device/inode and non-writable-by-others ancestry.
   Root free space must stay at least 10,737,418,240 B for this stage. The
   native operational floor of 6,979,321,856 B remains unchanged.
2. Absent global/trn21 STOP and breaker; no train/eval/G2/guard process,
   shared lock or Neuron holder; A0 science label/output and all three r2
   targets/checkout paths absent. If a target becomes present unexpectedly,
   stop without writing. A narrow direct path check and a separately scoped
   label scan are required; an unrelated unreadable tmpfs directory does not
   become a false broad-scan PASS.
3. Both retained checkpoint size/SHA pairs and all six historical warm-cache
   directories' device/inode identities match the contract. Preserve these
   assets; never prune, copy, relocate, reset or clean them to make staging
   pass. The local capsule must still be regular mode `0600`, exact bytes and
   SHA; both local CPU tools must still match their pinned bytes/SHA.

The local synthetic tests reread the 6,338-member pinned capsule and reject
link/special entries, duplicate or escaping names, PAX path overrides,
credential-like filenames and unsafe `.git/config` remote/filter/include/
credential/hook settings. A separate Verifier must inspect that scan and the
archive provenance. On the host, the only archive eligible for extraction is
the persistently cross-session-verified SHA-identical r2 ext4 copy. The
Operator must repeat the member/config safety scan on that exact file or
provide a separately reviewed SHA-equivalence proof to the scanned pinned
bytes **before** extraction. A missing or suspicious member is HOLD. No
unreviewed dynamic scanning snippet may be substituted.

Only after this complete receipt, create the **new and previously absent**
checkout `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`
mode `0700`. Extract with no overwrite, no same-owner and no same-permissions
semantics. Do not touch the old planned checkout path. If checkout creation
or extraction partially fails, preserve it for diagnosis; no cleanup/retry.
The host stage receipt must prove exact Git HEAD
`e65a3097a5b00f6178188a935bb2acb9a5042c6c`, one reachable shallow
commit, no remotes/alternates, safe config, `git fsck` connectivity, clean
status including untracked files, root `train.py` SHA
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
and `prepare.py` SHA
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
Science output under that checkout must remain absent. Source/stage facts,
artifact receipts, root disk and protected assets must be retrieved into the
repo and reviewed by an **independent Verifier**.

Only **after** that review may a separate Supervisor contemplate CPU G2 with
new label `fs-20260925-g2-a0-c5c-exactenv-trn21-r2` and new root
`/home/ubuntu/trainium-v2-g2-20260925-trn21-r2`, both absent. CPU G2 requires
its own exact environment, fresh preflight, request and one-shot authority;
the old r1 G2 label/root are not reused. Native A0 remains HOLD after any
stage/G2 PASS until the separate root-free monitor/owned-guard interlock,
boundary observer and native request pass independent review and one-attempt
authorization. No BPB or submission claim follows from source staging.

## Local checks performed by Implementer only

`python3 -B research/v2/authorizations/trn21-c5c-a0-stage-r2/test_stage_r2.py`
checks pinned local capsule/tool and e65 source hashes, literal remote command
quoting, new-path local collision absence, 6,338 tar members, synthetically
exercises exclusive no-follow write/readback/no-overwrite/failure-no-cleanup,
and tests cross-session receipt rejection/acceptance logic. It does **not**
execute SSH, prove host ext4 persistence, inspect live trn21 or authorize CPU
G2/native work. An independent Verifier must repeat and adversarially review
it before any Operator stage assignment.
