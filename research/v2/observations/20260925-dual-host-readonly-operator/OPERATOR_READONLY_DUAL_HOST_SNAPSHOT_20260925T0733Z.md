# Operator dual-host read-only status snapshot — 2026-09-25 07:33 UTC

**Disposition: point-in-time observation only.** Both SSH probes returned 0 with
empty stderr and parsed valid bounded receipts. The observed hosts were idle by
the process, `neuron-ls`, and `fuser` checks below. This is not a launch
preflight, source-transfer review, one-shot assignment, breaker reset, or
training/evaluation authority. The canonical controller state was unchanged
around each probe (SHA-256
`9e859fdf29b684dff4040fe27f25cef28a40203bdc321e65475ab6b69a38716b`):
`launch_authorized=false`, `hosts.trn21.launch_allowed=false`,
`hosts.trn22.launch_allowed=false`; global, trn21 and trn22 breaker `tripped`
fields were all `false`. The four checked local STOP paths were absent.

| Host | Root-space observation UTC | Free bytes | Free inodes | Mount | Matched processes | Neuron/device/lock holders | Host STOPs |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| trn21 | 07:33:52.090952 | 57,897,381,888 | 31,563,301 | ext4 `/dev/root`, `259:1`, mount ID 27 | none; 0 unreadable cmdlines | none; `neuron-ls` valid, 1 device | none among 290 paths |
| trn22 | 07:33:28.604264 | 58,455,748,608 | 31,660,021 | ext4 `/dev/root`, `259:1`, mount ID 27 | none; 0 unreadable cmdlines | none; `neuron-ls` valid, 1 device | none among 262 paths |

Both `/dev/root` and `/dev/nvme0n1p1` resolved to block major:minor `259:1`
on both hosts. The trn21 boot ID was
`1903f7f0-8e0d-4011-9566-346e4ba01a0f`; trn22 was
`bf52dfdd-1dba-485a-9ee6-e5a27fb41ba6`. Each mount namespace was
`mnt:[4026531832]`. The existing zero-byte host lock was ext4 device
`66305`, owner `1000:1000`, mode `0664`, inode `596823` on trn21 and
`832589` on trn22. Its `fuser` returned rc 1/no PID; no lock was acquired.
The separate prior lock-mode receipts still govern any future transfer policy.

## Checkout identity

| Host and checkout | Directory identity | Git HEAD and cleanliness | Source SHA-256 |
| --- | --- | --- | --- |
| trn21 exact stage-r2 e65 | device `66305`, inode `3168319`, mode `0700`, UID:GID `1000:1000` | `e65a3097a5b00f6178188a935bb2acb9a5042c6c`; clean, status rc 0/empty stdout and stderr | `train.py` `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`; `prepare.py` `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` |
| trn22 exact A0 r2 e65 | device `66305`, inode `2432738`, mode `0775`, UID:GID `1000:1000`; `.git` is a 148-byte regular file, inode `2432740`, mode `0664` | same e65 commit; clean, status rc 0/empty stdout and stderr | same pinned c5c/prepare hashes |
| trn22 existing A0 r1 | device `66305`, inode `2432499`, mode `0775`, UID:GID `1000:1000` | same e65 commit; cleanliness not checked | same pinned c5c/prepare hashes |

The exact planned 446b/r4 checkout paths
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r4`
and
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r4`
were absent. No top-level non-symlink directory named
`trainium_freshstart_*` with `_r9` or `446b` was discovered on either host.
This is a bounded `/home/ubuntu` name scan, not proof of host-wide absence.
The expanded trn22 scan also found the existing A0 r1/r2 checkouts above.
One unrelated warmdown060 `_r4` checkout was present at
`8dead6599d6004fd4a8ac9749202124d1bdb8fd3`; its `train.py` hash is
`bc1d0b75e9964c2cccf27865d22ef287f7032984be4b10a8f8490b95e2ed1e7c`.
The e65 launcher hash on the checked A0 checkouts was
`822d4317408a09ee0dd6ab3eaaa4f706f0a173741659ea5e47cc12e45e4f1bef`;
the current handoff already excludes that older launcher from native A0.
No checkout was created, modified, or used to train.

## trn22 linked-worktree permission follow-up

A separate read-only diagnostic at 07:36:43 UTC resolved the trn22 e65 r2
checkout's 148-byte `.git` pointer to
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r1/.git/worktrees/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2`.
The pointer file SHA-256 was
`dc53f6887bc5f9c2c1bbdbbd60911cdbeb076f7c57bafd796ee5b96f69ecbf1b`;
the resolved gitdir was an existing directory on device `66305`, inode
`2432736`, UID:GID `1000:1000`, mode `0775`. The r1 source directory was
device `66305`, inode `2432499`, mode `0775`; its `.git` directory was inode
`2432500`, mode `0775`, and its `.git/worktrees` directory was inode `2432735`,
mode `0775`. Every checked access ACL xattr on `/`, `/home`, `/home/ubuntu`,
the r1 and r2 directories, pointer file and resolved gitdir ancestry was
absent (`ENODATA`); default ACL xattrs on those directories were also absent.
The local `/etc/passwd` and `/etc/group` files showed only `ubuntu` (UID 1000)
with primary GID 1000 and no explicit GID-1000 group members. Thus no other
**local-file** account sharing GID 1000 was found; external NSS accounts,
same-UID processes and privileged access were not excluded. The probe did
not open or lock Git's index or modify metadata. Its
[raw receipt](OPERATOR_RAW_TRN22_READONLY_GITDIR_ACL_20260925T073638Z.json)
has SHA-256
`b878de92aa1be748599079ec1a1295a70dff3a4ee8856fc1d17208298801f3bc`;
the streamed [diagnostic source](remote_gitdir_acl.py) SHA-256 was
`676c1ed5c80b37a56df7f90a80150fe29ebc0d7c6e65813513af8b7f4f4b8499`.

## Transport and evidence custody

The probe used the existing pinned strict-ED25519 SSH identity checks:
known-hosts SHA-256
`e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3`,
trn21 fingerprint `SHA256:G4eS9Bib/yRt+bpWWOUV0SgF/pqTfhLITafrA2ToCfc`,
and trn22 fingerprint `SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U`.
Only `exec python3 -B - <host>` ran remotely, with the committed
[`remote_snapshot.py`](remote_snapshot.py) source streamed on stdin. The
local [`collect_snapshot.py`](collect_snapshot.py) archived redacted argv,
source/transport hashes, raw stdout/stderr byte counts, SHA-256 and base64,
parsed receipts, and before/after controller flags. The final paired remote
source SHA-256 was
`4a8b8ed413240c229ce943dd9bd8d6781bbddf7c348aa76b5b496f5c5d14f70a`.
Neither raw stream was suppressed by the credential-pattern filter; neither
contained SSH stderr. No PEM bytes were read, and no credentials, tokens or
environment values were printed or archived. No remote write, process signal,
AWS API, marker use, training, evaluation, or submission occurred.

Final expanded raw receipts:

- [trn21](OPERATOR_RAW_TRN21_READONLY_20260925T073347Z.json), SHA-256 `e573fa14da5743285b7d349b65660d89afde84e0dc9584dd1d8cd6dca8e192b5`.
- [trn22](OPERATOR_RAW_TRN22_READONLY_20260925T073323Z.json), SHA-256 `c8a44adfea5b7be7f12603dc3f2ef97c32b1f1495d199e8ac42cc281591f94dd`.

Two earlier probe-only contacts preceded the requested trn22 e65 expansion;
they are preserved for complete contact accounting: [trn21 at 07:32](OPERATOR_RAW_TRN21_READONLY_20260925T073218Z.json),
SHA-256 `e42d3f5e7e5074a9c8f5900f029124a8ad951a7f098befdeac8bd185afe8dc06`,
and [trn22 at 07:32](OPERATOR_RAW_TRN22_READONLY_20260925T073201Z.json),
SHA-256 `a5d21dda7be15f941d7fb7ea8329724b3cf46cd82c5dc984c3fb2194eb28addd`.
Their streamed source digest was
`b3495be1b563df54a48c7ed6a352b1b14d4ac1b2459f0354e723f04b200b1421`;
the later source added the exact trn22 e65 path, the broader c5c/A0 name
inventory, and a read-only Git cleanliness check. The final paired receipts
above supply the expanded finding. Any future transfer or launch requires
its own fresh exact preflight, independent review, and applicable assignment.
