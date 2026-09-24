# Disk decision packet — 25 September 2026, 02:06 IST

Decision needed from Poojith: increase **both existing root EBS volumes from 200 to 250 GiB**.
No AWS operation, filesystem resize, deletion, or breaker reset has been performed.

## Verified inventory

Read-only checks at 20:33 UTC on 24 September:

| Host | Root EBS volume ID (NVMe serial normalized) | Available root bytes | Available GiB |
| --- | --- | ---: | ---: |
| trn21 | `vol-09b9c7946aeaa8c57` | 6,901,858,304 | 6.43 |
| trn22 | `vol-0d64a9cafbb4912e8` | 8,010,612,736 | 7.46 |

Both have a 214,748,364,800-byte (200 GiB) `/dev/nvme0n1` root disk,
partition `/dev/nvme0n1p1`, and an ext4 filesystem mounted at `/`.
The volume IDs are derived from the device serials, not an AWS API lookup; confirm the
instance attachment in the Melbourne (`ap-southeast-4`) console before changing either volume.

Evidence: [read-only inventory and checkpoint identities](../agentic/operations/TRN2_DISK_DECISION_PACKET_2026-09-25.json),
SHA-256 `1028e54c974e7605395fe11bf78be2a300ce8a6c479328a33c6aad53fa0d2f2d`.

## Requested cleanup: no current eligible targets

The quoted 3,893,264,384-byte opportunity came from a **historical** inventory.
`/home/ubuntu/workspace/autoloop_runs` is now empty. Consequently the current cleanup
list is empty, recoverable space from that list is zero, and there are no current
directory/archive hash pairs to approve. Recommending deletion of those old paths
would be misleading.

The historical removals and matching remote/local log, launch, ledger and verdict hashes
are already documented in [the completed R8 cleanup receipt](../agentic/operations/TRN21_DETERMINISTIC_CLEANUP_2026-09-23_R8.md).
No replacement deletion targets are inferred. A future cleanup list requires new exact
targets, archive proof, and your fresh approval.

Both extant LR0.015 checkpoints on trn21 are retained:

- Clean confirmation: `.../trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/train/out/final.pt`,
  469,831,009 bytes, SHA-256 `5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9`.
- Research control: `.../trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/sandbox/out/final.pt`,
  469,834,849 bytes, SHA-256 `aaa4b7c0ff58c56df564954c21f8abdd786930bfaa1cc447f6fb8e5b0da83535`.

The JSON receipt contains the unabbreviated absolute checkpoint paths.

## Human action: +50 GiB per host

Recommended: snapshot the existing volumes, then modify each exact volume from 200 to
250 GiB. Keep its existing type/IOPS/throughput; do not stop or replace either instance.
Wait until the modification is `optimizing` or `completed`, and verify the guest sees
the new disk size before extending the partition.

Run these commands **on each corresponding host**, only after the AWS resize:

```bash
lsblk -b -o NAME,PATH,TYPE,SIZE,FSTYPE,MOUNTPOINTS,PKNAME,SERIAL /dev/nvme0n1
findmnt -T / -o TARGET,SOURCE,FSTYPE,OPTIONS,SIZE,USED,AVAIL
sudo /usr/bin/growpart /dev/nvme0n1 1
sudo /usr/sbin/resize2fs /dev/nvme0n1p1
df -B1 /
```

Before the two `sudo` commands, confirm the parent disk is 268,435,456,000 bytes
(250 GiB), and `/` is still ext4 on `/dev/nvme0n1p1`. If either differs, stop and report
the output instead of substituting guessed devices. Both executables were found at
the exact paths above. These steps follow the [AWS EBS extension instructions](https://docs.aws.amazon.com/ebs/latest/userguide/recognize-expanded-volume-linux.html).

## Consequence if no action is taken

trn21 stays disk-blocked. trn22 has space for a projected warm dense run only after
all its other gates pass; table jobs and cold S0 replays remain blocked. S0's noon-IST
target is therefore conditional on disk expansion and successful runtime checks, not
a promise of readiness. No threshold is lowered to improve utilization.
