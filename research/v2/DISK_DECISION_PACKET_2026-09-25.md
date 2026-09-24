# Disk decision packet — 25 September 2026, 02:06 IST

## Urgent update — 25 September, about 03:00 IST

The first guarded warm c5c A0 run on trn22 exhausted its compile-space projection
**before any optimizer step**. The Operator observed only 6,449,438,720 bytes free,
below the immutable 6.5-GiB floor (6,979,321,856 bytes), authenticated the run's
guard identity, and stopped that owned guard. Its postflight ended at 21:28:07 UTC
with rc 1 and `host_clean=false` solely for disk; the timing observer saw zero steps.
No checkpoint/evaluation result, deletion, cache cleanup, or relaunch occurred.
The hash-verified [terminal receipt](incidents/fs-20260925-v2-c5c-a0-control-trn22-r1/operator-terminal.json)
and [independent review](../agentic/verification/20260925-a0-trn22-disk-safety-review.md)
are now archived. Preflight-to-postflight root free space fell by 1,500,942,336
bytes during compile. This is a failed safety gate, not a model score.

**Both hosts are now blocked:** trn21 was already under the floor and trn22 remains
under it at the latest checked 6,448,951,296 bytes free. The same +50-GiB-per-host EBS request
below is now needed before any further training. A reviewed new-label A0 retry and
human acknowledgement of the trn22 safety breaker are also required after the
volumes are expanded; resizing alone is not authorization to restart.

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

Even reclaiming all approximately 1.50 GB attributed to the failed A0 compile would
only restore trn22 to about 7.95 GB free. The failed attempt had already consumed
1,500,942,336 bytes **before step 1**; merely reproducing that partial compile while
staying above the 6.5-GiB floor requires at least 8,480,264,192 bytes at dispatch.
That is a lower bound, not enough for a full run or checkpoint. The separately
measured historical full-dense footprint is 1,854,160,896 bytes. A new-label retry
package is being prepared with a provisional, more conservative **at least 10 GiB
free** predispatch gate and continuous disk monitoring. Cleanup alone cannot justify
relaunching the same shape; volume expansion remains the practical dependency.

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

Both hosts stay disk-blocked: trn22's observed compile footprint exceeded its earlier
projection, and its breaker is tripped. Dense controls, table jobs and cold S0 replays
remain blocked until the volume expansion and separate gate review. S0's noon-IST
target has slipped; there is no submission-ready package. No threshold is lowered to
improve utilization.
