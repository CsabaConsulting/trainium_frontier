# trn21 c5c A0 source-stage read-only forensic addendum

Operator: `trn21_g1_r2_guarded_eval_operator`
Date: 2026-09-25 UTC
Status: HOLD continues; this is not a stage or G2 authorization.

This addendum supplements `research/v2/authorizations/fs-20260925-v2-c5c-a0-control-trn21-r1.source-stage-hold.md` at commit `c345f235a3b5796b95f51b24ad465843fedcfcc6`, SHA-256 `1710449665cf889349a2527cf793a1a247ab88ecc903a9c6aaf2cf7823f46135`.

Three **separate**, newly opened pinned SSH sessions used the same `-i /home/poojith-devan/Desktop/Trainium/trn2.pem -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20` trn21 transport. Each ran read-only `python3 -c` to capture hostname, UID, boot ID, `/proc/self/ns/mnt`, `stat('/dev/shm')`, `findmnt -J -T /dev/shm`, and `lstat` of only the exact archive path. Each SSH process returned exit `0` with empty stderr.

| UTC observation | Host / UID | Boot ID | Mount namespace | `/dev/shm` identity and options | Exact archive target |
|---|---|---|---|---|---|
| `00:52:18.893059` | `ip-172-31-34-85` / `1000` | `1903f7f0-8e0d-4011-9566-346e4ba01a0f` | `mnt:[4026531832]` | tmpfs, major:minor `0:26`, stat device `26`, inode `1`, `rw,nosuid,nodev,inode64` | absent |
| `00:52:36.066081` | `ip-172-31-34-85` / `1000` | same | `mnt:[4026531832]` | same | absent |
| `00:52:40.773511` | `ip-172-31-34-85` / `1000` | same | `mnt:[4026531832]` | same | absent |

The exact target checked was `/dev/shm/trainium-c5c-e65-one-commit-20260925.trn21.tar.gz`. The mount namespace matches the earlier post-transfer checks at `00:45:02` and `00:45:55`. Thus **follow-up SSH sessions are stable** in host, boot, namespace, and tmpfs identity, and the archive remained absent. The transfer session's mount namespace was **not** captured, so this does not prove it used the same namespace or explain the disappearance. Root cause remains **unknown**.

One preceding SSH probe had a local/remote shell-quoting error (`SyntaxError`; overall command exit `2`) before it collected a valid observation; it performed no requested file operation. The three rows above are successful, separate sessions after that error.

The exact previous transfer's remote `python3 -c` writer source bytes and full argv payload were not retained in this Operator's compacted context or stored tool-result values. The earlier receipt records its transport argv prefix, described open/write/fsync/hash/lstat mechanism, exit status, and terminal stdout, but the source cannot now be inspected to prove the absence of cleanup/unlink, mount or namespace changes, temporary-directory context, or post-JSON path mutation. None of those actions was observed in the read-only follow-ups, but they cannot be excluded from the unavailable transfer-source record. No transfer retry, remote write, extraction, CPU G2, native training, evaluation, deletion, or AWS action was performed during this addendum.
