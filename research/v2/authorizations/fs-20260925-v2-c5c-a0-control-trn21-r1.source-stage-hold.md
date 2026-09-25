# trn21 c5c A0 source-stage HOLD (Operator receipt)

Date: 2026-09-25 UTC
Operator: `trn21_g1_r2_guarded_eval_operator`
Status: **HOLD / incomplete source stage**. No CPU G2, native training, evaluation, or AWS action was performed.

## Authority and immutable identities

- Supervisor source-stage-only assignment: `research/v2/authorizations/fs-20260925-v2-c5c-a0-control-trn21-r1.source-stage-assignment.json`, SHA-256 `24a58fbe947537d3a4f0fc377e08638a6f0b6f48697988a5ed0b9958f77e75a3`, at controller commit `ff372851db79aa2e8be058e68e4e19df9a8215a8`.
- Independent stage/G2 static PASS: `research/agentic/verification/20260925-trn21-c5c-a0-r1-stage-g2-static-review.md`, SHA-256 `63505225afcf381c173f306c17b33f2831d67cd75bf3017ff2a57b38df2ec4d6`. Read-only feasibility receipt SHA-256 `90b77b5f220da9e72dbe1932b27b87b953ba85865793ea9ebc3318997f330df1` and independent review SHA-256 `ef1df8346555088b14fd575f85699b08aa9163efc68e10b582e41f2ee3099c40`.
- Local source capsule: `/dev/shm/trainium-c5c-e65-one-commit-20260925.tar.gz`, regular mode `0600`, 104,005,787 bytes, SHA-256 `32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264`. Its read-only tar-member scan found 6,338 members, zero links/special members, zero unsafe/traversal/duplicate/PAX-override paths, and zero credential/data-like filenames or high-confidence secret markers. The two local CPU tools matched the reviewed SHAs (`preflight_dryrun.py`: `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`; `contained.py`: `0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d`). Neither tool was transferred.
- Planned source commit `e65a3097a5b00f6178188a935bb2acb9a5042c6c`, `train.py` SHA-256 `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`, `prepare.py` SHA-256 `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`. These were **not** staged or checked on trn21 in this attempt.

## Fresh prewrite evidence

At `2026-09-25T00:41:39.039573Z`, a read-only trn21 preflight observed hostname `ip-172-31-34-85`, UID `1000`, HOME `/home/ubuntu`, root ext4 on `/dev/nvme0n1p1` with 58,676,260,864 bytes free (required 10,737,418,240), and `/dev/shm` tmpfs with 66,987,511,808 bytes free. The two protected checkpoint size/SHA pairs matched (`469834849`/`aaa4b7c0ff58c56df564954c21f8abdd786930bfaa1cc447f6fb8e5b0da83535`; `469831009`/`5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9`). All six warm-cache identities matched the reviewed filesystem/inodes: `neff 2945951`, `local 2945952`, `hlo 2945953`, `inductor 2945954`, `compiler 2945955`, `debug 2945956` on device `66305`. Neuron process count was 0; `/dev/neuron0` and shared host-lock `fuser` each returned no PIDs; no matched train/eval/G2/guard process, STOP, or breaker. All exact planned archive, checkout, tool, G2, science-output and log paths were absent. Controller STOP/breakers were closed, label unconsumed, and queued trn21 jobs 0.

The broad recursive `/dev/shm` name scan returned rc 1 because an unrelated root-owned `/dev/shm/eic-hostkey-CUD5mlzT` directory was unreadable. As separately reported to the Supervisor, the scoped top-level `/dev/shm` label scan and direct `lstat` of every exact allowed path passed at `2026-09-25T00:42:24.174591Z`: all were absent and neither G2 nor science label matched. The Supervisor authorized stage-only continuation with this caveat; the rc 1 was not silently treated as a broad-scan PASS.

## One attempted transfer and failed persistence check

Only one host-write attempt occurred. The local archive was streamed as stdin over the pinned trn21 SSH transport to remote Python. The command's recorded transport argv was `ssh -i /home/poojith-devan/Desktop/Trainium/trn2.pem -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com python3 -c <remote-writer> < /dev/shm/trainium-c5c-e65-one-commit-20260925.tar.gz`. `<remote-writer>` denotes the Python payload passed to `-c`, not a literal shell argument; its exact source bytes were not saved. It opened only `/dev/shm/trainium-c5c-e65-one-commit-20260925.trn21.tar.gz` with `O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW` and mode `0600`, wrote stdin in 4 MiB chunks, `fsync`ed, then counted, hashed, and `lstat`ed the file.

The SSH key contents were neither printed nor recorded. The transfer process exited `0` with no reported stderr and printed this final stdout JSON at `2026-09-25T00:44:23.931744Z`:

```json
{"bytes": 104005787, "device": 26, "exact": true, "inode": 6397, "mode": "0o600", "path": "/dev/shm/trainium-c5c-e65-one-commit-20260925.trn21.tar.gz", "sha256": "32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264", "utc": "2026-09-25T00:44:23.931744+00:00"}
```

The next **separate, read-only SSH process**, roughly five seconds later, attempted `lstat`/SHA of that exact path and exited `1` with `FileNotFoundError: [Errno 2] No such file or directory: '/dev/shm/trainium-c5c-e65-one-commit-20260925.trn21.tar.gz'`. Thus exit `0` and matching SHA in the transfer process do **not** establish persistence to a later session. No retry or overwrite was attempted.

Read-only follow-up at `2026-09-25T00:45:55.609065Z` again observed hostname `ip-172-31-34-85`, UID `1000`, HOME `/home/ubuntu`, boot ID `1903f7f0-8e0d-4011-9566-346e4ba01a0f`, uptime `837542.06` seconds (no recent reboot), mount namespace `mnt:[4026531832]`, `/dev/shm` device `26`, inode `1`, tmpfs options `rw,nosuid,nodev,inode64`, total/free `66,987,511,808` bytes, used `0`, and the exact target absent with no matching top-level tmpfs name. Root remained ext4 on `/dev/nvme0n1p1`, with 58,676,199,424 bytes free. An earlier read-only forensic check at `00:45:02.504535Z` saw the same boot ID/mount namespace and absent target. Read-only attempts to inspect `/proc/1/ns/mnt` returned `PermissionError`; the **transfer session's mount namespace was not recorded**.

## Conclusion and containment

Root cause is **unknown**. The evidence does not distinguish external deletion from per-session filesystem isolation or another mechanism. It does not support a claim that the capsule was persistently staged. The remote source checkout and CPU tool paths were never created, and no extraction, G2, native job, eval, pruning/deletion, breaker reset, or AWS action occurred. The precise archive target was absent at last observation. This label remains unconsumed. Do not retry the transfer or proceed to CPU G2/native work under this receipt; a separately reviewed corrective transfer plan and fresh authorization are required.
