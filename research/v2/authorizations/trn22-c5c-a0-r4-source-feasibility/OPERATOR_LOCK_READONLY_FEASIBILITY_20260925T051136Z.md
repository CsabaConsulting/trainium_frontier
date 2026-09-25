# trn22 existing-lock read-only feasibility receipt

Observed `2026-09-25T05:11:47.576478Z`. **PASS for the existing-lock inspection and brief nonblocking flock at that instant only.** The final-r3 transfer writer remains under the independent [late-label HOLD addendum](../trn22-c5c-a0-r4-transfer-only/INDEPENDENT_CONTINUITY_FINAL_R3_LATE_LABEL_HOLD_ADDENDUM.md); this check neither repairs nor authorizes it. No Supervisor transfer assignment, marker or journal existed before or after. No transfer, checkout, CPU G2, training, evaluation, cleanup, breaker reset, AWS action, or submission was performed.

The Operator reused the existing worktree, branching from canonical `8c005717913bd65f76e0834c0f53a7f904c63e7f` to `evidence/20260925-trn22-lock-readonly-operator-0508`; no new worktree or cleanup. The reviewed r3 request SHA-256 was `a2b1ad44023f32c010c8d169e25986830fa60a26d0162d281a4c0154b447e7a0`, strict-host-key transport `1fa3afb41dfbf8288e53c4aa43dd16b348013416e708d8432e97364b31556b51`, and read-only remote probe `e93efd99c0a58aa32537d945ddc8fc29f9d7ce0dabb89ed93526da30421f4086`. The controlling HOLD addendum SHA-256 was `94de8bc3d12a4c36cdad6d2160c1db081d4ea0c0035b6024bf7471e389ffaa0b`. The outside-Git capsule remained 104,154,875 bytes/SHA-256 `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`; pinned known-hosts SHA-256 `e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3` and ED25519 fingerprint `SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U` passed. Local validation inspected private-key metadata only; its contents were neither printed nor archived. Local controller state SHA-256 `c85be764f5f9f53953bfc3119f3fc4104d6e306fe051607993f615ffdf4e9693` had clear global/trn22 breakers and local STOPs at both local gates.

## Exact read-only sequence

All three SSH sessions used the request's pinned endpoint and strict host-key options from `stage_r4_transport.build_ssh_argv("probe")`, with `/dev/null` SSH config, no stdin and `python3 -B` to suppress remote bytecode writes. The first and third commands were `exec python3 -B -c <exact reviewed r3 remote source> probe`; the middle command was `exec python3 -B -c <exact diagnostic_source archived in the raw record>`. The command strings were shell-quoted and round-tripped through `shlex.split` before execution. The identity-file path is redacted in the archive; the source bytes and exact argv/remote-command hashes are preserved.

| Phase | SSH rc / stderr bytes | Raw stdout bytes / SHA-256 | Executed argv SHA-256 |
| --- | --- | --- | --- |
| Full reviewed safety probe before | `0 / 0` | `14,467` / `d5a0ac8889b0c35deecd8bee42c165695be557f069c1b05ef3a6dec87fc23c6a` | `7ca79c9760f1b97bff8fbad81ef4bdc8c0e602eb205b3ee473ce0b1a8dbd3826` |
| Existing-lock no-follow diagnostic | `0 / 0` | `3,474` / `5e0129ffcd1d0afee366ab84f084d42e25d4bbb97df796efdf1ba6f6c7fd2215` | `5e7ea4abbddc8f347bc73e7da815bda3cc2f5687eb5279e13b21415058a87db5` |
| Full reviewed safety probe after | `0 / 0` | `14,467` / `eab05ad82aa2378955aa43c660f8522b41fa1549dad3e27f65e1f01197ea7124` | `7ca79c9760f1b97bff8fbad81ef4bdc8c0e602eb205b3ee473ce0b1a8dbd3826` |

Each stderr SHA-256 was the empty-stream hash `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; none timed out. The exact diagnostic source SHA-256 was `1499ca19e1c52d94c8ffd71110f2510436c28ae7afa21b946880c5a34ab4f9d7`, archived in full beside all three base64 raw stdout/stderr streams and command hashes. It contains no create, truncate, write, delete, rename, chmod or chown operation.

## Exact existing path and flock observation

The diagnostic opened `/`, `/home`, `/home/ubuntu`, `.cache`, and `trainium_freshstart` descriptor-relatively with `O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`, comparing no-follow named metadata with `fstat` at each step. It then opened only the existing `device0.lock` with `O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC`; no `O_CREAT` or writable flag was used.

| Exact component | Type; symlink? | UID:GID | Mode | Device / inode | Links | Bytes |
| --- | --- | --- | --- | --- | ---: | ---: |
| `/home/ubuntu/.cache` | directory; no | `1000:1000` | `0700` | `66305 / 295736` | 6 | 4,096 |
| `/home/ubuntu/.cache/trainium_freshstart` | directory; no | `1000:1000` | `0775` | `66305 / 832588` | 3 | 4,096 |
| `/home/ubuntu/.cache/trainium_freshstart/device0.lock` | regular file; no | `1000:1000` | `0664` | `66305 / 832589` | 1 | 0 |

One `fcntl.flock(fd, LOCK_EX|LOCK_NB)` call succeeded on the read-only lock descriptor and was immediately followed by `LOCK_UN`. The remote receipt reports `attempt_duration_ns=8,335` and `held_ns=1,341`; the lock was released before further metadata or host checks. Named and opened metadata (including size, mode, device/inode, link count, mtime and ctime) matched before and after. This proves a brief acquisition was possible then, not that a future contender cannot appear or that the proposed writer is safe. In particular, the `0775` parent and `0664` lock modes are reported for independent repair review, not silently treated as private permissions.

Both surrounding reviewed full probes validated the pinned host `ip-172-31-32-253`, UID 1000, unchanged boot ID `bf52dfdd-1dba-485a-9ee6-e5a27fb41ba6` and mount namespace `mnt:[4026531832]`, ext4/root aliases, protected checkpoint and six cache identity pins, old-path inventory, absent future targets and 12 absent host STOPs. Before and after, matched/unreadable process lists were empty, the r4 science-label scan had zero matches, the shared-lock and `/dev/neuron0` fuser checks returned 1 with no PIDs, and `neuron-ls` had no non-diagnostic PID. Root free bytes were 58,453,852,160 before and 58,453,848,064 after (4 KiB lower, cause not attributed here); both exceeded the bound planning floor. The custom diagnostic's own host/boot/mount identity matched both probes.

## Evidence custody and limit

`OPERATOR_RAW_LOCK_READONLY_20260925T051136Z.json` is 106,454 bytes, SHA-256 `2355c6814dc3e847fd9882738065c049f0a39bf078ce99992bb4ac1526f88305`. It contains the exact diagnostic source, per-phase redacted argv and exact hashes, return codes, durations, full base64 stdout/stderr with checked byte counts and SHA-256, parsed before/lock/after receipts, and local gates. A credential-pattern scan found no private-key header, AWS key, token, `.env` path or unredacted identity-file path. The raw file remains owner-private locally; the committed copy contains no credential material.

**Terminal disposition: PASS_POINT_IN_TIME_EXISTING_LOCK_FLOCK_FEASIBILITY only.** It must not be interpreted as transfer authority or a fix for the controlling writer HOLD; all live gates and the proposed lock protocol require independent review before any future assignment.
