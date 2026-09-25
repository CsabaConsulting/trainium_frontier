# trn22 r4 source-stage feasibility — Operator read-only observation

**Observed:** 2026-09-25T02:42:28.357805+00:00. **Verdict:**
`READ_ONLY_FEASIBLE_AT_OBSERVATION_TIME_ONLY`. The frozen r4 archive and
checkout have **not** been transferred or created. This is planning evidence,
not a fresh pre-transfer preflight, an independent package review, a source
transfer/checkout assignment, CPU G2, native dispatch, evaluation or a score.

## Evidence and scope

- Controller base at observation: `af50558bafea0830a7474acbeba5b7e8f0bb7c80`
  on `agent/control-plane-20260921`. Its `research/v2/state.json` SHA-256 was
  `2edf7660514dd8b288c4cb8a1ec7cc96113fb74d998fe05f831c120effc28040`;
  `launch_authorized=false`, trn22 `launch_allowed=false`, and local global
  correctness/trn22 breakers untripped. Local `research/agentic/STOP` and
  `research/v2/STOP` were absent. These local facts do not authorize a host
  action.
- Plan inspected: Implementer HOLD-only source-stage contract at
  `4570f4e87d5e052a6d8e402326e146384759b8cf`, contract SHA-256
  `79878b16f459d4c12b32cb0ebb85d205dad86d584ae09654e0e8161920deab69`.
  It expressly has `source_transfer_authorized=false` and
  `source_checkout_authorized=false`. The local capsule bytes and manifest
  were **not** inspected by this Operator.
- The exact read-only probe source is `remote_readonly_probe.py`, SHA-256
  `fe6ace073432444cd2848fdcdf5f55a3e6652463446eacaba310251c61e7fc38`.
  Its complete curated host stdout is
  `HOST_SNAPSHOT_20260925T024228Z.json`, SHA-256
  `eb43d2100a933c2f4684c106d867d61014df057342168d4b9b861fd171051981`.
  The probe used only metadata reads, a read-only checkpoint rehash, bounded
  process/label scans and read-only device-holder queries; SSH exited 0 and
  returned exactly one parseable JSON object without diagnostics. No private
  key contents, credentials, environment values or process command lines were
  read into the receipt.
- Transport: existing documented trn22 endpoint
  `ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com`, identity
  file referenced by path only at `/home/poojith-devan/Desktop/Trainium/trn2.pem`
  (regular, mode 0600), `BatchMode=yes`, `IdentitiesOnly=yes`,
  `StrictHostKeyChecking=yes`, `HostKeyAlgorithms=ssh-ed25519`, explicit existing
  `/home/poojith-devan/.ssh/known_hosts`, and `GlobalKnownHostsFile=/dev/null`.
  The pinned ED25519 entry reported fingerprint
  `SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U`. The remote command was
  only `exec python3 -`, with the committed probe source streamed on stdin.

## Fresh host observations

| Gate | Observation |
| --- | --- |
| Identity | Hostname `ip-172-31-32-253`, effective UID 1000; boot ID `bf52dfdd-1dba-485a-9ee6-e5a27fb41ba6`; mount namespace `mnt:[4026531832]`. |
| Persistent parent | `/` and `/home/ubuntu` cover the same root ext4 mount, `/dev/root`, major:minor `259:1`, mount ID 27. `/`, `/home`, and `/home/ubuntu` are non-symlink, non-group/world-writable directories. `/home/ubuntu` is UID 1000, device 66305, inode 295711, mode 0750. |
| Space | Root free **58,453,950,464 bytes** and **31,660,018 inodes**, above the contract's pre-transfer 13,177,973,911-byte / 10,000-inode floors. The later post-stage floor remains 10,737,418,240 bytes and must be freshly rechecked after each stage. |
| Exclusive paths | The exact new 104,154,875-byte capsule target, r4 checkout, separately held monitor target, r4 checkout output/log/process/postflight/steps paths, and observer output/stderr paths were all absent. A bounded `/home/ubuntu` same-filesystem scan found zero names containing the exact science label. |
| Protected assets | Retained checkpoint remains regular, single-link, UID 1000, device 66305, inode 2417091, 469,831,009 bytes, SHA-256 `1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b`. All six warm-cache child directories retain device 66305, UID 1000 and pinned inodes 2430420–2430425. The checkpoint had no fuser holder. |
| Safety state | Twelve scoped host STOP paths absent. Zero matched train/eval/G2/guard/launcher processes; zero unreadable process command lines. `neuron-ls -j -a` parsed successfully with zero non-diagnostic PIDs. `/dev/neuron0` and the shared `device0.lock` each had fuser rc 1 and zero holders. |

The result is a **point-in-time read-only feasibility PASS** for planning an
independently reviewed, exact-target **transfer-only** request. Any future
transfer must use a separately reviewed writer and a new one-attempt Supervisor
assignment, then perform its own immediate full preflight. Any target drift,
different key/host/boot/mount/parent identity, safety stop, lower disk space or
ambiguous check is HOLD. No cleanup, retry, overwrite, breaker reset, AWS
mutation, checkout, monitor start, CPU probe, native run or eval occurred here.
