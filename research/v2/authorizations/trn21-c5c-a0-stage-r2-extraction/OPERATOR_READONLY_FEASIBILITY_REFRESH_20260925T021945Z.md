# Operator read-only feasibility refresh — trn21 c5c A0 source checkout

**Observation:** 2026-09-25T02:19:45.753673+00:00 (host UTC). **Verdict:**
`READ_ONLY_FEASIBLE_AT_OBSERVATION_TIME`; this is not a checkout preflight or
authority for any host write, CPU G2, native training, evaluation, AWS action,
cleanup, breaker reset, adoption, or submission. A later action requires a
fresh, separate preflight and exact one-shot authorization.

## Evidence binding and transport

- Controller base: `09692c4eaac6d5bb9db68a6df92e30fc5e0cb520` on
  `agent/control-plane-20260921`. Local state SHA-256 at observation:
  `3e2c2f9a2064453f04646e5ad7b6161001dcb13117ec502068e881affc269f7b`.
- Independently PASS transfer result: canonical `d8c16692e668e801a76134857ce743c25992c291`,
  `research/v2/authorizations/trn21-c5c-a0-stage-r2/INDEPENDENT_TRANSFER_RESULT_RECEIPT.md`,
  SHA-256 `bfafcb56ff449708dc31a1ebbe9023448018ea505e37a15847ae78bc3a3a2916`.
  Its terminal snapshot was at 2026-09-25T01:48:36.797405+00:00.
- Re-executed only that committed terminal receipt's
  `final_postflight_probe_exact_ssh_command` through `shlex.split`/`subprocess.run`,
  with the same pinned identity path and `BatchMode=yes`, `IdentitiesOnly=yes`,
  `StrictHostKeyChecking=yes`. The local identity file was regular, mode 0600;
  its contents were never read or printed. Exact SSH command string SHA-256:
  `1adf37c9a79cfc8a476f433a2285fe9335d79f623dade4364412fb15df0e441c`.
  SSH exit 0; stdout SHA-256
  `794163da7b354fde379e2a7ca77da1ceeb010551ac8f5dd575c66a1aefdf0b6d`;
  stderr empty. This read-only probe used no host write. The archived command
  itself is the full replayable command and scoped probe source; no command
  bytes were changed for this refresh.

## Fresh host observations and comparison

| Gate | Observed | Comparison with independent transfer PASS |
|---|---|---|
| Host/UID/boot/mount | `ip-172-31-34-85`, UID 1000, boot `1903f7f0-8e0d-4011-9566-346e4ba01a0f`, `mnt:[4026531832]`; ext4 `/dev/root`, mount ID 27, major:minor 259:1 | Same |
| Safe parent | `/home/ubuntu` device 66305, inode 295711, UID 1000, mode 0750; `/` and `/home` root-owned and non-group/world-writable | Same |
| Root free | 58,571,898,880 B, above 10,737,418,240-B floor | 36,864 B below the earlier 58,571,935,744-B observation; gate still passes |
| Exclusive paths | New checkout, its science output/log/steps paths, and planned G2 root absent; both science and stage label scans have zero matches | Same absence |
| Safety state | Six scoped host STOP paths absent; local `research/agentic/STOP` and `research/v2/STOP` absent; local global-correctness and trn21 breakers both untripped; local science ledger has zero label entries | Same safety conclusion at observation; current state still has `launch_allowed=false` |
| Live activity | Zero matched train/eval/G2/guard processes; unreadable cmdline count 0; `neuron-ls -j -a` valid with no non-diagnostic PID; `/dev/neuron0` and shared `device0.lock` fuser each rc 1/no holders | Same idle conclusion |

The exact three persistent ext4 transfer targets remain single-link, UID-1000,
mode-0600 regular files on device 66305, with the prior independently verified
identities:

| Target basename under `/home/ubuntu` | Bytes | SHA-256 | Inode |
|---|---:|---|---:|
| `trainium-c5c-e65-one-commit-20260925.trn21-stage-r2.tar.gz` | 104,005,787 | `32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264` | 296045 |
| `trainium-v2-g2-preflight-dryrun-20260925-trn21-stage-r2.py` | 23,550 | `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8` | 294946 |
| `trainium-v2-g2-contained-20260925-trn21-stage-r2.py` | 9,185 | `0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d` | 296647 |

Both protected checkpoints are still regular, single-link, UID-1000 files on
device 66305, with no fuser holder: inode 2947631 / 469,834,849 B /
`aaa4b7c0ff58c56df564954c21f8abdd786930bfaa1cc447f6fb8e5b0da83535`,
and inode 2938565 / 469,831,009 B /
`5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9`.
The warm-cache root remains device 66305/inode 2945950; its six child
directories retain the pinned device 66305 and inodes `neff` 2945951,
`local` 2945952, `hlo` 2945953, `inductor` 2945954, `compiler` 2945955,
and `debug` 2945956.

The probe and local checks were read-only. No path was created, deleted,
overwritten, extracted, or retried on trn21. This feasibility observation
expires for any later checkout decision; it must not be promoted to an
independent result review or dispatch authorization.
