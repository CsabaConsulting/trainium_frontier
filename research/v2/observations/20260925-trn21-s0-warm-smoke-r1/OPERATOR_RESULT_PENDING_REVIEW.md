# trn21 S0 warm 120-second compile smoke: inconclusive five-step exit

Operator evidence only, 2026-09-25 UTC. This is a consumed one-use diagnostic
label, not a full-budget S0 replay, a quality comparison, or submission
clearance. A different reviewer must examine these receipts before result use.

- Unique label: `fs-20260925-s0-warm-compile-smoke-trn21-r1`.
- Host root: `/home/ubuntu/trainium_s0_warm_smoke_20260925_trn21_r1`.
- Supervisor assignment at controller `f6a6fda1`, independently reviewed;
  previous full S0 warm assignment remained disabled. The latest human direction
  cancelled S0 full cold replays and kept only this already-running smoke.
- Exact unchanged S0 `train.py` SHA-256:
  `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb`.
  Trusted imported `prepare.py` SHA-256:
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  A0 training and fresh public20M evaluation were both terminal exit 0;
  fresh preflight found no device/lock/Neuron/process holder, no STOP, a unique
  absent run root, and 56,024,952,832 bytes free. Only the exact source file
  was copied into the new private root, and host SHA/import/cache checks passed.
- Four `python3 -u train.py --max-train-seconds 120` ranks were observed
  (PIDs 2897506–2897509) after detached launch at
  `2026-09-25T14:30:30.979207Z`. LNC2 x4 and the exact sole science CLI
  flag appear in the host launch manifest. `TRN_EXP_LABEL` was unique;
  `NEURON_COMPETITION_R1_STEP_LOG` pointed only to the private, exclusively
  precreated 0600 rank-0 log at
  `/home/ubuntu/trainium_s0_warm_smoke_20260925_trn21_r1/receipts/step-boundaries-rank0.log`.
- Torchrun exited `0` at `2026-09-25T14:40:00.945386Z`, with final
  distributed barrier/checkpoint path in the log. The rank-0 step log is
  exactly 10 bytes containing five distinct completed boundary numbers,
  `[0, 1, 2, 3, 4]`; its inode is 3,175,408. Logged losses were finite,
  9.0112 to 8.5971. No `nki_kernel_global` exception appeared.
- The checkpoint remains on host at `out/final.pt`: 469,830,945 bytes,
  inode 3,175,834, SHA-256
  `56ab480b77addaf1e8d86dec48539788c960a0e7c4c2b2b84b4a008c42149fe4`.
  A separate read-only CPU `torch.load(..., weights_only=True,
  map_location='cpu')` loaded 62 state tensors, all finite, with metadata
  `step=5`, `world_size=4`. The binary checkpoint was neither transferred nor
  removed.
- **Verdict: INCONCLUSIVE_FOR_CODE_CACHE, not PASS.** The assignment required
  at least 20 distinct completed boundaries and clean exit/checkpoint; only
  the latter was met. Step 0 took 307.64 s and step 1 took 231.31 s due to
  compile/startup. At step 4, after the startup clock reset, S0's stop rule
  still considers `max(recent_step_times[-20:])` plus checkpoint reserve;
  that exceeds the 120 s diagnostic budget and triggers the fifth-step exit.
  The clean exit shows the S0 warm path can get through those compilations,
  but it neither proves cache hits caused that nor supports a BPB claim.
- Cache occupancy before to after: job-pinned NEFF 138 to 150 files
  (341,344,946 to 510,306,537 bytes), HLO 354 to 360 files
  (325,886 to 327,720 bytes); persistent `/var/tmp/neuron-compile-cache`
  remained 818 files/284,030,856 bytes. Additional S0 graphs were compiled.
  No direct cache-hit evidence was observed; do not infer a warm-vs-cold cause.
- Separate read-only postflight ended `2026-09-25T14:41:14.828624Z` with no
  matched train/eval process, non-diagnostic Neuron PID, `/dev/neuron0` or
  device-lock holder; 54,226,755,584 bytes free. No cleanup or same-label
  retry was performed.

Retrieved immutable host-file hashes (SHA-256):

| Artifact | SHA-256 |
|---|---|
| `host-launch.json` | `2c6b467751632d72e7a3f368c41700e4efc466c7afa27be741a9828701703166` |
| `host-training.log` | `1d959cdc834596face3c0b5cda4e5490bac66a3db9ff8f3adba516c79b3cd6d4` |
| `host-run-exit-code.txt` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `host-step-boundaries-rank0.log` | `026d8ad3dfa1f2aa9da7964947ddedd4e83c6fc008206ebf898699dea80f9804` |
| `preflight-2026-09-25T142935.995234+0000.json` | `b7baf376019ef19d0bff847a4e5fffcecd63d2eb9518f666b0cbaa80d3da7ef9` |
| `stage-2026-09-25T142946.778463+0000.json` | `a53bfeadcc6ee89900af742af1846448c2247f81d29a08ec8c5e4fe9ccdf88f5` |
| `poststage-2026-09-25T143011.799365+0000.json` | `da03553135d56b130a24be59f2e7cd313ecaef173d78bbf819515b464cc7e17d` |
| `launch-2026-09-25T143023.205497+0000.json` | `1e801558753928c2c442bceab22795e4fba513599eea2a24213d57c8efacd344` |
| postflight `OPERATOR_RAW_TRN21_READONLY_20260925T144109Z.json` | `73a3ae55544f628485a95f3c8b5488abd9ffff3b4478a375e149dcaecf834491` |

The final postflight file is in the sibling
`research/v2/observations/20260925-dual-host-readonly-operator/` directory.
