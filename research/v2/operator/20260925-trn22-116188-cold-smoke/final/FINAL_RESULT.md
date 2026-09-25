# trn22 116188 cold 120-second compile smoke: Operator terminal receipt

**Disposition under the reviewed assignment: INCONCLUSIVE_FOR_CODE_CACHE.**
The diagnostic exited zero and saved a loadable checkpoint, but recorded only
five distinct completed rank-0 training boundaries instead of the required
20. The specified persistent compile-cache URL remained empty, so true-cold
attribution also fails its required proof. This run has no public BPB and
does not validate or invalidate S0 for submission. This Operator receipt is
not an independent postlaunch review.

## Exact run and observed result

- One-use label `fs-20260925-116188-cold-compile-smoke-trn22-r1`; host
  `trn22`; root
  `/home/ubuntu/trainium_116188_cold_smoke_20260925_trn22_r1`.
  Proposal/assignment independently reviewed, then Supervisor enabled;
  the actual launcher `e000ae28` received a separate independent PASS.
- Exact unmodified `train.py` SHA-256
  `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`;
  imported trusted `prepare.py` SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Full-chip `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone
  --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift
  --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32
  --clip-after-reduce --seq-len 1024 --no-eval-public --max-train-seconds 120`.
  `env -i` set the seven new job-owned cache paths, unique label and rank-0
  step-log path; no external `NEURON_CC_FLAGS` override. The live compiler
  inherited the file's baked `--optlevel=1 --auto-cast matmult
  --auto-cast-type bf16`, with no `--cache_dir`.
- Fresh preflight was `2026-09-25T14:21:30Z`; root/caches were absent,
  disk free was 56,619,393,024 bytes, prior A0 fresh eval was terminal,
  no Neuron process/device/lock holder was present, and controller breakers,
  STOP, source, prepare, data and tokenizer checks passed. The private
  root/seven empty cache dirs and exclusive empty rank-0 step log (inode
  `2621580`, mode `0600`) were staged `14:21:35Z`.
- Driver training started `14:21:51Z`, reported launch `14:21:54Z`, and
  finished `14:36:26Z` on 2026-09-25 UTC. Torchrun/all four workers
  returned zero; the exact `train.exitcode` file contains `0`. The rank-0
  step log has only `0,1,2,3,4`, one per completed optimizer step. Finite
  losses printed in `train.log` were 9.0112, 8.7822, 8.6919, 8.6435,
  8.5971. Its summary says `num_steps: 5`, `total_tokens_M: 0.7`,
  `startup_allowance_est: 867.5`, `charged_seconds_est: 0.4`,
  `over_budget: False`, `total_seconds: 845.6`, and `compiled: True`.
- The five-step stop is explained by the **unchanged source** at lines
  1436–1439: once minimum five steps are reached, it stops when current
  charged time plus `max(recent_step_times[-20:])` plus the 15-second
  checkpoint reserve reaches `max_train_seconds`. The first two cold
  compilation-heavy step times were 480.53 and 359.84 seconds, so the
  120-second smoke could not reach the assigned 20-step threshold even after
  normal subsequent steps were about 0.49 seconds. No source or budget was
  changed mid-run and the label will not be retried.
- Checkpoint remains on host at `out/final.pt`: 469,831,009 bytes, SHA-256
  `24fa2f71069e054b1e3ae5597d38666b5dc34e6c272f02b9a12e600f4d92b7ad`.
  A separate read-only `torch.load(..., weights_only=True, map_location='cpu')`
  succeeded, yielding `meta`, `model_config`, and `model_state`.

## Cache and postflight

The active compiler's `/proc` environment contained the pinned private
`NEURON_COMPILE_CACHE_URL` and `NEURON_COMPILE_CACHE_DIR`; no alternate
`--cache_dir` was observed. During execution fresh job-owned `neff`,
`neff_local`, `hlo` and transient debug compiler files appeared, and the
`neuronx-cc` log showed continuing passes, not a failure. The default
`/var/tmp/neuron-compile-cache` directory was absent when checked at
`14:33:33Z`. However, the pinned persistent URL directory had **zero files
after the run** and its mtime remained the staging time `14:21:35Z`. The
assignment therefore forbids a true-cold-cache PASS, regardless of the
private Neff/HLO files. Final file counts across the seven private dirs:

| Directory | Files |
| --- | ---: |
| persistent URL | 0 |
| compiler | 0 |
| neff | 128 |
| neff_local | 128 |
| hlo | 354 |
| debug | 0 |
| inductor | 0 |

At `14:37:17Z`, `ps` found no torchrun/train/compiler process,
`fuser /dev/neuron0` found no holder, the existing host device lock was
free, and `/` had 54,810,644,480 bytes available. No checkpoint, cache or
debug artifact was cleaned. The raw host logs copied below match these
host-side SHA-256 values:

| Raw artifact | SHA-256 |
| --- | --- |
| `driver.log` | `f07ea417578acd1f2199f42b4786ef384035427e94cc9878c283940f8ea7a19f` |
| `train.log` | `63c6420304f04beee3bd4aefdf13a292659131027fc41ed2318463695cc47ee6` |
| `step-boundaries-rank0.log` | `026d8ad3dfa1f2aa9da7964947ddedd4e83c6fc008206ebf898699dea80f9804` |
| `train.exitcode` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |

No public evaluation was run or claimed for this short diagnostic. The
postlaunch reviewer and Analyst should record the failed/inconclusive gate,
not a BPB comparison or adoption.
