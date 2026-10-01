# Deadline experiments — 2026-09-30

Current best is the unchanged phase1-final submission at b2405bd7e41dab9bc94bd43bf088fd25ca2c5aed. Latest official score recorded there is 0.9645; the recorded leaderboard cutoff is 0.9555. These experiments are unproven attempts to close the 0.0090 gap. Never replace the final submission using a component benchmark or training loss alone.

Base submission SHA256: 2b4b1ac0d7cd39628ee20282122797abd5f88a5bfc30d7d5115b45289d172894.
Official prepare.py SHA256: 5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb. It is unmodified. The complete public validation shard is evaluated with --eval-tokens 0; the CLI default is only 2M tokens and is not a comparable final result.

## Five arms

| Host | Candidate | Isolated hypothesis |
| --- | --- | --- |
| trn21 | middle | Keep eight layers and all parameters, run blocks 2–5 at half token resolution; use original even-position RoPE and causal residual interpolation back to full resolution before block 6. |
| trn22 | mlp | Share feed-forward residuals across adjacent positions in blocks 2–5, alternating causal parity; retain full attention in every selected attention block. |
| trn23 | mlp6 | Extend the previous sharing to blocks 1–6, retaining full first/last feed-forward blocks. |
| trn24 | rmsfp32 | Keep table values/gradients BF16 and all learning-rate/collective rules; accumulate the .999 RMSprop second moment and compute denominator/update in FP32 so small decays do not round away. |
| trn25 | mimetic | Isolate structured orthogonal V/O initialization from the previously unsuccessful five-setting bundle. Match RMS scales and preserve global RNG and every unrelated initialized parameter. |

All use seed 42, the pinned 30-minute training clock, four ranks, LNC2 for training, LNC1 for scoring, --tbl_freeze_tokens 0, and the same fixed compiler flags as the final submission. No seed sweep, external data, pretrained weights, evaluator changes, or inference-only alternate model.

## Correctness evidence

The remote native-venv CPU precheck passed: all 65 baseline/control parameter gradients and outputs are bitwise equal; all three sharing arms have exact causal prefix invariance, batch independence, finite gradients, and checkpoint reload parity. RMSprop matches an explicit FP32 reference for 100 steps and reproduces the original BF16 loss of .999 decay. Structured V/O initialization changes only the ten intended matrices and preserves global RNG. This proves structural correctness on CPU, not native performance or quality.

## Ownership and cleanup

Each job gets an immutable committed remote checkout, a unique label, and a source-hashed launch receipt saved locally before SSH. The launcher holds the common /home/ubuntu/.cache/trainium_freshstart/device0.lock from preflight through postflight. Preflight cleans only processes marked as descendants of the previous completed owned job and refuses unknown device holders or training/compiler jobs. It records SDK, memory, disk, inodes, runtime/device state, temperatures and kernel logs. Requires 16GiB host memory and 8GiB disk free. Each job has empty separate compiler/Inductor caches. No shared-cache purge, driver reload, package changes, or broad process kill.

A bounded timeout and guardian surround CPU checks, training and official public evaluation. Guardian always attempts marker/UID/process-birth-checked descendant cleanup, then records postflight and releases the common lock. Evidence/caches/checkpoints are preserved. Never launch another job while its lock is held.

## Restart / preservation

Worktree: trainium_frontier_deadline_20260930, branch research/deadline-hailmary-20260930. Source and experiment receipts are pushed to the user fork; heavy checkpoint/cache files remain under ignored out/. Requests and source manifest are under deadline/. Remote working directory names start trainium_freshstart_deadline_20261001; inspect exact request files before resuming. Logs and postflight receipts are in research/run-logs/. Mirror selected out/<label>/train.log, eval.log and final.pt locally, and commit/push new small evidence promptly. Never overwrite the phase1-final submission without an independently scored win and a valid 30-minute/cold-start record.

Official deadline is 2026-10-01 06:59 UTC (Sep30 23:59 PDT). The user must upload/submit a winning artifact; this campaign does not submit to the competition portal.
