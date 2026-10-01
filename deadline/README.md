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

## Matched scoring follow-up

The initial driver also requests the entire validation shard (CLI --eval-tokens 0). This is a separate score: earlier winning receipts were explicitly capped at 20,971,520 tokens, not the entire shard. After the first guardian releases the host, evaluate_match.py scores the saved checkpoint on that exact historical one-process/LNC2/context1024/20M configuration with source-owned import compiler defaults. It also runs the unchanged official native causality test and a native batch-independence check. Compare its matched score with 0.96433779426305065, not the whole-shard score with a 20M baseline. The unchanged competition portal remains the authority for the official score.

The local deadline/preserve.py process mirrors all five hosts every ~45 seconds and commits/pushes only requests and this wave's evidence every other cycle. Its current explicit stop time is 07:15 UTC, allowing final evidence and cleanup to survive a GUI restart. It never merges branches or stages model/source edits, and skips a backup if the index already has someone else's staged work. Its PID/log are under out/preservation-process.json and out/preservation.log. deadline/checkpoint_watch.py separately retrieves completed immutable checkpoints, verifies their recorded SHA256 and length, and writes small tracked backup receipts. Heavy weights remain under ignored out/checkpoints/ and on the originating host.

## NKI recovery of the pair-sharing arms (05:28 UTC)

The PyTorch sharing implementation was rejected on actual native timing: four-block sharing ~111k tokens/s (1.18s/131072-token step), six-block sharing ~125k tokens/s (1.05s), versus ~328k for the full computation arms. Backward compilation grew to over a million instructions and millions of spill-DMA descriptors. This is a compiler/layout failure, not evidence against the algorithm's achievable BPB. Both runs were intentionally stopped after timing samples; no final quality/checkpoint claim is made. Both guardians completed scoped cleanup with host_clean=true and no failures.

candidates/mlpnki and mlp6nki retain the exact respective sharing architectures but replace selection/expansion and both backward operations with four simple NKI kernels. Tiles use 128 token partitions and contiguous channels; at width1024 this gives 2KiB contiguous BF16 free dimension. SPMD assigns complete batch rows to distinct physical cores. Odd parity explicitly zeroes the first position and omits the out-of-row final repeat; its backward counts only the final valid consumer. No atomic update or reduction across cores is needed. Scratch bound is under 2MiB per core; no performance claim is made from this bound.

Both new files passed CPU model checks with deliberately nonzero MLP output matrices: logits/loss and all parameter gradients are bitwise equal to their original sharing files. Both hosts also passed native correctness: all four operations, both parities, and three sizes including production [32,1024,1024] matched CPU references exactly. The gate-release receipts show no failures. The new driver refuses training if that gate fails or the test process leaves a device holder. Kernel checks use separate caches; subsequent training caches must be fresh. Each new full run scores exactly public20971520/context1024/LNC2 with source-owned import flags. These are new guarded jobs after the previous clean postflights, not modifications to a running experiment. First steady native timing recovered ~328k tokens/s for four-block sharing and ~358k for six-block sharing; final BPB is still pending.

## Matched scoring and final preregistered wave (05:54 UTC)

The original middle/rmsfp32/mimetic runs all completed training within their unchanged budget and saved checkpoints. Initial whole-shard evaluation was intentionally ended to prioritize comparable scoring before the submission deadline. Each stopped guardian released its host with host_clean=true and no failures; no whole-shard score is claimed. All three checkpoints were copied locally and their hashes verified. Separate matched scoring started at 05:47 UTC on all three hosts, through the same locked preflight/postflight protocol.

Two additional architectures are preregistered and have passed CPU strict-causality, batch independence, finite gradients, checkpoint reload and disabled-parent bitwise output/gradient checks with nonzero MLP outputs. The unchanged official static submission check also passed. middle6 coarsens blocks 1–6, keeping blocks 0 and 7 full resolution; its U-Net skips remain shape-compatible (0->7 full, other pairs coarse). middle6rms adds only the previously preregistered FP32 table moment. Training clocks, seed, data, all parameters and evaluator stay as in the frozen submission. Neither candidate has a native quality claim yet.

deadline/last_wave.py launches these on trn21/trn24 only after their matched evaluation has completed successfully and postflight has released the device cleanly. It declines a launch after 06:02 UTC to reserve submission time. Exact requests and immutable source pins are under deadline/requests/. Its PID/log are under out/last-wave-process.json and out/last-wave.log. Stop time is a launch cutoff, not an authorization to overlap or omit validation. The final source is promoted only after actual native scoring and correctness checks establish a gain.
