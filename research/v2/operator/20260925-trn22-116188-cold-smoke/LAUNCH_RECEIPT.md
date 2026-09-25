# trn22 116188 cold 120-second compile smoke: live Operator receipt

Status at `2026-09-25T14:22:45Z`: **TRAINING/COMPILING; no step or final verdict yet.**
This is a diagnostic only. It cannot establish BPB or clear S0 for submission.
No cleanup, old-cache mutation, AWS lifecycle action, submission, or retry
occurred.

- Assignment: `research/v2/assignments/20260925-trn22-116188-cold-compile-smoke-r1.json`, controller `f6a6fda1`, conditionally enabled one-use label
  `fs-20260925-116188-cold-compile-smoke-trn22-r1`. Independent proposal/assignment PASS `b3ce7b93`; actual launcher at Operator branch `e000ae28` separately reviewed PASS by distinct Verifier before host writes.
- The preceding trn22 A0 training and fresh public20M evaluation were both terminal. Its `driver.log` had `EVALUATED_2026-09-25T14:16:06Z` and the exact `eval/rehearsal.json` had `mode=eval`, `returncode=0`; no Neuron job/device/lock holder remained.
- Fresh preflight `2026-09-25T14:21:30Z`: root absent; no Neuron job, device or lock holder; global-correctness and trn22 controller breakers false; no STOP; source, trusted prepare, public data/tokenizer, and disk checks passed; `/` free 56,619,393,024 bytes. The launcher asserted the one-use assignment had not expired.
- New unique host root `/home/ubuntu/trainium_116188_cold_smoke_20260925_trn22_r1` and seven empty cache dirs were created `2026-09-25T14:21:35Z`. The job-owned rank-0 step log was precreated empty/exclusive, inode `2621580`, mode `0600`; no old cache was touched.
- Copied source SHA-256: `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`; trusted imported `prepare.py` SHA-256: `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`, both verified on host.
- Actual training started `2026-09-25T14:21:51Z`; detached driver PID `3031206`; launcher reported `LAUNCHED_2026-09-25T14:21:54Z`. At 14:22:13 UTC one torchrun and four `train.py` workers, PIDs `3031226`–`3031229`, occupied `/dev/neuron0`; the exact argv was the 116188 dense flags plus only `--max-train-seconds 120`, LNC2, full four-rank chip. No step boundary existed yet. At 14:22:45 UTC root source/trusted-prepare SHAs still matched; step log remained empty; persistent/compiler/debug/inductor cache dirs had no files, while private `neff`/`neff_local` each had 3 and `hlo` had 9 newly created files. Cold attribution remains **pending** effective persistent-URL proof.

On completion retrieve raw train/driver/step logs, exit code, checkpoint identity, compiler/cache evidence and quiescent postflight. A clean exit with fewer than 20 distinct completed rank-0 step boundaries is inconclusive; any compiler failure is a failed diagnostic. Do not run public evaluation for this smoke.
