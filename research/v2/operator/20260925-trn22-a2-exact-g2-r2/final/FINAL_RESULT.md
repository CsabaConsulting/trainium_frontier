# trn22 A2 exact-size CPU G2 — raw Operator result

**Scope:** one authorized CPU/gloo synthetic two-step G2 of the A2 table arm on trn22. This is **not** Neuron training, public evaluation, BPB evidence, adoption, or submission approval. Independent numeric review remains pending.

## Identity and launch

- One-use assignment: `research/v2/assignments/20260925-c5c-a2-exact-cpu-g2-trn22-r2.json`, SHA-256 `196075917e95edb664704d7fbbbd82a1242c50c12751eee9fcf4bfb2b22ca7fd`, enabled in controller commit `5a10c117`; independent assignment review `3554eba0`.
- Source and G2 tool were transferred from clean Operator worktree HEAD `6034746a90b3155d1d239bf5e3325740016fc487`, with local before/after and host SHA-256 checks: `train.py` `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`; `preflight_dryrun.py` `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`. Trusted host `prepare.py`: `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Host root: `/home/ubuntu/trainium_c5c_a2_exact_g2_20260925_trn22_r2`; private TMPDIR: its `tmp/` child. The earlier unmarked prelaunch HOLD and Supervisor-authorized correction are recorded in `../PRELAUNCH_HOLD_20260925T1451Z.md`. There was no cleanup, changed source/tool, changed flags, or second attempt marker.
- Fresh gates passed at `2026-09-25T14:53:02Z`: 54,810,226,688 bytes root disk free; `MemAvailable` 129,172,242,432 bytes; no train/eval/G2 process, Neuron holder, STOP or breaker; host lock free; only the expected empty private `torchinductor_ubuntu/` existed in TMPDIR; canonical TMPDIR and all pinned hashes matched.
- Exclusive `ATTEMPT_ONCE.json` was created at `14:53:49.427164Z`; driver launch `14:53:52Z`. Command was the assigned `preflight_dryrun.py` with `--steps 2 --nproc 4 --micro-rows 2 --micro-tokens 256 --timeout 3600 --trace-rmsprop --keep`, followed by the packet's exact A2 flags (`--ngram-ve-layers 1,-1`, bigram+trigram, multiplier 64, dimension 128, RMSProp, and inherited dense flags including `--matrix-lr 0.015`). Environment included private TMPDIR, seed 42, `NEURON_LOGICAL_NC_CONFIG=2`, assigned compiler flags and 1800-second competition limit. The exact argv/env/host paths are preserved in `ATTEMPT_ONCE.json` and `g2.json`.

## Terminal evidence

- Tool ran `14:53:49–14:54:16Z`; launcher and tool exit status `0`; aggregate `g2.json` says `PASS`, `torchrun_rc=0`, `timed_out=false`, elapsed 26.7 s, plan SHA-256 `7cfeabba0d5ade8cebc24b103ad1297e190ad9c3a365edacba68e5c23a4937a1`.
- Four rank JSON files are present and report `PASS`, two finite losses per rank, identical plan SHA, four expected tables (`b1,t1,b5,t5`), no unchanged table, two RMSProp calls, and all untouched rows unchanged. These are **raw tool claims for independent Verifier review**, not an Operator numeric sign-off. Rank 0 reports the strict-load checkpoint check; the schema does not put that check in ranks 1–3.
- Host checkpoint `/home/ubuntu/trainium_c5c_a2_exact_g2_20260925_trn22_r2/tmp/g2-a86oqytn/out/final.pt` is 1,008,809,893 bytes, SHA-256 `1f3e95b40aba32b468fcff7612ab85302fafcaed139b38995622db8fe9a7e6d4`. It was retrieved to `/home/poojith-devan/Desktop/Trainium-A2-G2-20260925-9XLucr/final.pt`; local size and SHA-256 match host. The 1 GB checkpoint was deliberately **not** committed to Git.
- Read-only postflight at `14:54:40Z`: no G2/train process or `/dev/neuron0` holder; lock free; 53,801,295,872 bytes root disk free. No host cleanup or follow-on launch occurred.

## Preserved raw files

The ten adjacent raw files (`ATTEMPT_ONCE.json`, `driver.log`, `launcher.log`, `exitcode`, `g2.json`, `torchrun.log`, and `rank0.json`–`rank3.json`) are byte-for-byte retrievals: each local SHA-256 was checked against its host original. The host keeps the full job root and checkpoint. This receipt makes no BPB or adoption claim.
