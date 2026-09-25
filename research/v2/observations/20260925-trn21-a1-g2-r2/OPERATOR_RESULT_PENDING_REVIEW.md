# trn21 A1 exact-size CPU G2 — Operator result, pending independent numeric review

- One-use label: `fs-20260925-c5c-a1-exact-g2-trn21-r2`; assignment `5a10c117` / SHA-256 `a34b15dfe9c9dfd20d3cca1b656d79f0540a82e2266ea0b8393a7f70101615ff`.
- Host run root: `/home/ubuntu/trainium_c5c_a1_exact_g2_20260925_trn21_r2` on trn21. Source SHA-256 `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`; G2 tool SHA-256 `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`; trusted `prepare.py` SHA-256 `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Scope: CPU/gloo synthetic two-step four-rank check only. No Neuron training, public evaluation, model adoption, or submission occurred under this assignment.

## Chronology and one-call accounting

- Fresh trn21 preflight passed at 2026-09-25 14:52:08 UTC (idle, no Neuron/device/lock holder or STOP/breaker, 54.23 GB root free, 131.46 GB MemAvailable, exact unique destination absent). A private host root was created and exact source/tool bytes transferred and checked.
- The first local `operator_a1_g2.py launch` invocation at 14:52:57 UTC failed in its **pre-marker preflight** with `AssertionError: TMPDIR_NOT_EMPTY` (exit 1; stderr 143 bytes, SHA-256 `4ef394cad3f450b63888790f579214fbbc5940e6ca4afe09ac1a54a59e1091bc`). No marker, launcher, result, log, or worker process existed. Read-only inspection proved the sole TMPDIR entry was the empty private `torchinductor_ubuntu/` created by the probe's own Torch import; source/tool/prepare hashes and all capacity/safety gates remained valid. The Supervisor authorized one continuation of this same unmarked staged attempt, changing only that operational guard. Nothing was deleted or retried after a marker.
- Corrected poststage preflight passed at 14:55:56 UTC. The **single actual G2 attempt** was marked and launched at 14:56:09.255 UTC (20:26:09 IST). Exact environment and argv are preserved in `raw-host-root/launch.json`. The tool started at 14:56:09.274 UTC and ended at 14:56:34.276 UTC; detached monitor exit code `0`, disk floor not breached, tool elapsed 21.4 seconds.
- Aggregate `g2.json`: `verdict=PASS`, `torchrun_rc=0`, `timed_out=false`, `nproc=4`, `steps=2`. Four raw rank files each say `status=PASS`. These are **raw Operator observations, not the independent numeric PASS** required by the packet. No BPB was measured.
- Distinct read-only postflight at 15:00:23 UTC found no matching training/G2 process, no Neuron process, no device or host-lock holder, no STOP; root free `53,486,759,936` bytes. Receipt: `../20260925-dual-host-readonly-operator/OPERATOR_RAW_TRN21_READONLY_20260925T150018Z.json`.

## Artifact preservation

All 14 host regular files were retrieved byte-for-byte to `raw-host-root/`; host and local SHA-256 values agree 14/14 (see `RAW_SHA256SUMS.txt`). The raw four-rank log and rank JSONs, aggregate JSON, launch/marker/monitor records, source/tool copies, and model-only checkpoint remain on the host. The 739,322,371-byte checkpoint was also retrieved locally at `raw-host-root/tmp/g2-sg1ufc20/out/final.pt`, SHA-256 `c548b0863bb577de90c5e3efef3eecf4006ef7951561417f5b2189fcf3493a67`; it is intentionally **not committed to Git** because of its size. The small raw artifacts and their hashes are committed for independent review. No host cleanup or subsequent launch was performed.

The next action belongs to a different Verifier: check all four raw rank records, losses, common plan, checkpoint load, table changes, RMS call/check ordering and exact numeric bounds before any native/full table run is considered.
