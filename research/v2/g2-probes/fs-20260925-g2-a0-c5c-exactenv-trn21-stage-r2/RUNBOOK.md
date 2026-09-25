# trn21 c5c A0 exact-source four-rank CPU G2 — stage-r2 HOLD draft

This is an Implementer-prepared **offline review target**, not an Operator
request or assignment. `GATE_CONTRACT.json` has
`cpu_g2_dispatch_authorized=false`; direct execution of `build_g2_argv.py`
exits `DISPATCH HOLD`. No SSH, host CPU probe, native training, evaluation,
AWS or breaker action occurred while preparing it.

The selected science remains the unchanged trn21 A0 control under label
`fs-20260925-v2-c5c-a0-control-trn21-r1`: e65 commit
`e65a3097a5b00f6178188a935bb2acb9a5042c6c`, root `train.py` SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
and unmodified `prepare.py` SHA-256
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The 15-element Track-A training suffix and four effective environment values
are exact matches to the conditional Supervisor-ratified proposal. A CPU G2
is a numerical/safety gate; it does not consume the science label or measure
BPB.

## Critical source dependency

The **only** candidate source checkout is
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`
on root ext4, owner-private `0700`. The old r1 checkout, old `/dev/shm`
capsule, and r1 G2 root are not fallbacks. The three persistent ext4 files
were independently transfer-reviewed, and the frozen extraction code was
independently implementation-reviewed, but no checkout result has passed.
The earlier disabled checkout-request draft `5fc762f8` / SHA-256
`69e570f9...` received an independent **HOLD**, not approval. Its full
receipt SHA-256 `8656f59af4ca658d558525b69ed4c2a3b8d54a382ccc88d8603f8c20a0bb1678`
identifies the blocked lineage. A repaired request, separate exact-request
review, fresh host preflight, one-attempt checkout-only assignment, complete
Operator postflight, and independent checkout-result PASS must precede *any*
CPU G2 assignment. The repaired request/result hashes are deliberately null
here; the eventual final CPU request must rebind them. Never infer checkout
or G2 authority from the earlier transfer-only assignment.

The two CPU tools are already persisted on ext4, outside the clean checkout:
`/home/ubuntu/trainium-v2-g2-preflight-dryrun-20260925-trn21-stage-r2.py`
(23,550 B, SHA-256 `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`,
inode 294946) and
`/home/ubuntu/trainium-v2-g2-contained-20260925-trn21-stage-r2.py`
(9,185 B, SHA-256 `0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d`,
inode 296647). Both were final-session persistence-verified as single-link,
UID-1000, mode-0600 regular files on device 66305. A future Operator must
recheck those exact identities with no symlink following before and after G2;
do not transfer replacements or copy either tool into the science checkout.

## Exact CPU-only shape for separate review

`build_g2_argv.py` deterministically constructs the exact 64-word contained
launch argv, and the wait, failure-only stop, and unconditional quiescence
argv. Their SHA-256-of-compact-JSON-array digests are respectively
`1810cdcde05dbda9dea063f160ceb25563d01bc2eb61b70623163d945e432eeb`,
`3a416123b7264585e23fbb3fcc75a1501a993d26fac9fe0580f7f47b2516c380`,
`9216b16faa88664f8d91bf80752351af25287134fb5ad635f06f960b55e17e10`,
and `c959acbee53e6743cf7f2a300d8d8ddb4477c0514c8313aa4596d9d1545e0edf`.
They are **review bindings, not commands to run now**.

The one new CPU label is
`fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2`. Only a later assigned
Operator may create its previously absent, owner-private `0700` ext4 scratch
root `/home/ubuntu/trainium-v2-g2-20260925-trn21-stage-r2` after a fresh
absence check. This root is the step-log parent and `TMPDIR`; the step-log
file itself must be absent before launch. The root, receipt, rank logs,
contained process directory and synthetic checkpoint stay *outside* the
clean e65 checkout. The reviewed preflight tool receives `--steps 3 --nproc
4 --timeout 900 --keep` and appends only `--device-type cpu --no-compile
--num-steps 3 --out-dir <g2-temp-workdir>/out` to the exact science argv.
Synthetic data are required; `--real-data`, native Neuron work and public
evaluation are forbidden. No science output/step/log path may appear.

Before even this CPU-only action, require an independent static PASS on the
**final** exact G2 package, a Supervisor CPU-only one-attempt assignment,
fresh root free space at least 10,737,418,240 B, intact source/Git/tool
hashes and inodes, preserved checkpoint/cache identities, a clean host with
no lock/Neuron/process holder, absent G2 root/label and absent science
output/log/steps, and untripped local/global breaker and STOP gates.

After an assigned attempt, archive the full contained launch/wait/optional
stop/quiescence transports, `c5c-g2.json`, all four rank reports, torchrun
log, step log, SHA-256 manifest, full failure if any, and a separate complete
read-only postflight. Quiescence is mandatory even after G2 failure; a
survivor/holder is a breaker investigation. A CPU PASS requires four rank
PASS reports, three finite losses per rank, exact source/prepare/argv and
effective environment, no cross-rank parameter mismatch, the checkpoint
`load_for_eval` check, and step lines `0,1,2` in order. A *different*
Verifier must compare the retrieved result with the pinned prior c5c CPU
anchor (SHA-256 `17ce1ec8...`) using the tool's `--compare`; `identical:
true` is a test, not an assumed cross-host property.

On any failure, HOLD and preserve the capsule, checkout, CPU tools,
protected checkpoints/caches, and any partial G2 root. No cleanup,
overwrite, same-label retry, cold-cache fallback, science G2 bypass, native
training or evaluation is authorized. Even a fully reviewed CPU PASS leaves
native A0 behind its own exact request, live monitor/observer, disk and
one-attempt authority gates.

Local Implementer check only (static/synthetic; no CPU training):

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2/test_g2_argv.py -v
```
