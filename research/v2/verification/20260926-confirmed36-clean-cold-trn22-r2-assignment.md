# One diagnosed clean cold retry: trn22 r2

Supervisor release, 2026-09-26 04:14 UTC. Only Operator `/root/trn22_queue_operator` may dispatch once.

- Label: `fs-20260926-confirmed36-clean-cold-trn22-r2`.
- Source: `e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43` (unchanged clean candidate; actual CPU G2 and ordinary checkpoint helper independently accepted).
- Packet `c1384be93c2d314b49dd574f4b936a732f0b1a26`; driver `3d378afe134eeb46e813b013aaaf55d42743b25f6175e80cf3087975aecc269d`; binding `932d545020b822acd00deb3b5c93b68a421187a6ec424e6853428180ac541261`.
- Transport `b7a90d94cb763c1caf7ef7688827e10eadf43554`, SHA `beb290532ff958d6021161d1beb933a36fbbd1fa6cf86cc8512819a0e2555607`.
- Root: `/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-cold-trn22-r2-packet`.

Root read the driver, transport and their tests and replayed 77 binding checks and 45 transport checks in normal and optimized Python. The actual pinned rehearsal harness containment predicate rejects the old root and accepts the corrected root. The remote transport must additionally check resolved containment before creating the new root. No harness, prepare.py, scientific source, launch shape, or submission command change is authorized. The only diagnostic argument is `--num-steps 5`.

The r1 failure is preserved: `AssertionError` at `assert sandbox.is_relative_to(root / "out")`, before torchrun. Root authenticated all 20 embedded files and explicit 253-row postflight census: the three actually observed parent IDs are absent, no related lineage/device job remains, and original lock 66305:832589 is free. Unobserved short-lived children are not claimed as observed. This is the single diagnosed retry; a second failure is parked, not automatically retried.

Operator must independently enforce exact remote SHA, source/prepare/harness pins, fresh ps + neuron-ls + fuser and STOP/breaker checks, disk >=13.5 GiB before staging, all six unique cold caches initially absent, old queues stopped, one job under the original lifetime flock. Preserve failed r1 artifacts. Run the bounded five-step cold diagnostic only. Collect live rank/parent/starttime/PPID/PGID/SID evidence, actual uncapped start-to-fifth timing and compiler errors, then final complete lineage/device/lock quiescence on every outcome.

No eval, automatic full run, adoption, upload, AWS action or cleanup. Full cold replay remains separately gated on successful four-rank cold startup <=900 seconds and an independently accepted terminal/postflight. Target <=870 seconds. The disabled corrected full-r2 packet may be prepared locally during this diagnostic.
