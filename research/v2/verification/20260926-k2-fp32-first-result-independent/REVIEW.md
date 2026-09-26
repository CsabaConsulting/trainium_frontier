# Independent FP32-first native diagnostic outcome

**AUTHENTICATED FAILED DIAGNOSTIC. No native qualification.**
Exact raw015bacec8fdbe47f1b63f01b160afdc264d780f9 (canonical8307a086),
bundle SHA db4fd0b148fd29930bc4925ba898a8f80108d1bd84c6ad3efc33e8694e15e2c6.
All **31 files /213475 decoded bytes** and their embedded hashes authenticate.

| Route, all four ranks | Forward | Weight gradient |
|---|---|---|
| Direct consumer | Not called | Exact |
| Eager custom autograd | Exact | Exact |
| Compiled custom autograd | Exact | **All zero; mismatch** |
| Compiled ordinary embedding | Exact | Exact |

Every route's seven recorded input tensors are unchanged before/after. These
are host-visible hashes, not values captured inside a compiled VJP; an observer
effect and the precise compiler/runtime cause remain unknown.

The fixture is unique IDs0..255, T256/V256/C128, FP32, seed42; it is the
**first custom-compiled fixture**, without preceding BF16/zero cases. This
reproduces the failure without requiring that earlier ordering. It does not
identify the exact lowering defect, qualify production integration, exercise
duplicate/cancellation coverage, or convert the original20-case failure to PASS.

## Independent numerical reconstruction

The audit does not import Torch/NKI or execute the supplied reference. It
independently reconstructs the fixture using a local seed42 integer stream and
IEEE little-endian binary32 packing. Each gradient is an exact multiple of1/32
in[-8,8]; each token owns one distinct row, so FP32 zero-plus-gradient is exact.

- Expected gradient/upstream: **32709 nonzero elements**, SHA
  dee95c1da97debf6a901e00260bd60a74e7be1992eca64b832c4f6633cccc671.
- Independently generated all-positive-zero gradient: SHA
  fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471.
- The compiled-custom actual hashes equal the latter on all ranks, with0
  nonzeros and independently computed maximum absolute error **8.0**.
- IDs/counts/starts/positions/group-bounds hashes are independently reconstructed.
  Ordinary/eager output comparisons equal their recorded weight-input hash,
  as unique identity-order embedding requires. The Torch-generated random
  weight stream is not independently regenerated; its supplied hash is checked
  for cross-route/pre/post/output consistency, not claimed as local tensor data.

The exact bd2bca7c… checker,2c8295a4… kernel and14572563… reference blobs were
rehashed against frozenfa1a3652, actual launch/staging/rank pins and the original
source commits. No candidate/checker/kernel changes or threshold relaxation.

## Failure and process accounting

Launch00:33:35.568703 UTC; child exit1 at00:34:04.113223, wall
**28.54430987604428 seconds**. Frozen watchdog600+60 seconds; this was a child
failure, not timeout expiration or official-quality training. Outer flock
exit1/terminal DIAGNOSTIC_CHILD_FAILED, complete=false, agree.

All four route receipts and wrapper-return receipts were written with rc1.
That is **not** the same as all workers exiting1: Torchrun subsequently records
rank0/PID3217200 and rank3/PID3217203 exit1, but sends SIGTERM to
rank1/PID3217201 and rank2/PID3217202, which end with **-15**. Those later process
exits do not erase the already retrieved complete comparisons.

Six absent/empty private cache roots were created before the fresh in-lock
ps+neuron-ls guard and child. Per-rank environment records match all bound
variables; their presence does not prove SDK cache use. Original lifetime
device0 lock66305:832589, disk floor, precise argv and source support authenticate.

At00:35:02.263547 UTC explicit postflight records all **nine observed PIDs**
absent,256 unique unfiltered process-identity rows with no related
PID/PPID/PGID/SID or compiler-named process, four-core/LNC2 device idle and the
original lock free. No Operator kill/retry. The postflight scope prose falsely
says timeout was unobserved; the earlier live snapshot **explicitly contains
timeout PID3217195**, and the actual postflight PID list includes it. This stale
sentence is preserved, not silently corrected in raw evidence. No compiler PID
was observed in the bounded snapshots; that is not proof none ran between them.

## Reproduction and scope

```sh
python3 -I -B research/v2/verification/20260926-k2-fp32-first-result-independent/audit.py
python3 -O -I -B research/v2/verification/20260926-k2-fp32-first-result-independent/audit.py
```

Both pass **461 checks**, with byte-identical outputs. Seven negative controls
refuse a falsely exact zero gradient, changed input with an unchanged=true flag,
missing fourth route, nonfinite error, surviving related process, wrong outer
hash and corrupted inner byte count. Audit acceptance uses no assert.

Audit SHA7b8677168b10a1326aba5cf94446b72474ecbfc09dc1282b45ae6904f31213a5;
result SHA7c26f536b102c114881702cbe3c6b128169273477a41de050831242acdb3ef20.
The separate ledger-candidate.json is **not appended** to the authoritative
ledger/state. The prior20-case gate stays failed; native correctness, timing,
quality, adoption and submission remain unqualified. No host, source, queue,
dataset, tensor allocation, retry or new launch authority is part of this audit.
