# K2 namespace native r3 — independent terminal audit

Disposition: **PARKED_COMPILER_FAILURE**. Raw authentication passes; the native component gate does not. No further retry, repair, full training, evaluation, quality/adoption or launch authority is created.

## Frozen evidence and reproduction

This local, stdlib-only review reads immutable Git objects: terminal archive `7c40e4f5` (canonical `f4be8dd7`) and live launch `95666f13` (canonical `e46774b0`), under `research/v2/operator/20260926-trn22-k2-namespace-native/`. It authenticates all 22 terminal text/byte/SHA records (70,941 bytes) and all 16 live records (49,775 bytes), the exact frozen binding `0abd35a071508679e4d184ca2aeb3ab1295cba3f`, candidate/source support, launch argv/environment, per-rank environment and observed PIDs, failure coverage, and archived postflight. No raw evidence or candidate was edited.

Run from the repository root:

```sh
python research/v2/verification/20260926-k2-namespace-native-result-independent/audit.py --repo /home/poojith-devan/Desktop/Trainium
python -O research/v2/verification/20260926-k2-namespace-native-result-independent/audit.py --repo /home/poojith-devan/Desktop/Trainium
```

Both runs exited 0 with **192 explicit checks**, identical JSON output. The script contains no tensor imports, network/host calls, writes or candidate execution. This is verification of recorded evidence, not a native reproduction or new source review.

## What actually failed

The exact source is `d0f0d7fbdbe07ddee39680ee1a2b79fdfac388ca`, consumer SHA `4939265bf73b3d5f42276e72e48f022df338226a40a9be5b79ed63bde6a4f5b8`; checker `89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850` and reference `14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f` are unchanged. Remote staged hashes, actual launch pins, all four pre-import rank records and rank 0's terminal pins agree with the frozen Git bytes.

Native child launch was 2026-09-25 23:06:48.856645 UTC. Child exit was 1 at 23:07:14.675602; driver terminal was 23:07:14.675852. Recorded monotonic child duration is **25.8187616559444 seconds**, consistent within 0.2 ms with the recorded UTC child interval, and well below the 600+60-second diagnostic bound. Child, driver and outer flock all failed with exit 1; this was not a timeout or official quality run.

Rank 0's first `cross_tile_large` BF16 case reaches `check_native.py:133`, the consumer call, and detects an asynchronous compiler failure at the immediately following synchronization (`:134`):

> [INTERNAL_ERROR] [NCC_IXGM002] Expected function sg0000 in subgraph 0 to have 9 basic blocks, but on core 1 it has 1 basic blocks

That text is present in `receipts/rank0.json.error` and `diagnostic.log`. It is a later failure than the original `nl.bfloat16` name-resolution error; the repair does **not** establish successful native compilation or numerical correctness. The precise kernel/compiler root cause is not established by this audit.

The separately retrieved 6,461-byte compiler log records NeuronX compiler `2.26.6360.0+6f180f47`, Trn2/LNC2 invocation, HLO conversion and tensorization. **It does not itself contain the NCC_IXGM002 fatal text.** Its later timestamp and absent rank attribution do not justify assigning it as rank 0's exact failing compilation. The 1,168-byte MLIR is inventory-only, not downloaded or inspected here. Do not infer a compiler fix from these records.

## Missing coverage is a failed gate

Only rank 0 has a checker terminal receipt: `REFUSE_OR_UNSUPPORTED`, first case still `STARTED`. Its sole completed check is the finite/exact upstream BF16 device roundtrip (384 nonzero values, matching SHA); this checks input transport, **not the consumer**. `direct_consumer`, repeat-call, gradient, FP32 cases and timings were not reached in that receipt. `timings` is empty and no first-call duration was finalized.

Torchrun records SIGTERM/-15 for worker PIDs 3184334, 3184335 and 3184336 after rank 0's exit 1. Ranks 1–3 have pre-import environment evidence but no terminal checker or wrapper-exit files. Their exact progress is unknown; they are neither numerical passes nor independently demonstrated instances of the same compiler error. No `completion-check.json` exists. The strict zero-tolerance/identical-byte criterion is unchanged, not waived.

Therefore there is **no native consumer exactness, component timing, whole-step speedup, model-gradient, BPB or adoption evidence**. Preserve the previous refusal and this distinct compiler failure. This completes the single diagnosed retry; the route remains parked.

## Guard and cleanup evidence

The actual launch uses the same `/home/ubuntu/.cache/trainium_freshstart/device0.lock`, inode 832589/device 66305. The authenticated in-lock `ps` and `neuron-ls` checks returned clean before launch; the live snapshot shows all four workers on the device and the original lock held. All six cache roots were recorded absent/empty at creation and exactly matched the bound rank environments, with compiler URL=DIR. This does not independently establish every SDK cache's use or a performance-relevant cold comparison.

At 23:08:34.750819, the explicit postflight covers all nine observed dispatch/flock/driver/timeout/torchrun/worker PIDs and records them absent; its PPID/PGID/SID/compiler-command query is empty, the device is idle and the original lock is free. The 23:09:05 retrieval snapshot also records no matching jobs or device processes. These are authenticated historical snapshots, not permission to assume the host remains idle. No manual kill/cleanup occurred in the archived postflight. The subsequently assigned clock experiment is separate and untouched by this audit.

Exact top-level hashes and all support pins are in `checks-normal.json` and `checks-optimized.json`. Terminal bundle SHA: `0467cd4d6bad1ee3a0578033a474273ea70b788c02679c2e63c6819555021c9b`.
