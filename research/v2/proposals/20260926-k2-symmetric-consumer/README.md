# K2 symmetric feature consumer — disabled source packet

Proposal-first commit: bafeb353d0afd3f3ca693423b79f0bee8fdf3e6d. New branch codex/20260926-k2-symmetric-consumer. Candidate exists only under research/v2/lanes/k2-symmetric-consumer; original resident-consumer files, every REFUSE, production train.py and host packets are untouched.

Kernel SHA256: 2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae.
Unchanged checker: 89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850.
Unchanged scalar oracle: 14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f.

## Narrow source difference

The factory now returns the JIT callable selected with [2]. Each program computes feature_first=program_id(0)*64; the kernel checks two programs, allocates half-width resident tiles, adds that compile-time offset to the full-base indirect gradient access pattern and stores only its 64-column output slice. Both retain exactly the old groups0/1 and runtime maxima loads, inactive select, per-occurrence FP32 addition and FP32 output. This is distinct multicore ownership, not merely re-arming the old one-program path.

The old cache-key error explicitly contains [1]; its compiler error is preserved verbatim. That narrows the factual launch observation but does not prove why lowering failed or that this new design compiles. Detailed contract and primary-source links are in PROPOSAL.md.

## Own bounded checks, not an independent verdict

Seven stdlib tests pass normal and optimized Python, each including27 strict full-output synthetic cases: all20 original dtype/case combinations plus7 asymmetric-count/feature-signature cases. They verify32768 exactly-once stores, 16384 per program, stable occurrence sequence, identical control schedules, retained grouped-sum counterexamples, nonfinite/unstable-metadata refusal, explicit grid selection through a fake JIT binding and default-disabled no-native-import behavior. Source AST checks tie program use to gather/store offset; they are NOT a native interpreter or compiler test.

The unchanged check_native.py disabled invocation returned DISABLED without Torch/NKI. No tensor/model/SDK import or native action occurred. The final test process has hard virtual-address-space128MiB; current-exec /proc peaks were34496/37328kB. Raw rusage maxRSS additionally reported34320/186136KiB; the larger inherited/historical metric is preserved, not substituted with a claimed measured kernel allocation. Original pre-limit test outputs are also retained in IMPLEMENTER_RAW_TESTS.json.

Replay locally, sequentially:
```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python3 -B research/v2/lanes/k2-symmetric-consumer/test_symmetric.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python3 -O -B research/v2/lanes/k2-symmetric-consumer/test_symmetric.py
```

Source-declared accumulator32KiB/program; accumulator/value/updated/selected four tiles128KiB/program plus small metadata. Compiler liveness, actual SBUF allocation and throughput remain unknown. Metadata and token-address work are duplicated across programs; narrower gather/store transactions may be slower. No output state HBM reload or reduction is introduced. No production routing implementation exists.

## Future review and native falsifier — not a launch binding

Supply a genuinely fresh isolated reviewer ONLY review-input/kernel-full-context.diff, neutral lane-spec.md, exact new kernel/checker/oracle files and review-input/original-native raw receipts. Include manifest for byte authentication. Do not send this README, implementation tests, test conclusions or proposal reasoning as reviewer guidance. The old raw rank0, terminal and compiler log are byte-exact extractions; absent ranks1–3 remain absent.

After separate fresh review and separately reviewed Operator binding, the existing four-rank LNC2 checker can be used unchanged with child arguments:
```text
check_native.py --enable-experimental --label fs-20260926-k2-symmetric-consumer-trn22-r1 --out-dir /home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-k2-symmetric-consumer-trn22-r1-packet/receipts --expected-kernel-sha256 2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae --expected-reference-sha256 14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f --timing-repeats 0
```
This is a proposed child argument contract, NOT a runnable approved remote command or host reservation. The future binding must use the original shared lifetime flock, fresh in-lock ps+neuron-ls, exact pins, fresh diagnostic caches, one synchronous600+60 child, per-rank raw environment/logs/graph evidence and explicit PID postflight. None was created, staged or executed here.

Acceptance for this tiny falsifier requires all4 rank receipts, every20 cases, all150 full-byte comparisons per rank and no nonfinite/error/incomplete status. Exit0, compile completion, one rank or CPU emulation alone is insufficient. Custom wrapper backward is tiny diagnostic-only, not production model integration. No whole-step timing, BPB, full training or source adoption claim follows. All earlier failures remain controlling historical evidence.

Next action: independent review of this new source package. Undo: do not assign or import it. No queue, cache, checkpoint, machine or official scoring change has occurred.
