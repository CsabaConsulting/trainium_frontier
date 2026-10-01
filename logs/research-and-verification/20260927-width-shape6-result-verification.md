# Independent width768 and 6×1024 result authentication

Verdict: `PASS_ARCHIVE_AUTHENTICATION_LOSS_NO_ADOPTION` for both candidates. This is a read-only result review, not launch, adoption, current machine admission, or submission authority.

| Candidate | Host | Public20M/context1024 BPB | Same-host control BPB | Delta (candidate − control) |
| --- | --- | ---: | ---: | ---: |
| Width768, source bba9c440 | trn22 | 1.0034097788590979 | 0.9775917108277573 (2931b207) | +0.02581806803134057 |
| 6×1024, source 2ec8322d | trn21 | 0.993288361068266 | 0.9782855553304796 (c2393540) | +0.015002805737786451 |

The independent stdlib reference authenticates all 40 UTF-8 member hashes and byte counts in each of four retrieved archives (160 total), exact package source and launch command, completed proposal binding, clean exact committed prelaunch snapshot, seed42, four LNC2 ranks, context1024/pack32, uncapped1800-second training, six initially empty private training-cache directories, zero run/eval exits, final checkpoint receipts, and archived postflight idle/lock identity. It does not import the operator's result parser.

Raw evaluator `total_nats / (ln(2) * num_bytes)` reproduces all four BPB values. Each fresh evaluator consumed20,971,520 tokens and83,462,186 bytes within its1200-second evaluator limit; evaluator source/hash/checkpoint path agree with its run. A fresh process does not imply cold evaluator caches.

Width768:3411 completed steps; cold startup549.7902538080234s; startup charge0; charged through process exit1788.334435072029s. Source/log-format inversion recovers677,642,240 actual tokens:1652 accumulation-1 steps and1759 accumulation-2 steps. All346 printed throughput rows agree with the phase/token accounting.

6×1024:3272 completed steps; cold startup533.4131320639281s; startup charge0; charged through process exit1788.7318382450612s. Source/log-format inversion recovers647,888,896 actual tokens:1601 accumulation-1 steps and1671 accumulation-2 steps. All332 printed throughput rows agree. These are source/log-derived totals, not locally loaded checkpoint metadata or early-phase extrapolations.

Width package is exactly c239 plus `N_EMBD=768`. Its trn22 comparator2931 differs from c239 only in `--batch-ramp` defaultFalse versusTrue; comparator's actual launch explicitly enabled `--batch-ramp`, so that difference is accounted for. No compiled-graph identity is claimed. Shape6 source is exactly the declared docstring/depth/attention-layer/six-layer-guard edits; its trn21 comparator is exact c239. The candidate packages match the reviewed local source snapshots byte-for-byte.

Checkpoint scope: matching positive byte count/SHA256 before and after evaluation and in training timing. The binary checkpoints are not embedded in these local bundles, so this review does not claim a local binary reload. Single seed/single run results are losses against their designated controls, not a cross-host significance claim. No ledger, queue, source, or host state was modified.

Exact hashes, raw scores, config, timing, token accounting, and the independently executed reference hash are in the adjacent JSON receipt.
