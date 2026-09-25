# Exact c5c A1 last-layer table gates — pending, no launch authority

This is an Implementer transcription of the human-approved A1 queue arm, not a new model
implementation or an Operator request. It pins **exact** `submissions/research_v2/train.py`
SHA-256 `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
`prepare.py`, the evaluator, `decide.py`, model source and launchers are unchanged. The
`gate-contract.json` state is pending and `dispatch_authorized=false`; the queued profile alone
is never authorization. Both hosts are currently disk-blocked and trn22 has a breaker. No host
action, disk change, breaker reset, artifact deletion or leaderboard submission is authorized.

## One causal factor and source-specific shape

Start with the exact c5c dense A0 flags in `gate-contract.json`. Replace only
`--no-ngram-ve` with:

```text
--ngram-ve --ngram-ve-layers=-1 --ngram-ve-table-mult 64 --ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16 --ngram-ve-clip-exclude
```

At depth 6, VE layers are 1, 3 and 5. The layer set `-1` resolves to 5, so only `b5` and
`t5` are made. At padded vocab 8192, each has `64 * 8192 = 524288` rows by 128 BF16 values:
128 MiB of parameters per table, 256 MiB total, before replicated gradients, RMSProp second
moments and compiler artifacts. A native HBM/disk projection is still **unknown**. Seed 42,
LNC2/world4 and 1800 training seconds are source defaults for the full run; no step/time
override belongs in the scored argv. `NEURON_CC_FLAGS` must include
`--optlevel=1 --auto-cast matmult --auto-cast-type bf16` as in the same-host control.

## Why old receipts do not transfer

- `workflow/evidence/g2-a1m8-impl1.json` exercises **multiplier 8** with synthetic/stub CPU
  data and an earlier IMPL-1 source. It cannot prove mult64 memory, compile, RMSProp trace or
  exact c5c semantics. Other upstream G2 receipts are claims until independently rerun.
- The independently CPU-reviewed `9191e858...` hybrid-repair source and its native-five-step
  lane are different bytes from c5c; their rank ownership and optimizer behavior are evidence
  for design, not a c5c/m64 native PASS. The old 35c9 A1 r2 died before optimizer step 1 with
  an ownership error and has no BPB verdict.
- The c5c no-table G2 comparison proves default-off identity only. It cannot exercise the
  RMSProp branch, the two large tables, Neuron compilation or full-step cost.

## Gate order (one host job at a time, distinct labels)

1. **Independent Verifier** checks this package, source/prepare/tool hashes, exact flags,
   source table/ownership paths, proposal completeness and `test_static.py`. It must not treat
   this Implementer test as its own review. Preserve a PASS/FAIL receipt.
2. After human EBS expansion and separate trn21 G1 calibration, an **Operator** establishes
   a valid same-host c5c A0 control: full 1800 s, seed42, fresh public20M/context1024, clean
   quiescence and archived result. The failed trn22 A0 before step1 is not a control. Bind its
   unique label, commit, score and steps in a separately reviewed run authorization.
3. On an idle, disk-safe trn21, run exact-source **four-rank CPU/G2** under external containment
   and a unique label. The G2 tool itself can leave a process group on timeout; quiescence is a
   separate required postcondition. From the clean repo root, the precise payload is:

```bash
python workflow/tools/preflight_dryrun.py --train-py submissions/research_v2/train.py --prepare-dir . --steps 2 --nproc 4 --timeout 3600 --trace-rmsprop --out /dev/shm/fs-20260925-v2-c5c-a1-m64d128-trn21-r1-g2.json -- --n-embd 1024 --ngram-ve --ngram-ve-layers=-1 --ngram-ve-table-mult 64 --ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16 --ngram-ve-clip-exclude --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015
```

   Require source SHA exactly c5c, unchanged `prepare.py` SHA, `prepare.stub=false`, verdict
   PASS on all four ranks, plan/owner summary present, precisely two table keys (`b5`, `t5`),
   both changed, zero cross-rank mismatches, strict `load_for_eval` checkpoint check, finite
   losses, two RMSProp group calls **per rank** and four table update checks per rank. Every
   touched-gradient median and recomputation error must be finite, every untouched row exact,
   and `max_abs_err_over_lr <= 0.03`. The G2 aggregate does **not** enforce all trace fields:
   independent Verifier must inspect the rank records. G2 uses synthetic data, a sliced CPU
   forward and `--no-compile`; a PASS is not native/Neuron authorization by itself.
4. Only after the exact-size G2 receipt passes, separately authorize a **30-step native
   Neuron smoke** on the same source/shape/topology. Its unique smoke label is in the gate
   contract; append only `--num-steps 30` to the A1 argv. Use the proven contained launcher,
   same compiled flags, source fingerprints and immutable disk floor. Capture per-rank optimizer
   integration/ownership records, 30 completed boundaries, finite losses/gradients, step-time
   and memory/disk peak, table change, rank agreement, strict checkpoint load and causal
   batch-independent eval. If the stock training logs cannot demonstrate rank table agreement,
   obtain a separately reviewed read-only/diagnostic receipt on the exact c5c source; do not
   infer it from a rank-0 checkpoint. The smoke is not a scored BPB result and cannot be a win.
5. Only after a separate Verifier PASS on native smoke, a valid **same-host c5c A0 control**
   and fresh Operator preflight, authorize the full A1 run under a new unique full label, with
   no `--num-steps` or `--max-train-seconds` override. Check disk ≥ immutable 6.5-GiB floor
   **plus** a conservative post-resize projected table/cache/checkpoint footprint; the current
   estimate is not authenticated after the A0 compile disk stop. Require idle cores, clean
   source, one host job, proper lock, exact c5c/prepare hashes, complete containment, finite
   model-only checkpoint, budget evidence, causality and fresh-process public20M/context1024.
6. Archive/hash proposal, commit, exact argv/environment, host, logs, metrics, checkpoint status,
   controls and verdict. A1 needs a same-host step ratio in `[0.90,1.10]`; otherwise it is
   CONFOUNDED. Delta ≤−0.0006 nominates a distinct trn22 replicate after that host's human
   breaker acknowledgement and same-host c5c control. Two negative deltas, mean ≤−0.0010,
   disagreement ≤0.0015 become ADOPT_PENDING_HUMAN_ACK only. A large win (≤−0.010 delta or
   BPB <0.975) stays HOLD for an independent cold replay/audit. `decide.py` is unchanged.

Any failed gate stops this package. A retry needs a fresh label and written diagnosis; it
never relaunches a failed label or bypasses the disk/breaker state.
