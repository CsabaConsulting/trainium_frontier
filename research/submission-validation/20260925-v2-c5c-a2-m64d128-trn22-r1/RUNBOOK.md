# Conditional exact c5c A2 table arm on trn22

Status: **completed proposal for independent review; no dispatch authority**. This
Implementer package transcribes the human-directed A2 profile from the seeded
research-v2 queue. It does not create a `job.json`, host request, G2 receipt,
smoke receipt, scored result or one-shot authorization. The package edits no
model source, `prepare.py`, evaluator, tokenizer, data, launcher or decision rule.
The exact source is `submissions/research_v2/train.py`, SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.

## Science and comparison

The c5c dense A0 argv has `--no-ngram-ve`. A2 replaces that token with the
queue's one predeclared table-module profile:

```text
--ngram-ve --ngram-ve-layers 1,-1 --ngram-ve-table-mult 64 --ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16 --ngram-ve-clip-exclude
```

Everything else, including `--matrix-lr 0.015`, width 1024, pack factor 32,
sequence length 1024, clip-after-reduce, seed 42, 1,800-second budget and
LNC2/four ranks, stays paired with the A0 control. This tests the complete
table profile versus dense A0. It does not isolate the addition of layer 1;
that attribution requires a valid **same-host trn22 A1** result.

The newest human controller brief directs A2/trn22 alongside A1/trn21, and
the current table-gate brief now records that amendment at its top.
`CURRENT_STATE.md`, `DECISION_LOG.md` and a historical lower paragraph of
the table brief retain an A1-first recommendation from the earlier science
sequence. The explicit newer human parallel priority governs candidate
preparation; the independent Verifier and Supervisor must check that
authority ordering before release. A1 is not an absolute prerequisite for
A2's separate profile screen under that direction, while all exact-size,
native, disk and control gates remain in force. This package itself is
still HOLD.

The queue predicts A2 minus A0 public-20M BPB at `-0.008`, with an 80% interval
`[-0.017,+0.003]`, P(improvement) `0.60` and 2.0 estimated host-hours, giving
EV `0.0024` BPB per host-hour. These are priors. The old-stack layer-1 run's
`-0.0118` step-matched BPB and roughly 5.86% four-rank throughput cost cannot
be relabeled as exact c5c evidence. Old 35c9 A1 failed before step 1; the
existing `m8`/stub receipts and different-source native work cannot prove
the full-size A2 path.

## Four-table geometry and unknown footprint

Depth 6 has VE layers 1, 3 and 5. The `1,-1` layer string resolves to 1 and
5; the trigram rule applies to both first and last VE layers. The exact keys
are `b1`, `t1`, `b5`, `t5`. Each table has `64 × 8192 = 524,288` rows by 128
BF16 values: 128 MiB of parameters. Four tables imply 512 MiB of table
parameters, about 512 MiB of dense BF16 gradients, and about 512 MiB of BF16
RMSProp second moments **per rank** once initialized. The roughly 1.5-GiB
sum per rank excludes temporary tensors, activations, the dense model and its
optimizer, compiler/cache artifacts, checkpoint serialization and all
four-rank effects. It is neither a measured HBM peak nor a root-disk bound.

The trn22 A0 r1 compile consumed 1,500,942,336 bytes of root free space
before optimizer step 0 and crossed the immutable 6.5-GiB floor. It left no
valid control, checkpoint or public20M result. The A0 r2 package's provisional
10-GiB prelaunch requirement is **not** an A2 allowance. Table compile/cache,
temporary, checkpoint and log growth need a fresh conservative projection
after the human expands the exact existing EBS root volume and the guest
filesystem. Leave an explicit unknown-growth reserve above 6.5 GiB, monitor
root free space through compile and training, and abort safely if the floor
is approached. Do not infer cleanup, cache pruning, AWS or breaker authority.

## Exact effective science environment and argv

The final stage must use a clean `env -i` and capture these **effective**
science values exactly; no other `NEURON_COMPETITION_R1_*` variable may be
present:

```text
NEURON_LOGICAL_NC_CONFIG=2
NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16
NEURON_COMPETITION_R1_SEED=42
NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800
```

The `--seed` and `--max-train-seconds` parser defaults read the last two
values. The complete host environment still needs a stage-bound executable
path, home, `PYTHONPATH` and six authenticated cache paths. Until those exact
host bindings and their SHA-256 are recorded in a final immutable request,
there is no runnable launch command. No `--seed`, `--max-train-seconds` or
`--num-steps` override belongs in the full scored argv.

The exact training argv array is in `GATE_CONTRACT.json`. Its rendered
full-run payload, after a clean checkout stages the c5c bytes at root
`train.py`, is:

```text
torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015 --ngram-ve --ngram-ve-layers 1,-1 --ngram-ve-table-mult 64 --ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16 --ngram-ve-clip-exclude --out-dir out/fs-20260925-v2-c5c-a2-m64d128-trn22-r1-full
```

`fs-20260925-v2-c5c-a2-m64d128-trn22-r1-full` and its distinct `-g2`
and `-smoke` labels had no exact-string collision in the controller's ledger,
old log or package paths at preparation. Recheck local and host labels,
checkout/output/log/step paths before staging. The seed queue's
`20260924-A2-...` label is an unexecuted template, not this label.

## Gate order

1. An **independent Verifier** reviews the proposal, exact source/queue/prepare
   and tool hashes, the authority ordering, table geometry, label search and
   no-source-edit diff. The Implementer cannot issue this receipt.
2. The **human** expands and verifies the trn22 root EBS volume and guest
   filesystem, and explicitly acknowledges/resets the trn22 disk breaker
   after the reviewed r1 diagnosis. A changed state field alone does not
   establish human provenance. Fresh read-only inventory must confirm the
   host, filesystem, free bytes, idleness, lock and empty Neuron holders.
3. An Operator stages a **new clean pinned checkout** with the exact c5c
   bytes copied to root `train.py`; rehashes `prepare.py`, tools, proposal and
   stage; verifies the six existing warm-cache identities without cleanup;
   proves all proposed labels/output paths absent; and binds a new A2-specific
   disk/HBM projection, exact complete `env -i` argv and the already-PASSed
   trn22 G1 receipt. G1 reproduced a retained historical checkpoint at
   `0.992968300659387` over 20,971,520 tokens/context 1024. That is
   evaluator calibration, not an A0 control or A2 score.
4. On a memory-qualified idle machine, run the **full-size c5c four-rank
   CPU/G2** payload under external containment and prove quiescence. The
   wrapper must set `TMPDIR` to capacity-verified temporary storage: the G2
   tool creates a temporary checkpoint and can leave descendants on timeout.
   `--out` alone does not move that working directory. The precise G2 tool
   argv, followed by the contract's `a2_argv`, is:

   ```text
   python workflow/tools/preflight_dryrun.py --train-py submissions/research_v2/train.py --prepare-dir . --steps 2 --nproc 4 --micro-rows 2 --micro-tokens 256 --timeout 3600 --trace-rmsprop --out /dev/shm/fs-20260925-v2-c5c-a2-m64d128-trn22-r1-g2.json -- --n-embd 1024 --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015 --ngram-ve --ngram-ve-layers 1,-1 --ngram-ve-table-mult 64 --ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16 --ngram-ve-clip-exclude
   ```

   Require exact c5c and unchanged `prepare.py` hashes, `prepare.stub=false`,
   four PASS ranks, finite losses, ownership plan/summary, keys
   `ngram_embeds.b1/t1/b5/t5.weight`, every table changed, no cross-rank
   mismatch, strict `load_for_eval` checkpoint check, exactly two RMSProp
   group calls and eight per-table update checks **per rank**, finite touched
   gradient medians and recomputation errors, exact untouched rows, and
   `max_abs_err_over_lr <= 0.03`. The aggregate G2 verdict does not enforce
   every trace field; a different Verifier must inspect all rank records.
   G2 imports the real `prepare.py` but uses synthetic data, a sliced CPU
   forward and no Neuron compilation.
5. After a separately reviewed G2 PASS and renewed host/disk preflight, a
   distinct one-shot authorization may permit a **30-step native smoke**
   on c5c/LNC2/four ranks with the same table argv plus
   `--num-steps 30 --out-dir out/fs-20260925-v2-c5c-a2-m64d128-trn22-r1-smoke`.
   Require compiled NKI/eager path integrity, 30 sequential boundaries,
   finite losses and gradients, per-rank table/optimizer agreement, peak HBM
   and disk evidence, model-only strict checkpoint load, causality and
   batch-independent evaluation, full containment and quiescence. If normal
   logs omit rank-level proof, obtain a separately reviewed exact-source
   diagnostic; do not infer equality from rank 0. Smoke is not a BPB result.
6. Only after the native gate, a valid archived **same-host trn22 c5c A0**
   full-budget/seed-42 fresh public20M result, a renewed conservative A2
   disk projection and clean Operator preflight may the Supervisor issue a
   **separate one-shot full-run authorization** bound to a finalized request
   SHA. A0 r1 is consumed and invalid; A0 r2 is still a pending static-held
   package. A failed A2 attempt requires diagnosis and a new label.
7. Retrieve logs, quiescence, checkpoint hash/status, exact env/argv and
   context-1024 fresh public20M evaluation over 20,971,520 tokens. Public2M
   may screen but cannot be a KEEP. The unchanged `decide.py` compares A2
   against the valid same-host A0: step ratio outside `[0.90,1.10]` is
   CONFOUNDED; delta `<= -0.0006` nominates a distinct trn21 replicate.
   Cross-host adoption needs both deltas negative, mean `<= -0.0010`, host
   disagreement `<= 0.0015`, then human acknowledgement. Delta `<= -0.010`
   or score `< 0.975` is HOLD for independent cold replay/audit. A same-host
   A1 is needed before claiming how much of A2's effect comes from layer 1.

No SSH, G2 execution, native work, training, evaluation, AWS/volume action,
breaker reset, protected deletion, staging or dispatch occurred while this
package was prepared.
