# Pending independent G2 then cold full1800 recipe

This packet does **not** authorize a launch, host contact, staging or a queue
change. The Implementer performed none. Supervisor/independent Verifier must
first bind `proposal.json` to one idle host, unique label/private root and the
existing verified contained driver. A second separate host cold replay remains
required before submission acceptance. Never reuse an old S0 assignment.

1. Review the narrow patch and source SHA `4083da77...21409`. Verify unchanged
   trusted prepare SHA `5807c46e...3bbb`, G2 SHA `6a545891...d0b8`, and adopted
   reference6ef7. Use a clean tree and preserve existing submissions/Downloads.
2. On an independently approved idle large-memory CPU host, run existing G2 on
   the exact candidate with **empty train argv**, steps3/ranks4, micro2×256,
   real trusted prepare import and deterministic synthetic inputs. No local
   PyArrow/stub-prepare support, no package install, no concurrent native job.
   Also run exact6ef7 with `cpu_g2.research_reference_argv`, then the existing
   `--compare` command; require both PASS and `identical: true`. This remains
   CPU/no-compile evidence, not a Neuron kernel check.

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NEURON_LOGICAL_NC_CONFIG=2 NEURON_COMPETITION_R1_SEED=42 NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800 <verified-python> <verified-tool>/preflight_dryrun.py --train-py <exact-candidate>/train.py --prepare-dir <trusted-prepare-directory> --steps 3 --nproc 4 --micro-rows 2 --micro-tokens 256 --timeout 3600 --keep --out <unique-receipt-root>/g2-port.json --
```

3. Independent Verifier reviews complete CPU losses/state/ownership/reload and
   trained-forward correctness, not just `PASS` text. Pending native/causality
   checks must be included in the separately reviewed host assignment; this
   packet does not waive them or invent an extra launcher.
4. Operator performs every current protocol preflight. Authenticate host
   idleness/device and advisory lock, STOP/breakers, disk>=6.5GiB plus footprint,
   clean source hashes, exact trusted imported prepare path, permitted shards
   and tokenizer, selected runtime/compiler versions and environment. No
   user/system configuration, instance, storage or protected artifact changes.
5. Bind seven private cold-cache paths listed in proposal. Prove absent then
   empty before launch, inspect effective compiler routing and rule out shared
   or inherited cache paths. Capture manifests/objects/compiler evidence. The
   prior116188 smoke had an empty persistent URL after compilation and remains
   INCONCLUSIVE: repeat of that observation cannot certify this candidate cold.
   If the native stack needs a different cache-routing setting, obtain an
   independently reviewed binding; do not guess or silently alter source/flags.
6. Launch exact candidate via the **existing verified contained driver**, once,
   bare train argv and full-chip LNC2×4. Source declares1800 and seed42; explicitly
   record `training_budget_seconds:1800` and allowed environment declarations
   `NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800`, `...SEED=42`. Clear/reject other
   inherited training overrides. External NEURON_CC_FLAGS stays unset; the
   unchanged source sets optlevel1/BF16 autocast. Do not substitute a step-limited
   smoke, warm cache, shortened budget, new data/evaluator or a new launcher.
7. Preserve an exclusive rank0 append-only boundary log and the runner's actual
   arrival timestamps. Record launch→fifth boundary, capped min(startup,900),
   launch→all-workers exit, charged-through-exit, the source-native startup and
   charged estimates, all exit codes, finite loss trace, distinct boundaries,
   checkpoint hash/meta and postflight quiescence. Unknown remains null.
   The 30s source margin and 15s checkpoint reserve are unchanged. Startup
   >900 consumes paid budget; >870 is a planning warning, not disqualification.
   A valid reserve can leave optimizer time below1800. Any measured charged
   time >1800 invalidates the budget gate despite a false source over_budget.
8. After strict checkpoint and trained-forward/causality gates, quiesce training
   and use the unchanged G1-certified fresh eval command in proposal, context1024,
   exactly20971520 public tokens, batch1, timeout1200. Retrieve exact metrics,
   raw logs, hashes and checkpoint through existing artifact machinery. No
   evaluator changes or inference-time updates. Preserve failure artifacts too.
9. Record under a new experiment label and existing append-only ledger; no
   rewriting old results. Apply fixed1800 rules and all unchanged BPB/two-host/
   big-win gates. Step count is diagnostic only; below60% of same-host control
   or timing spikes/stalls trigger investigation. One replay is not a causal
   improvement claim or submission approval. Leave submission merges/uploads
   and any cleanup/AWS action to the human.
