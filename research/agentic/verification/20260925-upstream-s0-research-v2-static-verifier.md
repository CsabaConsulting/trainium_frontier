# Independent static verification: upstream S0 and research-v2

Date: 2026-09-25  
Role: independent Verifier  
Canonical report base: `33774742dc49de7a5d6f5fd539c1530430306763`  
External source checkout: `/home/poojith-devan/Documents/ChatGPT/trainium-research-context-audit-20260925`  
External source HEAD: `4bd08d0f199c0917a65a332c416a6577c925520f`

Canonical installation was independently observed at
`8b9fc841ed817eac263c7087c3bce7c9f008bda1`, whose direct parent is the report base. Git-object
hashing confirmed that its two installed bases and two installed products have the exact identities
listed below.

## Verdict

**STATIC PASS WITH REQUIRED INDEPENDENT EXECUTION GATES.** I found no static correctness blocker to:

1. running an A0 control on the exact research-v2 bytes with all three new mechanisms off; or
2. advancing S0 to its independent CPU/G2 and Neuron gates.

This is not a G2 receipt, a Neuron result, a run authorization, or a submission approval. The upstream
CPU receipts were read as claims and were not accepted as independent proof. No host, training,
evaluation, AWS, or submission action was performed during this review.

## Authenticated source identities

| Object | SHA-256 |
|---|---|
| exact research base `04-train-research-35c9.py` | `35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2` |
| research-v2 `workflow/proposals/research-v2/train.py` | `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` |
| exact clean base `05-train-submission-clean-116188.py` | `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d` |
| S0 `workflow/candidates/S0/train.py` | `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb` |
| G2 harness `workflow/tools/preflight_dryrun.py` | `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8` |
| verifier role contract | `918dfd80880d2a92958a30a8fca60d7eed220db6e0c61f43890df69d9578073e` |

The S0 transform reproduced `ad223cdc...` byte-for-byte from `116188cf...`. The research-v2 transform
reproduced `c5c410af...` byte-for-byte from `35c9ef79...`, including asserted intermediate IMPL-1
SHA-256 `9e5dce344674b8217d8749ca3e2ce8c1961a73a9c10c1c6759f645a62ed66783`.

## S0 review

I read the complete 530-line diff against `116188cf...` and the resulting training, checkpoint, and
evaluation paths.

- The bare launch is compatible with argparse and fixes the intended science in-file: sequence length
  1024, width 1024, pack factor 32, matrix LR 0.015, total batch 131072, seed 42, and 1800 seconds.
- Training imports only the trusted training-side `prepare` APIs and uses `make_dataloader(...,
  'train', ...)`. The in-process public validation and causality paths and their imports are absent.
- The same `GPT`, `Block`, attention, and MLP classes are used for training and `load_for_eval`.
  There is no `self.training` arithmetic branch, monkeypatch, batch/rank special case, or collective in
  evaluation forward.
- `load_for_eval` uses `torch.load(..., weights_only=True)`, reconstructs the saved config, widens only
  non-persistent rotary buffers, strict-loads `model_state`, moves to the requested device, enters eval
  mode, and freezes all parameters.
- The MLP rewrite flattens only the token dimensions and applies the same two linear operations and
  ReLU-squared math. CPU equality is plausible from source, but the Neuron execution and compilation
  path remains an execution gate.
- Removed optimizer serialization is not a production regression: the final checkpoint is model-only,
  and no optimizer resume caller exists in this run path.

No static blocker was found. S0 still requires the independent G2 lockstep below, one Neuron lockstep,
causality/batch-independence checks on the actual device checkpoint, and the already-required two cold
public-20M replays. I make no native-numerics or BPB claim here.

## research-v2 review

I read the complete 150-line diff against `35c9ef79...`, the IMPL-1 patch, and the live optimizer,
gradient synchronization, loader, checkpoint, and eval call sites.

### Default-off A0 path

All additions are bypassed under the A0 argv:

- `NGRAM_VE_FREEZE_EPOCH=0`, `FRESH_TAIL_FRAC=0.0`, and `REQUEUE_IDENTITY=False`;
- the original trusted `make_dataloader` branch is selected;
- the freeze block is not entered;
- there is no RMSProp group with `--no-ngram-ve`, so both added RMSProp branches are false; and
- the original AdamW/Muon ownership plan and update code execute unchanged.

Thus no source-level blocker was found for an A0 control. Independent three-step G2 lockstep against
exact 35c9 remains mandatory; upstream receipts do not satisfy that gate.

### IMPL-1 replicated RMSProp production path

The two-hunk repair is internally consistent:

- known `kind='rmsprop'` groups are excluded from owner planning; unknown kinds still fail;
- the source optimizer retains the complete replicated RMSProp group on every rank;
- the normal training loop AVG-reduces all gradients before clipping and stepping for the approved
  clip-after-reduce table argv;
- the RMSProp group is separately clipped and is stepped exactly once through `_rmsprop_step(source)`
  when `update_adamw` is true;
- RMSProp parameters are absent from the owned send/receive exchange, while AdamW/Muon ownership is
  unchanged; and
- final checkpoints are model-only. Hybrid optimizer resume is not used and is not made a gate here.

This clears static review only. The exact full-size A2 trace below must independently show calls,
updates, cross-rank agreement, checkpoint strict-load, and finite losses before any native table smoke.

### IMPL-2 and IMPL-3

- Freeze is default-off, rejects a negative epoch and the incompatible flat-gradient-buffer mode, and
  drops table gradients before gradient reduction and optimizer step. The source documents that its
  buffered epoch label may lead consumption by about 512 documents. This feature is not needed for A0.
- Fresh-tail is default-off, bounds its fraction to `[0, 0.5]`, rejects conflicts with the other data
  orders, preserves the per-rank row-group partition, and rejects an empty tail. Its in-file loader is
  not assumed identical to the organizer loader; any fresh-tail arm must use the paired
  `--requeue-identity` control and a real-shard order receipt. This feature is not needed for A0.

## Exact minimal independent G2 commands

Run these only on an idle host with the canonical install checked out cleanly. Each invocation must be
the payload of a distinct external contained label matching the receipt name, under the host's one-run
lock, with its output on tmpfs. Do not run them concurrently. The G2 harness starts a new session but,
on its own timeout, does not terminate the whole process group. Therefore the outer containment layer
must TERM/KILL every tagged descendant and prove quiescence before the receipt is accepted.

### A0: exact 35c9 versus research-v2, three steps

```bash
python workflow/tools/preflight_dryrun.py --train-py 04-train-research-35c9.py --prepare-dir . --steps 3 --nproc 4 --timeout 3600 --out /dev/shm/fs-20260925-g2-a0-35c9-trn22-r1.json -- --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015
python workflow/tools/preflight_dryrun.py --train-py submissions/research_v2/train.py --prepare-dir . --steps 3 --nproc 4 --timeout 3600 --out /dev/shm/fs-20260925-g2-a0-research-v2-trn22-r1.json -- --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015
python workflow/tools/preflight_dryrun.py --compare /dev/shm/fs-20260925-g2-a0-35c9-trn22-r1.json /dev/shm/fs-20260925-g2-a0-research-v2-trn22-r1.json > /dev/shm/fs-20260925-g2-a0-35c9-vs-research-v2-trn22-r1.json
```

Both receipts must be `PASS` with the exact source hashes above, four `PASS` ranks, finite losses,
zero cross-rank mismatches, strict checkpoint/load-for-eval success, and equal plan hashes. The compare
receipt must say `identical: true`, with identical losses and no parameter digest differences.

### S0: exact 116188 versus bare S0, three steps

```bash
python workflow/tools/preflight_dryrun.py --train-py 05-train-submission-clean-116188.py --prepare-dir . --steps 3 --nproc 4 --timeout 3600 --out /dev/shm/fs-20260925-g2-s0-116188-trn22-r1.json -- --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
python workflow/tools/preflight_dryrun.py --train-py submissions/candidates/S0/train.py --prepare-dir . --steps 3 --nproc 4 --timeout 3600 --out /dev/shm/fs-20260925-g2-s0-bare-trn22-r1.json --
python workflow/tools/preflight_dryrun.py --compare /dev/shm/fs-20260925-g2-s0-116188-trn22-r1.json /dev/shm/fs-20260925-g2-s0-bare-trn22-r1.json > /dev/shm/fs-20260925-g2-s0-116188-vs-bare-trn22-r1.json
```

Apply the same receipt gates. The compare must report `identical: true` across the ownership plan,
all rank losses, state keys, and every compared parameter digest. This is still CPU/G2 evidence: the
harness appends `--no-compile` and cannot establish Neuron compilation or NKI equality.

### Full-size A2 replicated-RMSProp trace, two steps

Run only after the four baseline/S0 receipts and two compare receipts above are sealed:

```bash
python workflow/tools/preflight_dryrun.py --train-py submissions/research_v2/train.py --prepare-dir . --steps 2 --nproc 4 --timeout 3600 --trace-rmsprop --out /dev/shm/fs-20260925-g2-a2m64-research-v2-trn22-r1.json -- --n-embd 1024 --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015 --ngram-ve --ngram-ve-layers=1,-1 --ngram-ve-table-mult 64 --ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16 --ngram-ve-clip-exclude
```

The exact train hash must be `c5c410af...`; every rank and the aggregate must be `PASS`; all four
table tensors must change; cross-rank mismatches must be empty; the checkpoint check must pass; and
each rank must record exactly two RMSProp group calls and eight per-table checks (four tables times two
steps). Every touched gradient/update diagnostic must be finite, every untouched row must remain exact,
and `max_abs_err_over_lr` must be at most `0.03` (predeclared BF16 storage/arithmetic tolerance). The
G2 aggregate does not enforce these trace-field conditions itself, so the independent receipt reviewer
must inspect them explicitly.

## Checks performed locally

- Read `workflow/AGENTS.md`, `START_HERE.md`, and `roles/verifier.md`, then the full relevant source,
  transforms, diffs, proposal/evidence claims, training/eval/load/optimizer/data call sites.
- Recomputed all six source/tool hashes in the identity table.
- Rebuilt S0 and research-v2 in memory/temp output and proved exact byte equality with their committed
  products.
- Parsed the four Python candidates/bases with `ast.parse`.
- Did not import Torch, allocate model tensors, run G2, trust upstream receipts as proof, or access a
  host. The local machine has insufficient free RAM for the four-rank G2 matrix.

## Scope and limitations

- Static review cannot prove Neuron compiler acceptance, NKI numerics, startup, throughput, causality on
  an actual checkpoint, or public BPB.
- The G2 harness deliberately uses CPU shims, synthetic data by default, a reduced forward slice, a
  fake clock, and `--no-compile`.
- The commands above establish only CPU/G2 gates. Native smoke, cold replay, fresh public evaluation,
  archive, and human submission authority remain separate.
