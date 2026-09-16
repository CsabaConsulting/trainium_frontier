# Execution runbook: sub-1.0 campaign (P0–P4) on trn21/trn22

Status when written (2026-09-16): code + scripts are committed locally in this
repo. **No remote step has run yet** — the authoring sandbox has no route to
the boxes (see RESUME-HERE.md). Run this from a machine with normal SSH.

## 0. Setup + stock-take (both boxes)

```bash
R1=ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com   # trn21 i-0329acb9a6244986b
R2=ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com    # trn22 i-0673e54d9d66e6a1e
KEY=~/.ssh/trn2.pem
ssh -i $KEY $R1 'ls -d ~/workspace/trainium_frontier && ~/workspace/native_venv/bin/python -c "import torch, neuronx_cc" 2>&1 | tail -1; neuron-ls 2>/dev/null | head -5; ls -lt ~/workspace/trainium_frontier/out 2>/dev/null | head -5'
# same for $R2
```

```bash
RD=~/workspace/trainium_frontier   # fix if step 0 shows another path
cd /home/csaba/repos/AWS/TrainiumFrontierCompetition/trainium_frontier
git pull --ff-only  # take the committed sub-1.0 work below
for R in $R1 $R2; do
  rsync -avz -e "ssh -i $KEY" --exclude out --exclude __pycache__ --exclude checkpoints \
    ./ $R:$RD/
done
# Smoke-test the sync: --help must exit 0 (a bare-% help bug once made EVERY
# invocation crash; fixed 2026-09-16, but re-verify after every rsync):
ssh -i $KEY $R1 "cd $RD && ../native_venv/bin/python train.py --help > /dev/null && echo HELP_OK"
```

Locked launch shape (from experiment-log.jsonl ground truth):
`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`.
All runs go through `research/run_experiment.py` (writes run-logs/<label>.log +
experiment-log.jsonl row). Full-budget: `--num-steps 1000000 --max-train-seconds 1800`.
Smoke: `--num-steps 300 --max-train-seconds 3600` (step-capped; never scored).
Seed discipline: `--seed 42` on everything (default); no seed hunting.

## P0. Eval reconciliation — R1 first, no training cost

Find the record checkpoint on R1:
```bash
ssh -i $KEY $R1 "cd $RD && ls -lt out/*/ckpt*.pt 2>/dev/null | head -5; ls -lt out/*/*.pt 2>/dev/null | head -5"
```
Then (single process is fine for eval):
```bash
ssh -i $KEY $R1 "cd $RD && NEURON_LOGICAL_NC_CONFIG=2 ../native_venv/bin/python \\
  research/reconcile_eval.py --checkpoint <BEST_CKPT> 2>&1 | tee research/run-logs/sub1-p0-reconcile.log"
```
Gate: the 4-cell table must reproduce the historical spread (raw-2048 vs
ema-2048 vs sidecar). Relabel every past number to its def before any other
conclusion. Bring the log back (see §Retrieval).

## P1. NKI attention forward — R1, engineering then smoke

Target: `CausalSelfAttention.forward` SDPA call site (train.py ~L2546).
Reference: local `nki-samples/` flash-attention examples (NOT synced to the
box — copy the one file you adapt from). Requirement: keep the rope+QK-norm
input contract (contiguous [B,T,H,D]) or beat it knowingly.
Smoke (300 steps), compare `tokens/s` vs the locked baseline smoke:
```bash
cd $RD && ../native_venv/bin/python research/run_experiment.py --label sub1-p1-nkiattn-smoke \\
  --solution sub1 --description "nki attention forward smoke" \\
  --num-steps 300 --max-train-seconds 3600 --nproc-per-node 4 --lnc 2
```
Gate: ≥ +3% step rate AND loss curve on top of baseline → full-budget run.
Else stop; record numbers.

## P2a. GQA smoke — R1 (code already in tree, flag-gated, CPU-verified)

```bash
cd $RD && ../native_venv/bin/python research/run_experiment.py --label sub1-p2a-mqa1-smoke \\
  --solution sub1 --description "gqa n_kv_heads=1 (MQA) smoke" \\
  --num-steps 300 --max-train-seconds 3600 --nproc-per-node 4 --lnc 2 \\
  --extra-args --n-kv-heads 1
```
Gates: (1) completes (NKI rope kernel fires on the repeated+contiguous K/V —
check the log, not just the loss); (2) steps/s ≥ baseline −5%…
→ then FULL budget: same label with `-full`, `--num-steps 1000000 --max-train-seconds 1800`.
Constraint (hard): at locked width n_head=5, only `--n-kv-heads 1` is legal;
2/4 need a composite n_head (768→6, 1024→8) i.e. a width change, separate arm.
Still must confirm on hardware: NKI softcap+CE "none"-reroute path is only
exercised with --byte-loss-weight; GQA alone doesn't touch it.

## P2b. MXFP8 spike — R1, timeboxed 2 h wall, kill fast

1. Legality: competition rules on dtype of matmuls (if unclear, ask organizers;
   do NOT burn silicon first).
2. Probe on box (CPU0, no training):
   `python -c "import torch; x=torch.randn(128,64).to(torch.float8_e4m3fn); print(x.dtype)"`
   then the same under `torch.device('neuron')` inside the venv.
3. Compile cost: one 100-step probe with the candidate FP8 scope; kill if
   compile eats >5% of an 1800 s budget. Quality run only if
   throughput-neutral or better.

## P3. Batch×LR — R2, full-budget only, after P0

TOTAL_BATCH_SIZE / DEVICE_BATCH_SIZE are module constants (train.py ~L226-227,
no CLI flag): each arm = edit + commit + run. Grid (seed 42, EMA eval):
batch ∈ {65536, 131072, 262144} × LR scale ∈ {0.7×, 1×, 1.4×} via
`--matrix-lr` (confirm exact flag names via `--help`, now working).
Note: batch alone at full budget already lost both directions
(65k→1.048832, 262k→1.051031, per train.py:227 comment) — this sweep tests the
JOINT interaction, 3–4 slots max, kill on first two negatives.

## P4. Data-pipeline variants — R2, smoke-first

- Reshuffle: `--epoch-shuffle --epoch-shuffle-seed <s>` (exists in tree).
- Byte-weight loss (new, CPU-verified): append `--extra-args --byte-loss-weight`
  to the locked shape; smoke first (loss-scale print `byte-loss-weight:` must
  appear in log), then full budget. NKI kernel path with the "none"-reroute
  MUST be confirmed in the smoke log before a full run.
- Checkpoint averaging (new, CPU-verified): after any keep,
  `python research/average_checkpoints.py --out out/<label>/avg.pt out/<label>/*.pt`
  then score avg.pt with reconcile_eval.py (P0) — near-zero cost.
- Repack: PACK_FACTOR variants (confirm flag via --help).

## Retrieval (after every run)

```bash
cd /home/csaba/repos/AWS/TrainiumFrontierCompetition/trainium_frontier
for R in $R1 $R2; do
  rsync -avz -e "ssh -i $KEY" $R:$RD/research/run-logs/ research/run-logs/
  rsync -avz -e "ssh -i $KEY" $R:$RD/research/experiment-log.jsonl /tmp/exp-$(echo $R | cut -d@ -f2 | cut -d. -f1).jsonl
done
# merge experiment-log rows (dedupe by label), bring back ONLY record checkpoints:
#   rsync -avz -e "ssh -i $KEY" $R:$RD/out/<record-label>/ ./checkpoints/<record-label>/
git add research/run-logs research/experiment-log.jsonl checkpoints/<record-label>
git commit -m "sub1: <label> <bpb> <keep/discard>"
```

## Box allocation

R1 = P0 → P1 → P2a → P2b (measurements/smokes, then full on win).
R2 = P3 → P4 (full-budget queue). Never run the same arm on both boxes;
the single queue of record is this file's checkbox list — check an item before
launching it.
