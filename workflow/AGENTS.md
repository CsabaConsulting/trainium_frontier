# AGENTS.md: Trainium Frontier Phase-1 autonomous research workflow (v2)

This file is the operating manual for autonomous coding agents (Codex or similar) working on this
project. Copy `workflow/` into the research repository's root and place this file at the repo root as
`AGENTS.md`. Codex reads it automatically. Every Codex session follows it. The per-role files in
`workflow/roles/` add role-specific duties.

**Goal:** the lowest official Phase-1 `val_bpb` by 2026-09-30 23:59 PDT (**2026-10-01 12:29 IST**).
**Best stack decided by 2026-09-27 23:00 IST; research freeze 2026-09-28 12:00 IST** (final package
ready). The days after that are buffer for submission and one retry. Top 10 advances to Phase 2; first
place is the prize.
**Metric we select on:** fresh-process public eval at context 1024 over 20,971,520 tokens ("public-20M").
Best estimate: official ≈ public-20M − 0.001 (±0.002, from one pair). The +0.005 rule applies to the 2M
prefix only.

---

## 0. Non-negotiable rules (violating any of these loses the competition or the team's trust)

1. Never modify `prepare.py`, the tokenizer, the data, the evaluator or scoring. Never read or infer
   validation data for training. No pretrained weights, no external data, no packages beyond the environment.
2. Every scored run: full chip (`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4`),
   1,800 s budget, seed 42, no eval-path tricks. `load_for_eval` returns the trained model, and its forward
   is pure, causal and batch-independent. No test-time training, no eval-time adaptation, no "evaluation
   model" that differs from the trained one.
3. **Human-only:** leaderboard submissions; any AWS action (instances, volumes, buckets, IAM, spend);
   resetting a correctness breaker; merging into the submission branch; changing decision thresholds.
   Agents *prepare* these and alert. They never do them.
4. Never print, copy, commit or transmit credentials, tokens, `.env`, PEM or private keys.
5. One training job per host. Comparisons are same-host only. Never launch without a passing G2
   preflight receipt and a clean Operator preflight.
6. Never weaken containment. A run is collected only after `contained.py quiesce` succeeds.
7. Never rewrite Git history, force-push, or relabel a failed run. A retry is a new label (`-r2`) with a
   diagnosis.
8. Separation of authority per run: whoever proposes does not verify; whoever implements does not verify
   or operate. Flag-only arms are verified *mechanically* (G2 plus exact bytes hash), not by the proposer.

## 1. Architecture: deterministic core, LLM lanes around it

```
                 +-------------------- deterministic core (scripts, no LLM) --------------------+
 queue.json  --> | select next job -> G2 preflight -> Operator preflight -> contained launch       |
 (approved)      | -> monitor -> contained stop/quiesce -> fresh 20M eval -> ledger append        |
                 | -> decide.py verdict -> archive/hash -> git commit/push -> alerts/summaries     |
                 +---------------------------------------------------------------------------------+
    ^ proposals          ^ implementations         ^ receipts           | verdicts, anomalies
    |                    |                         |                    v
 [Researcher]      [Implementer]             [Verifier]          [Analyst]    [Porter]   (Codex lanes)
```

- **Hosts are the scarce resource.** Target at least 90% host utilization. The deterministic core never
  waits on an LLM. If no approved job is ready, it runs the idle policy (§6).
- **Codex effort goes to the lanes**, in parallel:
  - generating and ranking hypotheses (Researcher);
  - writing code arms (Implementer, one branch per arm);
  - independent review (Verifier);
  - anomaly analysis and summaries (Analyst);
  - keeping the clean submission file in sync (Porter).
- **Decisions are mechanical:** `workflow/tools/decide.py`. An LLM may *explain* a verdict and never
  *override* one.

## 2. Repository layout (v2)

```
workflow/AGENTS.md                      # this file (also copy to repo root)
workflow/CODEX_PROMPT.md                # master prompt to paste into Codex sessions (goal, phases, roles)
workflow/roles/{supervisor,researcher,implementer,verifier,operator,analyst,porter}.md
workflow/tools/preflight_dryrun.py      # G2: exact argv through train.py's own __main__ on CPU/gloo x4
workflow/tools/decide.py                # verdicts + control drift (rules v2-2026-09-24)
workflow/tools/contained.py             # launch/stop/quiesce with descendant tagging (+cgroup)
workflow/tools/run_job.py               # ONE job end to end on a host (preflight..archive), receipts + ledger + verdict
workflow/tools/test_run_job.sh          # end-to-end self-test of run_job.py with fake train/eval (no Neuron)
workflow/proposals/research-v2/         # research file for all Track-A arms (35c9 + IMPL-1/2/3, default-off)
workflow/candidates/S0/                 # review-clean dense submission candidate + EVIDENCE.md
workflow/templates/*.json               # proposal, run record, verdict, submission package
research/v2/queue.json                  # approved, ordered jobs (seeded from reports/2026-09-24-queue-v2.json)
research/v2/ledger.jsonl                # append-only run records (schema: templates/run-record.json)
research/v2/experiments/<label>/        # proposal.json, g2.json, preflight.json, launch.json, run.log(.gz),
                                        # stop.json, quiescence.json, eval.json, record.json, verdict.json
research/v2/state.json                  # controller lease, host states, breaker states
research/v2/summaries/<utc>.md          # six-hourly summaries
```

Checkpoints never go into Git. They go to the content-addressed archive `archive/sha256/<hash>.pt`, and
the record stores the hash and URI.

## 3. The run lifecycle (every job, no exceptions)

**On the host, steps 4–11 are one command:**
`python workflow/tools/run_job.py --host-config research/v2/host.json --job research/v2/experiments/<label>/job.json`
(templates: `workflow/templates/host.json`, `workflow/templates/job.json`). It refuses to start without a
PASS G2 receipt for the exact bytes and argv, and it refuses until `eval_args_verified` is set from the G1
receipt. Exit codes: 0 done; 3 containment breaker; 4 preflight refused; 5 training failed; 6 eval failed;
7 archive failed. Re-run `workflow/tools/test_run_job.sh` after any change to the tools.


| # | Step | Command / artifact | Gate |
|---|---|---|---|
| 1 | Proposal | `experiments/<label>/proposal.json` (template) | Researcher writes it; a *different* session verifies it (code arms) |
| 2 | Implementation (code arms only) | branch `exp/<label>`; one causal change; patch + new SHA-256 | Verifier receipt: diff review + G2 PASS + lockstep with the flag off (bit-identical to base) |
| 3 | **G2 preflight** | `python workflow/tools/preflight_dryrun.py --train-py <file> --out experiments/<label>/g2.json -- <exact argv>` | `verdict == PASS` for the exact bytes and argv |
| 4 | Operator preflight | idle check, `fuser /dev/neuron*` empty, disk ≥ 6.5 GiB free (plus predicted footprint), `git status --porcelain` clean, SHA-256 of `train.py` equals the proposal's, `NEURON_CC_FLAGS` contains `--optlevel=1` or is unset | all pass → `preflight.json` |
| 5 | Launch | `contained.py launch --label L --dir D --cgroup -- env NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py <argv> --out-dir out/L` | `launch.json` written; workers visible within 5 min |
| 6 | Monitor | tail `run.log`; watch step time, loss, `epoch` field | NaN, loss > 100, or startup > 870 s → mark and let the file's own guard stop it; never kill silently |
| 7 | Stop/quiesce | `contained.py wait`, then `contained.py quiesce --files out/L/final.pt D/run.log` (on failure: `contained.py stop`, then quiesce) | exit 0, else breaker plus human alert |
| 8 | Eval | fresh process: `prepare.py eval-public` on `out/L/final.pt`, context 1024, 20,971,520 tokens (flags per G1 receipt), under `contained.py` with label `L-eval` | `eval.json`; finite; tokens and seq as required |
| 9 | Record | append `record.json` to `ledger.jsonl` | schema-valid |
| 10 | Verdict | `python workflow/tools/decide.py --ledger research/v2/ledger.jsonl --arm <arm_id>` → `verdict.json` | mechanical |
| 11 | Archive | copy the checkpoint to the archive; verify the hash remotely; delete on the host unless it is a current control or submission candidate (≤2 per host) | hash match |
| 12 | Commit | `git add research/v2/experiments/L research/v2/ledger.jsonl`; commit; push with 4× backoff | pushed |
| 13 | Follow-up | NOMINATE → enqueue the cross-host replicate first; ADOPT_PENDING_HUMAN_ACK → alert the human, enqueue new-base controls; HOLD → independent cold replay plus audit; CONFOUNDED → re-plan; DISCARD → close | |

## 4. Decision rules (implemented in `decide.py`; do not reinterpret)

| Rule | Threshold |
|---|---|
| Valid run | evaluated; `over_budget=False`; causality passed; git clean; public-20M at context 1024; finite |
| Control | same host, same base SHA-256; pooled mean of the last ≤3 valid controls; ≥2 preferred |
| Step window | arm/control steps ∈ [0.90, 1.10] unless `kind=throughput_arm`, else CONFOUNDED |
| Nominate (one host) | Δ ≤ −0.0006 |
| Discard (one host) | Δ ≥ +0.0005. In the ambiguous band, discard unless the kind is high-prior (`table_arm`, `data_order_arm`), which gets one replicate |
| Adopt (two hosts) | both Δ < 0, mean ≤ −0.0010, disagreement ≤ 0.0015 → ADOPT_PENDING_HUMAN_ACK |
| Discard (two hosts) | mean > −0.0003 |
| Big-win hold | Δ ≤ −0.010 or public-20M < 0.975 → independent cold replay plus audit before anyone reports it |
| Control drift | a control more than 0.0010 from the host/base mean → alarm; set that control's status to `drift_quarantined`; rerun |

After an adoption changes the base, every later comparison needs ≥1 (ideally 2) fresh controls per host on
the new base bytes. Earlier flag results are **stale** on the new base and must be re-screened before
stacking.

## 5. What to run: current state and ranked queue (seeded 2026-09-24)

Facts established so far (full detail and evidence in `reports/2026-09-24-frontier-research-program.md`
and `workflow/evidence/`):

- Best confirmed: dense 6×1024, matrix LR 0.015, public-20M **0.99195** (two runs). Clean file `116188…`
  is bit-identical to 35c9 with the dense flags over 3 CPU training steps (`lockstep3-35c9-vs-116188-base.json`).
- Exact 35c9 **cannot** run A1/A2 tables, `--relu2-tau`, `--n-embd 1152` or `--mlp-ratio 5` (G2 FAIL
  receipts). **Use the research-v2 file** (`workflow/proposals/research-v2/train.py`, SHA-256 `c5c410af…`)
  for every Track-A arm. It is 35c9 plus IMPL-1 (table unblock), IMPL-2 (`--ngram-ve-freeze-epoch`) and
  IMPL-3 (`--fresh-tail-frac`, `--requeue-identity`), all default-off and CPU-proven bit-identical to
  35c9 at default. It still needs a Verifier receipt plus a Neuron smoke.
- **S0** (`workflow/candidates/S0/`, SHA-256 `ad223cdc…`, bare launch command) is the review-clean dense
  submission candidate, proven numerics-identical to 116188 on CPU. Its remaining gates are in its
  `EVIDENCE.md`.
- Dense trains about 1.7 epochs with the whole second epoch inside the LR cooldown. On the old stack, steps
  past the epoch boundary hurt. The value of extra steps for dense is **unknown** until D1 runs.

**Queue order** (seed `research/v2/queue.json` from `reports/2026-09-24-queue-v2.json`):

1. **Gates:**
   - G1: read `prepare.py` for the shard default, eval context and prefix semantics, and startup enforcement; count shards on both hosts.
   - IMPL-1: Verifier receipt, then the Neuron smoke.
2. **A0 controls:** 35c9 base on each host (then pooled ≥2).
3. **D1 epoch-value diagnostic:** the control with `--max-train-seconds T_e`, where T_e is the time of the first `epoch 2` log line.
4. **A2 on trn22 and A1 on trn21** (IMPL-1 bytes), then cross-host replicates.
5. **FT fresh-tail order** (IMPL-3) and **A2f/A1f epoch-2 table freeze** (IMPL-2), as the table and D1 results dictate.
6. **Flag bank on the current base** (each G2-PASSed; one causal change each):
   - `--matrix-lr 0.0175` / `0.0125`
   - `--warmdown-ratio 0.85`
   - `--embedding-lr 0.45`
   - `--mlp-sandwich-norm`
   - `--weight-decay 0.35`
   - `--final-lr-frac 0.0`
   - `--epoch-shuffle` (after verifying epoch-1 batch identity)
   - idle-fill only: `--head-gate --head-gate-norm`, `--x0-gate`, `--out-pool`
7. **After a table adoption:**
   - **B2** touched-row update (static-shape all-gather, lazy RMSProp with catch-up decay): exactness first.
   - **A3** tables on all VE layers (1,3,5); **A4** mult 128.
   - **A-eps** (table RMSProp ε 1e-10 → 1e-15) if the Neuron smoke shows median |table grad| ≲ 1e-10.
   - Re-bracket matrix LR and warmdown.
   - **SC-1** depth 8 × 768 plus the adopted stack, declared as a stack confirmation.

**Parked** (do not run without new evidence and a human OK): LNC1/world 8; any FP8; new NKI kernels without a
profile; `--relu2-tau` (needs code); Recursive's WD pulses, Demon, momentum/β₂ ramps and window/RoPE changes;
engram, out-bigram, MUDD, DCMHA, RWKV/Mamba, U-Net skips, weight EMA, batch ramp, byte-loss, byte-WTE,
SwiGLU, adam-every-N; anything on the old 640 stack; seed hunting; 2M screening; 300-step or 900-s quality
smokes.

## 6. Keeping both hosts busy (idle policy)

When a host frees, the core picks the first applicable item:

1. a NOMINATE replicate for this host;
2. a due control refresh (≥8 runs or ≥12 h since the last control on this base);
3. the next ready queue item for this host (G2-PASSed, preflight-clean);
4. a clean-file cold replay of the current submission candidate;
5. a diagnostic: D2 profile (30 min), the 2M-vs-20M calibration eval, or the memorization eval of stored checkpoints;
6. a base replicate.

Alert the human if a host sits idle for more than 15 minutes.

Research runs use a persistent warm compile cache (`NEURON_COMPILE_CACHE_DIR=/var/neuron-cache/research`,
recorded in the receipt, LRU-pruned only while the host is quiescent). Submission candidates additionally
get a **cold** cache run, and startup must be ≤ 840 s.

## 7. Generating new hypotheses (Researcher lane; this is where new gains come from)

Every proposal must include the following. Missing fields → rejected mechanically:

- **one causal change**, and the exact argv or patch;
- a **mechanism** tied to evidence: a ledger row, a primary source with commit or DOI, or a diagnostic;
- a **ledger search**: aliases searched in `ledger.jsonl`, the old `experiment-log.jsonl`, DECISION_LOG
  and the report's stop list, plus why this is not a duplicate;
- an **expected Δ with an 80% interval**, P(Δ ≤ −0.001), host-hours, and **EV = P·|Δ|/host-hour**;
- **Trainium feasibility**: compile/graph change, SBUF/ownership/world-size constraints, cold-startup
  risk, memory;
- the **cheapest falsifier**, and what happens if positive or negative.

Priorities for new ideas, in order:

1. **Repeated-data and table-memory interactions:** fresh-tail order; epoch-2 table freeze or LR drop;
   table ε; table init (Recursive uses uniform(−s, s), the team uses zeros); K=2 multi-hash (only after
   B2).
2. **Re-testing "negatives" that were confounded by the epoch boundary** on the old stack (qk-shift,
   sandwich norm, BF16 policies). They are untested on dense, not refuted.
3. **Optimizer/schedule single-factor brackets on the newest base.**
4. **Throughput work only after D1 shows extra steps help (>0.005 per epoch) *and* D2 profiles the
   bottleneck.**

Reject: bundles of unrelated changes; anything that needs evaluation-path changes; anything depending on
validation data; mechanisms that only win at 300 steps.

## 8. Porting wins into the clean submission file (Porter lane)

1. Specialize: take the research bytes plus the adopted flags → constants; delete unreachable branches and
   flags; keep one forward shared by training and `load_for_eval`.
2. Prove equivalence:
   - G2 on both files with their respective argv;
   - `preflight_dryrun.py --compare research.json clean.json` must report `identical: true` over ≥3
     steps (same plan hash, same losses, same parameter hashes);
   - then a Neuron 100-step lockstep (steps 0–1 identical, later steps within 1e-4 relative).
3. Two **cold** replays with the bare submission launch command (fresh cache dir): both within ±0.0008 of
   the research mean, startup ≤ 840 s, `over_budget=False`.
4. Independent reviewer-style audit (`workflow/roles/verifier.md` checklist B). The clean file must stay
   within 12 h of the research best.
5. Package `submissions/candidates/<date>-<sha8>/`: `train.py`, `launch-command.txt`, `SHA256SUMS`, the
   evidence index, and the audit receipt. **Alert the human. The human submits.**

## 9. Alerts, summaries, breakers

- **Alert immediately:**
  - a record with non-empty `anomalies` (containment had to stop a straggler);
  - a milestone crossed;
  - ADOPT_PENDING_HUMAN_ACK, HOLD, or submission-ready;
  - any correctness failure (NaN, over-budget, causality, dirty git, hash mismatch, survivors, neuron holders);
  - disk < 6.5 GiB;
  - a host idle > 15 min;
  - control drift;
  - cold startup > 840 s.
- **Every 6 h:** a summary to `research/v2/summaries/` (utilization per host, runs by verdict, best
  confirmed public-20M, pending human decisions, breakers, disk, next 5 queue items). Commit it.
- **Breakers:**
  - 2 consecutive launch failures on one host → block that host.
  - Any correctness failure → pause code arms globally until a human resets.
  - 3 FAILED verdicts in 6 h → global pause.
  - Disk floor → block the host.

## 10. Timeline to the freeze (IST)

| When | Must be true |
|---|---|
| Sep 24 night | G1 done; controls on both hosts; IMPL-1 verified; A1/A2 screens launched |
| Sep 25 ~03:00 | Table decision #1 (two-host) |
| Sep 25 ~10:00–12:00 | S0 package (cleaned 116188, numerics-identical) ready for the human |
| Sep 26 evening | S1 package if a table stack is adopted and clean-ported (≥0.004 better than S0) |
| Sep 27 12:00 | **No new mechanisms after this.** Only replicates, stacking of already-adopted wins, and cold replays |
| Sep 27 23:00 | **Best stack decided:** two hosts × 2 replicates confirmed; the Porter's clean port already in progress |
| Sep 28 12:00 | **Freeze:** S2 package complete (clean-port equivalence, 2 cold replays, audit) |
| Sep 28 evening | Recommended human submission of S2 |
| Sep 29–30 | Buffer only: a retry or resubmission if anything fails. Hosts run confirmation replicates, nothing new. Hard deadline Oct 1 12:29 IST |
