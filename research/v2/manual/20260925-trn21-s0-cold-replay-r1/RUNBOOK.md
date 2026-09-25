# trn21 S0 cold replay: disabled manual Operator handoff

This packet is **OFFLINE/DISABLED**. It is an Implementer handoff, not an Operator request or launch authorization. An independent Verifier must check this packet; the Supervisor must check a completed proposal, current human break-glass authority, and assign a distinct Operator. Do not run this packet as a script. Do not contact trn21 until that assignment. This first replay does not make S0 submission-ready: `workflow/candidates/S0/EVIDENCE.md` still requires a second cold replay, Neuron equivalence and an independent final audit.

## Exact source and evidence

| Item | Pin |
|---|---|
| Unique run label | `fs-20260925-s0-cold-replay-trn21-r1` |
| Host / new run directory | `trn21` / `/home/ubuntu/trainium_s0_cold_20260925_trn21_manual_r1` |
| Local source | `workflow/candidates/S0/train.py`, SHA-256 `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb` at source commit `e10e51c99dfe3285d3f5172ed4a0cc3bd4b8fc3f` |
| Exact submission command | `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py` (no user argv) |
| Existing CPU G2 | `workflow/evidence/lockstep3-S0-bare.json`, SHA-256 `37d10c67d9cfa7f601ad9ad0ef461cce5aee1c355370a8526136be8e0c5a166b`, verdict PASS, `argv=[]`, `nproc=4` on **VM CPU/stub prepare** only |
| Trusted evaluator source | `prepare.py`, SHA-256 `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`; exact eval args in `research/v2/g1.json` |
| Human authority | `research/v2/BREAK_GLASS_2026-09-25.md`; it waives the stalled staging pipeline, not source/host/G2/eval/proposal gates |

The 2026-09-24 trn21 `116188` full runs (`0.9920722144854677`, `0.9918353324459732` public-20M) are the **comparison history**, not S0 evidence. S0 is only CPU-proven equivalent so far. No new BPB is claimed here.

## Operator sequence after independent authorization

1. **Local lock.** Verify the four hashes above, confirm this label does not occur in the v2 ledger or on trn21, confirm a completed S0 replay proposal and separate Verifier/Supervisor approval. Record their commit/receipt hashes. Never retry this label after any failure; diagnose and propose `-r2` instead.
2. **Fresh host preflight, before writing.** Confirm SSH alias `trn21` resolves to the expected machine; verify its host identity, existing trusted `prepare.py` SHA, no other `torchrun`/`train.py`/`prepare.py` or Neuron holder, no competing launcher, STOP/breaker state, and root free bytes. Require **at least 6.5 GiB free plus projected fresh-cache/checkpoint/log growth**; low space is HOLD. Confirm `NEURON_CC_FLAGS` is unset or contains `--optlevel=1`, capture its allowlisted value, and check existing tracked source checkout is clean. Do not reset a breaker, clear cache, kill a process, overwrite a worktree or delete artifacts.
3. **Plain transfer to a new location.** Require the proposed run directory to be absent, then create it with restricted permissions (failure/exists = HOLD). Plain `scp` or `rsync` **only the exact S0 `train.py`** to `/home/ubuntu/trainium_s0_cold_20260925_trn21_manual_r1/train.py`. The Operator must compute SHA-256 on the host and require the exact `ad223...125bb` digest **before launch**. This is an untracked, separate run directory; do not mutate the Sep-24 checkout. For Python's `import prepare`, use `PYTHONPATH` pointing to the already verified trusted checkout containing the pinned `prepare.py`; do not edit or replace it. Verify the imported module's `__file__` and hash in the intended training interpreter before launch.
4. **Cold cache and exact run.** In that new directory, require no `out/final.pt` and no pre-existing cache directory. Create and verify six distinct empty cache directories under this run's `cache/`, and bind `NEURON_COMPILE_CACHE_DIR`, `TORCH_NEURONX_NEFF_CACHE_DIR`, `TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR`, `TORCH_NEURONX_HLO_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and `TORCH_NEURONX_DEBUG_DIR` to them, as in the Sep-24 rehearsal. Record paths, owner and empty state; never reuse a warm cache. Preserve existing `NEURON_CC_FLAGS` only if compliant and verified. Start exactly one full-chip job with the command above, cwd equal to the new run directory; the file defaults are seed 42, 1,800 s and `out/final.pt`. Capture stdout/stderr, start/end UTC, PID/exit code and full **allowlisted** launch environment. No extra `train.py` flags, pipes, scientific code edits or alternate shapes. Observe the run to completion; do not infer success from an active process.
5. **Training gate.** Require exit 0, a clean worker/process/Neuron postflight, `out/final.pt` present and unchanged while collected, finite losses/checkpoint, a valid step count and `over_budget: False` from the actual log. Record startup estimate and compare to the cold ≤840 s gate. A crash, NaN, stale output, missing checkpoint, surviving holder or budget failure is a failed run, never a promising BPB. Preserve the checkpoint and caches; no cleanup is authorized.
6. **Fresh-process public-20M evaluation.** With the pinned trusted `prepare.py` and S0 path, in a **new process** run G1's exact command (shown below). Require a finite score, `num_tokens=20971520`, context 1024, no eval timeout/overrun. Archive the raw eval log, parsed JSON and checkpoint SHA-256. Do not use in-process training evaluation or public-2M as the decision score. Recheck checkpoint hash after evaluation.
7. **Durable result.** Retrieve source identity, preflight, transfer/host hash, cache path/empty proof, command/environment, full training log, step/budget summary, checkpoint status/hash, fresh eval log/parsed result and postflight into `research/v2/experiments/<label>/`. Append the complete success or failure to `research/v2/ledger.jsonl` using the v2 run-record schema, commit/push evidence without secrets. The Analyst compares S0 to same-host `116188` history, but no win/adoption is declared from one replay. Report steps, public-20M BPB and delta, or the actual sanitized failure.

The trusted eval command, with absolute paths substituted and cwd in the already verified checkout holding `prepare.py`, is:

```text
/home/ubuntu/workspace/native_venv/bin/python prepare.py eval-public --train-py /home/ubuntu/trainium_s0_cold_20260925_trn21_manual_r1/train.py --checkpoint /home/ubuntu/trainium_s0_cold_20260925_trn21_manual_r1/out/final.pt --device neuron --batch-size 1 --seq-len 1024 --eval-tokens 20971520 --timeout-seconds 1200
```

This command is a plan, not a claim that host paths still exist or that a host check passed. The independent Operator must verify interpreter, checkout, data availability and all paths live before any run.
