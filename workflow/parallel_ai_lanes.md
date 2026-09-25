# Parallel AI lanes: what other AIs can do while Codex runs the hosts

**Rules for every lane:**
- Code only. No hardware.
- Each lane works in its own folder or branch and hands back a patch plus evidence.
- Codex's Verifier decides whether a patch goes onto a host.
- Never share keys, PEM files or host access with these AIs.

**Shared context (paste at the top of every lane prompt):**

```
Context: AWS Trainium Frontier Phase 1. Train an LM from scratch in exactly 30 min on one
Trn2 chip (trn2.3xlarge, launch: NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone
--nproc_per_node=4 train.py), minimizing val_bpb on ClimbMix with a fixed vocab-8192 BPE.
Only train.py may change. No pretrained weights, no external data, no eval tricks.
load_for_eval must return the trained model with a pure, causal forward.
Repo: https://github.com/poojithdevan4D/trainium-research-context (branch
claude/trainium-frontier-research-nkp5zd).
Files:
- workflow/proposals/research-v2/train.py: research file, 35c9 + table unblock
- 05-train-submission-clean-116188.py: the file that scored 0.9901 official
- workflow/tools/preflight_dryrun.py: CPU checker (gloo, 4 ranks)
Current model: dense 6 layers x 1024, SEQ_LEN 1024, pack 32, batch 131072 tokens/step,
~3,650 steps, ~268k tok/s, ~1.7 epochs over ~288M tokens, Muon matrix LR 0.015.
Public-20M 0.9915. Leaderboard: #1 0.9464, top-10 line 0.9824.
Deliver: one patch against research-v2/train.py, behind a new default-OFF flag, so default
behaviour is bit-identical. Also: a CPU G2 PASS receipt from preflight_dryrun.py with the
flag on, proof that flag-off is bit-identical over 3 steps, and a short README with the
expected effect and risks.
```

---

## Lane 1: Throughput (FP8 projections). The biggest proven lever on Trn2.
The #3 team measured −0.019 on this hardware, mostly from kernels plus FP8.

```
Task: add --fp8-proj: a row-scaled FP8 (e4m3) forward for the MLP c_fc/c_proj and the
attention q/k/v/o projections, with the backward kept in bf16 (straight-through w.r.t. the
quantization), fp32 master weights, no loss scaling. Use torch._scaled_mm or the Neuron-
supported FP8 matmul path. Check AWS Neuron docs for Trn2 FP8 support in torch-neuronx and
cite the doc. Also provide a microbenchmark script the Operator can run on Trn2, measuring
full-step time with the flag on vs off, with the pass rule ">= 3% faster whole step".
```

## Lane 2: The #3 team's architecture
```
Task: add --unet-8x1024: depth 8 x width 1024, U-Net skip connections (layer i output
added to layer n-1-i input through a zero-init learned scalar), and attention in only a
subset of layers (flag: attention layers list, default "0,2,4,6,7"; the others are MLP-only).
Read https://ritwikareddykancharla.github.io/projects/trainium-of-thought.html first and
match their description where it is specified; list every assumption where it is not.
Keep step-0 bit-identity where possible (zero-init skips).
```

## Lane 3: Recursive schedule package (free in step time)
```
Task: add --recursive-schedule: in one flag, port the schedule pieces of Recursive's
optimized_from_karpathy.py (github.com/recursive-org/first-steps-toward-automated-ai-research,
nanochat_autoresearch/solutions/, Apache-2.0; include an attribution comment):
- Muon warmdown 0.95 (floor 0.05), Adam warmdown 0.65
- Muon momentum decay to ~0.79 over the warmdown, NorMuon beta2 ramp 0.95 -> 0.97
- time-based, matching our lr_multiplier convention
Do NOT port the WD pulses in this lane. List exact constants with source line numbers.
```

## Lane 4: Sparse table updates (B2). Needed only if tables win tonight.
```
Task: add --ngram-sparse-update for the hashed n-gram value tables. Per step: collect the
unique touched row indices per rank, all_gather indices and row gradients with a static
padded cap (Neuron dislikes dynamic shapes), use index_add_ (not scatter; duplicate
indices), accumulate in fp32, and apply lazy RMSProp with catch-up decay only to touched
rows. Prove it matches the dense-update path exactly on CPU for 3 steps at a small table
size. Check whether torch-neuronx supports all_to_all; if it does, offer that variant too.
Reference: nanoGPT_speedrun/from_best/train_gpt.py in the Recursive repo ("sharded_sparse").
```

## Lane 5: Research scout (web only, no code)
```
Task: find every public detail about how the top teams on the AWS Trainium Frontier
leaderboard train: Them Bones (0.9464), Team Awesome, Trainium of Thought, msamuel,
Team Quanta, Next Token, Team DLYog, Eldrun, Yuntian Deng, competitive larping.
Search GitHub, X/Twitter, LinkedIn, personal blogs, and the Devpost forum
(trainium-frontier.devpost.com/forum_topics). Also collect every organizer rule
clarification:
- submission quota
- what "QUEUED" status means
- startup allowance
- eval context
- what reviewers flag
Public information only; never try to access private repos or accounts. Output a table:
team, technique, measured delta, source link.
```

## Lane 6: Profiler and analyst (offline, using logs Codex already has)
```
Task: using the run logs and ledger in the research repo, answer three questions:
(1) where step time goes (forward, backward, optimizer, all-reduce), from any profile or
    timing lines;
(2) how loss changes across the epoch-1 -> epoch-2 boundary in the dense runs;
(3) which of the old "confounded" results would change verdict if the epoch-2 penalty is
    table-specific.
Output: a ranked list of where the next 10% of throughput or 0.005 BPB most likely comes
from, with evidence.
```

---

## How to run this without chaos
- **One lane per AI.** Claude Code, Gemini, Cursor, Copilot and Grok can each take one. Give each the shared context plus its lane block.
- **Every patch goes to Codex's Verifier.** No external AI launches anything.
- **Priority order, if you only have 2 or 3 AIs free:** Lane 1 (FP8), Lane 2 (architecture), Lane 5 (scout).
- **Deadline for patches:** Sep 26, 12:00 IST, so they can be raced on Sep 26 and 27.
- **Send Claude** anything you're unsure about: a patch, a scout finding, a submission candidate.
