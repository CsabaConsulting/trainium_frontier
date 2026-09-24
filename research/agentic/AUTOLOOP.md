# AUTOLOOP — unattended code-and-flag experiment controller

Implemented 2026-09-22 under `research/agentic/STANDING_AUTONOMOUS_AUTHORIZATION.md`. Code lives in
`research/agentic/autoloop/`; CLI is `research/agentic/autoloop.py`; tests are in
`research/agentic/tests/`. Nothing in this document waives `AGENTS.md`.

## 2026-09-23 fixed full-only mode

When `plan_mode` is `fixed_full_only`, `frontier-full-only-queue.json` supersedes the general ladder
below. The controller never calls the Researcher, never creates a smoke run, and rejects any
launchable proposal outside the seven predeclared full arms. It migrates the schema-2 runtime
losslessly, changing the verified local-conv r3 record to `parked` with
`launch_authorized=false`; usage, history, run counts, failures, and breakers are retained.

The first cohort is exact control on `trn22` and MATRIX_LR 0.025 on `trn21`. Both independent
verification receipts and both clean preflights must exist before either launch. Later arms are
strictly serial. Every arm uses a direct one-constant commit, the exact submission argv, unique
empty train/eval compile caches, and a separate context-1024 public evaluation over exactly
20,971,520 tokens / 83,462,186 bytes. Cross-host throughput is diagnostic and cannot affect a
verdict. Result/verdict/log evidence must be locally and remotely verified before the exact
checkpoint hash is pruned; prune and exact label-owned cache-cleanup receipts are then committed,
pushed, and reverified before the state closes. If push is disabled, the checkpoint and caches are
retained. The controller never clears a breaker automatically: the pinned pushed R8 cleanup receipt
must be recorded through the explicit Operator recovery transition.

## 2026-09-24 final-week control-plane instrumentation

The disk safety boundary is unchanged: production config remains `min_free_gib=7.4`, and the
existing validator still refuses values below the protocol floor. Cleanup remains an Operator
action. `TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=900` is recorded separately from training time. Every
collected result now carries `startup_cache_diagnostics`, derived only from its frozen manifest and
runner ledger row: the available startup estimate, cap comparison, compile flag, unique cache
identities, and an explicit statement that cache hit/miss telemetry is unavailable. No train or
evaluation source is instrumented.

Every controller cycle writes one fsynced JSONL utilization sample per configured host to ignored
runtime state at `research/agentic/runtime/idle-utilization.jsonl`. A sample distinguishes actual
process/device busy state from an idle host that is launch-blocked (for example by disk). The
cycle report calls out idle streaks longer than ten minutes; inventory failures are recorded as
`unknown`, never guessed busy or idle.

`FINAL_WEEK_RESEARCH_ADDENDUM_2026-09-24.md` is included in Researcher and Verifier context. The
new `evaluation_policy` schema is intentionally disabled with an empty base hash and no authorized
control labels/sweeps, because the canonical-root rebase is a separate reviewed change. Enabling it
without a 64-hex current-base hash and explicit same-host current-control labels is invalid. Its
deterministic assessor enforces:

- context-1024, 2,097,152-token screens only nominate full20M confirmation at delta <= -0.0005;
- delta >= +0.0005 is discarded; the middle band is discarded unless the record carries a
  predeclared active-sweep identifier;
- a screen has no ADOPT/KEEP transition;
- ADOPT requires exactly two distinct, strictly negative 20,971,520-token runs of identical
  candidate bytes, each paired to a same-host current-base control, with mean delta <= -0.0008.

The current pinned base and fixed queue are not changed by this instrumentation patch.

## What it does

A deterministic Python loop that, on each cycle:

1. Stops if `research/agentic/STOP` exists, a circuit breaker is open, the competition cutoff has
   passed, or the config is invalid.
2. Reconciles running work: polls launched runs, collects declared artifacts with SHA-256 checks,
   analyzes against the matched control, archives evidence to git.
3. If a box is idle and everything prior is archived, advances that box one rung on the ladder:
   `control smoke → control full → candidate smoke → (KEEP) candidate full → verdict`.
4. When a new candidate is needed, asks the Researcher for one flag or bounded `train.py` code
   proposal, then requires an independent Verifier (different model family) to pass the proposal.
5. For code proposals, an isolated Implementer returns exact text replacements which are applied
   only in a fresh worktree at the authoritative commit. Deterministic syntax/diff/scope checks and
   a second independent implementation review must pass before a candidate commit can be staged.

Models never see a shell, the SSH key, the manifest, or hardware. They return JSON; the controller
validates and decides. Code candidates may modify exactly `train.py`; the patch, checks, verifier
decision, candidate branch, and commit are durable before the Operator receives the commit.

## Pinned, non-overridable facts (`autoloop/config.py`)

| item | value |
|---|---|
| commit | `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a` |
| hosts | `trn21` → `ubuntu@ec2-16-26-246-98…`, `trn22` → `ubuntu@ec2-16-26-154-27…` |
| key | `./trn2.pem`, must be mode 600, checked with `stat` only, never opened |
| remote python | `/home/ubuntu/workspace/native_venv/bin/python` |
| seed | 42 (manifest + argv + `NEURON_COMPETITION_R1_SEED`) |
| shapes | LNC=2/nproc=4 (default, 295 ledger rows) or LNC=1/nproc=8; nothing else |
| budget | `--max-train-seconds` ≤ 1800; smoke tier 300 s |
| CC flags | must contain `--optlevel=1` |
| forbidden flags | `--out-dir --seed --init-from --num-steps --max-train-seconds --eval-tokens --eval-public --compile-only --device-type` |
| forbidden remote paths | anything containing `wave195`, `trainium_freshstart_20260918`, `trainium_frontier` |

## Remote layout (all new; existing directories are never read, written, or cleaned)

```
/home/ubuntu/autoloop/                 controller scratch
  bundles/<commit12>.bundle            git bundle of the pinned commit (sha256-verified)
  launchers/<label>/                   remote_launcher.py, launch.json, pid, runner.log, done.json
  locks/<box>.lock (dir), <box>.owner  per-box mutual exclusion
/home/ubuntu/workspace/autoloop_runs/
  <commit8>--<proposal_id>--<label>/   ONE fresh clone per run; never reused
```

Staging proof required before any launch (else `RemoteError` → breaker):
run dir did not exist; `git clone --no-checkout <bundle>`; `git checkout --detach <commit>`;
`rev-parse HEAD == commit`; `status --porcelain` empty; `sha256(train.py)` equals
`git show <commit>:train.py` locally. Launch re-checks HEAD/porcelain, the lock owner, and a
fresh inventory (no `train.py|phase_timer|reconcile|torchrun|run_experiment` processes, silent
`fuser /dev/neuron0`, free disk ≥ 8 GiB, data cache present, pinned python present).

Launch is `remote_launcher.py <launch.json>` run through the pinned python. The launcher refuses
if HEAD/porcelain mismatch, if `--no-commit` or `--optlevel=1` is missing, or if the label's
pid/done files already exist; it double-forks so training survives SSH exit and writes
`done.json` with the runner's exit code.

## Remote command allowlist

Only argv lists pass `guard_remote_argv`: programs `git test mkdir rmdir rm cat tail sha256sum df ps
fuser hostname stat ls true` plus the pinned python (launcher only). `git` is limited to
`clone/checkout/rev-parse/status/log/show`, only inside the runs base, never forced. No shell
metacharacters and no `kill`, `sudo`, `aws`, `ec2`, `ssh`, `rsync`, or `curl`. `rmdir` is allowed
only for lock directories. `rm` accepts exactly one manifest-bound `final.pt` inside a fresh
autoloop run, and only through hash-verified `prune_checkpoint`; it cannot remove directories or
any other file. `scp` uploads only into the scratch dir and downloads only from autoloop dirs.

## Ladder and verdict rules (`autoloop/analyst.py`)

- Correctness gates (any failure = `FAILED`, branch closed, no relaunch): runner exit 0, ledger
  `status == keep`, `over_budget False`, `causality_passed True`, `eval_over_timeout False`, finite
  `public_val_bpb`, `final.pt` present, `git_dirty` explained only by the run's own log/ledger.
- Smoke: candidate steps/s ≥ control × (1 − 0.02), plus every machine-readable proposal-specific
  `smoke_gate` threshold, → `KEEP` (eligible for full) else `DISCARD`. Proposal gates can only
  tighten the global floor.
- Full: steps within 10 % of control else `CONFOUNDED`; `Δbpb ≤ −0.002` → `KEEP` else `DISCARD`.
- Controls are re-run per (box, shape, commit) by this loop; historical rows are evidence only.
- Thresholds live in `caps` and are frozen into each proposal's `*-verdict.json`.

## Durable record

```
research/agentic/experiments/<proposal_id>/
  proposal.json  verification.json
  implementation.patch  implementation-checks.json  implementation.json  # code proposals
  <tier>-inventory.json  <tier>-manifest.json  <tier>-staged.json  <tier>-launch.json
  <tier>-result.json  <tier>-verdict.json  <tier>-verdict.md
  artifacts/<tier>/{<label>.log, experiment-log.jsonl, runner.log, done.json, launch.json}
```

Each tier is committed (`git add/commit` of that directory only; plain `git push` only if
`git_push: true`) **before** the next launch anywhere. Closing a proposal appends a row to
`DECISION_LOG.md`. Runtime state (`runtime/autoloop_state.json`) and bundles are git-ignored.
Raw model replies are retained only under ignored `runtime/model-replies/`; they are never committed.
After result/verdict evidence is committed and pushed, the controller hash-verifies and prunes
smoke checkpoints, control checkpoints, and non-winning full checkpoints. A full candidate `KEEP`
checkpoint remains for human review. Each prune receipt is committed separately.

Verified full-run winners also trigger `.github/workflows/trainium-winner-alert.yml`. The workflow
opens one deduplicated GitHub issue assigned to the account named by the repository variable
`TRAINIUM_ALERT_ASSIGNEE`. GitHub handles the private email routing; no recipient address, SMTP
credential, Bedrock call, or OpenAI call is stored in or made by the workflow. It triggers only for
`full-verdict.json` with `decision == KEEP` and `bpb_delta <= -0.002`; submission remains manual.

## Restart semantics

State is written atomically before every remote step. On restart: `launched:*` is polled, never
relaunched (the remote launcher also refuses a second launch of the same label); `collected:*` is
analyzed; `analyzed:*` is archived; `staged:*` reuses the exact persisted immutable checkout and
label only after `op.launch` revalidates the controller lock, idle/device/disk inventory, HEAD,
porcelain, and source hash. A pre-existing launcher directory makes an ambiguous launch fail
closed. Labels are globally unique across the local ledger, archived ledgers, and state.

## Circuit breakers (no new launches; running training is left to finish)

STOP file · cutoff 2026-09-30 23:59 PT · invalid config · daily/total run caps · daily LLM
call/spend caps · archive/push failure · lock held by another controller · two consecutive
infrastructure failures on a box · any `RemoteError`/`ManifestError` (staging proof, hash
mismatch, missing artifact, tampered manifest) · state schema mismatch.

Malformed/unfinished Researcher output creates a 30-minute backoff instead of spending on every
two-minute poll. A deliberate `none` response backs off for one hour.

Fable's always-on reasoning is given a 64,000-token response ceiling. A live 16,000-token call
exhausted its allowance entirely in hidden reasoning and returned no proposal text. Failed-call
receipts retain stop reason and token usage under ignored `runtime/model-replies/` for diagnosis.

The standing unattended configuration permits at most 48 launches per UTC day and 384 total
through the cutoff. The independent Bedrock ceiling remains $10/day and 60 model calls/day.

## Commands

```bash
cp research/agentic/autoloop.example.json research/agentic/autoloop.json   # edit model ids
.agent-venv/bin/python research/agentic/autoloop.py status      # local gates only
.agent-venv/bin/python research/agentic/autoloop.py dry-run --tier full  # fixed mode; no remote call
.agent-venv/bin/python research/agentic/autoloop.py preflight   # read-only remote inventory
.agent-venv/bin/python research/agentic/autoloop.py once        # one cycle
.agent-venv/bin/python research/agentic/autoloop.py daemon      # until STOP/breaker
python -m unittest discover -s research/agentic/tests -v        # mocked tests
```

## Zero-token notifications

`.github/workflows/trainium-winner-alert.yml` creates an assigned GitHub issue after a durable,
comparable full-run KEEP with BPB delta at most -0.002. GitHub's normal issue-assignment email is
the delivery channel; no OpenAI model or API token is involved.

The local `trainium-autoloop-watchdog.timer` checks every five minutes for an inactive controller
or an open controller circuit breaker. It creates one deduplicated assigned GitHub issue when human
action is required. An intentional `research/agentic/STOP` suppresses the inactive-service alert.
The watchdog is read-only with respect to AWS, Trn2 workers, experiment state, and submissions.

## Human checklist before the first unattended cycle

1. `git fetch` so commit `9a4caa7b` is present locally (`status` reports it).
2. Confirm which DNS name is `trn21` and which is `trn22`; `preflight` prints each hostname.
   Swap the mapping in `config.py` if the protocol box ids are reversed.
3. Confirm `/home/ubuntu/.cache/neuron_competition_r1` is the live data cache on both hosts
   (otherwise set `NEURON_COMPETITION_R1_CACHE_DIR` in the manifest env — this is a code change).
4. Run `preflight`; both boxes must be clean and idle with ≥ 8 GiB free.
5. Run `bedrock_team.py doctor` for both model ids; the loop refuses same-family
   Researcher/Verifier.
6. Run `once` and inspect `experiments/control-*/smoke-*` before allowing `daemon`.

## Limitations and assumptions (read before trusting a result)

- The controller and both control ladders were live-tested on 2026-09-22. Code-change behavior is
  covered by an isolated worktree/commit unit test and still fails closed at every remote gate.
- The trn21/trn22 ↔ DNS mapping is an assumption from the order the hosts were given.
- The data cache location on the hosts is assumed from `.gitignore`'s comment; `preflight` fails
  closed if the directory is missing but cannot verify its contents.
- `research/run_experiment.py` at the pinned commit is assumed to accept the argv shape in
  `manifest.build_manifest` (`--label --solution --description --num-steps --max-train-seconds
  --nproc-per-node --lnc --no-commit --extra-args …`). That matches the current `main` runner.
- Flag existence is checked by regex over `train.py` at the pinned commit; a flag registered by
  an unusual helper could be missed (rejected, never wrongly accepted).
- Controls and candidates on a box may run on different days; the protocol's same-day rule for
  rate comparisons is recorded (timestamps in results) but not enforced beyond same box/commit.
- Retry attempts overwrite `<tier>-manifest.json`; the abandoned staged directory remains on the
  host under its `-rN` name and is never cleaned by the loop.
- The loop cannot kill anything. A run that exceeds the controller timeout is marked `FAILED`
  and observed; the box will fail inventory until it exits, which opens the breaker after two
  cycles.
- Per-run cost: each new box requires a 5-minute and a 30-minute control before any candidate.
- Leaderboard submission, AWS lifecycle, and merges to the submission candidate remain human
  actions. Development code candidates are isolated branches and are never merged automatically.
