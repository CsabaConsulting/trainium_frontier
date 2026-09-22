# AUTOLOOP — unattended flag-only experiment controller

Implemented 2026-09-22 under `research/agentic/STANDING_AUTONOMOUS_AUTHORIZATION.md`. Code lives in
`research/agentic/autoloop/`; CLI is `research/agentic/autoloop.py`; tests are in
`research/agentic/tests/`. Nothing in this document waives `AGENTS.md`.

## What it does

A deterministic Python loop that, on each cycle:

1. Stops if `research/agentic/STOP` exists, a circuit breaker is open, the competition cutoff has
   passed, or the config is invalid.
2. Reconciles running work: polls launched runs, collects declared artifacts with SHA-256 checks,
   analyzes against the matched control, archives evidence to git.
3. If a box is idle and everything prior is archived, advances that box one rung on the ladder:
   `control smoke → control full → candidate smoke → (KEEP) candidate full → verdict`.
4. When a new candidate is needed, asks the Researcher model for **one flag-only proposal**
   (existing `train.py` flags at the pinned commit; no code edits), then requires an
   **independent Verifier** (different model family) to return `PASS` after deterministic checks.

Models never see a shell, the SSH key, the manifest, or hardware. They return JSON; the controller
validates and decides.

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

Only argv lists pass `guard_remote_argv`: programs `git test mkdir rmdir cat tail sha256sum df ps
fuser hostname stat ls true` plus the pinned python (launcher only). `git` is limited to
`clone/checkout/rev-parse/status/log/show`, only inside the runs base, never forced. No shell
metacharacters, no `rm`, `kill`, `sudo`, `aws`, `ec2`, `ssh`, `rsync`, `curl`. `rmdir` is allowed
only for lock directories. `scp` uploads only into the scratch dir and downloads only from
autoloop directories.

## Ladder and verdict rules (`autoloop/analyst.py`)

- Correctness gates (any failure = `FAILED`, branch closed, no relaunch): runner exit 0, ledger
  `status == keep`, `over_budget False`, `causality_passed True`, `eval_over_timeout False`, finite
  `public_val_bpb`, `final.pt` present, `git_dirty` explained only by the run's own log/ledger.
- Smoke: candidate steps/s ≥ control × (1 − 0.02) → `KEEP` (eligible for full) else `DISCARD`.
- Full: steps within 10 % of control else `CONFOUNDED`; `Δbpb ≤ −0.002` → `KEEP` else `DISCARD`.
- Controls are re-run per (box, shape, commit) by this loop; historical rows are evidence only.
- Thresholds live in `caps` and are frozen into each proposal's `*-verdict.json`.

## Durable record

```
research/agentic/experiments/<proposal_id>/
  proposal.json  verification.json
  <tier>-inventory.json  <tier>-manifest.json  <tier>-staged.json  <tier>-launch.json
  <tier>-result.json  <tier>-verdict.json  <tier>-verdict.md
  artifacts/<tier>/{<label>.log, experiment-log.jsonl, runner.log, done.json, launch.json}
```

Each tier is committed (`git add/commit` of that directory only; plain `git push` only if
`git_push: true`) **before** the next launch anywhere. Closing a proposal appends a row to
`DECISION_LOG.md`. Runtime state (`runtime/autoloop_state.json`) and bundles are git-ignored.

## Restart semantics

State is written atomically before every remote step. On restart: `launched:*` is polled, never
relaunched (the remote launcher also refuses a second launch of the same label); `collected:*` is
analyzed; `analyzed:*` is archived; `staged:*` (crash between stage and launch) is abandoned and
re-staged under a new `-rN` label into a new directory. Labels are globally unique across the
local ledger, archived ledgers, and state.

## Circuit breakers (no new launches; running training is left to finish)

STOP file · cutoff 2026-09-30 23:59 PT · invalid config · daily/total run caps · daily LLM
call/spend caps · archive/push failure · lock held by another controller · two consecutive
infrastructure failures on a box · any `RemoteError`/`ManifestError` (staging proof, hash
mismatch, missing artifact, tampered manifest) · state schema mismatch.

The standing unattended configuration permits at most 48 launches per UTC day and 384 total
through the cutoff. The independent Bedrock ceiling remains $10/day and 60 model calls/day.

## Commands

```bash
cp research/agentic/autoloop.example.json research/agentic/autoloop.json   # edit model ids
.agent-venv/bin/python research/agentic/autoloop.py status      # local gates only
.agent-venv/bin/python research/agentic/autoloop.py dry-run --tier smoke -- --weight-ema
.agent-venv/bin/python research/agentic/autoloop.py preflight   # read-only remote inventory
.agent-venv/bin/python research/agentic/autoloop.py once        # one cycle
.agent-venv/bin/python research/agentic/autoloop.py daemon      # until STOP/breaker
python -m unittest discover -s research/agentic/tests -v        # mocked tests
```

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

- **Nothing here has been executed.** Files were written offline with no Bash, SSH, or Python
  available in this session. The unit tests are complete but unrun; run them first.
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
- Leaderboard submission, AWS lifecycle, merges to the submission candidate, and any code change
  remain human actions. The loop has no code path for them.
