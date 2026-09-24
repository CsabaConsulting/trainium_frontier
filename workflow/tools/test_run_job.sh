#!/usr/bin/env bash
# Local-only end-to-end and adversarial checks for run_job.py. No Neuron or host access.
set -u
W="$(cd "$(dirname "$0")" && pwd)"
T="$(mktemp -d)"
R="$T/repo"
mkdir -p "$R/research/v2/experiments" "$T/archive"
cd "$R"
git init -q
git config user.email test@example.invalid
git config user.name run-job-test
printf "print('train')\n" > train.py
printf "# canonical prepare fixture\n" > prepare.py
printf "out/\n" > .gitignore
: > research/v2/ledger.jsonl
touch research/v2/experiments/.keep
git add train.py prepare.py .gitignore research
git commit -qm source
SRC_HEAD="$(git rev-parse HEAD)"
TRAIN_SHA="$(sha256sum train.py | cut -d' ' -f1)"
PREP_SHA="$(sha256sum prepare.py | cut -d' ' -f1)"
HOSTNAME_NOW="$(hostname)"

python3 - "$T/host.json" "$T/archive" "$T/device.lock" "$HOSTNAME_NOW" "$W" <<'PY'
import json, sys
out, archive, lock, hostname, tools = sys.argv[1:]
json.dump({
    "host": "test-host", "expected_hostname": hostname, "host_lock_path": lock,
    "python": "python3", "out_root": "out", "archive_dir": archive,
    "disk_floor_gib": 0.01, "use_cgroup": False, "test_mode": True,
    "quiescence_window_s": 0.05, "stop_grace_s": 0.2,
    "test_ignore_other_training_processes": True,
    "env": {"NEURON_LOGICAL_NC_CONFIG": "2"},
    "eval_args": ["--seq-len", "1024", "--eval-tokens", "20971520", "--timeout-seconds", "60"],
    "eval_args_verified": True,
    "train_cmd_override": ["python3", f"{tools}/fake/fake_train.py"],
    "eval_cmd_override": ["python3", f"{tools}/fake/fake_eval.py"],
    "causality_cmd_override": ["python3", f"{tools}/fake/fake_causality.py"],
    "protected_labels": [],
}, open(out, "w"))
PY

mkjob_raw() {
  label=$1 arm=$2 kind=$3 train_sha=${4:-$TRAIN_SHA}
  d="research/v2/experiments/$label"
  mkdir -p "$d"
  python3 - "$d" "$label" "$arm" "$kind" "$train_sha" "$TRAIN_SHA" "$PREP_SHA" "$SRC_HEAD" <<'PY'
import hashlib, json, pathlib, sys
d, label, arm, kind, train_sha, base_sha, prep_sha, source_commit = sys.argv[1:]
d = pathlib.Path(d)
argv = ["--seq-len", "1024", "--seed", "42", "--max-train-seconds", "1800", "--no-eval-public"]
g2 = {"verdict": "PASS", "train_py_sha256": base_sha, "argv": argv}
(d / "g2.json").write_text(json.dumps(g2, sort_keys=True) + "\n")
g2_sha = hashlib.sha256((d / "g2.json").read_bytes()).hexdigest()
job = {
    "label": label, "arm_id": arm, "kind": kind, "host": "test-host",
    "train_py": "train.py", "train_sha256": train_sha, "base_sha256": base_sha,
    "prepare_py_sha256": prep_sha, "git_commit": source_commit, "argv": argv,
    "g2_receipt": f"research/v2/experiments/{label}/g2.json", "g2_receipt_sha256": g2_sha,
    "footprint_gib": 0.01, "timeout_s": 1810, "eval_timeout_s": 120, "causality_timeout_s": 30,
}
(d / "job.json").write_text(json.dumps(job, sort_keys=True) + "\n")
PY
}

commit_job() {
  git add "research/v2/experiments/$1"
  git commit -qm "job $1"
}

commit_evidence() {
  git add research/v2
  git commit -qm "evidence $1"
}

runj() {
  label=$1; shift
  set +e
  env "$@" python3 "$W/run_job.py" --host-config "$T/host.json" \
    --job "$R/research/v2/experiments/$label/job.json" --repo "$R" \
    > "$T/$label.out" 2> "$T/$label.err"
  rc=$?
  set -e
  python3 - "$label" "$rc" "$T/$label.out" <<'PY'
import json, sys
label, rc, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(path))
print(f"{label:<12} rc={rc} " + str({k: d.get(k) for k in
      ("public_20m", "verdict", "anomalies", "failed_stage", "error") if d.get(k) not in (None, [])}))
PY
  return "$rc"
}

set -e

# Happy path: real JSON metrics, two controls, one arm, archive-before-prune.
mkjob_raw C1 A0 control; commit_job C1
runj C1 FAKE_BPB=0.9920
test -f out/C1/final.pt && test -f research/v2/experiments/C1/archive.json
commit_evidence C1

mkjob_raw C2 A0 control; commit_job C2
runj C2 FAKE_BPB=0.9918
commit_evidence C2

mkjob_raw F6 F6 flag_arm; commit_job F6
runj F6 FAKE_BPB=0.9905
test "$(python3 -c 'import json; print(json.load(open("research/v2/experiments/F6/verdict.json"))["verdict"])')" = NOMINATE
test ! -e out/F6/final.pt && test -f research/v2/experiments/F6/prune.json
commit_evidence F6

# A stopped escaped descendant is recorded as an anomaly and raises a human alert.
mkjob_raw E1 F7 flag_arm; commit_job E1
runj E1 FAKE_BPB=0.9930 FAKE_MODE=escape
python3 - <<'PY'
import json
r = json.load(open("research/v2/experiments/E1/record.json"))
v = json.load(open("research/v2/experiments/E1/verdict.json"))
assert r["status"] == "evaluated" and r["anomalies"] and v.get("human_alert")
PY
commit_evidence E1

# A launched failing label is consumed: an exact relaunch is refused before Popen.
mkjob_raw B1 F8 flag_arm; commit_job B1
if runj B1 FAKE_MODE=badloss; then exit 1; else test $? -eq 5; fi
test -f research/v2/experiments/B1/launch-train.json
commit_evidence B1
launch_sha="$(sha256sum research/v2/experiments/B1/launch-train.json)"
if runj B1; then exit 1; else test $? -eq 4; fi
test "$(sha256sum research/v2/experiments/B1/launch-train.json)" = "$launch_sha"
commit_evidence B1-repeat-refusal

# Wrong bytes and effective-env overrides fail before launch.
mkjob_raw X1 F9 flag_arm 0000000000000000000000000000000000000000000000000000000000000000; commit_job X1
if runj X1; then exit 1; else test $? -eq 4; fi
test ! -e research/v2/experiments/X1/launch-train.json
commit_evidence X1

mkjob_raw ENV1 F10 flag_arm; commit_job ENV1
if runj ENV1 NEURON_CC_FLAGS=--optlevel=0; then exit 1; else test $? -eq 4; fi
test ! -e research/v2/experiments/ENV1/launch-train.json
commit_evidence ENV1

mkjob_raw CFGENV F10b flag_arm; commit_job CFGENV
python3 - "$T/host.json" "$T/bad-env.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["env"]["NEURON_CC_FLAGS"] = "--optlevel=0"
json.dump(d, open(sys.argv[2], "w"))
PY
set +e
python3 "$W/run_job.py" --host-config "$T/bad-env.json" \
  --job "$R/research/v2/experiments/CFGENV/job.json" --repo "$R" \
  > "$T/CFGENV.out" 2> "$T/CFGENV.err"
cfg_rc=$?
set -e
test "$cfg_rc" -eq 4 && test ! -e research/v2/experiments/CFGENV/launch-train.json
commit_evidence CFGENV

# Full cleanliness includes untracked source. A preflight-only refusal remains retryable.
mkjob_raw DIRTY F10c flag_arm; commit_job DIRTY
printf 'untracked\n' > unexpected-source.txt
if runj DIRTY; then exit 1; else test $? -eq 4; fi
test ! -e research/v2/experiments/DIRTY/claim.json
rm unexpected-source.txt
commit_evidence DIRTY-refusal
runj DIRTY FAKE_BPB=0.9922
commit_evidence DIRTY-retry

# The canonical host lock is exclusive across distinct labels. Exactly one runner launches;
# the lock-refused (preflight-only) label remains retryable after the active evidence is committed.
mkjob_raw LOCK1 FLOCK1 flag_arm; commit_job LOCK1
mkjob_raw LOCK2 FLOCK2 flag_arm; commit_job LOCK2
env FAKE_MODE=slow python3 "$W/run_job.py" --host-config "$T/host.json" \
  --job "$R/research/v2/experiments/LOCK1/job.json" --repo "$R" \
  > "$T/LOCK1.out" 2> "$T/LOCK1.err" &
lock1_pid=$!
for _ in $(seq 1 100); do
  test -f research/v2/experiments/LOCK1/launch-train.json && break
  sleep 0.02
done
if runj LOCK2; then exit 1; else test $? -eq 4; fi
set +e
wait "$lock1_pid"
lock1_rc=$?
set -e
test "$lock1_rc" -eq 7
test -f research/v2/experiments/LOCK1/launch-train.json
test ! -f research/v2/experiments/LOCK2/launch-train.json
commit_evidence LOCK-contention
runj LOCK2 FAKE_BPB=0.9921
commit_evidence LOCK2-retry

# Missing/partial evaluation JSON and false causality never enter the ledger.
mkjob_raw EVBAD F11 flag_arm; commit_job EVBAD
if runj EVBAD FAKE_EVAL_MODE=missing; then exit 1; else test $? -eq 6; fi
test -f out/EVBAD/final.pt && ! grep -q '"label": "EVBAD"' research/v2/ledger.jsonl
commit_evidence EVBAD

mkjob_raw CBAD F12 flag_arm; commit_job CBAD
if runj CBAD FAKE_CAUSALITY=fail; then exit 1; else test $? -eq 6; fi
test -f out/CBAD/final.pt && ! grep -q '"label": "CBAD"' research/v2/ledger.jsonl
commit_evidence CBAD

# Archive/record/ledger/verdict injected failures all preserve the source checkpoint.
for stage in archive record ledger verdict; do
  label="I-${stage}"
  mkjob_raw "$label" "I-$stage" flag_arm; commit_job "$label"
  python3 - "$T/host.json" "$T/host-$stage.json" "$stage" <<'PY'
import json, sys
src, dst, stage = sys.argv[1:]
d = json.load(open(src)); d["inject_failure_stage"] = stage
json.dump(d, open(dst, "w"))
PY
  set +e
  python3 "$W/run_job.py" --host-config "$T/host-$stage.json" \
    --job "$R/research/v2/experiments/$label/job.json" --repo "$R" \
    > "$T/$label.out" 2> "$T/$label.err"
  rc=$?
  set -e
  if test "$rc" -ne 7 || test ! -f "out/$label/final.pt"; then
    printf 'injected %s failure did not preserve checkpoint (rc=%s)\n' "$stage" "$rc" >&2
    exit 1
  fi
  commit_evidence "$label"
done

# Production G2 cannot be replaced by the tiny local fixture.
python3 - "$W" "$TRAIN_SHA" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import run_job
sha = sys.argv[2]
job = {"train_sha256": sha, "argv": ["--seed", "42"]}
try:
    run_job.validate_g2({"verdict": "PASS", "train_py_sha256": sha, "argv": job["argv"]}, job, "x" * 64, False)
except run_job.JobError:
    pass
else:
    raise SystemExit("tiny production G2 unexpectedly accepted")
PY

# Every malformed evaluator variant is refused; no token/context facts are synthesized.
python3 - "$W" "$T" <<'PY'
import json, math, pathlib, sys
sys.path.insert(0, sys.argv[1])
import run_job
tmp = pathlib.Path(sys.argv[2]); base = {
  "val_bpb": .99, "eval_seconds": 1.0, "eval_timeout": 60,
  "eval_over_timeout": False, "num_tokens": 20971520, "num_bytes": 1000,
  "total_nats": .99 * math.log(2) * 1000,
}
variants = []
for change in ("extra", "wrong_tokens", "nan", "wrong_math"):
    row = dict(base)
    if change == "extra": row["seq_len"] = 1024
    if change == "wrong_tokens": row["num_tokens"] -= 1
    if change == "nan": row["val_bpb"] = float("nan")
    if change == "wrong_math": row["total_nats"] += 1
    variants.append((change, json.dumps(row)))
variants.append(("multiple", json.dumps(base) + "\n" + json.dumps(base)))
for name, text in variants:
    path = tmp / f"eval-{name}.log"; path.write_text(text)
    try:
        run_job.parse_eval(path, 60)
    except run_job.JobError:
        pass
    else:
        raise SystemExit(f"malformed eval accepted: {name}")
PY

# Missing/duplicate/non-finite training summaries fail closed.
python3 - "$W" "$T" <<'PY'
import pathlib, sys
sys.path.insert(0, sys.argv[1])
import run_job
tmp = pathlib.Path(sys.argv[2])
base = '''step 00000 | loss 3.1 | lrm 1 | dt 0.5s | tok/s 1 | remaining_steps 1 | remaining_time 1s | epoch 1
---
training_seconds: 1790
budget_seconds: 1800
max_train_seconds: 1800
startup_allowance_est: 100
charged_seconds_est: 1791
over_budget: False
num_steps: 10
total_tokens_M: 1
world_size: 4
seq_len: 1024
num_params_M: 1
'''
bad = {"missing": base.replace("world_size: 4\n", ""),
       "duplicate": base + "num_steps: 10\n",
       "nonfinite": base.replace("training_seconds: 1790", "training_seconds: nan")}
for name, text in bad.items():
    path = tmp / f"train-{name}.log"; path.write_text(text)
    try:
        run_job.parse_training_log(path)
    except run_job.JobError:
        pass
    else:
        raise SystemExit(f"malformed summary accepted: {name}")
PY

# Confinement and symlinked roots are rejected without touching their targets.
python3 - "$W" "$R" "$T" <<'PY'
import pathlib, sys
sys.path.insert(0, sys.argv[1])
import run_job
repo, tmp = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
try:
    run_job.confined(repo, "../escape", "test")
except run_job.JobError:
    pass
else:
    raise SystemExit("traversal accepted")
target = tmp / "real-root"; target.mkdir()
link = tmp / "linked-root"; link.symlink_to(target, target_is_directory=True)
try:
    run_job.ensure_plain_dir(link)
except run_job.JobError:
    pass
else:
    raise SystemExit("symlink root accepted")
PY

# Reusing a correct content-addressed archive never replaces it; symlink checkpoints are refused.
python3 - "$W" "$T" <<'PY'
import hashlib, os, pathlib, sys
sys.path.insert(0, sys.argv[1])
import run_job
tmp = pathlib.Path(sys.argv[2]); out = tmp / "archive-unit-out"; out.mkdir()
ck = out / "final.pt"; ck.write_bytes(b"fixed-checkpoint")
st = os.lstat(ck)
ident = {"device": st.st_dev, "inode": st.st_ino, "size": st.st_size,
         "mtime_ns": st.st_mtime_ns, "sha256": hashlib.sha256(ck.read_bytes()).hexdigest()}
cfg = {"archive_dir": str(tmp / "archive-unit")}
dst, _ = run_job.archive_checkpoint(cfg, ck, ident)
before = os.stat(dst)
dst2, receipt = run_job.archive_checkpoint(cfg, ck, ident)
after = os.stat(dst2)
assert receipt["reused_existing"] and before.st_ino == after.st_ino and before.st_mtime_ns == after.st_mtime_ns
real = out / "real.pt"; real.write_bytes(b"x")
link = out / "linked.pt"; link.symlink_to(real)
try:
    run_job.checkpoint_identity(link, out)
except run_job.JobError:
    pass
else:
    raise SystemExit("symlink checkpoint accepted")
PY

rows="$(wc -l < research/v2/ledger.jsonl)"
left="$(python3 - "$W" <<'PY'
import sys
sys.path.insert(0, sys.argv[1]); import contained
labels = ["C1", "C2", "F6", "E1", "B1", "X1", "ENV1", "CFGENV", "DIRTY",
          "LOCK1", "LOCK2", "EVBAD", "CBAD"]
live = {x: contained.tagged_pids(x) for x in labels}
print({k: v for k, v in live.items() if v})
PY
)"
test "$left" = "{}"
test "$rows" -eq 7
test "$(sha256sum "$W/decide.py" | cut -d' ' -f1)" = a47bedd258697b0791f48dcac3098d9071b5aaf00a7664d9e39987b4052adc66
printf 'PASS rows=%s no_leaked_processes scratch=%s\n' "$rows" "$T"
