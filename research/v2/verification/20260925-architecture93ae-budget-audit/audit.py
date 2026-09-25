"""Read-only, stdlib-only checks of archived evidence and exact timer AST."""
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
COMMIT = "1e21c3d797eed764f108c41546b270065ad5650c"
SOURCE_SHA = "93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5"
PREP_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
NATIVE = ROOT / "research/v2/operator/20260925-trn22-l2-93ae-cpu-r1/native"
checks = []


def check(condition, name):
    if not condition:
        raise RuntimeError(name)
    checks.append(name)


source = subprocess.check_output(
    ["git", "show", COMMIT + ":workflow/proposals/research-v2/train.py"], cwd=ROOT
)
check(hashlib.sha256(source).hexdigest() == SOURCE_SHA, "exact_93ae_source")
check(hashlib.sha256((ROOT / "prepare.py").read_bytes()).hexdigest() == PREP_SHA,
      "unchanged_prepare")
archive = NATIVE / "steady-progress-1814"
retrieval = json.loads((archive / "retrieval.json").read_text())
for name, info in retrieval["files"].items():
    raw = (archive / name).read_bytes()
    check(len(raw) == info["bytes"] and hashlib.sha256(raw).hexdigest() == info["sha256"],
          "archive_hash_" + name)
launch = json.loads((archive / "training-launch.json").read_text())
check(launch["train_sha256"] == SOURCE_SHA and launch["prepare_sha256"] == PREP_SHA,
      "launch_source_binding")
check("NEURON_COMPETITION_R1_STEP_LOG" not in launch["environment"], "no_official_step_observer")
argv = launch["command"]
check(argv[argv.index("--max-train-seconds") + 1] == "1800" and
      "--no-eval-public" in argv and "--unet-8x1024" in argv,
      "declared_training_budget_and_separate_eval")
log = (archive / "training.log").read_text()
steps = re.findall(r"^step (\d+) .*?dt ([0-9.]+)s .*?remaining_time (\d+)s", log, re.M)
check([int(row[0]) for row in steps[:5]] == list(range(5)), "first_five_present")
durations = [float(row[1]) for row in steps[:5]]
check(durations == [847.90, 695.40, .49, .49, .49], "first_five_durations")
# Rounded to .01 seconds; allow .005 per interval. These disjoint step
# intervals omit extra startup/inter-step work, so are only a lower bound.
startup_lower_bound = sum(durations) - .005 * 5
check(startup_lower_bound > 900 and steps[4][2] == "1111", "startup_cap_exceeded")
check("TRAINING_END_UTC=" not in log and "run-exit-code.txt" in retrieval["missing"],
      "exit_compliance_still_pending_at_snapshot")

tree = ast.parse(source)
boundary = [n for n in ast.walk(tree) if isinstance(n, ast.If) and
            ast.unparse(n.test) == "step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED"]
check(len(boundary) == 1, "unique_fifth_boundary_timer")
timer = compile(ast.Module(body=boundary[0].body, type_ignores=[]), "exact93ae_timer", "exec")
scenarios = []
for startup in [0., 800., 870., 899., 900., 1000., 1559., 3000.]:
    origin = 12345.
    env = {"time": SimpleNamespace(monotonic=lambda: origin + startup),
           "_PROCESS_STARTED": origin, "STARTUP_LAUNCH_MARGIN_SECONDS": 30.,
           "TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS": 900,
           "recent_step_times": [847.9, 695.4, .49, .49, .49]}
    exec(timer, env)
    charged = origin + startup - env["budget_started"]
    expected = max(0., startup + 30. - 900.)
    check(charged == expected and charged >= max(0., startup - 900.) and
          env["startup_allowance_est"] == min(startup, 900.) and
          env["recent_step_times"] == [], "timer_scenario_" + str(startup))
    if startup >= 900:
        check(env["budget_started"] == origin + 870., "cap_does_not_refund_excess_" + str(startup))
    scenarios.append({"synthetic_startup": startup, "charged_at_fifth": charged})

start_text = re.search(r"^TRAINING_START_UTC=(.+)$", log, re.M).group(1)
deadline = dt.datetime.fromisoformat(start_text) + dt.timedelta(seconds=2700)
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
    NATIVE / "host_driver.py", archive / "retrieval.json", archive / "training-launch.json",
    archive / "training.log", NATIVE / "startup-first-step-observations.json"]}
print(json.dumps({"status": "PASS_STATIC_ARCHIVED_EVIDENCE_ONLY", "check_count": len(checks),
                  "checks": checks, "source_sha256": SOURCE_SHA, "prepare_sha256": PREP_SHA,
                  "first_five_logged_dt_sum": round(sum(durations), 3),
                  "rounded_dt_only_startup_lower_bound": round(startup_lower_bound, 3),
                  "training_start_utc": start_text,
                  "conservative_logged_wall_2700s_reference_utc": deadline.isoformat(),
                  "scenarios": scenarios, "evidence_sha256": hashes}, indent=2))
