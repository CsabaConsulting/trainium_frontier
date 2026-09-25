"""One fresh-process public-20M evaluation for completed trn21 A0 control."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SNAPSHOT = ROOT / "research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py"
ASSIGNMENT = Path("/home/poojith-devan/Desktop/Trainium/research/v2/assignments/20260925-trn21-a0-breakglass-r1.json")
DEST = "/home/ubuntu/trainium_a0_warm_20260925_trn21_manual_r1"
E65 = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2"
EVAL_CACHE = "/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/eval/caches"
DATA = "/home/ubuntu/.cache/neuron_competition_r1"
LOCK = "/home/ubuntu/.cache/trainium_freshstart/device0.lock"
TRAIN_SHA = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREP_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
REMOTE = r'''
import datetime, hashlib, json, os, pathlib, socket, subprocess, sys
DEST = pathlib.Path(C["dest"])
E65 = pathlib.Path(C["e65"])
CACHE = pathlib.Path(C["cache"])
DATA = pathlib.Path(C["data"])
LOCK = pathlib.Path(C["lock"])
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(4 * 1024 * 1024)
            if not b: break
            h.update(b)
    return h.hexdigest()
def run(args): return subprocess.run(args, capture_output=True, text=True, timeout=30)
assert socket.gethostname() == "ip-172-31-34-85", "HOSTNAME"
assert os.geteuid() == 1000, "UID"
assert os.statvfs("/").f_bavail * os.statvfs("/").f_frsize >= 10 * 1024**3, "DISK"
for p in ("/home/ubuntu/research/agentic/STOP", "/home/ubuntu/research/v2/STOP", "/home/ubuntu/trainium_freshstart_20260918/research/agentic/STOP", "/home/ubuntu/trainium_freshstart_20260918/research/v2/STOP", str(E65 / "research/agentic/STOP"), str(E65 / "research/v2/STOP")):
    assert not os.path.lexists(p), "STOP"
assert sha(DEST / "train.py") == C["train_sha"], "TRAIN_SOURCE"
assert sha(E65 / "prepare.py") == C["prep_sha"], "PREPARE_SOURCE"
assert (DEST / "run-exit-code.txt").read_text().strip() == "0", "TRAIN_NOT_SUCCESSFUL"
assert (DEST / "training.log").is_file() and (DEST / "training.log").stat().st_size > 1000, "TRAIN_LOG"
checkpoint = DEST / "out/final.pt"
assert checkpoint.is_file() and checkpoint.stat().st_size > 1000000, "CHECKPOINT"
checkpoint_sha = sha(checkpoint)
assert CACHE.is_dir() and all((CACHE / name).is_dir() for name in ("neff", "local", "hlo", "inductor", "compiler", "debug")), "EVAL_CACHE"
assert DATA.is_dir() and (DATA / "data/shard_06542.parquet").is_file(), "PUBLIC_DATA"
for p in ("/dev/neuron0", str(LOCK)):
    f = run(["fuser", p]); assert f.returncode == 1 and not f.stdout.strip(), "HOLDER"
nl = run(["neuron-ls", "-j", "-a"])
assert nl.returncode == 0 and not [x for d in json.loads(nl.stdout) for x in d.get("neuron_processes", []) if not str(x.get("command", "")).startswith("neuron-ls")], "NEURON_PROCESS"
hits = []
for p in pathlib.Path("/proc").iterdir():
    if not p.name.isdigit() or int(p.name) == os.getpid(): continue
    try: b = (p / "cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError): continue
    words = [x.decode("utf-8", "replace") for x in b.split(b"\0") if x]
    if any(any(term in w for term in ("torchrun", "train.py", "prepare.py", "run_experiment.py")) for w in words): hits.append(int(p.name))
assert not hits, "HOST_PROCESS"
assert not (DEST / "eval-launch.json").exists(), "ALREADY_EVALUATED"
assert not (DEST / "fresh-public20m.log").exists(), "EVAL_LOG_EXISTS"
facts = {"utc": utc(), "checkpoint_sha256": checkpoint_sha, "checkpoint_bytes": checkpoint.stat().st_size, "cache_root": str(CACHE), "host_idle": True}
if C["mode"] == "preflight":
    print(json.dumps({"status": "PASS_EVAL_PREFLIGHT", **facts}, sort_keys=True))
elif C["mode"] == "launch":
    env = {
        "HOME": "/home/ubuntu", "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
        "NKI_VENV_PATH": "/home/ubuntu/workspace/native_venv", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(E65),
        "NEURON_COMPETITION_R1_CACHE_DIR": str(DATA), "NEURON_LOGICAL_NC_CONFIG": "2",
        "NEURON_CC_FLAGS": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
        "TORCH_NEURONX_NEFF_CACHE_DIR": str(CACHE / "neff"), "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": str(CACHE / "local"),
        "TORCH_NEURONX_HLO_CACHE_DIR": str(CACHE / "hlo"), "TORCHINDUCTOR_CACHE_DIR": str(CACHE / "inductor"),
        "NEURON_COMPILE_CACHE_DIR": str(CACHE / "compiler"), "TORCH_NEURONX_DEBUG_DIR": str(CACHE / "debug"),
    }
    argv = ["/usr/bin/flock", "-n", str(LOCK), "/usr/bin/timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "1800s", "/home/ubuntu/workspace/native_venv/bin/python", str(E65 / "prepare.py"), "eval-public", "--train-py", str(DEST / "train.py"), "--checkpoint", str(checkpoint), "--device", "neuron", "--batch-size", "1", "--seq-len", "1024", "--eval-tokens", "20971520", "--timeout-seconds", "1200"]
    monitor = "import datetime,os,subprocess,sys\nprint('EVAL_START_UTC='+datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)\nrc=subprocess.call(" + repr(argv) + ",cwd=" + repr(str(DEST)) + ",env=os.environ.copy())\nopen(" + repr(str(DEST / "eval-exit-code.txt")) + ",'x').write(str(rc)+'\\n')\nprint('EVAL_END_UTC='+datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)\nsys.exit(rc)\n"
    started = utc()
    with open(DEST / "eval-launch.json", "x") as f:
        json.dump({"schema": "trn21-a0-manual-fresh-public20m-v1", "utc": started, "train_sha256": C["train_sha"], "prepare_sha256": C["prep_sha"], "checkpoint_sha256": checkpoint_sha, "checkpoint_bytes": checkpoint.stat().st_size, "command": argv, "environment": env}, f, sort_keys=True)
        f.write("\n")
    with open(DEST / "fresh-public20m.log", "x") as out:
        proc = subprocess.Popen(["/usr/bin/python3", "-I", "-B", "-c", monitor], cwd=DEST, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print(json.dumps({"status": "EVAL_STARTED", "utc": started, "monitor_pid": proc.pid, "checkpoint_sha256": checkpoint_sha}, sort_keys=True))
else: raise RuntimeError("MODE")
'''


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"preflight", "launch"}:
        raise SystemExit("mode: preflight|launch")
    mode = sys.argv[1]
    assignment = json.loads(ASSIGNMENT.read_text())
    assert assignment["train_sha256"] == TRAIN_SHA and assignment["prepare_sha256"] == PREP_SHA
    spec = importlib.util.spec_from_file_location("pinned_snapshot", SNAPSHOT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    before = module.local_state()
    assert not before["local_stops_present"]
    argv, _, _, transport_sha = module.load_transport("trn21")
    argv[-1] = "exec /usr/bin/python3 -I -B -"
    cfg = {"mode": mode, "dest": DEST, "e65": E65, "cache": EVAL_CACHE, "data": DATA, "lock": LOCK, "train_sha": TRAIN_SHA, "prep_sha": PREP_SHA}
    source = ("import json\nC = json.loads(" + repr(json.dumps(cfg, sort_keys=True)) + ")\n" + REMOTE).encode()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    result = subprocess.run(argv, input=source, capture_output=True, timeout=180, check=False)
    ended = dt.datetime.now(dt.timezone.utc).isoformat()
    receipt = {"schema": "trn21-a0-manual-eval-operator-transport-v1", "mode": mode, "started_utc": started, "ended_utc": ended, "assignment_sha256": hashlib.sha256(ASSIGNMENT.read_bytes()).hexdigest(), "remote_source_sha256": hashlib.sha256(source).hexdigest(), "ssh_returncode": result.returncode, "ssh_stdout": result.stdout.decode("utf-8", "replace"), "ssh_stderr": result.stderr.decode("utf-8", "replace"), "local_before": before, "local_after": module.local_state(), "transport_sha256": transport_sha}
    p = HERE / ("eval-" + mode + "-" + started.replace(":", "").replace("+00:00", "Z") + ".json")
    with p.open("x") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"receipt": str(p), "ssh_returncode": result.returncode, "stdout": receipt["ssh_stdout"], "stderr_bytes": len(result.stderr)}, sort_keys=True))
    if result.returncode != 0 or result.stderr:
        raise SystemExit("eval preflight or launch failed; inspect receipt and do not reuse label")


if __name__ == "__main__":
    main()
