"""One-use trn21 S0 120-second warm-cache compile smoke; pending assignment PASS."""

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
ASSIGNMENT = Path("/home/poojith-devan/Desktop/Trainium/research/v2/assignments/20260925-trn21-s0-warm-compile-smoke-r1.json")
SCIENCE = ROOT / "workflow/candidates/S0/train.py"
TRAIN_SHA = "ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb"
PREP_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
LABEL = "fs-20260925-s0-warm-compile-smoke-trn21-r1"
REMOTE = r'''
import datetime, hashlib, json, os, pathlib, socket, subprocess, sys
DEST = pathlib.Path(C["dest"])
STEP_LOG = pathlib.Path(C["step_log"])
A0 = pathlib.Path(C["a0"])
PREP = pathlib.Path(C["prepare"])
DATA = pathlib.Path("/home/ubuntu/.cache/neuron_competition_r1")
LOCK = pathlib.Path("/home/ubuntu/.cache/trainium_freshstart/device0.lock")
CACHES = {key: pathlib.Path(value) for key, value in C["caches"].items()}
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()
def run(args, env=None): return subprocess.run(args, capture_output=True, text=True, timeout=45, env=env)
def occupancy(path):
    count = total = 0
    for root, dirs, files in os.walk(path):
        count += len(files)
        for name in files:
            try: total += os.stat(os.path.join(root, name)).st_size
            except FileNotFoundError: pass
    return {"files": count, "bytes": total}
def host_check():
    assert socket.gethostname() == "ip-172-31-34-85", "HOSTNAME"
    assert os.geteuid() == 1000, "UID"
    assert STEP_LOG == DEST / "receipts/step-boundaries-rank0.log", "STEP_LOG_PATH"
    free = os.statvfs("/").f_bavail * os.statvfs("/").f_frsize
    assert free >= 10 * 1024**3, "DISK"
    assert (A0 / "run-exit-code.txt").is_file() and (A0 / "eval-exit-code.txt").is_file(), "A0_NOT_TERMINAL"
    for p in ("/home/ubuntu/research/agentic/STOP", "/home/ubuntu/research/v2/STOP", "/home/ubuntu/trainium_freshstart_20260918/research/agentic/STOP", "/home/ubuntu/trainium_freshstart_20260918/research/v2/STOP", str(PREP.parent / "research/agentic/STOP"), str(PREP.parent / "research/v2/STOP")):
        assert not os.path.lexists(p), "STOP"
    assert sha(PREP) == C["prep_sha"], "PREPARE_FILE"
    assert DATA.is_dir() and all((DATA / "data" / f"shard_{i:05d}.parquet").is_file() for i in range(5)), "TRAIN_DATA"
    assert (DATA / "data/shard_06542.parquet").is_file(), "PUBLIC_VAL_DATA"
    assert (DATA / "tokenizer/tokenizer.pkl").is_file() and (DATA / "tokenizer/token_bytes.pt").is_file(), "TOKENIZER"
    native = "/home/ubuntu/workspace/native_venv/bin/python"
    imported = run([native, "-B", "-c", "import hashlib,json,prepare; from pathlib import Path; p=Path(prepare.__file__).resolve(); print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cache_dir':str(prepare.CACHE_DIR)}))"], env={"HOME": "/home/ubuntu", "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin", "PYTHONPATH": str(PREP.parent), "PYTHONDONTWRITEBYTECODE": "1"})
    assert imported.returncode == 0, "PREPARE_IMPORT"
    imported_obj = json.loads(imported.stdout.strip())
    assert imported_obj["path"] == str(PREP) and imported_obj["sha256"] == C["prep_sha"] and imported_obj["cache_dir"] == str(DATA), "PREPARE_IMPORT_IDENTITY"
    expected = {"TORCH_NEURONX_NEFF_CACHE_DIR": 2945951, "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": 2945952, "TORCH_NEURONX_HLO_CACHE_DIR": 2945953, "TORCHINDUCTOR_CACHE_DIR": 2945954, "NEURON_COMPILE_CACHE_DIR": 2945955, "TORCH_NEURONX_DEBUG_DIR": 2945956}
    observed = {}
    for key, inode in expected.items():
        p = CACHES[key]
        assert p.is_dir() and p.stat().st_ino == inode, "CACHE_IDENTITY_" + key
        observed[key] = {"path": str(p), "inode": inode, **occupancy(p)}
    assert observed["TORCH_NEURONX_NEFF_CACHE_DIR"]["files"] > 0 and observed["TORCH_NEURONX_HLO_CACHE_DIR"]["files"] > 0, "WARM_CACHE_EMPTY"
    url = CACHES["NEURON_COMPILE_CACHE_URL"]
    assert url.is_dir(), "PERSISTENT_CACHE_URL"
    persistent = {"path": str(url), "inode": url.stat().st_ino, **occupancy(url)}
    assert persistent["files"] > 0 and persistent["bytes"] > 0, "PERSISTENT_CACHE_EMPTY"
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
    label_scan = run(["find", "/home/ubuntu", "-xdev", "-name", "*" + C["label"] + "*", "-print"])
    assert label_scan.returncode == 0 and not label_scan.stdout.strip(), "LABEL_ALREADY_PRESENT"
    if C["mode"] in ("preflight", "stage"):
        assert not os.path.lexists(DEST), "DEST_EXISTS"
    else:
        assert DEST.is_dir() and DEST.stat().st_uid == 1000, "DEST_BAD"
        assert sha(DEST / "train.py") == C["train_sha"], "STAGED_TRAIN"
        assert not (DEST / "launch.json").exists() and not (DEST / "training.log").exists(), "ALREADY_LAUNCHED"
        assert not os.path.lexists(STEP_LOG.parent) and not os.path.lexists(STEP_LOG), "STEP_LOG_EXISTS"
    return {"utc": utc(), "free_bytes": free, "prepare_import": imported_obj, "warm_caches": observed, "persistent_cache": persistent, "idle": True}
facts = host_check()
if C["mode"] == "preflight":
    print(json.dumps({"status": "PASS_PREFLIGHT", **facts}, sort_keys=True))
elif C["mode"] == "stage":
    os.mkdir(DEST, 0o700)
    print(json.dumps({"status": "DEST_CREATED", **facts}, sort_keys=True))
elif C["mode"] == "poststage":
    print(json.dumps({"status": "PASS_POSTSTAGE", **facts}, sort_keys=True))
elif C["mode"] == "launch":
    os.mkdir(STEP_LOG.parent, 0o700)
    step_fd = os.open(STEP_LOG, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(step_fd)
    assert STEP_LOG.stat().st_size == 0 and STEP_LOG.stat().st_uid == 1000 and STEP_LOG.stat().st_mode & 0o777 == 0o600, "STEP_LOG_PRECREATE"
    env = {
        "HOME": "/home/ubuntu", "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
        "NKI_VENV_PATH": "/home/ubuntu/workspace/native_venv", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(PREP.parent), "NEURON_LOGICAL_NC_CONFIG": "2", "TRN_EXP_LABEL": C["label"],
        "NEURON_COMPETITION_R1_STEP_LOG": str(STEP_LOG),
        **{key: str(path) for key, path in CACHES.items()},
    }
    assert "NEURON_CC_FLAGS" not in env and [x for x in env if x.startswith("NEURON_COMPETITION_R1_")] == ["NEURON_COMPETITION_R1_STEP_LOG"], "INHERITED_OVERRIDE"
    argv = ["/usr/bin/flock", "-n", str(LOCK), "/usr/bin/timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "2400s", "/home/ubuntu/workspace/native_venv/bin/torchrun", "--standalone", "--nproc_per_node=4", "train.py", "--max-train-seconds", "120"]
    monitor = "import datetime,os,subprocess,sys\nprint('TRAIN_START_UTC='+datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)\nrc=subprocess.call(" + repr(argv) + ",cwd=" + repr(str(DEST)) + ",env=os.environ.copy())\nopen(" + repr(str(DEST / "run-exit-code.txt")) + ",'x').write(str(rc)+'\\n')\nprint('TRAIN_END_UTC='+datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)\nsys.exit(rc)\n"
    started = utc()
    with open(DEST / "launch.json", "x") as f:
        json.dump({"schema": "trn21-s0-warm-120s-smoke-detached-launch-v1", "label": C["label"], "source_sha256": C["train_sha"], "prepare_sha256": C["prep_sha"], "utc": started, "cwd": str(DEST), "command": argv, "environment": env, "rank0_step_log": str(STEP_LOG), "rank0_step_log_inode": STEP_LOG.stat().st_ino, "rank0_step_log_precreated_empty_exclusive": True, "success_gate": "rank0 >=20 boundaries and torchrun all-four-ranks exit 0 with checkpoint/final barrier", "cache_before": facts["warm_caches"], "persistent_cache_before": facts["persistent_cache"], "effective_compiler_flags_expected_from_source": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16", "not_a_full_budget_run": True}, f, sort_keys=True)
        f.write("\n")
    with open(DEST / "training.log", "x") as out:
        proc = subprocess.Popen(["/usr/bin/python3", "-I", "-B", "-c", monitor], cwd=DEST, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print(json.dumps({"status": "STARTED", "utc": started, "monitor_pid": proc.pid, "source_sha256": C["train_sha"], "label": C["label"]}, sort_keys=True))
else: raise RuntimeError("MODE")
'''


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"preflight", "stage", "poststage", "launch"}:
        raise SystemExit("mode: preflight|stage|poststage|launch")
    mode = sys.argv[1]
    assignment = json.loads(ASSIGNMENT.read_text())
    assert assignment["label"] == LABEL and assignment["train_sha256"] == TRAIN_SHA
    assert assignment["trusted_prepare_sha256"] == PREP_SHA and assignment["launch_authorized"] is True
    assert assignment["training_command"] == "NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --max-train-seconds 120"
    assert assignment["step_log_path"] == assignment["run_root"] + "/receipts/step-boundaries-rank0.log"
    assert digest(SCIENCE) == TRAIN_SHA
    assert dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(assignment["expires_at_utc"].replace("Z", "+00:00"))
    spec = importlib.util.spec_from_file_location("pinned_snapshot", SNAPSHOT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    before = module.local_state()
    assert not before["local_stops_present"]
    argv, _, _, transport_sha = module.load_transport("trn21")
    argv[-1] = "exec /usr/bin/python3 -I -B -"
    cfg = {"mode": mode, "dest": assignment["run_root"], "step_log": assignment["step_log_path"], "a0": "/home/ubuntu/trainium_a0_warm_20260925_trn21_manual_r1", "prepare": assignment["trusted_prepare_host_path"], "caches": assignment["warm_cache_paths"], "label": LABEL, "train_sha": TRAIN_SHA, "prep_sha": PREP_SHA}
    source = ("import json\nC = json.loads(" + repr(json.dumps(cfg, sort_keys=True)) + ")\n" + REMOTE).encode()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    result = subprocess.run(argv, input=source, capture_output=True, timeout=180, check=False)
    ended = dt.datetime.now(dt.timezone.utc).isoformat()
    receipt = {"schema": "trn21-s0-warm-120s-smoke-operator-transport-v1", "mode": mode, "started_utc": started, "ended_utc": ended, "source_sha256": TRAIN_SHA, "assignment_sha256": digest(ASSIGNMENT), "remote_source_sha256": hashlib.sha256(source).hexdigest(), "ssh_returncode": result.returncode, "ssh_stdout": result.stdout.decode("utf-8", "replace"), "ssh_stderr": result.stderr.decode("utf-8", "replace"), "local_before": before, "local_after": module.local_state(), "transport_sha256": transport_sha}
    p = HERE / (mode + "-" + started.replace(":", "").replace("+00:00", "Z") + ".json")
    with p.open("x") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"receipt": str(p), "ssh_returncode": result.returncode, "stdout": receipt["ssh_stdout"], "stderr_bytes": len(result.stderr)}, sort_keys=True))
    if result.returncode != 0 or result.stderr:
        raise SystemExit("preflight or remote action failed; inspect receipt; do not reuse a consumed label")


if __name__ == "__main__":
    main()
