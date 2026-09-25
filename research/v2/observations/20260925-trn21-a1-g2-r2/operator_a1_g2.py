"""One-use, CPU-only A1 G2 Operator under the exact enabled r2 assignment."""

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
CLEAN = Path("/home/poojith-devan/Desktop/Trainium-trn21-a1-g2-operator")
ASSIGNMENT = CLEAN / "research/v2/assignments/20260925-c5c-a1-exact-cpu-g2-trn21-r2.json"
PACKET = CLEAN / "research/v2/g2-probes/20260925-c5c-table-exact-g2-minimal/PACKETS.HOLD.json"
SOURCE = CLEAN / "submissions/research_v2/train.py"
TOOL = CLEAN / "workflow/tools/preflight_dryrun.py"
HEAD = "5a10c117decab7228f4e9db0d38dba4b9a608848"
ASSIGNMENT_SHA = "a34b15dfe9c9dfd20d3cca1b656d79f0540a82e2266ea0b8393a7f70101615ff"
PACKET_SHA = "23b357834922bf410f46d78d68f91b578536930c5528983317eb1988bdd6178f"
SOURCE_SHA = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
TOOL_SHA = "6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8"
PREP_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
LABEL = "fs-20260925-c5c-a1-exact-g2-trn21-r2"
MIN_DISK = 34359738368
MIN_MEM = 68719476736
FLOOR_DISK = 6979321856

REMOTE = r'''
import datetime, hashlib, json, os, pathlib, socket, subprocess, sys
R = pathlib.Path(C["run_root"])
S = pathlib.Path(C["source_host_path"])
T = pathlib.Path(C["tool_host_path"])
TMP = pathlib.Path(C["tmpdir"])
PREP = pathlib.Path(C["trusted_prepare_dir"]) / "prepare.py"
LOCK = pathlib.Path("/home/ubuntu/.cache/trainium_freshstart/device0.lock")
A0 = pathlib.Path("/home/ubuntu/trainium_a0_warm_20260925_trn21_manual_r1")
S0 = pathlib.Path("/home/ubuntu/trainium_s0_warm_smoke_20260925_trn21_r1")
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(4 * 1024 * 1024)
            if not b: break
            h.update(b)
    return h.hexdigest()
def run(argv, env=None, timeout=45):
    return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=timeout)
def check():
    assert socket.gethostname() == "ip-172-31-34-85" and os.geteuid() == 1000, "IDENTITY"
    st = os.statvfs("/")
    free = st.f_bavail * st.f_frsize
    assert free >= C["min_disk"] and st.f_favail >= 10000, "DISK_OR_INODES"
    mem = next(int(x.split()[1]) * 1024 for x in open("/proc/meminfo") if x.startswith("MemAvailable:"))
    assert mem >= C["min_mem"], "MEM_AVAILABLE"
    for p in ("/home/ubuntu/research/agentic/STOP", "/home/ubuntu/research/v2/STOP", "/home/ubuntu/trainium_freshstart_20260918/research/agentic/STOP", "/home/ubuntu/trainium_freshstart_20260918/research/v2/STOP", str(PREP.parent / "research/agentic/STOP"), str(PREP.parent / "research/v2/STOP")):
        assert not os.path.lexists(p), "HOST_STOP"
    assert (A0 / "run-exit-code.txt").read_text().strip() == "0" and (A0 / "eval-exit-code.txt").read_text().strip() == "0", "A0_NOT_SUCCESSFUL"
    assert (S0 / "run-exit-code.txt").read_text().strip() == "0", "S0_NOT_TERMINAL"
    assert PREP.is_file() and sha(PREP) == C["prep_sha"], "TRUSTED_PREPARE"
    for p in ("/dev/neuron0", str(LOCK)):
        f = run(["fuser", p]); assert f.returncode == 1 and not f.stdout.strip(), "DEVICE_OR_LOCK_HOLDER"
    n = run(["neuron-ls", "-j", "-a"])
    assert n.returncode == 0 and not [x for d in json.loads(n.stdout) for x in d.get("neuron_processes", []) if not str(x.get("command", "")).startswith("neuron-ls")], "NEURON_PROCESS"
    hits = []
    for p in pathlib.Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid(): continue
        try: b = (p / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        words = [x.decode("utf-8", "replace") for x in b.split(b"\0") if x]
        if any(any(term in w for term in ("torchrun", "train.py", "prepare.py", "preflight_dryrun.py", "run_experiment.py")) for w in words): hits.append(int(p.name))
    assert not hits, "HOST_PROCESS"
    scan = run(["find", "/home/ubuntu", "-xdev", "-name", "*" + C["label"] + "*", "-print"])
    assert scan.returncode == 0 and not scan.stdout.strip(), "LABEL_PRESENT"
    absent = [R, pathlib.Path(C["attempt_marker"]), pathlib.Path(C["g2_out"]), pathlib.Path(C["launcher_log"]), S, T, TMP]
    if C["mode"] in ("preflight", "stage"):
        assert all(not os.path.lexists(p) for p in absent), "RUN_PATH_ALREADY_PRESENT"
    else:
        assert R.is_dir() and R.stat().st_uid == 1000 and R.stat().st_mode & 0o777 == 0o700, "RUN_ROOT"
        assert all(p.is_dir() and p.stat().st_uid == 1000 and p.stat().st_mode & 0o777 == 0o700 for p in (S.parent, T.parent, TMP)), "PRIVATE_CHILDREN"
        assert sha(S) == C["source_sha"] and sha(T) == C["tool_sha"], "STAGED_SHA"
        assert not any(TMP.iterdir()), "TMPDIR_NOT_EMPTY"
        assert all(not os.path.lexists(p) for p in (pathlib.Path(C["attempt_marker"]), pathlib.Path(C["g2_out"]), pathlib.Path(C["launcher_log"]), R / "launch.json", R / "run-exit-code.txt")), "ATTEMPT_ALREADY_PRESENT"
        envlist = ["PATH=/home/ubuntu/workspace/native_venv/bin:/usr/bin:/bin", "HOME=/home/ubuntu", "TMPDIR=" + str(TMP), "PYTHONDONTWRITEBYTECODE=1"]
        probe = run(["/usr/bin/env", "-i", *envlist, C["cpu_python"], "-B", "-c", "import json,sys,tempfile,torch; print(json.dumps({'python':sys.executable,'torch':torch.__version__,'tmpdir':tempfile.gettempdir()}))"], timeout=60)
        assert probe.returncode == 0, "INTERPRETER_OR_TORCH"
        interpreter = json.loads(probe.stdout.strip().splitlines()[-1])
        assert interpreter["python"] == C["cpu_python"] and interpreter["tmpdir"] == str(TMP), "PRIVATE_TMPDIR_RESOLUTION"
    return {"utc": utc(), "free_bytes": free, "free_inodes": st.f_favail, "mem_available_bytes": mem, "host_idle": True, "prepare_sha256": C["prep_sha"], "interpreter": interpreter if C["mode"] not in ("preflight", "stage") else None}
facts = check()
if C["mode"] == "preflight":
    print(json.dumps({"status": "PASS_PREFLIGHT", **facts}, sort_keys=True))
elif C["mode"] == "stage":
    os.mkdir(R, 0o700)
    for p in (S.parent, T.parent, TMP): os.mkdir(p, 0o700)
    print(json.dumps({"status": "PRIVATE_ROOT_CREATED", **facts}, sort_keys=True))
elif C["mode"] == "poststage":
    print(json.dumps({"status": "PASS_POSTSTAGE", **facts}, sort_keys=True))
elif C["mode"] == "launch":
    envlist = ["PATH=/home/ubuntu/workspace/native_venv/bin:/usr/bin:/bin", "HOME=/home/ubuntu", "TMPDIR=" + str(TMP), "PYTHONDONTWRITEBYTECODE=1", "NEURON_LOGICAL_NC_CONFIG=2", "NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16", "NEURON_COMPETITION_R1_SEED=42", "NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800"]
    science = [C["cpu_python"], "-B", str(T), "--train-py", str(S), "--prepare-dir", str(PREP.parent), "--steps", "2", "--nproc", "4", "--micro-rows", "2", "--micro-tokens", "256", "--timeout", "3600", "--trace-rmsprop", "--keep", "--out", C["g2_out"], "--", *C["science_argv"]]
    argv = ["/usr/bin/flock", "-n", str(LOCK), "/usr/bin/timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "4200s", "/usr/bin/env", "-i", *envlist, *science]
    started = utc()
    marker = pathlib.Path(C["attempt_marker"])
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"label": C["label"], "created_utc": started, "assignment_sha256": C["assignment_sha"], "source_sha256": C["source_sha"], "tool_sha256": C["tool_sha"], "one_use": True}, f, sort_keys=True)
        f.write("\n")
    with open(R / "launch.json", "x") as f:
        json.dump({"schema": "trn21-a1-cpu-g2-one-use-launch-v1", "label": C["label"], "utc": started, "assignment_sha256": C["assignment_sha"], "packet_sha256": C["packet_sha"], "source_sha256": C["source_sha"], "tool_sha256": C["tool_sha"], "prepare_sha256": C["prep_sha"], "command": argv, "environment": envlist, "preflight": facts, "disk_floor_bytes": C["floor_disk"], "cpu_only": True}, f, sort_keys=True)
        f.write("\n")
    monitor = """import datetime,json,os,signal,subprocess,sys,time
ARGV = """ + repr(argv) + """
ROOT = """ + repr(str(R)) + """
FLOOR = """ + repr(C["floor_disk"]) + """
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def free():
    s=os.statvfs('/'); return s.f_bavail*s.f_frsize
print('G2_START_UTC='+utc(), flush=True)
p=subprocess.Popen(ARGV,cwd=ROOT,start_new_session=True)
minimum=free(); breached=False
while p.poll() is None:
    minimum=min(minimum,free())
    if minimum < FLOOR:
        breached=True
        os.killpg(p.pid,signal.SIGTERM)
        try: p.wait(timeout=60)
        except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
        break
    time.sleep(5)
rc=p.wait()
minimum=min(minimum,free())
with open(ROOT+'/run-exit-code.txt','x') as f: f.write(str(rc)+'\\n')
with open(ROOT+'/monitor-result.json','x') as f: json.dump({'ended_utc':utc(),'exit_code':rc,'disk_floor_breached':breached,'min_root_free_bytes':minimum},f,sort_keys=True); f.write('\\n')
print('G2_END_UTC='+utc(),flush=True)
sys.exit(rc)
"""
    with open(C["launcher_log"], "x") as log:
        proc = subprocess.Popen(["/usr/bin/python3", "-I", "-B", "-c", monitor], cwd=R, env={"HOME": "/home/ubuntu", "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print(json.dumps({"status": "G2_STARTED", "utc": started, "monitor_pid": proc.pid, "label": C["label"], "source_sha256": C["source_sha"], "tool_sha256": C["tool_sha"]}, sort_keys=True))
else: raise RuntimeError("MODE")
'''


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"preflight", "stage", "poststage", "launch"}:
        raise SystemExit("mode: preflight|stage|poststage|launch")
    mode = sys.argv[1]
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=CLEAN, capture_output=True, text=True, check=True).stdout.strip() == HEAD, "CLEAN_HEAD"
    assert not subprocess.run(["git", "status", "--porcelain"], cwd=CLEAN, capture_output=True, text=True, check=True).stdout, "DIRTY_SOURCE_WORKTREE"
    assert digest(ASSIGNMENT) == ASSIGNMENT_SHA and digest(PACKET) == PACKET_SHA, "AUTHORITY_SHA"
    assert digest(SOURCE) == SOURCE_SHA and digest(TOOL) == TOOL_SHA and digest(CLEAN / "prepare.py") == PREP_SHA, "LOCAL_BYTES"
    assignment = json.loads(ASSIGNMENT.read_text())
    packet = json.loads(PACKET.read_text())
    arm = next(p for p in packet["packets"] if p["arm"] == "A1" and p["host"] == "trn21")
    assert assignment["launch_authorized"] is True and assignment["neuron_training_authorized"] is False and assignment["label"] == LABEL, "ASSIGNMENT_DISABLED_OR_WRONG"
    assert assignment["packet_sha256"] == PACKET_SHA and assignment["source_sha256"] == SOURCE_SHA and assignment["g2_tool_sha256"] == TOOL_SHA, "ASSIGNMENT_PINS"
    assert arm["g2_label"] == LABEL and arm["run_root"] == assignment["run_root"], "PACKET_IDENTITY"
    assert dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(assignment["expires_at_utc"].replace("Z", "+00:00")), "EXPIRED"
    assert (CLEAN / "research/v2/verification/20260925-trn21-c5c-a0-r1-result-independent-PASS.md").is_file(), "A0_REVIEW_MISSING"
    assert (CLEAN / "research/v2/verification/20260925-c5c-a2-a1-exact-cpu-g2-assignments-independent-PASS.md").is_file(), "ASSIGNMENT_REVIEW_MISSING"
    spec = importlib.util.spec_from_file_location("pinned_snapshot", SNAPSHOT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    before = module.local_state()
    assert not before["local_stops_present"] and before["breaker_tripped"]["global_correctness"] is False and before["breaker_tripped"]["trn21"] is False, "LOCAL_STOP_OR_BREAKER"
    argv, _, _, transport_sha = module.load_transport("trn21")
    argv[-1] = "exec /usr/bin/python3 -I -B -"
    cfg = {"mode": mode, "label": LABEL, "run_root": arm["run_root"], "attempt_marker": arm["attempt_marker"], "launcher_log": arm["launcher_log"], "trusted_prepare_dir": arm["trusted_prepare_dir"], "source_host_path": arm["source_host_path"], "tool_host_path": arm["tool_host_path"], "tmpdir": arm["tmpdir"], "g2_out": arm["g2_out"], "science_argv": arm["argv"], "cpu_python": packet["common"]["cpu_python"], "min_disk": MIN_DISK, "min_mem": MIN_MEM, "floor_disk": FLOOR_DISK, "source_sha": SOURCE_SHA, "tool_sha": TOOL_SHA, "prep_sha": PREP_SHA, "assignment_sha": ASSIGNMENT_SHA, "packet_sha": PACKET_SHA}
    source = ("import json\nC=json.loads(" + repr(json.dumps(cfg, sort_keys=True)) + ")\n" + REMOTE).encode()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    done = subprocess.run(argv, input=source, capture_output=True, timeout=180, check=False)
    ended = dt.datetime.now(dt.timezone.utc).isoformat()
    receipt = {"schema": "trn21-a1-cpu-g2-operator-transport-v1", "mode": mode, "started_utc": started, "ended_utc": ended, "clean_worktree_head": HEAD, "clean_status": True, "assignment_sha256": ASSIGNMENT_SHA, "packet_sha256": PACKET_SHA, "source_sha256": SOURCE_SHA, "tool_sha256": TOOL_SHA, "remote_source_sha256": hashlib.sha256(source).hexdigest(), "transport_sha256": transport_sha, "ssh_returncode": done.returncode, "ssh_stdout": done.stdout.decode("utf-8", "replace"), "ssh_stderr": done.stderr.decode("utf-8", "replace"), "local_before": before, "local_after": module.local_state()}
    path = HERE / (mode + "-" + started.replace(":", "").replace("+00:00", "Z") + ".json")
    with path.open("x") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"receipt": str(path), "ssh_returncode": done.returncode, "stdout": receipt["ssh_stdout"], "stderr_bytes": len(done.stderr)}, sort_keys=True))
    if done.returncode != 0 or done.stderr:
        raise SystemExit("G2 preflight/action failed; inspect receipt; no retry after attempt marker")


if __name__ == "__main__":
    main()
