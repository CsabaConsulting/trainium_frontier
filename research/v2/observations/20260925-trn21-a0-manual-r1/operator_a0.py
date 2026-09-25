"""One-label trn21 A0 break-glass transport; no science source edits."""

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
SCIENCE = Path("/home/poojith-devan/Desktop/Trainium-r10-science-local-20260925/train.py")
ASSIGNMENT = Path("/home/poojith-devan/Desktop/Trainium/research/v2/assignments/20260925-trn21-a0-breakglass-r1.json")
TRAIN_SHA = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREP_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
DEST = "/home/ubuntu/trainium_a0_warm_20260925_trn21_manual_r1"
E65 = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2"
CACHE = "/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train"
S0 = "/home/ubuntu/trainium_s0_cold_20260925_trn21_manual_r1"
DATA = "/home/ubuntu/.cache/neuron_competition_r1"
LABEL = "fs-20260925-v2-c5c-a0-control-trn21-r1"
LOCK = "/home/ubuntu/.cache/trainium_freshstart/device0.lock"
ARGV = ["--n-embd", "1024", "--no-ngram-ve", "--no-qk-shift", "--compile-sdpa-direct", "--nki-local-conv", "--bf16-norm-output", "--pack-factor", "32", "--clip-after-reduce", "--seq-len", "1024", "--no-eval-public", "--matrix-lr", "0.015"]
REMOTE = r'''
import datetime, hashlib, json, os, pathlib, re, socket, stat, subprocess, sys
MODE = C["mode"]
DEST = pathlib.Path(C["dest"])
E65 = pathlib.Path(C["e65"])
CACHE = pathlib.Path(C["cache"])
S0 = pathlib.Path(C["s0"])
DATA = pathlib.Path(C["data"])
LOCK = pathlib.Path(C["lock"])
def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(4 * 1024 * 1024)
            if not b: break
            h.update(b)
    return h.hexdigest()
def run(args):
    return subprocess.run(args, capture_output=True, text=True, timeout=30)
def check():
    assert socket.gethostname() == "ip-172-31-34-85", "HOSTNAME"
    assert os.geteuid() == 1000, "UID"
    st = os.statvfs("/")
    free = st.f_bavail * st.f_frsize
    assert free >= 10 * 1024**3, "DISK"
    for p in ("/home/ubuntu/research/agentic/STOP", "/home/ubuntu/research/v2/STOP", "/home/ubuntu/trainium_freshstart_20260918/research/agentic/STOP", "/home/ubuntu/trainium_freshstart_20260918/research/v2/STOP", str(E65 / "research/agentic/STOP"), str(E65 / "research/v2/STOP")):
        assert not os.path.lexists(p), "STOP"
    assert sha(E65 / "train.py") == C["train_sha"], "E65_TRAIN"
    assert sha(E65 / "prepare.py") == C["prep_sha"], "PREPARE"
    gh = run(["git", "-C", str(E65), "rev-parse", "HEAD"])
    assert gh.returncode == 0 and gh.stdout.strip() == "e65a3097a5b00f6178188a935bb2acb9a5042c6c", "E65_COMMIT"
    gs = run(["git", "-C", str(E65), "status", "--porcelain=v1"])
    assert gs.returncode == 0 and not gs.stdout.strip(), "E65_DIRTY"
    expected = {"neff": 2945951, "local": 2945952, "hlo": 2945953, "inductor": 2945954, "compiler": 2945955, "debug": 2945956}
    assert CACHE.is_dir() and CACHE.stat().st_ino == 2945950, "WARM_CACHE_ROOT"
    for name, inode in expected.items():
        p = CACHE / name
        assert p.is_dir() and p.stat().st_ino == inode, "WARM_CACHE_" + name
    assert (S0 / "run-exit-code.txt").read_text().strip() == "1", "S0_NOT_FAILED"
    assert not (S0 / "out/final.pt").exists(), "S0_CHECKPOINT_UNEXPECTED"
    assert DATA.is_dir(), "DATA_CACHE"
    for shard in range(5):
        assert (DATA / "data" / f"shard_{shard:05d}.parquet").is_file(), "TRAIN_SHARD"
    assert (DATA / "data/shard_06542.parquet").is_file(), "PUBLIC_VAL_SHARD"
    # /proc scan excludes the diagnostic SSH process because its source is stdin.
    hits = []
    own = os.getpid()
    for p in pathlib.Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == own: continue
        try: b = (p / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        words = [x.decode("utf-8", "replace") for x in b.split(b"\0") if x]
        if any(any(term in w for term in ("torchrun", "train.py", "prepare.py", "phase_timer", "reconcile", "run_experiment.py")) for w in words):
            hits.append(int(p.name))
    assert not hits, "HOST_PROCESS"
    for p in ("/dev/neuron0", str(LOCK)):
        f = run(["fuser", p])
        assert f.returncode == 1 and not f.stdout.strip(), "HOLDER"
    nl = run(["neuron-ls", "-j", "-a"])
    assert nl.returncode == 0, "NEURON_LS"
    devices = json.loads(nl.stdout)
    holders = [x.get("pid") for d in devices for x in d.get("neuron_processes", []) if not str(x.get("command", "")).startswith("neuron-ls")]
    assert not holders, "NEURON_PROCESS"
    if MODE in ("preflight", "stage"):
        assert not os.path.lexists(DEST), "DEST_EXISTS"
    else:
        assert DEST.is_dir() and DEST.stat().st_uid == 1000, "DEST_BAD"
        assert sha(DEST / "train.py") == C["train_sha"], "STAGED_TRAIN"
        assert not (DEST / "launch.json").exists(), "ALREADY_LAUNCHED"
        assert not (DEST / "training.log").exists(), "LOG_EXISTS"
    return {"utc": utc(), "free_bytes": free, "host": socket.gethostname(), "warm_cache_inodes_match": True, "idle": True, "source_match": True, "dest_absent": not os.path.lexists(DEST)}
facts = check()
if MODE == "preflight":
    print(json.dumps({"status": "PASS_PREFLIGHT", **facts}, sort_keys=True))
elif MODE == "stage":
    os.mkdir(DEST, 0o700)
    print(json.dumps({"status": "DEST_CREATED", **facts}, sort_keys=True))
elif MODE == "poststage":
    print(json.dumps({"status": "PASS_POSTSTAGE", **facts}, sort_keys=True))
elif MODE == "launch":
    env = {
        "HOME": "/home/ubuntu", "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
        "NKI_VENV_PATH": "/home/ubuntu/workspace/native_venv", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(E65), "NEURON_COMPETITION_R1_CACHE_DIR": str(DATA),
        "NEURON_LOGICAL_NC_CONFIG": "2", "NEURON_CC_FLAGS": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
        "NEURON_COMPETITION_R1_SEED": "42", "NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS": "1800",
        "NEURON_COMPETITION_R1_OUT_DIR": str(DEST / "out"),
        "TORCH_NEURONX_NEFF_CACHE_DIR": str(CACHE / "neff"), "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": str(CACHE / "local"),
        "TORCH_NEURONX_HLO_CACHE_DIR": str(CACHE / "hlo"), "TORCHINDUCTOR_CACHE_DIR": str(CACHE / "inductor"),
        "NEURON_COMPILE_CACHE_DIR": str(CACHE / "compiler"), "TORCH_NEURONX_DEBUG_DIR": str(CACHE / "debug"),
    }
    argv = ["/usr/bin/flock", "-n", str(LOCK), "/usr/bin/timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "3600s", "/home/ubuntu/workspace/native_venv/bin/torchrun", "--standalone", "--nproc_per_node=4", "train.py", *C["argv"]]
    monitor = "import datetime,os,subprocess,sys\nprint('TRAIN_START_UTC='+datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)\nrc=subprocess.call(" + repr(argv) + ",cwd=" + repr(str(DEST)) + ",env=os.environ.copy())\nopen(" + repr(str(DEST / "run-exit-code.txt")) + ",'x').write(str(rc)+'\\n')\nprint('TRAIN_END_UTC='+datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)\nsys.exit(rc)\n"
    started = utc()
    with open(DEST / "launch.json", "x") as f:
        json.dump({"schema": "trn21-a0-manual-detached-launch-v1", "label": C["label"], "source_sha256": C["train_sha"], "prepare_sha256": C["prep_sha"], "utc": started, "cwd": str(DEST), "command": argv, "environment": env, "mode": "warm_cache_manual_one_attempt"}, f, sort_keys=True)
        f.write("\n")
    with open(DEST / "training.log", "x") as out:
        proc = subprocess.Popen(["/usr/bin/python3", "-I", "-B", "-c", monitor], cwd=DEST, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print(json.dumps({"status": "STARTED", "utc": started, "monitor_pid": proc.pid, "source_sha256": C["train_sha"], "label": C["label"]}, sort_keys=True))
else:
    raise RuntimeError("MODE")
'''


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"preflight", "stage", "poststage", "launch"}:
        raise SystemExit("mode: preflight|stage|poststage|launch")
    mode = sys.argv[1]
    assignment = json.loads(ASSIGNMENT.read_text())
    assert digest(SCIENCE) == TRAIN_SHA
    assert assignment["train_sha256"] == TRAIN_SHA
    assert assignment["prepare_sha256"] == PREP_SHA
    assert assignment["argv"] == ARGV
    assert assignment["label"] == LABEL and assignment["launch_authorized"] is True
    assert dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(assignment["expires_at_utc"].replace("Z", "+00:00"))
    spec = importlib.util.spec_from_file_location("pinned_snapshot", SNAPSHOT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    before = module.local_state()
    assert not before["local_stops_present"]
    argv, display, identity, tsha = module.load_transport("trn21")
    argv[-1] = "exec /usr/bin/python3 -I -B -"
    cfg = {"mode": mode, "dest": DEST, "e65": E65, "cache": CACHE, "s0": S0, "data": DATA, "lock": LOCK, "label": LABEL, "argv": ARGV, "train_sha": TRAIN_SHA, "prep_sha": PREP_SHA}
    source = ("import json\nC = json.loads(" + repr(json.dumps(cfg, sort_keys=True)) + ")\n" + REMOTE).encode()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    result = subprocess.run(argv, input=source, capture_output=True, timeout=150, check=False)
    ended = dt.datetime.now(dt.timezone.utc).isoformat()
    after = module.local_state()
    receipt = {"schema": "trn21-a0-manual-operator-transport-v1", "mode": mode, "started_utc": started, "ended_utc": ended, "source_sha256": TRAIN_SHA, "assignment_sha256": digest(ASSIGNMENT), "remote_source_sha256": hashlib.sha256(source).hexdigest(), "ssh_returncode": result.returncode, "ssh_stdout": result.stdout.decode("utf-8", "replace"), "ssh_stderr": result.stderr.decode("utf-8", "replace"), "local_before": before, "local_after": after, "transport_sha256": tsha}
    p = HERE / (mode + "-" + started.replace(":", "").replace("+00:00", "Z") + ".json")
    with p.open("x") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"receipt": str(p), "ssh_returncode": result.returncode, "stdout": receipt["ssh_stdout"], "stderr_bytes": len(result.stderr)}, sort_keys=True))
    if result.returncode != 0 or result.stderr:
        raise SystemExit("transport or remote gate failed; inspect local receipt; do not retry a consumed label")


if __name__ == "__main__":
    main()
