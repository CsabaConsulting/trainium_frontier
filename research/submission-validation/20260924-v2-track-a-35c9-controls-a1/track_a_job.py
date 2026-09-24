#!/usr/bin/env python3
"""Run one immutable 35c9 control/A1 arm and emit sealed 2M evidence."""

import argparse
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import artifact_io as aio
import contracts as c

PYTHON = "/home/ubuntu/workspace/native_venv/bin/python"
PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def enable_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise SystemExit(f"cannot enable child subreaper: errno={ctypes.get_errno()}")
    observed = ctypes.c_int()
    if libc.prctl(PR_GET_CHILD_SUBREAPER, ctypes.byref(observed), 0, 0, 0) != 0 or observed.value != 1:
        raise SystemExit("child subreaper verification failed")


def reap_group(pgid):
    reaped = []
    while True:
        try:
            pid, _status = os.waitpid(-pgid, os.WNOHANG)
        except ChildProcessError:
            break
        if pid <= 0:
            break
        reaped.append(pid)
    return reaped


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise SystemExit(f"cannot inspect process group {pgid}") from error


def wait_group_gone(pgid, seconds, reaped):
    deadline = time.monotonic() + seconds
    while True:
        reaped.extend(reap_group(pgid))
        if not group_exists(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def quiesce_group(process, term_grace=15.0, kill_grace=10.0):
    """Reap one private session and reject even rc=0 when cleanup was required."""
    pgid = process.pid
    leader_returncode = process.poll()
    reaped = []
    cleanup_required = group_exists(pgid)
    term_sent = False
    kill_sent = False
    if cleanup_required:
        try:
            os.killpg(pgid, signal.SIGTERM)
            term_sent = True
        except ProcessLookupError:
            pass
        if leader_returncode is None:
            try:
                leader_returncode = process.wait(timeout=term_grace)
            except subprocess.TimeoutExpired:
                pass
        gone = wait_group_gone(pgid, term_grace, reaped)
        if not gone:
            try:
                os.killpg(pgid, signal.SIGKILL)
                kill_sent = True
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    leader_returncode = process.wait(timeout=kill_grace)
                except subprocess.TimeoutExpired:
                    pass
            gone = wait_group_gone(pgid, kill_grace, reaped)
        if not gone:
            raise SystemExit(f"process group {pgid} survived TERM/KILL escalation")
    if process.poll() is None:
        try:
            leader_returncode = process.wait(timeout=0.2)
        except subprocess.TimeoutExpired as error:
            raise SystemExit("leader remained after process-group quiescence") from error
    else:
        leader_returncode = process.returncode
    reaped.extend(reap_group(pgid))
    if group_exists(pgid):
        raise SystemExit(f"process group {pgid} reappeared after quiescence")
    return {
        "pgid": pgid, "leader_returncode": leader_returncode,
        "cleanup_required": cleanup_required, "term_sent": term_sent,
        "kill_sent": kill_sent, "quiescent": True,
        "reaped_descendants": sorted(set(reaped)),
        "term_grace_seconds": float(term_grace), "kill_grace_seconds": float(kill_grace),
    }


def run_capture(argv, cwd, env, timeout, term_grace=15.0, kill_grace=10.0):
    enable_subreaper()
    started = time.monotonic()
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        lifecycle = quiesce_group(process, term_grace, kill_grace)
        output, _ = process.communicate()
    else:
        lifecycle = quiesce_group(process, term_grace, kill_grace)
    effective_returncode = process.returncode
    if lifecycle["cleanup_required"] and effective_returncode == 0:
        effective_returncode = 125
    return effective_returncode, output, time.monotonic() - started, timed_out, lifecycle


def run_training(argv, sandbox, env, timeout, term_grace=15.0, kill_grace=10.0):
    enable_subreaper()
    raw_log = sandbox / "training.raw.log"
    step_log = sandbox / "steps.jsonl"
    started = time.monotonic()
    observed = []
    with raw_log.open("xb") as output:
        process = subprocess.Popen(argv, cwd=sandbox, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout:
                timed_out = True
                break
            try:
                steps = step_log.read_text().splitlines()
            except FileNotFoundError:
                steps = []
            while len(observed) < len(steps):
                observed.append({"step": int(steps[len(observed)]), "observed_seconds": elapsed})
            time.sleep(0.1)
        lifecycle = quiesce_group(process, term_grace, kill_grace)
        output.flush()
        os.fsync(output.fileno())
    elapsed = time.monotonic() - started
    effective_returncode = process.returncode
    if lifecycle["cleanup_required"] and effective_returncode == 0:
        effective_returncode = 125
    return effective_returncode, aio.regular_bytes(raw_log), elapsed, timed_out, observed, lifecycle


def fresh_caches(root, namespace):
    result = {}
    for key, name in (
        ("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"),
        ("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "local"),
        ("TORCH_NEURONX_HLO_CACHE_DIR", "hlo"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("NEURON_COMPILE_CACHE_DIR", "compiler"),
        ("TORCH_NEURONX_DEBUG_DIR", "debug"),
    ):
        path = root / "caches" / namespace / name
        path.mkdir(parents=True, exist_ok=False)
        if list(path.iterdir()):
            raise SystemExit("fresh cache not empty")
        result[key] = str(path)
    return result


def cache_observations(paths):
    observations = {}
    for key, value in sorted(paths.items()):
        path = Path(value)
        files = []
        total = 0
        for candidate in sorted(path.rglob("*")):
            if candidate.is_symlink():
                raise SystemExit("cache symlink rejected")
            if candidate.is_file():
                size = candidate.stat().st_size
                total += size
                files.append({"relative": str(candidate.relative_to(path)), "bytes": size})
        observations[key] = {"path": value, "bytes": total, "file_count": len(files), "files": files}
    return observations


def finite_scan(checkpoint, env):
    source = (
        "import hashlib,json,os,sys,torch\n"
        "p=sys.argv[1];fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);h=hashlib.sha256();z=0\n"
        "while True:\n b=os.read(fd,8388608)\n if not b:break\n h.update(b);z+=len(b)\n"
        "os.close(fd);v=torch.load(p,map_location='cpu',weights_only=True);s=[v];n=0\n"
        "while s:\n x=s.pop()\n if isinstance(x,dict):s.extend(x.values())\n elif isinstance(x,(list,tuple)):s.extend(x)\n elif isinstance(x,torch.Tensor):\n  n+=1\n  assert not (x.is_floating_point() or x.is_complex()) or torch.isfinite(x).all().item()\n"
        "print(json.dumps({'checkpoint_bytes':z,'checkpoint_sha256':h.hexdigest(),'finite':True,'map_location':'cpu','tensor_count':n,'weights_only':True},sort_keys=True))\n"
    )
    argv = [PYTHON, "-c", source, str(checkpoint)]
    return argv, run_capture(argv, checkpoint.parent, env, 600)


def make_archive(store):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in sorted(c.ARCHIVE_MEMBERS):
            data = store.read(name)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def verify_sources(request, core, head):
    source = ROOT / c.SOURCE_PATH
    if aio.digest(aio.regular_bytes(source)) != c.SOURCE_SHA:
        raise SystemExit("working immutable source mismatch")
    source_object = subprocess.check_output(["git", "show", f"{c.SOURCE_COMMIT}:{c.SOURCE_PATH}"], cwd=ROOT)
    if aio.digest(source_object) != c.SOURCE_SHA:
        raise SystemExit("immutable source Git object mismatch")
    if aio.digest(aio.regular_bytes(ROOT / "prepare.py")) != c.PREPARE_SHA:
        raise SystemExit("prepare.py mismatch")
    if aio.digest(aio.regular_bytes(ROOT / "README.md")) != c.RULES_SHA:
        raise SystemExit("competition rules mismatch")
    for relative, expected in request["source_sha256"].items():
        working = aio.regular_bytes(ROOT / relative)
        core_blob = subprocess.check_output(["git", "show", f"{core}:{relative}"], cwd=ROOT)
        head_blob = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
        if working != core_blob or core_blob != head_blob or aio.digest(working) != expected:
            raise SystemExit(f"protected source mismatch: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=c.ARMS, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--runtime-job", type=Path, required=True)
    args = parser.parse_args()
    arm = args.arm
    spec = c.ARMS[arm]
    if socket.gethostname() != spec["hostname"] or git("symbolic-ref", "--short", "HEAD") != c.BRANCH:
        raise SystemExit("wrong host/branch")
    head = git("rev-parse", "HEAD")
    rel = HERE.relative_to(ROOT)
    auth_path = (ROOT / args.authorization).absolute()
    request_path = (ROOT / args.request).absolute()
    runtime_path = (ROOT / args.runtime_job).absolute()
    if (auth_path != HERE / f"{arm}-authorization.json"
            or request_path != HERE / f"{arm}-request.json"
            or runtime_path != ROOT / "research/run-logs" / f"{spec['label']}.job.json"):
        raise SystemExit("runtime path contract mismatch")
    auth_bytes = aio.regular_bytes(auth_path)
    request_bytes = aio.regular_bytes(request_path)
    auth = c.json_object(auth_bytes, "authorization", canonical=True)
    request = c.json_object(request_bytes, "request", canonical=True)
    core = auth.get("verified_core_commit", "")
    if not c.HEX40.fullmatch(core or "") or subprocess.run(["git", "merge-base", "--is-ancestor", core, head], cwd=ROOT).returncode:
        raise SystemExit("verified core is not an ancestor")
    allowed_finalization = {str(rel / f"{name}-{suffix}.json") for name in c.ARMS for suffix in ("authorization", "request")}
    changed = set(git("diff", "--name-only", core, head).splitlines())
    if changed != allowed_finalization:
        raise SystemExit("finalization path set mismatch")
    request_sha = c.validate_authorization(auth, request, arm, core)
    authorization_sha = aio.digest(auth_bytes)
    concrete = c.build_concrete(request, authorization_sha, request_sha, head)
    runtime_bytes = aio.regular_bytes(runtime_path)
    runtime = c.json_object(runtime_bytes, "runtime job", canonical=False)
    runtime_sha = c.validate_runtime(runtime, concrete, runtime_bytes)
    verify_sources(request, core, head)
    control_identity = None
    if spec["role"] == "a1":
        control_identity = c.validate_control_evidence(c.GitReader(auth["control_evidence"]), auth["control_evidence"])
    free_bytes = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if c.disk_disposition(free_bytes) != "ALLOWED":
        raise SystemExit(f"disk gate: {c.disk_disposition(free_bytes)}")
    store = aio.Store(ROOT, spec["label"])
    work = ROOT / "out" / f"{spec['label']}-work"
    if store.path.exists() or work.exists():
        raise SystemExit("non-overwrite output/work path already exists")
    store.create()
    work.mkdir(mode=0o700)
    sandbox = work / "sandbox"
    sandbox.mkdir(mode=0o700)
    shutil.copy2(ROOT / c.SOURCE_PATH, sandbox / "train.py")
    shutil.copy2(ROOT / "prepare.py", sandbox / "prepare.py")
    if aio.digest(aio.regular_bytes(sandbox / "train.py")) != c.SOURCE_SHA or aio.digest(aio.regular_bytes(sandbox / "prepare.py")) != c.PREPARE_SHA:
        raise SystemExit("sandbox source copy mismatch")
    child_env = dict(c.ENVIRONMENT)
    child_env["TRAINIUM_FRESH_START_RUN_ID"] = spec["label"]
    train_caches = fresh_caches(work, "train")
    child_env.update(train_caches)
    child_env["NEURON_COMPETITION_R1_OUT_DIR"] = str(sandbox / "out")
    child_env["NEURON_COMPETITION_R1_STEP_LOG"] = str(sandbox / "steps.jsonl")
    check_argv = [PYTHON, "prepare.py", "check", "--train-py", "train.py", "--launch-command", spec["launch"]]
    check_rc, check_output, check_wall, check_timeout, check_lifecycle = run_capture(
        check_argv, sandbox, child_env, 120
    )
    store.write("contract-check.log", check_output)
    store.write_json("contract-check-process.json", {
        "schema": "track-a-contract-check-process-v1", "argv": check_argv,
        "returncode": check_rc, "timed_out": check_timeout, "wall_seconds": check_wall,
        "process_group": check_lifecycle,
    })
    if check_rc or check_timeout:
        raise SystemExit("organizer contract check failed")
    c.validate_clean_lifecycle(check_lifecycle)
    train_rc, train_output, train_wall, train_timeout, observations, train_lifecycle = run_training(
        c.train_argv(arm), sandbox, child_env, 3000
    )
    store.write("training.log", train_output)
    try:
        step_bytes = aio.regular_bytes(sandbox / "steps.jsonl")
    except FileNotFoundError:
        step_bytes = b""
    store.write("training-steps.jsonl", step_bytes)
    allowance = min(observations[4]["observed_seconds"], 900.0) if len(observations) >= 5 else None
    charged = train_wall - allowance if allowance is not None else None
    training_process = {
        "schema": "track-a-training-process-v1", "argv": c.train_argv(arm),
        "launch_command": spec["launch"], "returncode": train_rc, "timed_out": train_timeout,
        "wall_seconds": train_wall, "startup_allowance_seconds": allowance,
        "charged_through_exit_seconds": charged,
        "charged_within_budget": charged is not None and charged <= 1800.0,
        "observations": observations, "seed": 42, "world_size": 4, "lnc": 2,
        "process_group": train_lifecycle,
    }
    store.write_json("training-process.json", training_process)
    if train_rc or train_timeout or len(observations) < 5:
        raise SystemExit("full training failed or lacked startup boundaries")
    if not training_process["charged_within_budget"]:
        raise SystemExit("training budget gate failed")
    c.validate_clean_lifecycle(train_lifecycle)
    checkpoint = sandbox / "out" / "final.pt"
    checkpoint_data = aio.regular_bytes(checkpoint)
    checkpoint_sha = aio.digest(checkpoint_data)
    checkpoint_bytes = len(checkpoint_data)
    scan_env = dict(c.ENVIRONMENT)
    scan_env["TRAINIUM_FRESH_START_RUN_ID"] = spec["label"]
    scan_argv, (scan_rc, scan_output, scan_wall, scan_timeout, scan_lifecycle) = finite_scan(checkpoint, scan_env)
    store.write("checkpoint-finite.log", scan_output)
    if scan_rc or scan_timeout:
        raise SystemExit("finite checkpoint scan failed")
    c.validate_clean_lifecycle(scan_lifecycle)
    try:
        scan = json.loads(scan_output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise SystemExit("finite receipt parse failed") from error
    if (set(scan) != {"checkpoint_bytes", "checkpoint_sha256", "finite", "map_location", "tensor_count", "weights_only"}
            or scan["checkpoint_sha256"] != checkpoint_sha or scan["checkpoint_bytes"] != checkpoint_bytes
            or scan["finite"] is not True or scan["weights_only"] is not True or scan["map_location"] != "cpu"
            or type(scan["tensor_count"]) is not int or scan["tensor_count"] <= 0):
        raise SystemExit("finite receipt mismatch")
    finite_receipt = {
        "schema": "track-a-finite-v1", **scan, "argv": scan_argv,
        "wall_seconds": scan_wall, "process_group": scan_lifecycle,
    }
    store.write_json("checkpoint-finite.json", finite_receipt)
    eval_env = dict(c.ENVIRONMENT)
    eval_env["TRAINIUM_FRESH_START_RUN_ID"] = spec["label"]
    eval_caches = fresh_caches(work, "eval")
    eval_env.update(eval_caches)
    eval_argv = [
        PYTHON, "prepare.py", "eval-public", "--train-py", "train.py", "--checkpoint", str(checkpoint),
        "--device", "neuron", "--batch-size", "1", "--seq-len", "1024",
        "--eval-tokens", str(c.SCREEN_TOKENS), "--timeout-seconds", "1200",
    ]
    eval_rc, eval_output, eval_wall, eval_timeout, eval_lifecycle = run_capture(
        eval_argv, sandbox, eval_env, 1200
    )
    store.write("screen2m.log", eval_output)
    if eval_rc or eval_timeout or eval_wall > 1200.0:
        raise SystemExit("fresh 2M evaluation failed")
    c.validate_clean_lifecycle(eval_lifecycle)
    metrics = c.parse_metric_output(eval_output.decode(errors="replace"))
    metric_receipt = {
        "schema": "track-a-screen2m-metrics-v1", "argv": eval_argv,
        "metrics": metrics, "wall_seconds": eval_wall, "process_group": eval_lifecycle,
    }
    store.write_json("screen2m.metrics.json", metric_receipt)
    if aio.identity(aio.regular_bytes(checkpoint)) != {"sha256": checkpoint_sha, "bytes": checkpoint_bytes}:
        raise SystemExit("checkpoint changed during finite/eval")
    decision = c.a1_decision(control_identity["screen2m_val_bpb"], metrics["val_bpb"]) if control_identity else None
    observations_receipt = {
        "schema": "track-a-cache-observations-v1", "train": cache_observations(train_caches),
        "eval": cache_observations(eval_caches), "cleanup_authorized": False,
        "artifacts_deleted": [], "caches_deleted": [],
    }
    store.write_json("cache-observations.json", observations_receipt)
    result = {
        "schema": "track-a-result-v1", "arm": arm, "role": spec["role"], "label": spec["label"],
        "host": spec["host"], "hostname": spec["hostname"], "source_commit": c.SOURCE_COMMIT,
        "source_path": c.SOURCE_PATH, "source_sha256": c.SOURCE_SHA, "prepare_sha256": c.PREPARE_SHA,
        "launch_command": spec["launch"], "seed": 42, "world_size": 4, "lnc": 2,
        "max_train_seconds": 1800, "smoke": False, "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha, "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_preserved": True, "checkpoint_prune_authorized": False,
        "cache_cleanup_authorized": False, "finite_scan": finite_receipt, "screen2m": metrics,
        "screening_only": True, "keep_authorized": False, "adoption_authorized": False,
        "a1_decision": decision, "ready_to_retrieve": False,
    }
    store.write_json("result.json", result)
    archive = make_archive(store)
    store.write("evidence.tar.gz", archive)
    members = store.inspect(c.ARTIFACTS)
    manifest = {"schema": "track-a-artifact-manifest-v1", "label": spec["label"], "members": members}
    manifest_identity = store.write_json("artifact-manifest.json", manifest)
    completion = {
        "schema": "track-a-completion-v1", "arm": arm, "role": spec["role"], "label": spec["label"],
        "host": spec["host"], "hostname": spec["hostname"], "finalization_head": head,
        "verified_core_commit": core, "authorization_sha256": authorization_sha,
        "request_sha256": request_sha, "runtime_job_sha256": runtime_sha,
        "source_sha256": c.SOURCE_SHA, "prepare_sha256": c.PREPARE_SHA,
        "artifact_manifest_sha256": manifest_identity["sha256"], "artifact_members": members,
        "archive_sha256": aio.digest(archive), "archive_bytes": len(archive),
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_bytes, "checkpoint_preserved": True,
        "checkpoint_prune_authorized": False, "cache_cleanup_authorized": False,
        "screen2m": metrics, "ready_to_retrieve": True,
    }
    completion_identity = store.write_json("ready-to-retrieve.json", completion)
    final = store.inspect(c.ARTIFACTS | {"artifact-manifest.json", "ready-to-retrieve.json"})
    if final["artifact-manifest.json"] != manifest_identity or final["ready-to-retrieve.json"] != completion_identity:
        raise SystemExit("final evidence revalidation failed")
    print(json.dumps({"result": result, "completion_sha256": completion_identity["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
