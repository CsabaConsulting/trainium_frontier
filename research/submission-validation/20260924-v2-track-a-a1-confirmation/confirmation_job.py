#!/usr/bin/env python3
"""Run one evidence-bound Track-A A1 confirmation arm on real Neuron."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import artifact_io as aio
import contracts as c
import process_lifecycle as lifecycle


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def file_identity(path):
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if stat.S_ISLNK(os.lstat(current).st_mode):
            raise SystemExit(f"checkpoint symlink rejected: {current}")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit(f"checkpoint is not regular: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        sha = hashlib.sha256()
        size = 0
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            sha.update(block)
            size += len(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (size != before.st_size or size != opened.st_size or size != after.st_size
            or before.st_dev != after.st_dev or before.st_ino != after.st_ino):
        raise SystemExit(f"checkpoint changed during identity read: {path}")
    return {"checkpoint_path": str(path), "checkpoint_sha256": sha.hexdigest(), "checkpoint_bytes": size}


def validate_checkpoint(expected):
    observed = file_identity(expected["checkpoint_path"])
    wanted = {field: expected[field] for field in ("checkpoint_path", "checkpoint_sha256", "checkpoint_bytes")}
    if observed != wanted or not c.valid_digest(observed["checkpoint_sha256"]) or observed["checkpoint_bytes"] <= 0:
        raise SystemExit("retained checkpoint identity mismatch")
    return observed


def disk_gate():
    info = os.statvfs(ROOT)
    free = info.f_bavail * info.f_frsize
    disposition = c.disk_disposition(free)
    if disposition != "ALLOWED":
        raise SystemExit(f"disk gate: {disposition}")
    return free


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
            raise SystemExit("fresh cache is not empty")
        result[key] = str(path)
    return result


def cache_observations(paths):
    result = {}
    for key, value in sorted(paths.items()):
        root = Path(value)
        files = []
        total = 0
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise SystemExit("cache symlink rejected")
            if path.is_file():
                size = path.stat().st_size
                total += size
                files.append({"relative": str(path.relative_to(root)), "bytes": size})
        result[key] = {"path": value, "bytes": total, "file_count": len(files), "files": files}
    return result


def verify_sources(request, core, head):
    source_object = subprocess.check_output(["git", "show", f"{c.SOURCE_COMMIT}:{c.SOURCE_PATH}"], cwd=ROOT)
    if aio.digest(source_object) != c.SOURCE_SHA:
        raise SystemExit("immutable source Git object mismatch")
    for relative, expected in request["source_sha256"].items():
        working = aio.regular_bytes(ROOT / relative)
        core_blob = subprocess.check_output(["git", "show", f"{core}:{relative}"], cwd=ROOT)
        head_blob = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
        if working != core_blob or core_blob != head_blob or aio.digest(working) != expected:
            raise SystemExit(f"protected source mismatch: {relative}")


def make_archive(store, members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in sorted(members):
            data = store.read(name)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def metric_argv(sandbox, checkpoint, tokens):
    return [
        c.PYTHON, "prepare.py", "eval-public", "--train-py", "train.py",
        "--checkpoint", str(checkpoint), "--device", "neuron", "--batch-size", "1",
        "--seq-len", "1024", "--eval-tokens", str(tokens), "--timeout-seconds", "1200",
    ]


def evaluate(store, work, sandbox, label, name, checkpoint, tokens, nbytes):
    disk_gate()
    before = validate_checkpoint(checkpoint)
    caches = fresh_caches(work, name)
    env = dict(c.ENVIRONMENT)
    env["TRAINIUM_FRESH_START_RUN_ID"] = label
    env.update(caches)
    argv = metric_argv(sandbox, before["checkpoint_path"], tokens)
    rc, output, wall, timed_out, group = lifecycle.run_capture(argv, sandbox, env, 1200)
    store.write(f"{name}.log", output)
    if rc or timed_out or wall > 1200.0:
        raise SystemExit(f"{name} evaluation failed")
    c.validate_clean_lifecycle(group)
    metrics = c.parse_metric_output(output.decode(errors="replace"), tokens, nbytes)
    receipt = {
        "schema": "track-a-confirmation-metric-v1", "label": label, **before,
        "argv": argv, "metrics": metrics, "wall_seconds": wall, "process_group": group,
    }
    store.write_json(f"{name}.metrics.json", receipt)
    if validate_checkpoint(checkpoint) != before:
        raise SystemExit(f"{name} checkpoint changed during evaluation")
    return receipt, caches


def finite_scan(checkpoint, env):
    source = (
        "import hashlib,json,os,sys,torch\n"
        "p=sys.argv[1];fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);h=hashlib.sha256();z=0\n"
        "while True:\n b=os.read(fd,8388608)\n if not b:break\n h.update(b);z+=len(b)\n"
        "os.close(fd);v=torch.load(p,map_location='cpu',weights_only=True);s=[v];n=0\n"
        "while s:\n x=s.pop()\n if isinstance(x,dict):s.extend(x.values())\n"
        " elif isinstance(x,(list,tuple)):s.extend(x)\n elif isinstance(x,torch.Tensor):\n"
        "  n+=1\n  assert not (x.is_floating_point() or x.is_complex()) or torch.isfinite(x).all().item()\n"
        "print(json.dumps({'checkpoint_bytes':z,'checkpoint_sha256':h.hexdigest(),'finite':True,'map_location':'cpu','tensor_count':n,'weights_only':True},sort_keys=True))\n"
    )
    argv = [c.PYTHON, "-c", source, str(checkpoint)]
    return argv, lifecycle.run_capture(argv, checkpoint.parent, env, 600)


def prior_summary(bundle):
    return {key: {
        "archive_commit": run["archive_commit"], "completion_sha256": run["completion_sha256"],
        "manifest_sha256": run["manifest_sha256"], "result_sha256": run["result_sha256"],
        "metrics_sha256": run["metrics_sha256"], "checkpoint_path": run["checkpoint_path"],
        "checkpoint_sha256": run["checkpoint_sha256"], "checkpoint_bytes": run["checkpoint_bytes"],
    } for key, run in bundle["runs"].items()}


def run_pair(store, work, sandbox, label, bundle):
    runs = bundle["runs"]
    inputs = prior_summary(bundle)
    input_evidence = {
        "schema": "track-a-confirmation-input-v1", "prior_runs": inputs,
        "screen_delta": bundle["screen_delta"], "trn21_confirmation": None,
    }
    store.write_json("input-evidence.json", input_evidence)
    control, control_caches = evaluate(
        store, work, sandbox, label, "control20m", runs["control-trn21"], c.FULL_TOKENS, c.FULL_BYTES
    )
    candidate, candidate_caches = evaluate(
        store, work, sandbox, label, "a1-20m", runs["a1-trn21"], c.FULL_TOKENS, c.FULL_BYTES
    )
    delta = c.decimal_value(candidate["metrics"]["val_bpb"], "trn21 A1 20M") - c.decimal_value(
        control["metrics"]["val_bpb"], "trn21 control 20M"
    )
    decision = {
        "schema": "track-a-trn21-pair-decision-v1",
        "delta_definition": "candidate_minus_same_host_control",
        "metric_population": "fresh20m_seq1024_tokens20971520_bytes83462186",
        "trn21_delta": str(delta), "strictly_negative": delta < 0,
        "verdict": "NOMINATE_TRN22_REPLICATE" if delta < 0 else "DISCARD",
        "adoption_authorized": False,
    }
    store.write_json("decision.json", decision)
    store.write_json("cache-observations.json", {
        "schema": "track-a-confirmation-cache-observations-v1",
        "control20m": cache_observations(control_caches), "a1_20m": cache_observations(candidate_caches),
        "cleanup_authorized": False, "caches_deleted": [],
    })
    return {
        "input_evidence": input_evidence,
        "metrics": {"control20m": control["metrics"], "a1_20m": candidate["metrics"]},
        "decision": decision, "candidate_checkpoint": None,
    }


def run_replicate(store, work, sandbox, label, bundle, confirmation):
    runs = bundle["runs"]
    inputs = prior_summary(bundle)
    confirmation_summary = {key: confirmation[key] for key in (
        "label", "archive_commit", "completion_sha256", "manifest_sha256", "result_sha256",
        "control20m_sha256", "a120m_sha256", "trn21_delta",
    )}
    input_evidence = {
        "schema": "track-a-confirmation-input-v1", "prior_runs": inputs,
        "screen_delta": bundle["screen_delta"], "trn21_confirmation": confirmation_summary,
    }
    store.write_json("input-evidence.json", input_evidence)
    control, control_caches = evaluate(
        store, work, sandbox, label, "control20m", runs["control-trn22"], c.FULL_TOKENS, c.FULL_BYTES
    )
    disk_gate()
    train_caches = fresh_caches(work, "train")
    env = dict(c.ENVIRONMENT)
    env["TRAINIUM_FRESH_START_RUN_ID"] = label
    env.update(train_caches)
    env["NEURON_COMPETITION_R1_OUT_DIR"] = str(sandbox / "out")
    env["NEURON_COMPETITION_R1_STEP_LOG"] = str(sandbox / "steps.jsonl")
    check_argv = [c.PYTHON, "prepare.py", "check", "--train-py", "train.py", "--launch-command", c.A1_LAUNCH]
    check_rc, check_output, check_wall, check_timeout, check_group = lifecycle.run_capture(
        check_argv, sandbox, env, 120
    )
    store.write("contract-check.log", check_output)
    store.write_json("contract-check-process.json", {
        "schema": "track-a-confirmation-contract-process-v1", "argv": check_argv,
        "returncode": check_rc, "timed_out": check_timeout, "wall_seconds": check_wall,
        "process_group": check_group,
    })
    if check_rc or check_timeout:
        raise SystemExit("organizer contract check failed")
    c.validate_clean_lifecycle(check_group)
    train_rc, train_output, train_wall, train_timeout, observations, train_group = lifecycle.run_training(
        c.train_argv(), sandbox, env, 3000
    )
    store.write("training.log", train_output)
    try:
        steps = aio.regular_bytes(sandbox / "steps.jsonl")
    except FileNotFoundError:
        steps = b""
    store.write("training-steps.jsonl", steps)
    allowance = min(observations[4]["observed_seconds"], 900.0) if len(observations) >= 5 else None
    charged = train_wall - allowance if allowance is not None else None
    training_process = {
        "schema": "track-a-confirmation-training-process-v1", "argv": c.train_argv(),
        "launch_command": c.A1_LAUNCH, "returncode": train_rc, "timed_out": train_timeout,
        "wall_seconds": train_wall, "startup_allowance_seconds": allowance,
        "charged_through_exit_seconds": charged,
        "charged_within_budget": charged is not None and charged <= 1800.0,
        "observations": observations, "seed": 42, "world_size": 4, "lnc": 2,
        "process_group": train_group,
    }
    store.write_json("training-process.json", training_process)
    if train_rc or train_timeout or len(observations) < 5 or not training_process["charged_within_budget"]:
        raise SystemExit("other-host full training failed/budget gate failed")
    c.validate_clean_lifecycle(train_group)
    checkpoint = sandbox / "out/final.pt"
    checkpoint_identity = file_identity(checkpoint)
    scan_env = dict(c.ENVIRONMENT)
    scan_env["TRAINIUM_FRESH_START_RUN_ID"] = label
    scan_argv, (scan_rc, scan_output, scan_wall, scan_timeout, scan_group) = finite_scan(checkpoint, scan_env)
    store.write("checkpoint-finite.log", scan_output)
    if scan_rc or scan_timeout:
        raise SystemExit("candidate finite checkpoint scan failed")
    c.validate_clean_lifecycle(scan_group)
    try:
        scan = json.loads(scan_output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise SystemExit("candidate finite receipt parse failed") from error
    if (set(scan) != {"checkpoint_bytes", "checkpoint_sha256", "finite", "map_location", "tensor_count", "weights_only"}
            or scan["checkpoint_sha256"] != checkpoint_identity["checkpoint_sha256"]
            or scan["checkpoint_bytes"] != checkpoint_identity["checkpoint_bytes"]
            or scan["finite"] is not True or scan["weights_only"] is not True
            or scan["map_location"] != "cpu" or type(scan["tensor_count"]) is not int
            or scan["tensor_count"] <= 0):
        raise SystemExit("candidate finite receipt mismatch")
    store.write_json("checkpoint-finite.json", {
        "schema": "track-a-confirmation-finite-v1", **scan, "argv": scan_argv,
        "wall_seconds": scan_wall, "process_group": scan_group,
    })
    candidate_expected = {**checkpoint_identity}
    screen, screen_caches = evaluate(
        store, work, sandbox, label, "a1-screen2m", candidate_expected, c.SCREEN_TOKENS, c.SCREEN_BYTES
    )
    full, full_caches = evaluate(
        store, work, sandbox, label, "a1-20m", candidate_expected, c.FULL_TOKENS, c.FULL_BYTES
    )
    decision = c.final_decision(
        confirmation["metrics"]["control20m"]["val_bpb"],
        confirmation["metrics"]["a1_20m"]["val_bpb"],
        control["metrics"]["val_bpb"], full["metrics"]["val_bpb"],
    )
    store.write_json("decision.json", decision)
    store.write_json("cache-observations.json", {
        "schema": "track-a-confirmation-cache-observations-v1",
        "control20m": cache_observations(control_caches), "train": cache_observations(train_caches),
        "a1_screen2m": cache_observations(screen_caches), "a1_20m": cache_observations(full_caches),
        "cleanup_authorized": False, "caches_deleted": [],
    })
    if file_identity(checkpoint) != checkpoint_identity:
        raise SystemExit("candidate checkpoint changed after evaluation")
    return {
        "input_evidence": input_evidence,
        "metrics": {"control20m": control["metrics"], "a1_screen2m": screen["metrics"],
                    "a1_20m": full["metrics"]},
        "decision": decision, "candidate_checkpoint": checkpoint_identity,
    }


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
    if (not c.HEX40.fullmatch(core or "")
            or subprocess.run(["git", "merge-base", "--is-ancestor", core, head], cwd=ROOT).returncode):
        raise SystemExit("verified core is not an ancestor")
    allowed = {str(rel / f"{name}-{suffix}.json") for name in c.ARMS for suffix in ("authorization", "request")}
    if set(git("diff", "--name-only", core, head).splitlines()) != allowed:
        raise SystemExit("finalization path set mismatch")
    authorization = c.validate_authorization(auth, request, arm, core)
    request_sha = authorization["request_sha256"]
    auth_sha = aio.digest(auth_bytes)
    concrete = c.build_concrete(request, auth_sha, request_sha, head)
    runtime_bytes = aio.regular_bytes(runtime_path)
    runtime = c.json_object(runtime_bytes, "runtime job")
    runtime_sha = c.validate_runtime(runtime, concrete, runtime_bytes)
    verify_sources(request, core, head)
    prior = authorization["prior"]
    local_keys = ("control-trn21", "a1-trn21") if arm == "pair-trn21" else ("control-trn22",)
    for key in local_keys:
        validate_checkpoint(prior["runs"][key])
    disk_gate()
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
    if (aio.digest(aio.regular_bytes(sandbox / "train.py")) != c.SOURCE_SHA
            or aio.digest(aio.regular_bytes(sandbox / "prepare.py")) != c.PREPARE_SHA):
        raise SystemExit("sandbox source copy mismatch")
    if arm == "pair-trn21":
        payload = run_pair(store, work, sandbox, spec["label"], prior)
    else:
        payload = run_replicate(store, work, sandbox, spec["label"], prior, authorization["confirmation"])
    for key in local_keys:
        validate_checkpoint(prior["runs"][key])
    result = {
        "schema": "track-a-confirmation-result-v1", "arm": arm, "role": spec["role"],
        "label": spec["label"], "host": spec["host"], "hostname": spec["hostname"],
        "source_sha256": c.SOURCE_SHA, "prepare_sha256": c.PREPARE_SHA,
        **payload, "checkpoint_inputs_preserved": True, "ready_to_retrieve": False,
    }
    store.write_json("result.json", result)
    artifacts = c.OUTPUT_ARTIFACTS[arm]
    archive_members = artifacts - {"evidence.tar.gz"}
    archive = make_archive(store, archive_members)
    store.write("evidence.tar.gz", archive)
    members = store.inspect(artifacts)
    manifest = {"schema": "track-a-confirmation-artifact-manifest-v1", "label": spec["label"], "members": members}
    manifest_identity = store.write_json("artifact-manifest.json", manifest)
    completion = {
        "schema": "track-a-confirmation-completion-v1", "arm": arm, "role": spec["role"],
        "label": spec["label"], "host": spec["host"], "hostname": spec["hostname"],
        "finalization_head": head, "verified_core_commit": core,
        "authorization_sha256": auth_sha, "request_sha256": request_sha,
        "runtime_job_sha256": runtime_sha, "source_sha256": c.SOURCE_SHA,
        "prepare_sha256": c.PREPARE_SHA, "artifact_manifest_sha256": manifest_identity["sha256"],
        "artifact_members": members, "archive_sha256": aio.digest(archive), "archive_bytes": len(archive),
        "input_evidence": payload["input_evidence"], "metrics": payload["metrics"],
        "decision": payload["decision"], "checkpoint_inputs_preserved": True,
        "candidate_checkpoint": payload["candidate_checkpoint"], "ready_to_retrieve": True,
    }
    completion_identity = store.write_json("ready-to-retrieve.json", completion)
    final = store.inspect(artifacts | {"artifact-manifest.json", "ready-to-retrieve.json"})
    if final["artifact-manifest.json"] != manifest_identity or final["ready-to-retrieve.json"] != completion_identity:
        raise SystemExit("final evidence revalidation failed")
    print(json.dumps({"result": result, "completion_sha256": completion_identity["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
