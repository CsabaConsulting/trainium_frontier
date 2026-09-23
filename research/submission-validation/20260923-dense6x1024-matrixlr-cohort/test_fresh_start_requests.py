#!/usr/bin/env python3
"""Parse launcher contracts without importing main or dispatching SSH."""
import hashlib, json, pathlib, re, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
REQUESTS = (
    ("control", "control-fresh-start-request.json", "control-job.json", False),
    ("candidate", "candidate-fresh-start-request.json", "candidate-job.json", False),
    ("control_r2", "control-r2-fresh-start-request.json", "control-r2-job.json", True),
    ("candidate_r2", "candidate-r2-fresh-start-request.json", "candidate-r2-job.json", True),
    ("control_r3", "control-r3-fresh-start-request.json", "control-r3-job.json", True),
    ("candidate_r3", "candidate-r3-fresh-start-request.json", "candidate-r3-job.json", True),
)
tracked_logs = subprocess.check_output(
    ["git", "ls-files", "research/run-logs"], cwd=ROOT, text=True
).splitlines()
for arm, request_name, job_name, is_r2 in REQUESTS:
    request_path = HERE / request_name
    request = json.loads(request_path.read_text())
    job = json.loads((HERE / job_name).read_text())
    assert request["host"] in ("trn21", "trn22")
    assert re.fullmatch(r"fs-[A-Za-z0-9_.-]+", request["label"])
    assert re.fullmatch(r"[0-9a-f]{40}", request["expected_commit"])
    subprocess.run(["git", "cat-file", "-e", request["expected_commit"] + "^{commit}"],
                   cwd=ROOT, check=True)
    tracked_at_commit = set(subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", request["expected_commit"]],
        cwd=ROOT, text=True,
    ).splitlines())
    remote = pathlib.PurePosixPath(request["remote"])
    assert remote.parent == pathlib.PurePosixPath("/home/ubuntu")
    assert remote.name.startswith("trainium_freshstart_")
    assert request["command"][0] == "timeout"
    assert request["environment"]["NEURON_LOGICAL_NC_CONFIG"] == "2"
    assert "--optlevel=1" in request["environment"]["NEURON_CC_FLAGS"]
    assert set(request["fingerprint_paths"]) == set(request["expected_sha256"])
    for path, expected in request["expected_sha256"].items():
        assert path in tracked_at_commit, (arm, "not tracked at expected commit", path)
        blob = subprocess.check_output(
            ["git", "show", f'{request["expected_commit"]}:{path}'], cwd=ROOT
        )
        actual = hashlib.sha256(blob).hexdigest()
        assert actual == expected, (path, actual, expected)
    post = request["post_run_core"]
    assert post["expected_commit"] == request["expected_commit"]
    assert post["prune_script_sha256"] == request["expected_sha256"]["research/submission-validation/20260923-dense6x1024-matrixlr-cohort/prune_checkpoint.py"]
    assert post["cwd"] == request["remote"]
    prune_source = (ROOT / "research/submission-validation/20260923-dense6x1024-matrixlr-cohort/prune_checkpoint.py").read_text()
    assert "--expected-core-commit" in prune_source and "--expected-script-sha256" in prune_source
    assert job["expected_commit"] == request["expected_commit"]
    assert job["fresh_start_request"] == str(request_path.relative_to(ROOT))
    assert job["fresh_start_request_sha256"] == hashlib.sha256(request_path.read_bytes()).hexdigest()
    if is_r2:
        assert arm in request["command"]
        assert "research/fresh_start_sync.py" in request["fingerprint_paths"]
        contract_revision = "r3" if arm.endswith("r3") else "r2"
        assert f"post-run-evidence-contract-{contract_revision}.json" in " ".join(request["fingerprint_paths"])
        if arm.endswith("r3"):
            closure = {f"research/fresh_start_{name}.py" for name in
                       ("launch", "sync", "guard", "host", "backup")}
            assert closure <= set(request["fingerprint_paths"])
        assert not any(request["label"] in path for path in tracked_logs), (
            arm, "retry label is already used", request["label"]
        )
print("fresh_start request launcher compatibility: PASS (no dispatch)")
