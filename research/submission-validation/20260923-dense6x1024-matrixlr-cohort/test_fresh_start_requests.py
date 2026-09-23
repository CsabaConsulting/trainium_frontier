#!/usr/bin/env python3
"""Parse launcher contracts without importing main or dispatching SSH."""
import hashlib, json, pathlib, re, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
for arm in ("control", "candidate"):
    request = json.loads((HERE / f"{arm}-fresh-start-request.json").read_text())
    assert request["host"] in ("trn21", "trn22")
    assert re.fullmatch(r"fs-[A-Za-z0-9_.-]+", request["label"])
    assert re.fullmatch(r"[0-9a-f]{40}", request["expected_commit"])
    subprocess.run(["git", "cat-file", "-e", request["expected_commit"] + "^{commit}"],
                   cwd=ROOT, check=True)
    remote = pathlib.PurePosixPath(request["remote"])
    assert remote.parent == pathlib.PurePosixPath("/home/ubuntu")
    assert remote.name.startswith("trainium_freshstart_")
    assert request["command"][0] == "timeout"
    assert request["environment"]["NEURON_LOGICAL_NC_CONFIG"] == "2"
    assert "--optlevel=1" in request["environment"]["NEURON_CC_FLAGS"]
    for path, expected in request["expected_sha256"].items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert actual == expected, (path, actual, expected)
print("fresh_start request launcher compatibility: PASS (no dispatch)")
