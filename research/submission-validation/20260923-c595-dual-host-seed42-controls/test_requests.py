#!/usr/bin/env python3
import hashlib,json,pathlib,re,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
CLOSURE={f"research/fresh_start_{name}.py" for name in ("launch","sync","guard","host","backup")}
for host,revision in (("trn21",""),("trn22",""),("trn21","-r2"),("trn22","-r2")):
    path=HERE/f"control-{host}{revision}-request.json";request=json.loads(path.read_text())
    job=json.loads((HERE/f"control-{host}{revision}-job.json").read_text())
    assert request["host"]==host and re.fullmatch(r"[0-9a-f]{40}",request["expected_commit"])
    assert request["environment"]["NEURON_CC_FLAGS"]=="--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
    assert request["environment"]["NEURON_LOGICAL_NC_CONFIG"]=="2"
    assert request["command"][:5]==["timeout","--foreground","--signal=TERM","--kill-after=60s","7200s"]
    sandbox=request["command"][request["command"].index("--sandbox-root")+1]
    archive=request["command"][request["command"].index("--archive")+1]
    assert sandbox.startswith(request["remote"]+"/out/") and archive.startswith(request["remote"]+"/out/archives/")
    assert CLOSURE<=set(request["fingerprint_paths"])
    if revision:
        assert {"research/submission-validation/20260923-c595-dual-host-seed42-controls/cleanup_caches.py",
                "research/submission-validation/20260923-c595-dual-host-seed42-controls/post_run_operator.py"} <= set(request["fingerprint_paths"])
    assert set(request["fingerprint_paths"])==set(request["expected_sha256"])
    tracked=set(subprocess.check_output(["git","ls-tree","-r","--name-only",request["expected_commit"]],cwd=ROOT,text=True).splitlines())
    for item,want in request["expected_sha256"].items():
        assert item in tracked
        blob=subprocess.check_output(["git","show",f'{request["expected_commit"]}:{item}'],cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest()==want,(item,hashlib.sha256(blob).hexdigest(),want)
    assert job["request_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
    assert job["expected_commit"]==request["expected_commit"]
    # Historical request labels may now have immutable dispatch/evidence records;
    # their request bytes and Git-object fingerprints above remain authoritative.

for host in ("trn21","trn22"):
    path=HERE/f"control-{host}-r2-v2-request.json";request=json.loads(path.read_text())
    job=json.loads((HERE/f"control-{host}-r2-v2-job.json").read_text())
    assert request["expected_commit"]=="067ff2326bbad1d4ed7060a802e7cee3e8448545"
    assert job["request_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(request["fingerprint_paths"])==set(request["expected_sha256"])
    post=request["post_run_core"]
    assert post["expected_commit"]==request["expected_commit"]
    for key in ("post_run_operator_sha256","post_run_contract_sha256","prune_script_sha256","cleanup_script_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}",post[key])
    for item,want in request["expected_sha256"].items():
        blob=subprocess.check_output(["git","show",f'{request["expected_commit"]}:{item}'],cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest()==want,(item,hashlib.sha256(blob).hexdigest(),want)
print("c595 immutable requests: PASS (no dispatch)")
