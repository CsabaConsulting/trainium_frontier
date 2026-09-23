#!/usr/bin/env python3
import hashlib,json,pathlib,re,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
CLOSURE={f"research/fresh_start_{name}.py" for name in ("launch","sync","guard","host","backup")}
for host in ("trn21","trn22"):
    path=HERE/f"control-{host}-request.json";request=json.loads(path.read_text())
    job=json.loads((HERE/f"control-{host}-job.json").read_text())
    assert request["host"]==host and re.fullmatch(r"[0-9a-f]{40}",request["expected_commit"])
    assert request["environment"]["NEURON_CC_FLAGS"]=="--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
    assert request["environment"]["NEURON_LOGICAL_NC_CONFIG"]=="2"
    assert request["command"][:5]==["timeout","--foreground","--signal=TERM","--kill-after=60s","7200s"]
    assert CLOSURE<=set(request["fingerprint_paths"])
    assert set(request["fingerprint_paths"])==set(request["expected_sha256"])
    tracked=set(subprocess.check_output(["git","ls-tree","-r","--name-only",request["expected_commit"]],cwd=ROOT,text=True).splitlines())
    for item,want in request["expected_sha256"].items():
        assert item in tracked
        blob=subprocess.check_output(["git","show",f'{request["expected_commit"]}:{item}'],cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest()==want,(item,hashlib.sha256(blob).hexdigest(),want)
    assert job["request_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
    assert job["expected_commit"]==request["expected_commit"]
    assert not any(request["label"] in name for name in subprocess.check_output(["git","ls-files","research/run-logs"],cwd=ROOT,text=True).splitlines())
print("c595 immutable requests: PASS (no dispatch)")
