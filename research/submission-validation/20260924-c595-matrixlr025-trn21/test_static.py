#!/usr/bin/env python3
import hashlib,importlib.util,json,math,os,pathlib,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
base=ROOT/"submissions/dense6x1024_direct_review";candidate=ROOT/"submissions/dense6x1024_direct_review_matrixlr025"
assert sha(base/"train.py")=="c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537"
assert sha(candidate/"train.py")=="4b36bb0da7950299c09f8bbbfc8d4a77c6bb73415a85f3bcf413cc2c4bab2994"
left=(base/"train.py").read_text().splitlines();right=(candidate/"train.py").read_text().splitlines()
assert len(left)==len(right) and [(a,b) for a,b in zip(left,right) if a!=b]==[("MATRIX_LR = 0.02","MATRIX_LR = 0.025")]
assert (base/"launch-command.txt").read_bytes()==(candidate/"launch-command.txt").read_bytes()
proposal=json.loads((HERE/"EXPERIMENT_PROPOSAL.json").read_text())
assert proposal["single_causal_change"]=="MATRIX_LR = 0.020 -> 0.025" and proposal["environment"]["seed"]==42
assert proposal["environment"]["max_train_seconds"]==1800 and proposal["run_mode"]=={"full_budget":True,"smoke":False,"one_run_per_machine":True}
assert proposal["selection_metric"]=={"name":"full20m_public_val_bpb","batch_size":1,"seq_len":1024,"num_tokens":20971520,"num_bytes":83462186,"control":0.993231940827686,"lower_is_better":True}
assert proposal["decision"]["keep_if_improvement_at_least"]==0.0015 and proposal["decision"]["replicate_if_keep_improvement_below"]==0.0025
spec=importlib.util.spec_from_file_location("retune_job",HERE/"retune_job.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
scrubbed=m.scrub({"NEURON_CC_FLAGS":"bad","RANK":"1","TORCHELASTIC_X":"x","NEURON_COMPETITION_R1_SEED":"9","SAFE":"yes"})
assert scrubbed=={"SAFE":"yes"}
total=5776000.0;num_bytes=83462186
good={"val_bpb":total/(num_bytes*math.log(2)),"num_tokens":20971520,"num_bytes":num_bytes,"total_nats":total,"eval_seconds":1.0,"eval_timeout":1200,"eval_over_timeout":False}
assert m.metrics(json.dumps(good))["num_tokens"]==20971520
for bad in ({**good,"num_tokens":1},{**good,"num_bytes":1},{**good,"val_bpb":0.9},{**good,"val_bpb":True},{**good,"eval_over_timeout":True}):
 try:m.metrics(json.dumps(bad))
 except (SystemExit,OSError):pass
 else:raise AssertionError("invalid metric accepted")
checkpoint_sha="a"*64
finite={"schema":"c595-strict-finite-v1","finite":True,"weights_only":True,"map_location":"cpu","tensor_count":3,"checkpoint_sha256":checkpoint_sha,"checkpoint_bytes":99}
assert m.parse_finite(json.dumps(finite)+"\n",checkpoint_sha,99)==finite
finite_attacks=["arbitrary stdout",json.dumps(finite)+"\nextra",json.dumps({**finite,"tensor_count":0}),json.dumps({**finite,"finite":False}),json.dumps({**finite,"weights_only":False}),json.dumps({**finite,"checkpoint_sha256":"b"*64}),json.dumps({**finite,"checkpoint_bytes":100}),json.dumps({**finite,"extra":1}),json.dumps({k:v for k,v in finite.items() if k!="map_location"}),json.dumps({**finite,"tensor_count":True})]
for attack in finite_attacks:
 try:m.parse_finite(attack,checkpoint_sha,99)
 except SystemExit:pass
 else:raise AssertionError("forged finite receipt accepted")
from decimal import Decimal
control=Decimal("0.993231940827686")
assert m.decision_from_serialized(str(control-Decimal("0.0015")))["decision_gate"]=="KEEP"
assert m.decision_from_serialized(str(control-Decimal("0.0015")+Decimal("0.000000000000001")))["decision_gate"]=="DISCARD"
assert m.decision_from_serialized(str(control-Decimal("0.0025")))["replication_required"] is False
assert m.decision_from_serialized(str(control-Decimal("0.0025")+Decimal("0.000000000000001")))["replication_required"] is True
assert m.decision_from_serialized(str(control-Decimal("0.0025")-Decimal("0.000000000000001")))["replication_required"] is False
for bad in ("NaN","Infinity","0",None):
 try:m.decision_from_serialized(bad)
 except SystemExit:pass
 else:raise AssertionError("invalid decimal accepted")
# Exercise the reused verified no-follow/O_EXCL/write-all artifact boundary.
a=m.artifact_module(); old=(a.ROOT,a.LABEL,a.OUTPUT)
with tempfile.TemporaryDirectory() as directory:
 root=pathlib.Path(directory); a.ROOT=root;a.LABEL="test";a.OUTPUT=root/"out/test";a.create_output(root,a.OUTPUT)
 a.durable(a.OUTPUT/"one",b"abcdef"); assert a.artifact_bytes("one")==b"abcdef"
 chunks=[]
 def short(fd,data):chunks.append(bytes(data[:2]));return min(2,len(data))
 assert a.write_all(9,b"abcdef",short)==6 and b"".join(chunks)==b"abcdef"
 try:a.write_all(9,b"x",lambda fd,data:0)
 except SystemExit:pass
 else:raise AssertionError("zero durable write accepted")
 (a.OUTPUT/"link").symlink_to(a.OUTPUT/"one")
 try:a.artifact_bytes("link")
 except (SystemExit,OSError):pass
 else:raise AssertionError("artifact symlink accepted")
 (a.OUTPUT/"link").unlink()
 identity=a.inspect_artifacts({"one"});(a.OUTPUT/"one").write_bytes(b"truncated")
 assert a.inspect_artifacts({"one"})!=identity
 (a.OUTPUT/"extra").write_bytes(b"x")
 try:a.inspect_artifacts({"one"})
 except SystemExit:pass
 else:raise AssertionError("extra artifact accepted")
with tempfile.TemporaryDirectory() as directory:
 root=pathlib.Path(directory); outside=root/"outside";outside.mkdir();(root/"out").symlink_to(outside, target_is_directory=True)
 a.ROOT=root;a.LABEL="test";a.OUTPUT=root/"out/test"
 try:a.create_output(root,a.OUTPUT)
 except (SystemExit,OSError):pass
 else:raise AssertionError("ancestor symlink accepted")
a.ROOT,a.LABEL,a.OUTPUT=old
# Full artifact sealing lifecycle reaches a durable ready receipt and remains fail-closed.
with tempfile.TemporaryDirectory() as directory:
 root=pathlib.Path(directory);a.ROOT=root;a.LABEL=m.LABEL;a.OUTPUT=root/"out"/m.LABEL;a.create_output(root,a.OUTPUT)
 for name in sorted(m.BASE_ARTIFACTS):a.durable(a.OUTPUT/name,("payload:"+name).encode())
 members=a.inspect_artifacts(m.BASE_ARTIFACTS)
 manifest={"schema":"c595-matrixlr025-artifact-manifest-v1","label":m.LABEL,"members":members}
 a.durable_json(a.OUTPUT/"artifact-manifest.json",manifest)
 manifest_sha=m.revalidate_manifest(a,manifest,members);assert len(manifest_sha)==64
 ready={"schema":"test-ready-v1","artifact_manifest_sha256":manifest_sha,"ready_to_retrieve":True}
 a.durable_json(a.OUTPUT/"ready-to-retrieve.json",ready)
 assert json.loads(a.artifact_bytes("ready-to-retrieve.json"))==ready
 assert set(a.inspect_artifacts(m.BASE_ARTIFACTS|{"artifact-manifest.json","ready-to-retrieve.json"}))==m.BASE_ARTIFACTS|{"artifact-manifest.json","ready-to-retrieve.json"}
 (a.OUTPUT/"training.log").write_bytes(b"mutation")
 try:m.revalidate_manifest(a,manifest,members)
 except SystemExit:pass
 else:raise AssertionError("post-manifest mutation accepted")
 (a.OUTPUT/"unexpected").write_bytes(b"extra")
 try:m.revalidate_manifest(a,manifest,members)
 except SystemExit:pass
 else:raise AssertionError("post-manifest extra accepted")
a.ROOT,a.LABEL,a.OUTPUT=old
source=(HERE/"retune_job.py").read_text()
assert '"--mode","train","--fresh-caches"' in source and '"--mode","eval"' in source
assert "smoke" not in source.lower().replace('"smoke":False','') or '"smoke":False' in source
assert '"prune_authorized":False' in source and '"cache_cleanup_authorized":False' in source
assert "weights_only=True" in source and "artifact-manifest.json" in source and "ready-to-retrieve.json" in source
print("c595 MATRIX_LR 0.025 exact-diff/proposal/driver gates: PASS")
