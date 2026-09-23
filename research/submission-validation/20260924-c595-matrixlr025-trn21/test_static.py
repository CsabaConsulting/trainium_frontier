#!/usr/bin/env python3
import hashlib,importlib.util,json,math,pathlib
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
total=5776000.0;num_bytes=83462186
good={"val_bpb":total/(num_bytes*math.log(2)),"num_tokens":20971520,"num_bytes":num_bytes,"total_nats":total,"eval_seconds":1.0,"eval_timeout":1200,"eval_over_timeout":False}
assert m.metrics(json.dumps(good))["num_tokens"]==20971520
for bad in ({**good,"num_tokens":1},{**good,"num_bytes":1},{**good,"val_bpb":0.9},{**good,"val_bpb":True},{**good,"eval_over_timeout":True}):
 try:m.metrics(json.dumps(bad))
 except SystemExit:pass
 else:raise AssertionError("invalid metric accepted")
source=(HERE/"retune_job.py").read_text()
assert '"--mode","train","--fresh-caches"' in source and '"--mode","eval"' in source
assert "smoke" not in source.lower().replace('"smoke":False','') or '"smoke":False' in source
assert "prune_authorized\":False" in source and "cache_cleanup_authorized\":False" in source
print("c595 MATRIX_LR 0.025 exact-diff/proposal/driver gates: PASS")
