#!/usr/bin/env python3
import importlib.util,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("r2",HERE/"eval_recovery_r2.py");r2=importlib.util.module_from_spec(spec);spec.loader.exec_module(r2)
assert r2.r.m.LABEL=="fs-20260924-c595-matrixlr025-eval-recovery-trn21-r2"
assert r2.r.m.OUTPUT==r2.ROOT/"out"/r2.r.m.LABEL
auth=json.loads((HERE/"authorization-r2.json").read_text());job=json.loads((HERE/"remote-job-r2.json").read_text())
assert auth["label"]==job["label"]==r2.r.m.LABEL
assert auth["checkpoint_sha256"]==job["checkpoint_sha256"]==r2.r.m.CHECKPOINT_SHA
assert auth["training_authorized"] is False and auth["checkpoint_prune_authorized"] is False
print("LR evaluator recovery r2 remote core: PASS")
