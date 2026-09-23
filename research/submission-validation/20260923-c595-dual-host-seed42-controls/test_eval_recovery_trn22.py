#!/usr/bin/env python3
import hashlib,importlib.util,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent
def load(name):
 spec=importlib.util.spec_from_file_location(name,HERE/(name+".py"));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
left=(HERE/"eval_recovery.py").read_text();right=(HERE/"eval_recovery_trn22.py").read_text()
replacements={
 "trn22":"trn21","ip-172-31-32-253":"ip-172-31-34-85",
 "trainium_freshstart_20260924_c595_control_trn22_r2":"trainium_freshstart_20260924_c595_control_trn21_r2",
 "fs-20260924-c595-control-trn22-r2":"fs-20260924-c595-control-trn21-r2",
 "fs-20260924-c595-eval-recovery-trn22-r1":"fs-20260924-c595-eval-recovery-trn21-r1",
 "b12fbc758f63aeedec09752bdaa43b5267d3d5a2eb410ab6fbe8f80f641435ea":"cefe116eee4652fbab0ebc045fe6f7ea0a39785aff42ffb415106bf6e6820085"}
normalized=right
for old,new in replacements.items():normalized=normalized.replace(old,new)
assert normalized==left,"trn22 harness changed outside declared host/checkpoint/label constants"
m=load("eval_recovery_trn22")
auth=json.loads((HERE/"eval-recovery-trn22-authorization.json").read_text())
assert auth["host"]=="trn22" and auth["checkpoint_path"]==str(m.CHECKPOINT)
assert auth["checkpoint_sha256"]==m.CHECKPOINT_SHA and auth["checkpoint_bytes"]==469831009
assert auth["training_authorized"] is False and auth["checkpoint_prune_authorized"] is False
print("c595 trn22 evaluator recovery exact-delta gates: PASS")
