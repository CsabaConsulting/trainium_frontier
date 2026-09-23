#!/usr/bin/env python3
import hashlib,importlib.util,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location("lr_recovery",HERE/"eval_recovery.py");r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
expected={"schema":"c595-strict-finite-v1","finite":True,"weights_only":True,"map_location":"cpu","tensor_count":62,"checkpoint_sha256":r.m.CHECKPOINT_SHA,"checkpoint_bytes":r.m.CHECKPOINT_BYTES}
canonical=json.dumps(expected,sort_keys=True)
assert r.parse_finite_output(canonical)==(expected,0)
warning="\n".join(["[W923 23:48:21.812477753 OperatorEntry.cpp:208] Warning: Warning only once for all operators,  other operators may also be overridden.",*r.WARNING_TAIL,canonical])
assert r.parse_finite_output(warning)==(expected,7)
attacks=["noise\n"+canonical,canonical+"\n"+canonical,warning.replace("aten::gather.out","aten::other.out"),json.dumps({**expected,"tensor_count":61},sort_keys=True),json.dumps({**expected,"finite":False},sort_keys=True),json.dumps({**expected,"extra":1},sort_keys=True),json.dumps({**expected,"checkpoint_bytes":1},sort_keys=True),"not json"]
for attack in attacks:
 try:r.parse_finite_output(attack)
 except SystemExit:pass
 else:raise AssertionError("forged finite output accepted")
assert r.m.TRAIN_SHA=="4b36bb0da7950299c09f8bbbfc8d4a77c6bb73415a85f3bcf413cc2c4bab2994"
assert r.m.CHECKPOINT_SHA=="b2c4818e95e4ae25643ab48eb9dd199edebfc4b065290d6c601244b758af69ba" and r.m.CHECKPOINT_BYTES==469831009
assert r.m.SOURCE_COMMIT=="3e74e30fd1b0e856ddea321f08735bbf4df37a18"
source=(HERE/"eval_recovery.py").read_text();assert "m.main()" in source and "m.finite_scan=finite_scan" in source
assert "training_authorized\":false" in (HERE/"authorization.json").read_text() and "checkpoint_prune_authorized\":false" in (HERE/"authorization.json").read_text()
print("LR evaluator-only recovery static/adversarial checks: PASS")
