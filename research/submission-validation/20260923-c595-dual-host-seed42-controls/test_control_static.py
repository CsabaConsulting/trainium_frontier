#!/usr/bin/env python3
import importlib.util, json, math, pathlib

HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("control_job",HERE/"control_job.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def metric(**updates):
    value={"val_bpb":0.99,"num_tokens":2097152,"num_bytes":8418678,"total_nats":1.0,
           "eval_seconds":1.0,"eval_timeout":1200,"eval_over_timeout":False}
    value.update(updates);return json.dumps(value)
good_primary=metric();good_secondary=metric(num_tokens=20971520,num_bytes=83462186)
assert module.parse_metrics("noise\n"+good_primary+"\n",2097152,8418678)["num_bytes"]==8418678
assert module.parse_metrics(good_secondary,20971520,83462186)["num_tokens"]==20971520
invalid=("",good_primary+good_primary,metric(val_bpb=math.nan),metric(val_bpb=True),metric(val_bpb="0.99"),
         metric(val_bpb=0),metric(num_tokens=True),metric(num_tokens=2097152.0),metric(num_bytes=True),
         metric(total_nats=math.inf),metric(total_nats=0),metric(eval_seconds="1"),metric(eval_seconds=0),
         metric(eval_timeout=True),metric(eval_timeout=1199),metric(eval_over_timeout=True),
         json.dumps({**json.loads(good_primary),"extra":1}))
for item in invalid:
    try:module.parse_metrics(item,2097152,8418678)
    except SystemExit:pass
    else:raise AssertionError("invalid metrics accepted")

loss_log="\n".join(f"step {step:05d} | loss {loss:.4f} | lrm 1.000 |" for step,loss in module.REFERENCE_LOSSES.items())
parsed=module.parse_losses(loss_log)
assert parsed["step_50_separate"]==4.8776 and parsed["maximum_absolute_delta"]==0
try:module.parse_losses(loss_log.replace("loss 5.2152","loss 5.2154"))
except SystemExit:pass
else:raise AssertionError("loss delta gate not enforced")

dirty={"RANK":"7","WORLD_SIZE":"8","MASTER_ADDR":"bad","TORCHELASTIC_RUN_ID":"bad",
       "NEURON_COMPETITION_R1_SEED":"9","NEURON_CC_FLAGS":"bad","PYTHONPATH":"bad",
       "PYTHONHOME":"bad","NEURON_RT_VISIBLE_CORES":"0","NEURON_RT_NUM_CORES":"1","KEEP":"yes"}
clean=module.scrub_eval_environment(dirty)
assert clean["KEEP"]=="yes" and "NEURON_CC_FLAGS" not in clean
assert not ({"RANK","WORLD_SIZE","MASTER_ADDR","TORCHELASTIC_RUN_ID","NEURON_COMPETITION_R1_SEED",
             "PYTHONPATH","PYTHONHOME","NEURON_RT_VISIBLE_CORES","NEURON_RT_NUM_CORES"}&set(clean))
source=(HERE/"control_job.py").read_text()
assert '[PYTHON, "prepare.py", "eval-public", "--train-py",' in source
assert '"--eval-tokens", "20971520", "--timeout-seconds", "1200"' in source
assert module.COMPILER=="--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
print("c595 control static/eval/loss/environment gates: PASS")
