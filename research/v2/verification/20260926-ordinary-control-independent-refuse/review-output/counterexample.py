import json, importlib.util
from pathlib import Path
out=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('p',out/'reviewed_probe.py'); p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
r=json.loads((out/'baseline-cpu-receipt.json').read_text())
# Preserve all oracle gradients, IDs and tangents. Give the claimed FP32 tensor
# zero raw bytes while retaining its claimed 256x128 shape.
empty=p.blob(b'','torch.float32',(256,128),r['blobs'])
r['weight']=empty
for entry in r['routes']:
    for key in ('expected_inputs','before_forward','before_backward','after_backward'):
        entry[key]['weight']=empty
    entry['expected_output']=empty
    entry['output']=p.comparison(empty,empty,r['blobs'])
accepted=p.validate_receipt(r,0)
(out/'counterexample-receipt.json').write_text(json.dumps(r,sort_keys=True))
print(json.dumps({'counterexample':'zero-byte weight and forward claim shape 256x128','validator_accepted':accepted,'weight':empty},indent=2))
