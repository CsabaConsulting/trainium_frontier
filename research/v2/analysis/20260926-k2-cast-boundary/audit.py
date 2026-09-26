"""Tiny CPU cast-chain analysis and archived evidence extraction; no native gate waiver."""
import ast
import base64
import hashlib
import json
import struct
import subprocess
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path('/home/poojith-devan/Desktop/Trainium')
torch.set_num_threads(1)
def need(value, name):
    if not value:
        raise RuntimeError(name)
def sha(value):
    return hashlib.sha256(value).hexdigest()
def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]
def bits(value):
    return format(struct.unpack('<I', struct.pack('<f', value))[0], '08x')
def bitlist(tensor):
    return [bits(x) for x in tensor.detach().float().reshape(-1).tolist()]
def recurrence(values):
    out = 0.0
    history = []
    for value in values:
        out = f32(out + f32(value))
        history.append(bits(out))
    return out, history

source = subprocess.check_output(['git','show','7419927c7417b5666a984ec9c2b849feadabaaa9:workflow/proposals/research-v2/train.py'], cwd=ROOT)
need(sha(source) == '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb', 'SOURCE_PIN')
tree = ast.parse(source)
gpt = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GPT')
forward = next(n for n in gpt.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
casts = [ast.unparse(n) for n in ast.walk(forward) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'to' and any(isinstance(a, ast.Name) and a.id == 'COMPUTE_DTYPE' for a in n.args)]
need('self.transformer.wte(idx).to(COMPUTE_DTYPE)' in casts, 'WTE_IMMEDIATE_CAST')
need('self.value_embeds[str(i)](idx).to(COMPUTE_DTYPE)' in casts, 'VE_IMMEDIATE_CAST')
dtype_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_dtype_for')
need(any(isinstance(n, ast.If) and ast.unparse(n.test) == "device.type == 'neuron'" and any(isinstance(r, ast.Return) and ast.unparse(r.value) == 'torch.bfloat16' for r in n.body) for n in dtype_fn.body), 'NEURON_BF16')

fixtures = {
    'fp32_min_positive': [2.0**-149],
    'bf16_min_positive': [2.0**-133],
    'normal_boundary': [2.0**-126],
    'normal_cancel_to_fp32_min': [2.0**-126 + 2.0**-149, -2.0**-126],
    'bf16_normal_cancel_to_bf16_min': [2.0**-126 + 2.0**-133, -2.0**-126],
    'duplicate_signed_bf16': [2.0**-133, -2.0**-133, 2.0**-133],
}
records = []
for name, original in fixtures.items():
    for sign in (1, -1):
        values = [f32(sign * x) for x in original]
        for dtype in (torch.bfloat16, torch.float32):
            weights = torch.nn.Parameter(torch.ones((2,1), dtype=torch.float32))
            ids = torch.zeros(len(values), dtype=torch.long)
            raw = F.embedding(ids, weights)
            out = raw.to(dtype)
            observed = {}
            def capture(key):
                def hook(gradient):
                    observed[key] = gradient.detach().clone()
                return hook
            out.register_hook(capture('output_tangent'))
            raw.register_hook(capture('embedding_tangent'))
            upstream = torch.tensor(values, dtype=dtype).reshape(-1,1)
            out.backward(upstream)
            rounded = upstream.float().reshape(-1).tolist()
            expected, history = recurrence(rounded)
            need(observed['output_tangent'].dtype == dtype, 'OUTPUT_TANGENT_DTYPE')
            need(observed['embedding_tangent'].dtype == torch.float32, 'WEIGHT_TANGENT_DTYPE')
            need(bitlist(observed['embedding_tangent']) == bitlist(upstream), 'CAST_BACKWARD_WIDENS_EXACT')
            need(bits(weights.grad[0,0].item()) == bits(expected), 'ORDERED_TINY_CPU_SUM')
            need(weights.grad[1,0].item() == 0, 'UNUSED_ROW_ZERO')
            records.append({'case':name,'sign':sign,'output_dtype':str(dtype),'input_fp32_bits':[bits(x) for x in values], 'output_tangent_bits':bitlist(observed['output_tangent']), 'embedding_tangent_bits':bitlist(observed['embedding_tangent']), 'ordered_fp32_intermediates':history,'final_gradient_bits':bits(expected)})

archive_path = ROOT / 'research/v2/operator/20260926-trn22-k2-registered/completed/results-bundle.json'
archive = json.loads(archive_path.read_text())
native = []
for rank in range(4):
    entry = archive['files'][f'receipts/rank{rank}.json']
    data = base64.b64decode(entry['base64'])
    need(len(data) == entry['bytes'] and sha(data) == entry['sha256'], 'RAW_RANK_HASH')
    receipt = json.loads(data)
    rows=[]
    for case in receipt['cases']:
        if not case['name'].endswith('subnormal'):
            continue
        rows.append({'case':case['name'],'dtype':case['dtype'],'status':case['status'], 'input_nonzero':case['checks']['upstream_roundtrip']['actual']['nonzero'], 'routes':[{ 'name':route['name'],'status':route['status'],'actual_nonzero':route.get('checks',{}).get('gradient',{}).get('actual',{}).get('nonzero'), 'expected_nonzero':route.get('checks',{}).get('gradient',{}).get('expected',{}).get('nonzero')} for route in case['routes']]})
    need(len(rows) == 4 and receipt['status'] == 'REGISTERED_FULL_CASES_MISMATCH_OR_ERROR', 'PRESERVE_FAILED_AGGREGATE')
    native.append({'rank':rank,'receipt_sha256':entry['sha256'],'original_status':receipt['status'],'cases':rows})

print(json.dumps({'status':'CPU_DOMAIN_ANALYSIS_COMPLETE_NOT_A_NATIVE_GATE','source_sha256':sha(source),'source_casts':casts,'torch':torch.__version__,'cpu_cases':records,'native_archive_sha256':sha(archive_path.read_bytes()),'native_archived_observations':native,'production_paths_in_scope':['wte','ordinary value embeddings; no ngram/table paths'], 'limitations':['Static production cast placement plus tiny CPU autograd, not a whole-model native runtime trace.','No all-input or production-size scatter proof, metadata performance or speed qualification.','Original unrestricted FP32 gate remains failed. A new BF16-only production contract requires a separate proposal and fresh independent review.'],'source_modified':False,'native_launched':False,'gate_waiver':False}, indent=2, allow_nan=False))
