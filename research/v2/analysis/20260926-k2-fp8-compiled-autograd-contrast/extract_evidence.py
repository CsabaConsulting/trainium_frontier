"""Archive/source-only contrast. No torch, SDK, tensor, host or production import."""
import base64, hashlib, json, subprocess
from pathlib import Path

R = Path('/home/poojith-devan/Desktop/Trainium')
checks = []
def sha(b): return hashlib.sha256(b).hexdigest()
def check(label, value):
    if not value: raise RuntimeError(label)
    checks.append(label)
def blob(ref, path): return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=R)
def unpack(x):
    b = x['text'].encode() if 'text' in x else base64.b64decode(x['base64'], validate=True)
    check('artifact bytes/hash', len(b) == x['bytes'] and sha(b) == x['sha256'])
    return b

p = 'research/v2/operator/20260926-trn21-k2-symmetric-native/completed/results-bundle.json'
raw = (R / p).read_bytes()
check('K2 immutable archive', raw == blob('7f0fbbde', p) and sha(raw) == '16dc0f7d84eed33eaaa821cb6afd21d00fc5b6d60b4ad1b9f93e55f3e07b9c7b')
bundle = json.loads(raw)
files = {n: unpack(x) for n, x in bundle['files'].items()}
k2 = []
for rank in range(4):
    r = json.loads(files[f'receipts/rank{rank}.json'])
    check('K2 rank status/scope', r['rank'] == rank and r['status'] == 'REFUSE_OR_UNSUPPORTED' and len(r['cases']) == 13 and not r['source_adoption'])
    selected = []
    for c in r['cases']:
        relevant = [{k:x[k] for k in ('name','exact','nonzero','max_abs','actual_sha256','expected_sha256')} for x in c['checks'] if x['name'] in ('upstream_roundtrip', 'direct_consumer', 'eager_autograd_backward', 'compiled_enclosing_autograd_backward')]
        selected.append(dict(name=c['name'], dtype=c['dtype'], status=c['status'], metadata_group_bounds=c['metadata_group_bounds'], checks=relevant))
    failed = r['cases'][-1]
    check('K2 isolated reported failure', failed['name'] == 'unique' and failed['dtype'] == 'torch.float32' and [(c['name'], c['nonzero']) for c in failed['checks'] if not c['exact']] == [('compiled_enclosing_autograd_backward', 0)])
    check('all earlier K2 cases exact', all(c['status'] == 'EXACT' and all(x['exact'] for x in c['checks']) for c in r['cases'][:-1]))
    k2.append(dict(rank=rank, receipt_sha256=sha(files[f'receipts/rank{rank}.json']), source=r['kernel_sha256'], checker=r['checker_sha256'], reference=r['reference_sha256'], torch=r['torch'], nki=r['nki_version'], case_summary=[dict(name=c['name'], dtype=c['dtype'], status=c['status'], compiled_nonzero=next((x['nonzero'] for x in c['checks'] if x['name']=='compiled_enclosing_autograd_backward'),None)) for c in r['cases']], unique_case_details=[c for c in selected if c['name']=='unique'] if rank==0 else 'See rank 0 and original hashed rank receipt.'))

prefix = 'research/v2/operator/20260925-trn21-l1-vjpdiag-r1/completed/'
retrieval = json.loads((R / prefix / 'retrieval.json').read_bytes())
fp8 = []
for rank in range(4):
    name = f'receipts/diagnostic/diagnostic-rank{rank}.json'
    raw = (R / prefix / name).read_bytes()
    check('FP8 immutable rank artifact', raw == blob('c39844eb', prefix + name) and sha(raw) == retrieval['files'][name]['sha256'] and len(raw) == retrieval['files'][name]['bytes'])
    r = json.loads(raw)
    check('FP8 unchanged HOLD scope', not r['full_component_gate_passed'] and not r['whole_step_measured'] and r['original_rms_ceiling'] == .01)
    def brief(x): return {k:x[k] for k in ('dtype','finite','nonzero','relative_rms','rms','max_abs')}
    fp8.append(dict(rank=rank, receipt_sha256=sha(raw), source=r['train_sha256'], checker=r['checker_sha256'], cases={n:dict(status=c['status'], grad_fn=c['grad_fn'], dx=brief(c['dx']), dw=brief(c['dw']), upstream_roundtrip=brief(c['upstream_roundtrip'])) for n,c in r['cases'].items()}, manual_native_vjp={k:brief(v) for k,v in r['manual_native_vjp'].items()}))

sources = {}
for ref, expected, spans in [
    ('a4aa4351','6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3',[(520,525),(1912,1914),(2226,2239),(3828,3831),(3843,3850)]),
    ('7419927c','36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb',[(539,544),(1948,1950),(2278,2294),(3896,3899),(3913,3920),(4930,4962)]),
    ('50a90137','921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858',[(1458,1473),(1575,1619),(1650,1655)])]:
    raw = blob(ref, 'workflow/proposals/research-v2/train.py')
    check('exact source ' + ref, sha(raw) == expected)
    lines = raw.decode().splitlines()
    sources[ref] = dict(sha256=sha(raw), excerpts=[dict(first=a,last=b,lines=lines[a-1:b]) for a,b in spans])
support = {}
for name, expected in [('consumer_nki.py','2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae'),('check_native.py','89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850'),('reference.py','14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f')]:
    path = 'research/v2/lanes/k2-symmetric-consumer/' + name
    raw = (R / path).read_bytes()
    check('exact K2 support ' + name, sha(raw) == expected)
    support[path] = sha(raw)
print(json.dumps(dict(scope='READ_ONLY_ANALYST_CONTRAST_NOT_NEW_GATE_OR_NATIVE_TEST', checks=len(checks), K2_archive_commit='7f0fbbde', K2_archive_sha256=sha((R / p).read_bytes()), K2_support=support, K2=k2, FP8=fp8, sources=sources, all_original_failures_preserved=True), indent=2, allow_nan=False))
