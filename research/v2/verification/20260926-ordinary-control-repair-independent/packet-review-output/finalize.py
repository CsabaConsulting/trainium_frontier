from pathlib import Path
import hashlib,json
out=Path(__file__).resolve().parent;root=out.parent;inp=root/'packet-input'
for prefix in ('aggregation','packet'):
 if (inp/(prefix+'-byte-repair-normal.json')).read_bytes()!=(inp/(prefix+'-byte-repair-optimized.json')).read_bytes():raise RuntimeError('SUPPLIED_RECEIPTS_DIFFER')
if (inp/'EXPERIMENT_PROPOSAL.json').read_bytes()!=(root/'SPEC.json').read_bytes():raise RuntimeError('SPEC_CHANGED')
v={'verdict':'PASS_SCOPED_DISABLED_MECHANICS','negative_cases_executed':30,'leaf_sha256':'1cc6ee395f88125b9c9d280562e918d002a1e1a10dc88d50501d1fb4a09a551c','host_driver_sha256':'7e24fc86bebcde3691b22f94fb6288c12fe009b2ea2fe59f2208af6cc760a463','canonical_provenance_authentication':'SUPERVISOR_SEPARATE','operator_assignment':'SEPARATE_REQUIRED','native_acceptance':False,'host_action_authorized_by_review':False,'original_K2_gate':'FAILED_PRESERVED'}
(out/'verdict.json').write_text(json.dumps(v,indent=2)+'\n')
files=[p for p in inp.iterdir() if p.is_file()]+[p for p in out.rglob('*') if p.is_file() and p.name!='manifest.json']
m={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
for n in ('VERDICT.md','verdict.json','manifest.json','run1.stdout','driver.stdout'):print(n,hashlib.sha256((out/n).read_bytes()).hexdigest())
