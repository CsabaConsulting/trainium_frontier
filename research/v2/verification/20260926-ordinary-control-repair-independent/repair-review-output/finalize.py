from pathlib import Path
import hashlib,json
out=Path(__file__).resolve().parent;root=out.parent
old=json.loads((root/'review-output/manifest.json').read_text())
for name,h in old.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise RuntimeError('ORIGINAL_CHANGED:'+name)
v={'verdict':'PASS_SCOPED_SOURCE_CPU','source_sha256':'1cc6ee395f88125b9c9d280562e918d002a1e1a10dc88d50501d1fb4a09a551c','original_review_immutable':True,'original_counterexample':'REJECTED_RAW_BYTE_COUNT','fresh_counterexample':'REJECTED_RAW_BYTE_COUNT','independent_exactness':'PASS','original_negatives':16,'repair_negatives':16,'native':'UNREVIEWED','packet':'UNREVIEWED','host_authority':False,'original_K2_gate':'FAILED_PRESERVED'}
(out/'verdict.json').write_text(json.dumps(v,indent=2)+'\n')
files=[p for p in root.iterdir() if p.is_file()]+[p for p in out.iterdir() if p.is_file() and p.name!='manifest.json']
m={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
for n in ('VERDICT.md','verdict.json','manifest.json','run1.stdout'):
 print(n,hashlib.sha256((out/n).read_bytes()).hexdigest())
