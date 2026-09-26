from pathlib import Path
import hashlib,json
out=Path(__file__).resolve().parent
verdict={'verdict':'REFUSE','source_sha256':'9c2774c5a675fba9e4612f66b65223d403c059cf7cf86d296720c16144ac348b','own_cpu_exactness':'PASS','conventional_negative_tests':16,'validator_false_acceptance':'zero-byte FP32 [256,128] weight and forward','counterexample_accepted':True,'native':'UNREVIEWED','packet':'NOT_SUPPLIED','original_K2_gate':'FAILED_PRESERVED','host_authority':False}
(out/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
manifest={str(p.relative_to(out.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.parent.rglob('*')) if p.is_file() and p.name!='manifest.json'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name in ['VERDICT.md','verdict.json','manifest.json','counterexample.py','counterexample.stdout','run1.stdout']:
 print(name,hashlib.sha256((out/name).read_bytes()).hexdigest())
