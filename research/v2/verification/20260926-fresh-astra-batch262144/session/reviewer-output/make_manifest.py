"""Hash final local artifacts and recheck all packet bytes; never read outside scope."""
import hashlib
import json
from pathlib import Path

root=Path('/work')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
for expected in json.loads((root/'input_hashes.json').read_text()):
    path=Path('/packet')/expected['path']
    assert path.stat().st_size==expected['bytes'] and digest(path)==expected['sha256'],str(path)
excluded={'artifact_manifest.json','manifest.command.json','manifest.stdout.txt','manifest.stderr.txt'}
entries=[{'path':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(root.iterdir()) if p.is_file() and p.name not in excluded]
payload={'schema':'independent-review-artifact-manifest-v1','root':str(root),'files':entries,
         'excluded':sorted(excluded),'exclusion_reason':'manifest cannot hash itself; its own command receipt/streams finish after manifest generation',
         'input_packet_rehashed_unchanged':True}
(root/'artifact_manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
for item in entries:
    p=root/item['path']
    assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
assert json.loads((root/'result.json').read_text())['verdict']=='PASS'
assert (root/'REVIEW.md').is_file()
print(json.dumps({'artifact_count':len(entries),'manifest_sha256':digest(root/'artifact_manifest.json'),'all_artifacts_reverified':True,'packet_unchanged':True},indent=2))
