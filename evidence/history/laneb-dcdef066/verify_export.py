"""Portable, read-only stdlib byte verification; no model/helpers/network/host execution."""
import argparse,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
DRAFT_PIN='2618fe90bd2580c76e08eca28c599401f1428c1ddcd224aa082713b6aad78ace'
SOURCE='dcdef066d8d0827f4ecdf0a80f8517e5348af28eea7acc06e03df4db49d4e90d'
COMMAND=b'NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0\n'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest-sha256',required=True);p.add_argument('--compare-canonical',action='store_true');a=p.parse_args()
 assert len(a.manifest_sha256)==64 and all(c in '0123456789abcdef'for c in a.manifest_sha256)
 raw=(H/'EXPORT_MANIFEST.json').read_bytes();assert sha(raw)==a.manifest_sha256,'EXPORT_MANIFEST_PIN'
 m=json.loads(raw);assert m['status']=='REVIEW_REQUEST_ONLY_NOT_UPLOAD_READY'
 assert m['draft_manifest_sha256']==DRAFT_PIN and m['source_sha256']==SOURCE
 assert m['upload_authorized']is m['submission_ready']is m['claude_accepted']is m['model_or_torch_execution']is False
 assert m['integrity_review_scope']=='PACKAGE_INTEGRITY_ONLY' and m['integrity_reviewer_fresh_session']is False
 expected=set(m['files']);actual={str(x.relative_to(H))for x in H.rglob('*')if x.is_file()and x.name!='EXPORT_MANIFEST.json'}
 assert actual==expected,'EXTRA_OR_MISSING_EXPORT_FILE'
 for name,pin in m['files'].items():
  rel=Path(name);assert not rel.is_absolute()and '..'not in rel.parts and not(H/rel).is_symlink()
  assert sha((H/rel).read_bytes())==pin,'EXPORTED_BYTE_PIN:'+name
 dm=json.loads((H/'frozen-draft/MANIFEST.json').read_bytes());assert sha((H/'frozen-draft/MANIFEST.json').read_bytes())==DRAFT_PIN
 assert {'frozen-draft/'+n for n in dm['files']}|{'frozen-draft/MANIFEST.json'}=={n for n in expected if n.startswith('frozen-draft/')}
 for name,pin in dm['files'].items():assert m['files']['frozen-draft/'+name]==pin
 assert (H/'train.py').read_bytes()==(H/'frozen-draft/train.py').read_bytes()and sha((H/'train.py').read_bytes())==SOURCE
 assert (H/'launch-command.txt').read_bytes()==(H/'frozen-draft/launch-command.txt').read_bytes()==COMMAND
 c=json.loads((H/'frozen-draft/CHECKLIST.json').read_bytes());assert [x['status']for x in c['items']]==['FAIL','FAIL','FAIL','PASS','FAIL','FAIL','FAIL','FAIL']and c['package_accepted']is False
 r=json.loads((H/'integrity-review/RESULT.json').read_bytes());assert r['verdict']=='PACKAGE_INTEGRITY_ONLY'and r['fresh_session']is False and r['manifest_sha256']==DRAFT_PIN and r['checks_passed']==586
 assert r['claude_compliance_acceptance']is None and r['submission_ready']is False
 compared=0
 if a.compare_canonical:
  for name in expected:
   origin=None
   if name.startswith('frozen-draft/'):origin=Path(m['canonical_draft'])/name.removeprefix('frozen-draft/')
   elif name.startswith('integrity-review/'):origin=Path(m['canonical_integrity_review'])/name.removeprefix('integrity-review/')
   elif name in('train.py','launch-command.txt'):origin=Path(m['canonical_draft'])/name
   if origin is not None:assert not origin.is_symlink()and origin.read_bytes()==(H/name).read_bytes(),'CANONICAL_COPY:'+name;compared+=1
 print(json.dumps(dict(status='EXPORT_BYTES_VERIFIED_REVIEW_REQUEST_ONLY',export_manifest_sha256=a.manifest_sha256,hashed_files=len(expected),canonical_copies_compared=compared,source_sha256=SOURCE,checklist_nonpassing_items=[1,2,3,5,6,7,8],integrity_scope='PACKAGE_INTEGRITY_ONLY',reviewer_fresh_session=False,claude_acceptance=None,submission_ready=False,upload_authorized=False,host_calls=0,model_or_torch_execution=False),indent=2))
if __name__=='__main__':main()
