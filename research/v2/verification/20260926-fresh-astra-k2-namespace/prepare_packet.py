"""Freeze compact input-only kernel review and reuse the reviewed operation."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
REF='d0f0d7fbdbe07ddee39680ee1a2b79fdfac388ca'
BASE='18b9365aabb63fa85e1720fdefc11177ca8e7853'
SOURCE='research/v2/lanes/k2-resident-consumer/consumer_nki.py'
PROPOSAL='research/v2/proposals/20260926-k2-native-name-binding/'
RAW_REF='94fb52c4'
RAW_PATH='research/v2/operator/20260926-trn22-k2-reporting-native/completed/results-bundle.json'
PINS={'baseline':'57b5b118e86eb9d78e26a011b276904aed9065868b8a12f39169e1257c3a1843','candidate':'4939265bf73b3d5f42276e72e48f022df338226a40a9be5b79ed63bde6a4f5b8'}
def blob(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=REPO)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    if (HERE/'PREPARED.json').exists():raise RuntimeError('No duplicate preparation')
    sources={'baseline':blob(BASE,SOURCE),'candidate':blob(REF,SOURCE)}
    if {k:sha(v) for k,v in sources.items()}!=PINS:raise RuntimeError('Source mismatch')
    supplied_diff=blob(REF,PROPOSAL+'kernel-full-context.diff')
    old=[];new=[]
    for line in supplied_diff.decode().splitlines(True):
        if line.startswith(('diff ','index ','--- ','+++ ','@@')):continue
        if line.startswith((' ','-')):old.append(line[1:])
        if line.startswith((' ','+')):new.append(line[1:])
    if ''.join(old).encode()!=sources['baseline'] or ''.join(new).encode()!=sources['candidate']:raise RuntimeError('Incomplete diff')
    bundle=json.loads(blob(RAW_REF,RAW_PATH))
    names=['terminal.json']+[f'receipts/rank{i}.json' for i in range(4)]
    receipts={}
    for name in names:
        raw=blob(REF,PROPOSAL+'original-native/'+name)
        entry=bundle['files'][name]
        if raw!=entry['text'].encode() or sha(raw)!=entry['sha256'] or len(raw)!=entry['bytes']:raise RuntimeError('Raw archive mismatch '+name)
        receipts[name]=raw
    packet=Path(tempfile.mkdtemp(prefix='trainium-fresh-k2-namespace-packet-20260926.'))
    run=Path(tempfile.mkdtemp(prefix='trainium-fresh-k2-namespace-review-20260926.'))
    (run/'output').mkdir()
    (packet/'kernel-full-context.diff').write_bytes(supplied_diff)
    (packet/'lane-spec.md').write_bytes((HERE/'lane-spec.md').read_bytes())
    for name,raw in receipts.items():
        dest=packet/'original-native'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    included=[{'path':str(p.relative_to(packet)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(packet.rglob('*')) if p.is_file()]
    manifest={'candidate_commit':REF,'baseline_commit':BASE,'source_path':SOURCE,'source_sha256':PINS,'raw_commit':RAW_REF,'included':included,'prior_history':False,'shared_branch':False,'excluded':['repository','implementation notes/tests/conclusions','prior reviews','unrelated training/model receipts','checker/reference source not changed by this patch']}
    encoded=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode();(packet/'INPUT_MANIFEST.json').write_bytes(encoded);pin=sha(encoded)
    template=(HERE.parent/'20260926-fresh-astra-startup-adjusted-schedule/session/operate.py').read_text()
    operation=template.replace('/tmp/trainium-fresh-startup-schedule-packet-20260926.loz9jxr3',str(packet)).replace('fef50271f26087f3a6ed286f217a05e27b96abc40eb1960a1d385813a13f11f4',pin)
    (run/'operate.py').write_text(operation)
    prepared={'packet':str(packet),'run':str(run),'input_manifest_sha256':pin,'candidate_commit':REF,'candidate_sha256':PINS['candidate'],'raw_commit':RAW_REF,'model_requested':'gpt-6-astra','operation_changes':['packet path','manifest pin'],'fresh_session_started':False,'task_file_count':len(included),'task_bytes':sum(x['bytes'] for x in included)}
    (HERE/'PREPARED.json').write_text(json.dumps(prepared,indent=2,sort_keys=True)+'\n')
    print(json.dumps(prepared))
if __name__=='__main__':main()
