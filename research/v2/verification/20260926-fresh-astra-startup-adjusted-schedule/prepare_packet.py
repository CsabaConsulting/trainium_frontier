"""Create one packet only after a frozen complete raw commit; never starts review."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
BASE='b4d83b5db66971f3c348837f04f53cba65aa6d33'
CANDIDATE='7419927c7417b5666a984ec9c2b849feadabaaa9'
SOURCE='workflow/proposals/research-v2/train.py'
RAW='research/v2/lanes/startup-adjusted-schedule/'
NAMES=('tiny-base.json','tiny-default.json','tiny-off.json','tiny-on779.json','tiny-on790.json','tiny-on1000.json','tiny-commands.json','failed-attempt1-base.json')
EXPECTED={'baseline':'3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c','candidate':'36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'}
def blob(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=REPO)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw-commit',required=True);args=parser.parse_args()
    if (HERE/'PREPARED.json').exists():raise RuntimeError('No duplicate review preparation')
    raw_commit=subprocess.check_output(['git','rev-parse',args.raw_commit+'^{commit}'],cwd=REPO,text=True).strip()
    sources={'baseline':blob(BASE,SOURCE),'candidate':blob(CANDIDATE,SOURCE)}
    if {k:sha(v) for k,v in sources.items()}!=EXPECTED or blob(raw_commit,SOURCE)!=sources['candidate']:raise RuntimeError('Source mismatch')
    receipts={name:blob(raw_commit,RAW+name) for name in NAMES}
    for name in NAMES[:6]:
        receipt=json.loads(receipts[name]);expected=EXPECTED['baseline' if name=='tiny-base.json' else 'candidate']
        if receipt['train_py_sha256']!=expected or receipt['torchrun_rc']!=0 or receipt['timed_out'] or len(receipt['ranks'])!=4:raise RuntimeError('Incomplete raw evidence '+name)
    packet=Path(tempfile.mkdtemp(prefix='trainium-fresh-startup-schedule-packet-20260926.'))
    run=Path(tempfile.mkdtemp(prefix='trainium-fresh-startup-schedule-review-20260926.'))
    (packet/'raw').mkdir();(run/'output').mkdir()
    diff=''.join(difflib.unified_diff(sources['baseline'].decode().splitlines(True),sources['candidate'].decode().splitlines(True),fromfile='baseline/train.py',tofile='candidate/train.py',n=100000))
    (packet/'baseline-to-candidate.diff').write_text(diff)
    (packet/'lane-spec.md').write_bytes((HERE/'lane-spec.md').read_bytes())
    for name,raw in receipts.items():(packet/'raw'/name).write_bytes(raw)
    included=[{'path':str(p.relative_to(packet)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(packet.rglob('*')) if p.is_file()]
    manifest={'candidate_commit':CANDIDATE,'baseline_commit':BASE,'raw_commit':raw_commit,'source_sha256':EXPECTED,'included':included,'prior_history':False,'shared_branch':False,'excluded':['repository','implementation tests/notes/conclusions','prior reviews','checkpoint files','full-size/native/quality claims']}
    encoded=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode();(packet/'INPUT_MANIFEST.json').write_bytes(encoded);pin=sha(encoded)
    template=(HERE.parent/'20260926-fresh-astra-k2-reporting-successor/session/operate.py').read_text()
    template=template.replace('/tmp/trainium-fresh-k2-reporting-packet-20260926.njo954e1',str(packet)).replace('47e5203a059347d6b4d6e5aa9202b0f1d5325ec3a81a08db01ce692dc81ba911',pin)
    (run/'operate.py').write_text(template)
    result={'packet':str(packet),'run':str(run),'input_manifest_sha256':pin,'candidate_sha256':EXPECTED['candidate'],'candidate_commit':CANDIDATE,'raw_commit':raw_commit,'fresh_session_started':False,'operation_changes':['packet path','manifest pin'],'model_requested':'gpt-6-astra'}
    (HERE/'PREPARED.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
