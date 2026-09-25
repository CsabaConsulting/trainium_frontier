"""Packet/operation authentication only; not independent source approval."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
p=json.loads((HERE/'PREPARED.json').read_text());packet=Path(p['packet']);run=Path(p['run']);checks=[]
def check(name,value):
    if not value:raise RuntimeError(name)
    checks.append(name)
def sha(raw):return hashlib.sha256(raw).hexdigest()
m=json.loads((packet/'INPUT_MANIFEST.json').read_text())
check('manifest pinned',sha((packet/'INPUT_MANIFEST.json').read_bytes())==p['input_manifest_sha256'])
expected={'kernel-full-context.diff','lane-spec.md','original-native/terminal.json'}|{f'original-native/receipts/rank{i}.json' for i in range(4)}
check('only diff spec five raw files',set(x['path'] for x in m['included'])==expected)
check('exact inventory',set(str(x.relative_to(packet)) for x in packet.rglob('*') if x.is_file())==expected|{'INPUT_MANIFEST.json'})
check('small packet',len(m['included'])==7 and sum(x['bytes'] for x in m['included'])==32019)
for item in m['included']:
    path=packet/item['path'];check('hash '+item['path'],not path.is_symlink() and path.stat().st_size==item['bytes'] and sha(path.read_bytes())==item['sha256'])
old=[];new=[]
for line in (packet/'kernel-full-context.diff').read_text().splitlines(True):
    if line.startswith(('diff ','index ','--- ','+++ ','@@')):continue
    if line.startswith((' ','-')):old.append(line[1:])
    if line.startswith((' ','+')):new.append(line[1:])
check('full old source',sha(''.join(old).encode())==m['source_sha256']['baseline'])
check('full new source',sha(''.join(new).encode())==m['source_sha256']['candidate']==p['candidate_sha256'])
check('no inherited history/branch',m['prior_history'] is m['shared_branch'] is False)
template=(HERE.parent/'20260926-fresh-astra-startup-adjusted-schedule/session/operate.py').read_text()
expected_operation=template.replace('/tmp/trainium-fresh-startup-schedule-packet-20260926.loz9jxr3',str(packet)).replace('fef50271f26087f3a6ed286f217a05e27b96abc40eb1960a1d385813a13f11f4',p['input_manifest_sha256'])
check('operation only path/pin changed',(run/'operate.py').read_text()==expected_operation)
pre=json.loads((run/'preflight.json').read_text());check('real outer-isolation preflight',pre['exit']==0 and pre['manifest_sha256']==p['input_manifest_sha256'])
prompt=json.loads((run/'prompt-input.json').read_text());check('fresh prompt composition',isinstance(prompt,list) and len(prompt)==5)
check('original failure retained',json.loads((packet/'original-native/terminal.json').read_text())['complete'] is False)
launch=None
if (run/'launch.json').exists():
    launch=json.loads((run/'launch.json').read_text());command=launch['command']
    check('exact fresh model',launch['model_requested']==command[command.index('-m')+1]=='gpt-6-astra')
    check('exact launched input',launch['manifest_sha256']==p['input_manifest_sha256'])
    check('new ephemeral isolated session',all(x in command for x in ('--ephemeral','--ignore-user-config','--ignore-rules','--skip-git-repo-check','--unshare-all')) and 'resume' not in command and 'fork' not in command)
    check('bounded no repository/history',launch['prior_history'] is launch['repository_mounted'] is False and launch['wall_cap_seconds']==1200)
result={'status':'PASS_INPUT_OPERATION_PROVENANCE_ONLY','checks':len(checks),'checks_passed':checks,'input_manifest_sha256':p['input_manifest_sha256'],'source_gate_decided_here':False,'started_utc':launch['started_utc'] if launch else None}
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args();encoded=json.dumps(result,indent=2)+'\n'
if args.output:args.output.write_text(encoded)
print(encoded,end='')
