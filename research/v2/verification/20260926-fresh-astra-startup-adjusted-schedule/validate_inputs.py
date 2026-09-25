"""Input/isolation provenance only; no source judgment or numerical test."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
prepared=json.loads((HERE/'PREPARED.json').read_text());packet=Path(prepared['packet']);run=Path(prepared['run'])
checks=[]
def check(name,value):
    if not value:raise RuntimeError(name)
    checks.append(name)
def sha(raw):return hashlib.sha256(raw).hexdigest()
manifest=json.loads((packet/'INPUT_MANIFEST.json').read_text())
check('manifest pin',sha((packet/'INPUT_MANIFEST.json').read_bytes())==prepared['input_manifest_sha256'])
current=('tiny-base.json','tiny-default.json','tiny-off.json','tiny-on779.json','tiny-on790.json','tiny-on1000.json','tiny-commands.json','failed-attempt1-base.json')
expected={'baseline-to-candidate.diff','lane-spec.md'}|{'raw/'+n for n in current}|{'raw/attempt2-stub2/'+n for n in current[:7]}
check('only approved diff/spec/raw files',set(x['path'] for x in manifest['included'])==expected)
check('exact inventory',set(str(p.relative_to(packet)) for p in packet.rglob('*') if p.is_file())==expected|{'INPUT_MANIFEST.json'})
for item in manifest['included']:
    path=packet/item['path'];check('hash '+item['path'],not path.is_symlink() and path.stat().st_size==item['bytes'] and sha(path.read_bytes())==item['sha256'])
old=[];new=[]
for line in (packet/'baseline-to-candidate.diff').read_text().splitlines(True):
    if line.startswith(('--- ','+++ ','@@')):continue
    if line.startswith((' ','-')):old.append(line[1:])
    if line.startswith((' ','+')):new.append(line[1:])
check('complete baseline source',sha(''.join(old).encode())==manifest['source_sha256']['baseline'])
check('complete candidate source',sha(''.join(new).encode())==manifest['source_sha256']['candidate'])
for name in current[:6]:
    d=json.loads((packet/'raw'/name).read_text())
    check('frozen complete raw '+name,d['torchrun_rc']==0 and d['timed_out'] is False and len(d['ranks'])==4 and all(r['schedule_observations']['startup_steps_excluded']==5 for r in d['ranks']))
check('no inherited history/branch',manifest['prior_history'] is manifest['shared_branch'] is False)
template=(HERE.parent/'20260926-fresh-astra-k2-reporting-successor/session/operate.py').read_text()
expected_operation=template.replace('/tmp/trainium-fresh-k2-reporting-packet-20260926.njo954e1',str(packet)).replace('47e5203a059347d6b4d6e5aa9202b0f1d5325ec3a81a08db01ce692dc81ba911',prepared['input_manifest_sha256'])
check('operation only path and pin changed',(run/'operate.py').read_text()==expected_operation)
preflight=json.loads((run/'preflight.json').read_text())
check('outer isolation preflight',preflight['exit']==0 and preflight['manifest_sha256']==prepared['input_manifest_sha256'])
prompt=json.loads((run/'prompt-input.json').read_text());check('fresh prompt input',isinstance(prompt,list) and len(prompt)==5)
launch=None
if (run/'launch.json').exists():
    launch=json.loads((run/'launch.json').read_text());command=launch['command']
    check('exact model/input',launch['model_requested']==command[command.index('-m')+1]=='gpt-6-astra' and launch['manifest_sha256']==prepared['input_manifest_sha256'])
    check('new isolated ephemeral CLI',all(flag in command for flag in ('--ephemeral','--ignore-user-config','--ignore-rules','--skip-git-repo-check','--unshare-all')) and 'resume' not in command and 'fork' not in command)
    check('no repo/history and bounded',launch['prior_history'] is launch['repository_mounted'] is False and launch['wall_cap_seconds']==1200)
result={'status':'PASS_INPUT_OPERATION_PROVENANCE_ONLY','checks':len(checks),'manifest_sha256':prepared['input_manifest_sha256'],'started_utc':launch['started_utc'] if launch else None,'reviewer_verdict':'PENDING','source_gate_decided_here':False,'checks_passed':checks}
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
text=json.dumps(result,indent=2)+'\n'
if args.output:args.output.write_text(text)
print(text,end='')
