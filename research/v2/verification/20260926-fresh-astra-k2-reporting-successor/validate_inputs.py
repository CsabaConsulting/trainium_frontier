"""Operator-only stdlib packet/isolation binding audit, not reviewer conclusions."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
prepared=json.loads((HERE/'PREPARED.json').read_text())
packet=Path(prepared['packet']);run=Path(prepared['run']);checks=[]
def check(name,ok):
    if not ok:raise RuntimeError(name)
    checks.append(name)
def sha(data):return hashlib.sha256(data).hexdigest()
encoded=(packet/'INPUT_MANIFEST.json').read_bytes();manifest=json.loads(encoded)
check('manifest pin',sha(encoded)==prepared['input_manifest_sha256'])
expected={'checker.diff','consumer_nki.py.diff','reference.py.diff','lane-spec.md','raw/checks-normal.json','raw/checks-optimized.json','raw/scalar-inputs.json'}
check('only seven approved task files',set(x['path'] for x in manifest['included'])==expected)
check('exact filesystem inventory',set(str(p.relative_to(packet)) for p in packet.rglob('*') if p.is_file())==expected|{'INPUT_MANIFEST.json'})
for row in manifest['included']:
    p=packet/row['path'];check('input '+row['path'],not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p.read_bytes())==row['sha256'])
def reconstruct(path):
    old=[];new=[]
    for line in path.read_text().splitlines(True):
        if line.startswith(('--- ','+++ ','@@')):continue
        if line.startswith((' ','-')):old.append(line[1:])
        if line.startswith((' ','+')):new.append(line[1:])
    return ''.join(old).encode(),''.join(new).encode()
old,new=reconstruct(packet/'checker.diff')
check('complete baseline checker',sha(old)==manifest['baseline_checker_sha256'])
check('complete candidate checker',sha(new)==manifest['candidate_source_sha256']['check_native.py'])
for name in ('consumer_nki.py','reference.py'):
    old,new=reconstruct(packet/(name+'.diff'));check('complete '+name,old==b'' and sha(new)==manifest['candidate_source_sha256'][name])
scalar=json.loads((packet/'raw/scalar-inputs.json').read_text())
check('numeric counterexample values only',set(scalar)=={'format','cases'} and len(scalar['cases'])==6 and all(set(row)=={'name','actual_hex','expected_hex'} for row in scalar['cases']))
check('fresh no history branch',manifest['prior_history'] is manifest['shared_branch'] is False and manifest['scalar_inputs_projection']['prior_conclusions_included'] is False)
template=(HERE.parent/'20260926-fresh-astra-architecture-recursive/session/operate.py').read_text()
expected_operation=template.replace('/tmp/trainium-fresh-architecture-recursive-packet-20260926.dsxaqcz_',str(packet)).replace('75a0dbd970b2397c8ad7e75ad564cf1deb28f1a1eed5e4a51e0a288d952895dd',prepared['input_manifest_sha256'])
check('operation only path/pin changed',(run/'operate.py').read_text()==expected_operation)
preflight=json.loads((run/'preflight.json').read_text())
check('isolation preflight exit0',preflight['exit']==0 and preflight['manifest_sha256']==prepared['input_manifest_sha256'])
prompt=json.loads((run/'prompt-input.json').read_text())
check('fresh prompt composition',isinstance(prompt,list) and len(prompt)==5)
launch=json.loads((run/'launch.json').read_text());command=launch['command']
check('exact model and bound manifest',command[command.index('-m')+1]==launch['model_requested']=='gpt-6-astra' and launch['manifest_sha256']==prepared['input_manifest_sha256'])
check('fresh new isolated CLI',all(flag in command for flag in ('--ephemeral','--ignore-user-config','--ignore-rules','--skip-git-repo-check','--unshare-all')) and 'resume' not in command and 'fork' not in command)
check('no repo/history mounted',launch['prior_history'] is launch['repository_mounted'] is False)
check('bounded one operation',launch['wall_cap_seconds']==1200 and '--disable' in command and 'multi_agent' in command)
print(json.dumps({'status':'PASS_INPUT_AND_OPERATION_BINDING_ONLY','checks':len(checks),'tests':checks,'manifest_sha256':prepared['input_manifest_sha256'],'started_utc':launch['started_utc'],'child_pid':launch['child_pid'],'reviewer_verdict':'PENDING','no_source_gate_decision_here':True},indent=2))
