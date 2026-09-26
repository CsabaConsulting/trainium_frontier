"""One bounded read-only observation of an exact retained packet binding."""
import ast,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=Path('/home/poojith-devan/Desktop/Trainium')
choice=sys.argv[1]
packet_path=HERE.parent/'20260926-final-sprint-p0/trn22-bound-packet.json' if choice=='trn22-p0-result' else HERE/(choice+'-bound-packet.json')
b=json.loads(packet_path.read_text())['binding'];out=HERE/(choice+'-live.json');assert not out.exists()
original=HERE.parent/'20260926-final-sprint-p0/observe_once.py';raw=original.read_bytes();assert hashlib.sha256(raw).hexdigest()=='acc5067b73fa106b6b5f2d40b7a1f9f422a930b66e73d246f6731ba53f51418d'
tree=ast.parse(raw);body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
body=body.replace("'train/rehearsal.json'):","'train/rehearsal.json','fresh-public20m-launch.json','fresh-public20m.log','eval/rehearsal.json','eval-exit-code.txt','checkpoint-after-postflight.json'):")
wire=('P='+repr({'binding':b})+'\n'+body).encode()
spec=importlib.util.spec_from_file_location('connector',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);assert not m.SENSITIVE.search(wire)
argv,*_=m.load_transport(b['host']);argv[-1]='exec /usr/bin/python3 -I -B -S -'
r=subprocess.run(argv,input=wire,capture_output=True,timeout=45);assert not m.SENSITIVE.search(r.stdout+r.stderr)
receipt={'returncode':r.returncode,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None,'payload_sha256':hashlib.sha256(wire).hexdigest()}
with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
o=receipt['remote'];print(json.dumps({'choice':choice,'returncode':r.returncode,'utc':o['utc'] if o else None,'workers':[{'pid':p['identity_before']['pid'],'start_ticks':p['identity_before']['start_ticks'],'rank':p.get('environment_whitelist',{}).get('RANK')} for p in o['processes'] if 'RANK' in p.get('environment_whitelist',{})] if o else [],'original_lock':o['original_lock'] if o else None,'training_launch':o['files']['training-launch.json'].get('text') if o else None,'terminal':o['files']['terminal.json'].get('text') if o else None,'error':r.stderr.decode(),'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
