"""Stdlib-only transport composition checks, no host or source imports."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
p=Path(__file__).with_name('stage-dispatch.py');text=p.read_text();tree=ast.parse(text)
body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
compile(body,'<remote>','exec');checks=0
def require(ok,message):
 global checks
 if not ok:raise RuntimeError(message)
 checks+=1
require(hashlib.sha256(p.read_bytes()).hexdigest()=='0333c5b8a238cc4d39cefffe845ba21b28bd7f8c7096fb9fc664f1207dc26803','transport pin')
for token in ('NORMAL_PYTHON_REQUIRED','EXACT_RELEASE_COMMIT_PATH_AND_PREDECESSOR_COMMIT_REQUIRED','RELEASE_PIN_MISMATCH','STATE_SCOPE_MISMATCH','PREDECESSOR_RAW_CHANGED','complete_process_census','PAYLOAD_SENSITIVE_MARKER','OUTPUT_SENSITIVE_MARKER','CPU_CHECKPOINT_SYNTHETIC_QUALIFICATION_COMPLETE'):
 require(token in text,token)
for token in ('(66305,832589)','LOCK_EX|fcntl.LOCK_NB','LABEL_ALREADY_CONSUMED','PRIOR_TERMINAL_CHANGED','PRIOR_OUTER_CHANGED','COMPLETE_PENDING_INDEPENDENT_REVIEW','OLD_QUEUE_PAUSE_MISSING','DISK_HEADROOM_PLUS_NEW_FOOTPRINT','BUSY_PS','BUSY_DEVICE','DEVICE_HOLDERS','CACHE_NOT_FRESH','len(set(cache_paths))==6','NEURON_COMPILE_CACHE_URL'):
 require(token in body,token)
require('trn21' not in body and '596823' not in body,'no stale host')
require('projected=3*1024**3' in body and 'free>=10.5*1024**3+projected' in body,'unchanged footprint floor')
require(body.index('fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)')<body.index('ps=subprocess.run'),'ps under original lock')
require(body.index('ls=subprocess.run')<body.index('fcntl.flock(lockfd,fcntl.LOCK_UN)'),'device under lock')
require(body.index('CACHE_NOT_FRESH')<body.index('root.mkdir'),'six caches absent before writes')
require(body.count('subprocess.Popen(')==1,'one dispatcher')
require("['--num-steps','5']" in text,'sole cold argv override')
r=subprocess.run([sys.executable,'-B',str(p),'--local-check'],capture_output=True,text=True)
require(r.returncode==0,r.stderr);o=json.loads(r.stdout)
require(o['files']==9 and not o['host_calls'] and not o['launch_authorized'],'local only')
require("full_commit='1491c1c2a18a2d86da8783cf632fc2290bfeafb1'" in text,'exact same-host terminal archive')
require("helper_prior=prior" in text and "(prior['lock_device'],prior['lock_inode'])==(66305,596823)" in text,'separate cross-host helper proof')
require("len(prior['known_prior_identities'])==19" in text,'nineteen full-job identities')
print(json.dumps({'status':'MECHANICAL_PREPARATION_PASS','checks':checks,'host_calls':False,'launch_authorized':False}))
