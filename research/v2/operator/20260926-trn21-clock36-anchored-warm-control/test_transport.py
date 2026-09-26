"""Local stdlib transport checks; no host, model or source execution."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
p=Path(__file__).with_name('stage-dispatch.py');text=p.read_text();tree=ast.parse(text)
body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
compile(body,'<remote>','exec');checks=0
def require(ok,message):
 global checks
 if not ok:raise RuntimeError(message)
 checks+=1
require(hashlib.sha256(p.read_bytes()).hexdigest()=='334abbd31b27b03a77d5630dc2353db6b5cdbac498af7d393235de81ed8610cc','transport pin')
for token in ('NORMAL_PYTHON_REQUIRED','EXACT_RELEASE_COMMIT_AND_PATH_REQUIRED','RELEASE_PIN_MISMATCH','STATE_SCOPE_MISMATCH','HOST_NOT_RELEASED','PREDECESSOR_RAW_CHANGED','complete_process_census','PAYLOAD_SENSITIVE_MARKER','OUTPUT_SENSITIVE_MARKER'):
 require(token in text,token)
for token in ('(66305,596823)','LOCK_EX|fcntl.LOCK_NB','LABEL_ALREADY_CONSUMED','PRIOR_TERMINAL_CHANGED','PRIOR_OUTER_CHANGED','CPU_CHECKPOINT_ONLY_COMPLETE_PENDING_INDEPENDENT_REVIEW','OLD_QUEUE_PAUSE_MISSING','DISK_FLOOR_PLUS_PROJECTED_FOOTPRINT','BUSY_PS','BUSY_DEVICE','DEVICE_HOLDERS','ANCHOR_STAT','ANCHOR_CHANGED','ANCHOR_HASH','RETAINED_CACHE_MISSING'):
 require(token in body,token)
require('projected=2*1024**3' in body and 'free>=10.5*1024**3+projected' in body,'approved footprint')
require(body.index('fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)')<body.index('ps=subprocess.run'),'ps under lock')
require(body.index('ls=subprocess.run')<body.index('fcntl.flock(lockfd,fcntl.LOCK_UN)'),'device under lock')
require(body.index('ANCHOR_HASH')<body.index('root.mkdir'),'readonly anchor before staging')
require(body.count('subprocess.Popen(')==1 and "plan['dispatch_command']" in body,'single exact dispatcher')
require("['parent_training_command']" in text and '==[3]' in text,'only declared script argument')
require("type(a[k]) is int and a[k]>2**53" in text,'raw nanoseconds stay Python integers')
require("os.chmod(dst,0o444)" in body and "dst=root/name" in body,'chmod only new staged files')
require('repair' not in body or 'no old file writes' in body,'no repair workflow')
r=subprocess.run([sys.executable,'-B',str(p),'--local-check'],capture_output=True,text=True)
require(r.returncode==0,r.stderr);o=json.loads(r.stdout)
require(o['files']==11 and not o['host_calls'] and not o['launch_authorized'],'local-only check')
print(json.dumps({'status':'MECHANICAL_TRANSPORT_PREPARATION_PASS','checks':checks,'host_calls':False,'launch_authorized':False}))
