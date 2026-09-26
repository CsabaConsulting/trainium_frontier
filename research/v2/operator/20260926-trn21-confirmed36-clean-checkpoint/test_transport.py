"""Local stdlib transport checks; no imports of source/helper or host access."""
import ast, hashlib, json, subprocess, sys
from pathlib import Path

p=Path(__file__).with_name('stage-dispatch.py');text=p.read_text();tree=ast.parse(text)
body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
compile(body,'<remote-body>','exec')
checks=0
def require(ok,message):
    global checks
    if not ok:raise RuntimeError(message)
    checks+=1
require(hashlib.sha256(p.read_bytes()).hexdigest()=='9246de36317ea03fcc498aa555649d04b4477bc4a21b6881f02cf0c3ade0719c','transport bytes')
for token in ('NORMAL_PYTHON_REQUIRED','EXACT_RELEASE_COMMIT_AND_PATH_REQUIRED','RELEASE_NOT_EXACT_PACKET','STATE_SCOPE_MISMATCH','HOST_NOT_RELEASED','complete_process_census','PAYLOAD_SENSITIVE_MARKER','OUTPUT_SENSITIVE_MARKER'):
    require(token in text,token)
for token in ('(66305,596823)','LOCK_EX|fcntl.LOCK_NB','LABEL_ALREADY_CONSUMED','PRIOR_DESCENDANT_PGID_SID_LIVE','CPU_PAIR_COMPLETE_PENDING_INDEPENDENT_REVIEW','PRIOR_OUTER_FAILED','OLD_QUEUE_PAUSE_MISSING','DISK_FLOOR_PLUS_PROJECTED_FOOTPRINT','AVAILABLE_RAM_FLOOR','BUSY_PS','BUSY_DEVICE','DEVICE_HOLDERS','RETAINED_INPUT_SHA','RETAINED_INPUT_IDENTITY','detached_command_template','input_pins','SUPERVISOR_ASSIGNMENT.md'):
    require(token in text,token)
require(body.index('fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)')<body.index("ps=subprocess.run"),'ps under lock')
require(body.index("ls=subprocess.run")<body.index('fcntl.flock(lockfd,fcntl.LOCK_UN)'),'device under lock')
require(body.index("holders=subprocess.run")<body.index('root.mkdir'),'fuser before writes')
require(body.index("RETAINED_INPUT_SHA")<body.index('root.mkdir'),'retained pins before writes')
require('projected=1024**3' in body and 'free>=10.5*1024**3+projected' in body,'disk footprint')
require("mem>=64*1024**3" in body,'RAM floor')
require(body.count('subprocess.Popen(')==1,'single dispatcher only')
require('832589' not in body and 'trn22' not in body,'no stale host')
require("/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1/runs/PAUSE_NEXT" in body,'original retired queue')
require(text.index("if sys.argv[1:]==['--local-check']")<text.index("m.load_transport('trn21')"),'local check before transport')
r=subprocess.run([sys.executable,'-B',str(p),'--local-check'],capture_output=True,text=True)
require(r.returncode==0,r.stderr)
result=json.loads(r.stdout)
require(result['retained_inputs']==17 and not result['host_calls'] and not result['launch_authorized'],'local-only result')
print(json.dumps({'checks':checks,'status':'MECHANICAL_TRANSPORT_CHECKS_PASS','host_calls':False,'launch_authorized':False}))
