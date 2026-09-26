"""Stdlib mechanical tests only; no host call or scientific execution."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
path=Path(__file__).with_name('stage-dispatch.py');text=path.read_text();tree=ast.parse(text)
body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
compile(body,'<remote-staging>','exec');checks=[]
def need(ok,name):
 if not ok:raise RuntimeError(name)
 checks.append(name)
need(hashlib.sha256(path.read_bytes()).hexdigest()=='adcb67146b1f542289b7cc21ac552c2bcd558e25d40d624c461a2d8ddc5127ec','exact prepared transport')
for token in ('NORMAL_PYTHON_REQUIRED','FROZEN_PACKET_CHANGED','EXACT_COLD_R2_ARCHIVE_COMMIT_RELEASE_COMMIT_AND_PATH_REQUIRED','RELEASE_CHANGED','RELEASE_PIN_MISMATCH','ACTUAL_COLD_R2_ARCHIVE_CHANGED','WRONG_COLD_PREDECESSOR','COLD_R2_NOT_QUIESCENT','ORIGINAL_LOCK_CHANGED','MISSING_ACTUAL_R2_IDS','SURVIVING_R2_LINEAGE','ACTUAL_COLD_FIFTH_OVER_900','ACTUAL_COLD_PROCESS_BUDGET','COLD_CHECKPOINT_BINDING','ACTUAL_FRESH_CACHE_RECEIPT','ACTUAL_COLD_SCIENCE_ARGV','HOST_NOT_RELEASED','STATE_SCOPE_MISMATCH','PAYLOAD_SENSITIVE_MARKER','OUTPUT_SENSITIVE_MARKER'):
 need(token in text,token)
need("rehearsal['step_boundaries']==5" in text and "[x['step'] for x in observed]==list(range(5))" in text,'actual exactly five boundaries')
need("times[4];assert fifth<=900" in text,'uncapped fifth not source allowance')
need("terminal['returncode']==0" in text and "outer['returncode']==0" in text and "rehearsal['returncode']==0" in text,'actual terminal outer and torchrun outcome')
need("len(workers)==4" in text and "{'0','1','2','3'}" in text and "env['WORLD_SIZE']=='4'" in text,'four actual observed ranks')
need('checkpoint_after_terminal' in text and "checkpoint['stable']" in text,'actual stable postflight checkpoint')
need('65b377e179c3330e9542348fa35be30ce5a19ece' in text,'real r2 four-rank receipt frozen')
need('d74ba2f8d367c61a00f172f9cb7ca0e1145140ca' in text,'exact full packet frozen')
need("for value in (b['label'],b['source_sha256'],pins['NATIVE_BINDING.disabled.json'],pins['host_driver.py'],prior_commit)" in text,'release explicitly binds actual cold raw commit')
for token in ('(66305,832589)','HOST_IDENTITY','LABEL_ALREADY_CONSUMED','OLD_QUEUE_PID_LIVE','PRIOR_COLD_RECEIPT_CHANGED','PRIOR_DIAGNOSTIC_TERMINAL_MISMATCH','PRIOR_DIAGNOSTIC_OUTER_NOT_EXITED','LOCK_EX|fcntl.LOCK_NB','DISK_HEADROOM_PLUS_NEW_FOOTPRINT','OLD_QUEUE_PAUSE_MISSING','PRIOR_PRIOR_DIAGNOSTIC_LINEAGE_REMAINS','BUSY_PS','BUSY_DEVICE','DEVICE_HOLDERS','ACTUAL_HARNESS_CONTAINMENT','HARNESS_PROJECT_MISMATCH','PHASE_SANDBOX_CONSUMED','CACHE_NOT_FRESH','TWELVE_PHASE_PRIVATE_CACHES','PRIOR_CHECKPOINT_CHANGED'):
 need(token in body,token)
need('trn21' not in body and '596823' not in body,'no stale host')
need("previous=Path(P['prior_root'])" in body,'actual current r2 predecessor not failed r1')
need('projected=3*1024**3' in body and 'free>=10.5*1024**3+projected' in body,'unchanged13.5GiB combined floor')
need(body.index('fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)')<body.index('ps=subprocess.run'),'fresh ps under original lock')
need(body.index('ls=subprocess.run')<body.index('fcntl.flock(lockfd,fcntl.LOCK_UN)'),'fresh device check under original lock')
need(body.index('ACTUAL_HARNESS_CONTAINMENT')<body.index('root.mkdir'),'resolved both-phase containment before writes')
need(body.index('CACHE_NOT_FRESH')<body.index('root.mkdir'),'all twelve absent cache paths before writes')
need(body.count('subprocess.Popen(')==1,'one unchanged dispatcher spawn')
need("(root/'metrics-eval').mkdir" in body,'cold eval metrics parent precreated')
need('CACHE_MISSING' not in body,'no inherited warm-directory preflight')
need("hashlib.file_digest(stream,'sha256')" in body,'retained cold checkpoint rehash before stage')
result=subprocess.run([sys.executable,'-B',str(path),'--local-check'],capture_output=True,text=True)
need(result.returncode==0,'local-only check exit: '+result.stderr);o=json.loads(result.stdout)
need(o['files']==9 and not o['host_calls'] and not o['launch_authorized'] and o['future_actual_cold_r2_archive_required'],'disabled preparation only')
print(json.dumps({'status':'LOCAL_MECHANICAL_TRANSPORT_PASS_NOT_INDEPENDENT_REVIEW','checks':len(checks),'names':checks,'host_calls':False,'launch_authorized':False},indent=2))
