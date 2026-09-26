"""Authenticate failed CPU qualification; no host/model changes."""
import base64,hashlib,json,pathlib
repo=pathlib.Path(__file__).resolve().parents[4]
dir=repo/'research/v2/operator/20260926-trn22-clock36-recursive-cpu-r2/completed'
bundlepath=dir/'results-bundle.json'
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
require(hashlib.sha256(bundlepath.read_bytes()).hexdigest()=='d2298e9cafa63d98e9aa54c8b324e42a5ac345f20f4144b811112900ba2b2a08','BUNDLE_SHA')
b=json.loads(bundlepath.read_text());raw={}
for name,f in b['files'].items():
 v=f['text'].encode() if 'text' in f else base64.b64decode(f['base64'],validate=True)
 require(len(v)==f['bytes'] and hashlib.sha256(v).hexdigest()==f['sha256'],'ARTIFACT:'+name);raw[name]=v
require(len(raw)==92 and sum(map(len,raw.values()))==234503,'COUNTS')
j=lambda n:json.loads(raw[n])
p=repo/'research/v2/proposals/20260926-clock36-recursive-exact-cpu-r2-trn22'
plan=json.loads((p/'CPU_PLAN.disabled.json').read_text())
arms={}
for name in ['reference','default','off','on']:
 require(raw[name+'/request.json']==(p/name/'request.json').read_bytes(),'REQUEST:'+name)
 req=j(name+'/request.json');r=j(name+'/g2.json');t=j(name+'/terminal.json')
 require(t['exit_code']==0 and t['verdict']=='PASS' and r['verdict']=='PASS','G2_FAILED:'+name)
 require(r['train_py_sha256']==req['source_sha256'] and r['argv']==req['argv'] and r['plan_sha256']==plan['expected_plan_sha256'],'G2_BINDING:'+name)
 require(len(r['ranks'])==4 and all(x['n_params']==65 for x in r['ranks']),'RANK_PARAM_COUNT:'+name)
 arms[name]={'source_sha256':r['train_py_sha256'],'exit_code':t['exit_code'],'verdict':r['verdict'],'terminal_utc':t['utc'],'checkpoint':j(name+'/checkpoint-identity.json')}
for phase in ['standard-compare-default','standard-compare-off','strict-receipts']:
 require(j(phase+'-exit.json')['exit_code']==0,'COMPARISON:'+phase)
require(j('checkpoint-compare-exit.json')['exit_code']==1,'EXPECTED_HELPER_FAILURE')
require(raw['checkpoint-compare.json']==b'','EMPTY_HELPER_STDOUT')
err=raw['checkpoint-compare.stderr'].decode()
require('RuntimeError: NONFINITE_TENSOR' in err and 'model_state' in err and 'tensor_record' in err,'EXACT_FAILURE')
term=j('terminal.json');require(term['exit_code']==1 and term['error']=='COMPARISON_FAILED:checkpoint-compare','TERMINAL_FAILURE')
postpath=dir/'explicit-process-postflight.json'
require(hashlib.sha256(postpath.read_bytes()).hexdigest()=='60c3e23c84093d715cb4a7eed980689d7531c79fb4eb4629f75a588327a1a31a','POSTFLIGHT_SHA')
post=json.loads(postpath.read_text())
require(post['all_observed_pids_absent'] is True and post['neuron_device_idle'] is True,'QUIESCENCE')
require(post['related_rows']==[] and post['fuser_rc']==1 and post['ps_rc']==0 and post['lock_competitor_rc']==0,'POSTFLIGHT')
require(post['lock_device']==66305 and post['lock_inode']==832589,'ORIGINAL_LOCK')
print(json.dumps({'status':'FAILED_CPU_QUALIFICATION_AUTHENTICATED_NOT_PROMOTED','files':len(raw),'raw_bytes':sum(map(len,raw.values())),'arms':arms,'terminal':term,'exact_error':err,'postflight_utc':post['utc'],'observed_pids':post['observed_pids'],'quiescent':True,'reference_worker_identity_gap':'Reference arm completed before first live process capture; four rank receipts exist, unseen worker IDs not fabricated. Broad final process/device checks cover readiness, not retroactive PID reconstruction.','cold_authorized':False},indent=2))
