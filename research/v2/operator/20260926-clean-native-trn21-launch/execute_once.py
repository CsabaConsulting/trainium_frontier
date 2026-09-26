"""One authorized exact-byte stage and detached dispatch; no retries."""
import base64, datetime as dt, hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode=True
C=Path('/home/poojith-devan/Desktop/Trainium')
HERE=Path(__file__).resolve().parent
ASSIGN='research/v2/verification/20260926-clean-native-trn21-assignment.md'
ASSIGN_SHA='2c95c557e4396213b42c0eada88bc6b6f5010d30de380bb96f560d884299f082'
LABEL='fs-20260926-confirmed36-clean-native-consistency-trn21-r1'
ROOT='/home/ubuntu/trainium_confirmed36_clean_native_consistency_20260926_trn21_r1'
PROJECT='/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def load(name,path,pin):
    require(sha(path.read_bytes())==pin,'PIN:'+name)
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def save(name,obj):
    with (HERE/name).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')
def main():
    require(sha((C/ASSIGN).read_bytes())==ASSIGN_SHA,'ASSIGNMENT')
    require(subprocess.run(['git','show','e0202f57:'+ASSIGN],cwd=C,capture_output=True,check=True).stdout==(C/ASSIGN).read_bytes(),'COMMITTED_ASSIGNMENT')
    con=load('connector',C/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py','7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899')
    transdir=C/'research/v2/proposals/20260926-confirmed36-clean-native-trn21-transport-r1'
    tr=load('transport',transdir/'transport.py','51e37ab4ca72a1cb48682a50dcf1eec64be101182f0c41dede5680a923997a8e')
    packet=C/'research/v2/proposals/20260926-confirmed36-clean-native-trn21-packet-r1/packet'
    runtime=C/'research/v2/operator/20260926-trn21-clean-native-runtime-readonly/runtime-inspection.json'
    require(sha(runtime.read_bytes())=='8ca95247b8f6a0699a4c198c514a4d6ff458e7f322963180d448fdf2af56e9da','RUNTIME')
    require(sha((transdir/'TRANSPORT_BINDING.disabled.json').read_bytes())=='40a838ba331baf20de6e24f0f4f90cb15947f61e555f55ac8489e3f6dddb6f32','TRANSPORT_BINDING')
    sr=(C/'research/v2/state.json').read_bytes();state=json.loads(sr);host=state['hosts']['trn21'];obs=con.local_state()
    require(host.get('launch_allowed') is True and host.get('next_authorized_label')==LABEL and host.get('next_authorized_assignment')==ASSIGN,'CONTROLLER_SCOPE')
    require(not obs['local_stops_present'] and obs['breaker_tripped']['global_correctness'] is False and obs['breaker_tripped']['trn21'] is False,'LOCAL_STOP_BREAKER')
    controller={'utc':con.utc(),'label':LABEL,'state_sha256':sha(sr),'local_stops_present':[],'host_launch_allowed':True,'global_correctness_breaker':False,'trn21_breaker':False}
    review='430ea3be744949e68c4721dc3bfd94ac81ede989beb287068fb677d1e6be7f47'
    require(sha((C/'research/v2/verification/20260926-clean-native-transport-repair-fresh/REPORT.md').read_bytes())==review,'REPAIR_REPORT')
    release={'label':LABEL,'host':'trn21','stage_authorized':True,'launch_authorized':True,'submission_authorized':False,'operator_assignment_sha256':ASSIGN_SHA,'mechanics_review_sha256':'a9e23827bdc75f06239e75192c7e839b8679076b30da807021f1cd444cb7cc8e','packet_manifest_sha256':'905eb5fb73281c4058c52f6d16f4824dd1146929fa80b64a7b9d8d7d41d61f8f','runtime_inspection_sha256':sha(runtime.read_bytes()),'runtime_pins':json.loads(runtime.read_bytes())['remote']['runtime_pins'],'fresh_controller_receipt':controller,'transport':{'stage_authorized':True,'retrieve_authorized':True,'source_sha256':'51e37ab4ca72a1cb48682a50dcf1eec64be101182f0c41dede5680a923997a8e','review_sha256':review,'operator_assignment_sha256':ASSIGN_SHA,'connection_receipt_sha256':'76716989128b156b6d3c75a355bc9071059aed11410f9d62d6a0eb94e2dbe874','remote_command':['/usr/bin/python3','-I','-B','-']}}
    save('RELEASE.json',release)
    def remote(source,name,timeout):
        argv,display,identity,pin=con.load_transport('trn21');argv[-1]='exec /usr/bin/python3 -I -B -';display[-1]=argv[-1]
        require(sha(json.dumps(argv,separators=(',',':')).encode())=='29bddb10fbd5fa7b0b351ba4e6a18fd8247c1754250f16911da887576fa9101e','EXACT_CONNECTION_ARGV')
        require(con.SENSITIVE.search(source) is None,'PAYLOAD_SENSITIVE')
        started=con.utc()
        r=subprocess.run(argv,input=source,capture_output=True,timeout=timeout,check=False)
        require(con.SENSITIVE.search(r.stdout+r.stderr) is None,'OUTPUT_SENSITIVE')
        result={'started_utc':started,'ended_utc':con.utc(),'returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode(),'source_sha256':sha(source),'safe_argv':display,'identity':identity}
        save(name,result);return result
    clean_prefix=("import subprocess\np="+repr(PROJECT)+"\nr=subprocess.run(['/usr/bin/git','-C',p,'rev-parse','HEAD'],capture_output=True,text=True,timeout=15)\nif r.returncode or r.stdout.strip()!='e65a3097a5b00f6178188a935bb2acb9a5042c6c': raise RuntimeError('PROJECT_HEAD:'+r.stderr)\ns=subprocess.run(['/usr/bin/git','-C',p,'status','--porcelain','--untracked-files=all'],capture_output=True,text=True,timeout=15)\nif s.returncode or s.stdout.strip(): raise RuntimeError('PROJECT_NOT_CLEAN:'+s.stdout+s.stderr)\n").encode()
    wire=clean_prefix+tr.make_wire(packet,HERE/'RELEASE.json',runtime,'stage')
    stage=remote(wire,'stage.json',60)
    require(stage['returncode']==0,'STAGE_FAILED:'+stage['stdout']+stage['stderr'])
    require(json.loads(stage['stdout'])['status']=='STAGED_ONLY_NOT_LAUNCHED','STAGE_STATUS')
    print(json.dumps({'phase':'STAGED','utc':stage['ended_utc']}),flush=True)
    release_sha=sha((HERE/'RELEASE.json').read_bytes())
    dispatch=clean_prefix+f'''
import hashlib,json,os,time,datetime
from pathlib import Path
root=Path({ROOT!r})
release=root/'RELEASE.json'
if hashlib.sha256(release.read_bytes()).hexdigest()!={release_sha!r}: raise RuntimeError('RELEASE_DRIFT')
manifest=json.loads((root/'MANIFEST.json').read_bytes())
for name,pin in manifest.items():
 raw=(root/name).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=pin['sha256'] or len(raw)!=pin['bytes']: raise RuntimeError('STAGED_PIN:'+name)
control=json.loads(release.read_bytes())['fresh_controller_receipt']
if not 0<=time.time()-datetime.datetime.fromisoformat(control['utc']).timestamp()<=60: raise RuntimeError('STALE_BEFORE_DISPATCH')
with (root/'dispatch-launch.log').open('x') as log:
 p=subprocess.Popen(['/usr/bin/python3','-I','-B',str(root/'dispatch_driver.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 info={{'dispatch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'label':{LABEL!r},'command':['/usr/bin/python3','-I','-B',str(root/'dispatch_driver.py')],'science_not_yet_observed':True}}
with (root/'operator-dispatch.json').open('x') as f:json.dump(info,f)
print(json.dumps(info),flush=True)
'''.encode()
    dispatched=remote(dispatch,'dispatch.json',30)
    require(dispatched['returncode']==0,'DISPATCH_UNKNOWN_OR_FAILED:'+dispatched['stdout']+dispatched['stderr'])
    print(dispatched['stdout'],flush=True)
if __name__=='__main__':main()
