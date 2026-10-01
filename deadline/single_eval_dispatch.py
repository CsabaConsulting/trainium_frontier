#!/usr/bin/env python3
"""Warm a guarded single-core scorer and copy verified checkpoints over the VPC.

A dedicated local SSH agent holds only the existing trn2 key. Agent forwarding
is scoped to the two user-owned source hosts. No SSH credentials are copied,
no authorized_keys are edited, and destination host keys come from existing
trusted SSH sessions. This controller never interrupts a training job.
"""
import concurrent.futures,datetime,fcntl,json,os,pathlib,shlex,subprocess,sys,time
from preserve import ROOT,SSH,HOSTS

REMOTE='/home/ubuntu/trainium_freshstart_deadline_singleeval_20261001'
TRAIN='/home/ubuntu/trainium_freshstart_deadline_heads_20261001'
PIN=(ROOT/'out/singleeval-source-commit.txt').read_text().strip()
CASES=[('trn21','trn25','172.31.41.184','headbias')]
READ=r'''
import pathlib,json,sys
r=pathlib.Path(sys.argv[1]);label=sys.argv[2]
f=r/'research/run-logs'/(label+'.log')
lines=f.read_text(errors='replace').splitlines()
a=[x.split('DEADLINE_TRAIN_COMPLETE ',1)[1] for x in lines if x.startswith('DEADLINE_TRAIN_COMPLETE ')]
t=r/'out'/label/'train.log'
print(json.dumps({'training':json.loads(a[-1]) if a else None,'within_budget':'over_budget:       False' in t.read_text(errors='replace')}))
'''
TRANSFER=r"""
import hashlib,json,pathlib,shlex,subprocess,sys
m=json.load(sys.stdin)
known=pathlib.Path(m['known_hosts_path']);known.parent.mkdir(parents=True,exist_ok=True)
known.write_text(m['known_hosts_line']+'\n');known.chmod(0o600)
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=yes','-o','HostKeyAlgorithms=ssh-ed25519','-o','UserKnownHostsFile='+str(known)]
p=pathlib.Path(m['training']['checkpoint'])
assert p.stat().st_size==m['training']['checkpoint_bytes']
assert hashlib.sha256(p.read_bytes()).hexdigest()==m['training']['checkpoint_sha256']
command=['rsync','-a','--partial','--timeout=40','-e',shlex.join(ssh),str(p),m['destination']+':'+m['target']+'.partial']
subprocess.run(command,check=True,timeout=120)
finish=r'''
import hashlib,json,pathlib,sys
m=json.load(sys.stdin);target=pathlib.Path(m['target']);partial=pathlib.Path(m['target']+'.partial')
assert partial.stat().st_size==m['training']['checkpoint_bytes']
assert hashlib.sha256(partial.read_bytes()).hexdigest()==m['training']['checkpoint_sha256']
partial.replace(target)
ready=target.with_suffix('.ready.json');temp=ready.with_suffix('.partial')
temp.write_text(json.dumps(m['training'],indent=2)+'\n');temp.replace(ready)
print(json.dumps({'checkpoint_ready':str(target),'bytes':target.stat().st_size,'verified':True}))
'''
subprocess.run([*ssh,m['destination'],'python3 -c '+shlex.quote(finish)],input=json.dumps(m),text=True,check=True,timeout=45)
"""

def emit(**data):
 print(json.dumps(dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data)),flush=True)

def ssh(host,command,**kw):
 return subprocess.run([*SSH,HOSTS[host],command],text=True,capture_output=True,timeout=kw.pop('timeout',45),**kw)

def run(case,env):
 source,host,private,variant=case;label='fs-20261001-deadline-'+variant+'-single-r7'
 p=subprocess.run(['rsync','-az','--timeout=35','-e',shlex.join(SSH),str(ROOT/'out/deadline-singleeval.bundle'),HOSTS[host]+':/home/ubuntu/deadline-singleeval.bundle'],capture_output=True,text=True,timeout=100)
 p.check_returncode()
 deploy="python3 -c "+shlex.quote("import pathlib,subprocess; r=pathlib.Path("+repr(REMOTE)+"); assert not r.exists(); subprocess.run(['git','clone','--quiet','/home/ubuntu/deadline-singleeval.bundle',str(r)],check=True); subprocess.run(['git','checkout','--quiet',"+repr(PIN)+"],cwd=r,check=True); (r/'out'/"+repr(label)+"/'incoming').mkdir(parents=True); print('SPARE_READY')")
 p=ssh(host,deploy,timeout=100);emit(host=host,deployed=p.stdout,stderr=p.stderr);p.check_returncode()
 request=dict(host=host,remote=REMOTE,expected_commit=PIN,label=label,purpose='Precompile unchanged inference on an idle spare; score size/hash-verified final checkpoint with native correctness gates',command=['timeout','--signal=TERM','--kill-after=30s','1600s','/home/ubuntu/workspace/native_venv/bin/python',REMOTE+'/deadline/evaluate_wait_lnc1.py','--variant',variant,'--checkpoint',REMOTE+'/out/'+label+'/incoming/final.pt','--label',label],environment=dict(PATH='/home/ubuntu/workspace/native_venv/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin',PYTHONPATH=REMOTE,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',NEURON_LOGICAL_NC_CONFIG='1',NEURON_COMPETITION_R1_CACHE_DIR='/home/ubuntu/.cache/neuron_competition_r1',NEURON_CC_FLAGS='--optlevel=3 --auto-cast matmult --auto-cast-type bf16'),fingerprint_paths=['deadline/evaluate_wait_lnc1.py','candidates/'+variant+'/train.py'])
 req=ROOT/'deadline/requests'/('request-singleeval-'+host+'.json');req.write_text(json.dumps(request,indent=2)+'\n')
 p=subprocess.run([sys.executable,str(ROOT/'research/fresh_start_launch.py'),str(req)],text=True,capture_output=True,timeout=120)
 emit(host=host,launch_returncode=p.returncode,stdout=p.stdout,stderr=p.stderr);p.check_returncode()
 p=ssh(host,'cat /etc/ssh/ssh_host_ed25519_key.pub');p.check_returncode()
 pub=p.stdout.strip().split();assert len(pub)>=2 and pub[0]=='ssh-ed25519'
 known_line=private+' '+pub[0]+' '+pub[1]
 known_path=TRAIN+'/out/singleeval-transfer-'+host+'/known_hosts'
 # Test authenticated VPC connectivity before waiting for the checkpoint.
 test='import pathlib,subprocess; p=pathlib.Path('+repr(known_path)+');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('+repr(known_line+'\n')+');subprocess.run('+repr(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=yes','-o','HostKeyAlgorithms=ssh-ed25519','-o','UserKnownHostsFile='+known_path,'ubuntu@'+private,'true'])+',check=True)'
 p=subprocess.run([*SSH,'-A',HOSTS[source],'python3 -c '+shlex.quote(test)],env=env,text=True,capture_output=True,timeout=30)
 emit(host=host,vpc_probe_returncode=p.returncode,stderr=p.stderr);p.check_returncode()
 trainlabel='fs-20261001-deadline-'+variant+'-r4'
 stop=datetime.datetime.fromisoformat('2026-10-01T06:58:00+00:00')
 while datetime.datetime.now(datetime.timezone.utc)<stop:
  p=ssh(source,'python3 -c '+shlex.quote(READ)+' '+shlex.quote(TRAIN)+' '+shlex.quote(trainlabel))
  if p.returncode:
   emit(host=host,source_read_error=p.stderr[-800:]);time.sleep(12);continue
  data=json.loads(p.stdout)
  if not data['training']:
   time.sleep(12);continue
  assert data['within_budget'],data
  transfer=dict(training=data['training'],known_hosts_path=known_path,known_hosts_line=known_line,destination='ubuntu@'+private,target=REMOTE+'/out/'+label+'/incoming/final.pt')
  p=subprocess.run([*SSH,'-A',HOSTS[source],'python3 -c '+shlex.quote(TRANSFER)],input=json.dumps(transfer),env=env,text=True,capture_output=True,timeout=180)
  emit(host=host,transfer_returncode=p.returncode,stdout=p.stdout,stderr=p.stderr);p.check_returncode()
  receipt=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_host=source,destination_host=host,training=data,transfer_stdout=p.stdout,transfer_returncode=p.returncode,private_key_copied=False,training_interrupted=False)
  (ROOT/'research/run-logs'/(label+'.checkpoint-transfer.json')).write_text(json.dumps(receipt,indent=2)+'\n')
  return
 raise TimeoutError('No final checkpoint by bounded transfer cutoff: '+variant)

def main():
 agent_path=ROOT/'out/lnc1.sock'
 assert len(str(agent_path).encode())<104,'Agent socket path too long'
 assert not agent_path.exists(),'Dedicated agent socket already exists'
 agent=subprocess.Popen(['ssh-agent','-D','-a',str(agent_path)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
 env=os.environ.copy();env['SSH_AUTH_SOCK']=str(agent_path)
 try:
  for _ in range(50):
   if agent_path.exists():break
   time.sleep(.1)
  p=subprocess.run(['ssh-add','/home/csaba/.ssh/trn2.pem'],env=env,text=True,capture_output=True,timeout=10)
  assert p.returncode==0,p.stderr
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   for f in concurrent.futures.as_completed([pool.submit(run,case,env) for case in CASES]):f.result()
 finally:
  agent.terminate();agent.wait(timeout=5)
  agent_path.unlink(missing_ok=True)

if __name__=='__main__':main()
