from pathlib import Path
import ast, base64, copy, hashlib, importlib.util, json, subprocess, sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent.parent;IN=ROOT/'packet-input';OUT=ROOT/'packet-review-output'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def check(ok,msg):
 if not ok:raise RuntimeError(msg)
def load(name):
 s=importlib.util.spec_from_file_location(name,IN/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=json.loads((IN/'NATIVE_BINDING.disabled.json').read_text()); h=load('host_driver');g=load('idle_guard');v=load('validate_receipts');p=load('ordinary_subnormal_probe')
results={'scope':'LOCAL_SYNTHETIC_AND_STATIC_ONLY','checks':[],'negatives':{}}
def passed(name):results['checks'].append(name)
for name,pin in {**b['support_sha256'],'host_driver.py':b['driver_sha256']}.items():check(sha(IN/name)==pin,'PIN:'+name)
passed('all_six_source_support_pins')
for a,k in [('ROOT','root'),('LABEL','label'),('EXPECTED_HOSTNAME','expected_hostname'),('PINS','support_sha256'),('COMMAND','command'),('CHECKER_ARGS','checker_args'),('ENV','environment')]:check(str(getattr(h,a))==str(b[k]),'CONSTANT:'+a)
check(b['host']=='trn21' and b['original_lock']==dict(path=h.LOCK,device=66305,inode=596823,verification=b['original_lock']['verification']),'LOCK_BINDING')
check(h.COMMAND[:8]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','600s','/home/ubuntu/workspace/native_venv/bin/torchrun','--standalone','--nproc_per_node=4'],'BOUND')
check(h.COMMAND[8]==str(h.ROOT/'rank_entry.py') and h.COMMAND[9:]==h.CHECKER_ARGS,'ARGV')
check(b['outer_command']==['/usr/bin/flock','-n',h.LOCK,'/usr/bin/python3','-I','-B',str(h.ROOT/'host_driver.py')],'OUTER')
check(all(b[k] is False for k in ('stage_authorized','launch_authorized','host_reserved','operator_assigned','native_qualification','submission_authorized')),'DISABLED_FLAGS')
passed('exact_root_host_label_argv_lifetime_flock_and_600_plus_60_bound')
# Authenticate all diff-added files against supplied review copies.
files={};current=None
for line in (IN/'mechanics-full-context.diff').read_text().splitlines(keepends=True):
 if line.startswith('+++ b/'):current=line.strip().rsplit('/',1)[1];files[current]=[]
 elif current and line.startswith('+') and not line.startswith('+++'):files[current].append(line[1:])
for name,lines in files.items():
 if (IN/name).exists():check(''.join(lines)==(IN/name).read_text(),'DIFF:'+name)
passed('full_context_diff_matches_supplied_sources')
src=(IN/'host_driver.py').read_text()
check(src.index('lock_fd = guard.require_idle_under_inherited_lock')<src.index('child_returncode = subprocess.call')<src.index('completion = validator.validate'),'GUARD_CHILD_AGGREGATE_ORDER')
check('pass_fds=(lock_fd,)' in src and 'if child_returncode != 0:' in src,'LOCK_INHERITANCE_CHILD_GATE')
lock_src=(IN/'idle_guard.py').read_text()
for fragment in ['actual != expected','len(matches) != 1',"'FLOCK' in line and 'WRITE' in line",'fcntl.LOCK_EX | fcntl.LOCK_NB']:
 check(fragment in lock_src,'LOCK_STATIC:'+fragment)
passed('static_inherited_lock_authentication_and_guard_before_child_before_aggregation')
def reject(name,fn):
 try:fn()
 except Exception as e:results['negatives'][name]=type(e).__name__+': '+str(e)
 else:raise RuntimeError('FALSE_ACCEPT:'+name)
device=dict(neuron_device=0,instance_type='trn2.3xlarge',nc_count=4,logical_neuroncore_config=2,neuroncore_ids=[0,1,2,3],neuron_processes=[])
def idle_case(kind):
 calls=[];dev=copy.deepcopy(device);ps='101 100 python /usr/bin/python3 host_driver.py\n100 1 flock /usr/bin/flock\n1 0 init /sbin/init\n';rc=0;err=''
 if kind=='busy_ps':ps+='102 1 python /usr/bin/python3 train.py\n'
 if kind=='busy_device':dev['neuron_processes']=[dict(pid=102)]
 if kind=='bad_device':dev['nc_count']=8
 if kind=='missing_own':ps='1 0 init /sbin/init\n'
 if kind=='bad_ps':ps='invalid\n'
 def run(command,**kw):
  calls.append(command);check(kw['timeout']==15 and kw['env']==g.CHECK_ENV,'IDLE_BOUND')
  if kind=='timeout':raise subprocess.TimeoutExpired(command,15)
  if command==g.PS:return SimpleNamespace(returncode=1 if kind=='ps_error' else 0,stdout=ps,stderr='warning' if kind=='ps_stderr' else '')
  check(command==g.NEURON_LS,'UNEXPECTED_EXEC')
  return SimpleNamespace(returncode=1 if kind=='device_error' else 0,stdout=json.dumps([dev]),stderr='')
 evidence={}
 try:g.check_idle(run,101,100,evidence)
 finally:
  if kind in ('ok','busy_ps','busy_device','bad_device'):check(calls==[g.PS,g.NEURON_LS],'BOTH_FRESH_CHECKS')
  (OUT/('idle-'+kind+'.json')).write_text(json.dumps({'calls':calls,'evidence':evidence},indent=2))
idle_case('ok');passed('synthetic_idle_ps_and_device')
for kind in ('busy_ps','busy_device','bad_device','missing_own','bad_ps','timeout','ps_error','ps_stderr','device_error'):reject('idle_'+kind,lambda:idle_case(kind))
# Use actual files under our review directory; no native or host commands.
testroot=OUT/'synthetic-packet';testroot.mkdir();(testroot/'receipts').mkdir()
(testroot/'ordinary_subnormal_probe.py').write_bytes((IN/'ordinary_subnormal_probe.py').read_bytes())
base=json.loads((ROOT/'repair-review-output/baseline-cpu-receipt.json').read_text())
records={}
for rank in range(4):
 r=copy.deepcopy(base);r.update(rank=rank,pid=1000+rank,argv=b['checker_args'],checker_sha256=b['support_sha256']['ordinary_subnormal_probe.py'],runtime=dict(torch='2.11.0+cpu',nki='0.5.0+28631259367.ga768afa6'),seed=42,native_qualification=False,environment={k:b['environment'][k] for k in ('NEURON_CC_FLAGS','NEURON_LOGICAL_NC_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
 records[f'receipts/rank{rank}.json']=r
 records[f'rank-environment-r{rank}.json']=dict(rank=rank,local_rank=rank,world_size=4,pid=1000+rank,checker_argv=b['checker_args'],checker_sha256=b['support_sha256']['ordinary_subnormal_probe.py'],environment=b['environment'])
 records[f'rank-wrapper-exit-r{rank}.json']=dict(rank=rank,returncode=0,error=None)
def run(data):
 for rel,obj in data.items():(testroot/rel).write_text(json.dumps(obj))
 return v.validate(testroot,b['label'],b['environment'],b['support_sha256'],b['checker_args'])
results['baseline']=run(records);passed('four_exact_ranks_synthetic_native_metadata_only')
def mismatch(d):
 r=d['receipts/rank3.json'];z=p.blob(bytes(131072),'torch.float32',(256,128),r['blobs']);r['weight']=z
 for entry in r['routes']:
  for key in ('expected_inputs','before_forward','before_backward','after_backward'):entry[key]['weight']=z
  entry['expected_output']=z;entry['output']=p.comparison(z,z,r['blobs'])
 p.validate_receipt(r,3)
mutations={
 'cross_rank_signature':mismatch,
 'wrong_rank':lambda d:d['receipts/rank0.json'].update(rank=1),
 'error_receipt':lambda d:d['receipts/rank0.json'].update(error='injected'),
 'wrong_leaf_pin':lambda d:d['receipts/rank0.json'].update(checker_sha256='0'*64),
 'wrong_argv':lambda d:d['receipts/rank0.json'].update(argv=[]),
 'wrong_runtime':lambda d:d['receipts/rank0.json']['runtime'].update(torch='wrong'),
 'wrong_label':lambda d:d['receipts/rank0.json'].update(label='wrong'),
 'wrapper_error':lambda d:d['rank-wrapper-exit-r0.json'].update(error='injected'),
 'wrapper_boolean_exit':lambda d:d['rank-wrapper-exit-r0.json'].update(returncode=False),
 'wrapper_env':lambda d:d['rank-environment-r0.json'].update(environment={}),
 'duplicate_pid':lambda d:(d['receipts/rank1.json'].update(pid=1000),d['rank-environment-r1.json'].update(pid=1000)),
 'missing_raw':lambda d:d['receipts/rank0.json']['blobs'].pop(d['receipts/rank0.json']['weight']['sha256']),
 'missing_route':lambda d:d['receipts/rank0.json']['routes'].pop(),
}
for name,mutate in mutations.items():
 d=copy.deepcopy(records);mutate(d);reject(name,lambda:run(d))
run(records)
(testroot/'receipts/rank3.json').rename(testroot/'saved-rank3.json')
reject('missing_rank_file',lambda:v.validate(testroot,b['label'],b['environment'],b['support_sha256'],b['checker_args']))
(testroot/'saved-rank3.json').rename(testroot/'receipts/rank3.json')
(testroot/'receipts/rank4.json').write_text('{}')
reject('extra_rank_file',lambda:v.validate(testroot,b['label'],b['environment'],b['support_sha256'],b['checker_args']))
(testroot/'receipts/rank4.json').rename(testroot/'extra-rank4.json')
(testroot/'receipts/rank0.json').write_text('{invalid')
reject('malformed_json',lambda:v.validate(testroot,b['label'],b['environment'],b['support_sha256'],b['checker_args']))
run(records)
disabled=subprocess.run([sys.executable,'-I','-B',str(IN/'ordinary_subnormal_probe.py')],capture_output=True,text=True)
check(disabled.returncode==0 and json.loads(disabled.stdout)==dict(status='DISABLED',native_imported=False),'DEFAULT_DISABLED')
results['disabled']={'stdout':disabled.stdout,'stderr':disabled.stderr,'returncode':disabled.returncode}
results['status']='PASS_SCOPED_DISABLED_MECHANICS';(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
