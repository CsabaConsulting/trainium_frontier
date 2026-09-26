"""One bounded read-only CPU/log observation of the actually observed compiler."""
import ast,hashlib,importlib.util,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
prior=json.loads((HERE/'trn21-first-steps-live.json').read_bytes())['remote']
rows=[p for p in prior['processes'] if any('walrus_driver' in arg for arg in p['argv'])]
assert len(rows)==1
row=rows[0];argv=row['argv'];log=argv[argv.index('--logfile')+1]
assert log.startswith(prior['root']+'/cold-training-caches/debug/neuronx-cc/') and log.endswith('/log-neuron-cc.txt')
body=r'''
import datetime,hashlib,json,os,signal,stat,time
signal.alarm(20)
def process():
 raw=open('/proc/'+str(P['pid'])+'/stat').read();r=raw[raw.rfind(')')+2:].split()
 args=open('/proc/'+str(P['pid'])+'/cmdline','rb').read().rstrip(b'\0').decode().split('\0')
 return {'pid':P['pid'],'start_ticks':int(r[19]),'state':r[0],'ppid':int(r[1]),'user_ticks':int(r[11]),'system_ticks':int(r[12]),'argv':args}
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'path':P['log'],'prior_pid':P['pid'],'prior_start_ticks':P['start_ticks']}
try:
 a=process();assert a['start_ticks']==P['start_ticks'] and a['argv']==P['argv'],'COMPILER_IDENTITY_CHANGED'
 out['before']=a
 fd=os.open(P['log'],os.O_RDONLY|os.O_NOFOLLOW)
 try:
  s=os.fstat(fd);assert stat.S_ISREG(s.st_mode),'LOG_NOT_REGULAR'
  prefix=os.read(fd,min(s.st_size,16384));os.lseek(fd,max(0,s.st_size-65536),os.SEEK_SET);tail=os.read(fd,65536)
  z=os.fstat(fd)
  out['log']={'bytes_before':s.st_size,'bytes_after':z.st_size,'mtime_ns':z.st_mtime_ns,'inode':z.st_ino,'prefix_sha256':hashlib.sha256(prefix).hexdigest(),'tail_sha256':hashlib.sha256(tail).hexdigest(),'prefix_text':prefix.decode(errors='replace'),'tail_text':tail.decode(errors='replace'),'sample_is_partial':s.st_size>65536}
 finally:os.close(fd)
 start=time.monotonic();time.sleep(.25);b=process();out['after']=b;out['cpu_sample_seconds']=time.monotonic()-start
 assert b['start_ticks']==a['start_ticks'] and b['argv']==a['argv'],'COMPILER_IDENTITY_CHANGED_DURING_SAMPLE'
 out['cpu_ticks_delta']=(b['user_ticks']+b['system_ticks'])-(a['user_ticks']+a['system_ticks']);out['ticks_per_second']=os.sysconf('SC_CLK_TCK');out['status']='OBSERVED'
except Exception as exc:out['status']='UNAVAILABLE_OR_CHANGED';out['error']=repr(exc)
print(json.dumps(out))
'''
p={'pid':row['identity_before']['pid'],'start_ticks':row['identity_before']['start_ticks'],'argv':argv,'log':log}
wire=('P='+repr(p)+'\n'+body).encode();ast.parse(wire)
cp=REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
assert hashlib.sha256(cp.read_bytes()).hexdigest()=='7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899'
spec=importlib.util.spec_from_file_location('connector',cp);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert not m.SENSITIVE.search(wire)
out=HERE/'trn21-live-compiler-progress.json';assert not out.exists()
transport,*_=m.load_transport('trn21');transport[-1]='exec /usr/bin/python3 -I -B -S -'
r=subprocess.run(transport,input=wire,capture_output=True,timeout=40);assert not m.SENSITIVE.search(r.stdout+r.stderr)
d={'returncode':r.returncode,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None,'payload_sha256':hashlib.sha256(wire).hexdigest()}
with out.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
o=d['remote'];print(json.dumps({'returncode':r.returncode,'utc':o.get('utc') if o else None,'status':o.get('status') if o else None,'cpu_ticks_delta':o.get('cpu_ticks_delta') if o else None,'cpu_sample_seconds':o.get('cpu_sample_seconds') if o else None,'log_bytes':o.get('log',{}).get('bytes_after') if o else None,'error':o.get('error') if o else r.stderr.decode(),'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
