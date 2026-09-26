"""Finite local observation schedule only: two known jobs, no launch or mutation."""
import concurrent.futures,datetime,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
START=datetime.datetime.fromisoformat('2026-09-26T10:47:30+00:00')
END=datetime.datetime.fromisoformat('2026-09-26T11:03:01+00:00')
def now():return datetime.datetime.now(datetime.timezone.utc)
def observe(host,tag):
 p=subprocess.run([sys.executable,'-B',str(HERE/'observe_light.py'),host,tag],capture_output=True,text=True,timeout=25)
 if p.returncode:
  return {'host':host,'LOCAL_OBSERVATION_ERROR':p.stderr,'stdout':p.stdout,'returncode':p.returncode,'not_a_host_terminal':True}
 return json.loads(p.stdout)
while now()<START:time.sleep(min(30,(START-now()).total_seconds()))
while now()<END:
 stamp=now();tag=stamp.strftime('%H%M%S')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
  futures=[ex.submit(observe,host,tag) for host in ('trn21','trn22')]
  for f in concurrent.futures.as_completed(futures):
   try:print(json.dumps(f.result()),flush=True)
   except BaseException as exc:print(json.dumps({'LOCAL_OBSERVATION_ERROR':repr(exc),'not_a_host_terminal':True}),flush=True)
 remaining=45-(now()-stamp).total_seconds()
 if remaining>0:time.sleep(min(remaining,max(0,(END-now()).total_seconds())))
print(json.dumps({'status':'BOUNDED_LOCAL_MONITOR_FINISHED','utc':now().isoformat(),'no_host_changes':True}),flush=True)
