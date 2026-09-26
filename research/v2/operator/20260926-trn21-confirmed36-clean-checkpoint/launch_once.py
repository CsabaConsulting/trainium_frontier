"""Record one exact scoped transport invocation; never automatically retry."""
import datetime,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
dest=HERE/'launch';dest.mkdir(exist_ok=False)
argv=[sys.executable,'-B',str(HERE/'stage-dispatch.py'),'a72617ec386ef4c98c87bcf893086cdd43e97487','research/v2/verification/20260926-confirmed36-clean-checkpoint-trn21-assignment.md']
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
r=subprocess.run(argv,capture_output=True,text=True,timeout=90)
receipt={'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'no_retry':True}
with (dest/'stage-transport.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
if r.returncode:print(json.dumps(receipt));raise SystemExit(r.returncode)
result=json.loads(r.stdout)
if result['rc']:print(json.dumps(result));raise SystemExit(result['rc'])
remote=json.loads(result['stdout'])
print(json.dumps({'launch':remote['launch'],'free_bytes':remote['preflight']['free_bytes'],'available_ram_bytes':remote['preflight']['available_ram_bytes'],'lock_device':remote['preflight']['lock_device'],'lock_inode':remote['preflight']['lock_inode'],'job_lines':remote['preflight']['job_lines'],'neuron_ls':remote['preflight']['neuron_ls']},indent=2))
