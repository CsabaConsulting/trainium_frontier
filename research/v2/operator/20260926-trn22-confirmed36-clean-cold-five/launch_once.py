"""Record one exact scoped cold-five invocation, with no automatic retry."""
import datetime,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
dest=HERE/'launch';dest.mkdir(exist_ok=False)
argv=[sys.executable,'-B',str(HERE/'stage-dispatch.py'),'364b76a5242d9b0232f5a8fbe96f2f523f8543fa','research/v2/verification/20260926-confirmed36-clean-cold-trn22-assignment.md','2e7edfc815bbbd707348b5af8c40395b9fc25260']
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
r=subprocess.run(argv,capture_output=True,text=True,timeout=90)
receipt={'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'no_retry':True}
with (dest/'stage-transport.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
if r.returncode:print(json.dumps(receipt));raise SystemExit(r.returncode)
result=json.loads(r.stdout)
if result['rc']:print(json.dumps(result));raise SystemExit(result['rc'])
remote=json.loads(result['stdout'])
print(json.dumps({'launch':remote['launch'],'free_bytes':remote['preflight']['free_bytes'],'six_caches_absent_before_harness':remote['preflight']['six_caches_absent_before_harness'],'lock_device':remote['preflight']['lock_device'],'lock_inode':remote['preflight']['lock_inode'],'job_lines':remote['preflight']['job_lines'],'neuron_ls':remote['preflight']['neuron_ls']},indent=2))
