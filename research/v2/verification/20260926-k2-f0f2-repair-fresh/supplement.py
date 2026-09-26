import sys,json,hashlib,subprocess
from pathlib import Path
R=Path(__file__).parent
P=Path('/home/poojith-devan/k2-f0f2-repair-review-EHvK9J')
sys.path.insert(0,str(P))
import monitor,common
base=dict(at=1,memavailable=4*common.GIB,rss=20*common.GIB,free=[6.5*common.GIB],cache=6*common.GIB,artifacts=2*common.GIB,logs=128*common.MIB)
monitor.guards(base,0)
checks={}
for key,value in dict(at=3,memavailable=4*common.GIB-1,rss=20*common.GIB+1,free=[6.5*common.GIB-1],cache=6*common.GIB+1,artifacts=2*common.GIB+1,logs=128*common.MIB+1).items():
    try:monitor.guards(dict(base,**{key:value}),0)
    except RuntimeError as exc:checks[key]=str(exc)
    else:raise RuntimeError('guard accepted '+key)
proc=subprocess.run([sys.executable,'-B',str(P/'host_driver.py')],capture_output=True,text=True,timeout=3)
if proc.returncode==0 or 'DISABLED_NO_SEPARATE_HOST_RELEASE' not in proc.stderr:raise RuntimeError('binding gate')
old=json.loads((P/'baseline/SOURCE_PINS.json').read_text());new=common.source_pins()
changed=[k for k in old if old[k]!=new[k]]
if changed!=['coordinator.py','host_driver.py','monitor.py']:raise RuntimeError('science changed')
manifest=hashlib.sha256((P/'SHA256SUMS').read_bytes()).hexdigest()
for line in (P/'SHA256SUMS').read_text().splitlines():
    sha,name=line.split(None,1)
    if hashlib.sha256((P/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('packet modified '+name)
(R/'supplement-results.json').write_text(json.dumps(dict(guards=checks,disabled_stderr=proc.stderr,changed_existing_sources=changed,manifest=manifest,all_frozen_files_verified=True),indent=2)+'\n')
print('supplement passed')
