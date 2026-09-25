#!/usr/bin/env python3
"""Synchronous bounded runner; always preserves stdout, stderr and execution status."""
import argparse, datetime, hashlib, json, os, pathlib, subprocess, sys, time
p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
p.add_argument('--script',required=True)
p.add_argument('--optimized',action='store_true')
a=p.parse_args()
w=pathlib.Path('/work'); source=pathlib.Path(a.script)
log=w/'raw'/a.name
snapshot=w/'raw'/(a.name+'.script.py')
snapshot.write_bytes(source.read_bytes())
command=[sys.executable]+(['-O'] if a.optimized else [])+['-B',str(snapshot),'--output',str(log)]
env=dict(os.environ)
limits={x:'1' for x in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
limits['PYTHONDONTWRITEBYTECODE']='1'
env.update(limits)
started=time.monotonic(); utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
code=None; error=None
with (w/'raw'/(a.name+'.stdout')).open('x') as out,(w/'raw'/(a.name+'.stderr')).open('x') as err:
    try:
        result=subprocess.run(command,env=env,stdout=out,stderr=err,timeout=120,check=False)
        code=result.returncode
    except BaseException as e: error=type(e).__name__+': '+str(e)
receipt=dict(command=command,environment_overrides=limits,started_utc=utc,elapsed_seconds=time.monotonic()-started,timeout_seconds=120,exit_code=code,runner_error=error,script_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),sequential=True)
with (w/'raw'/(a.name+'.execution.json')).open('x') as f: json.dump(receipt,f,indent=2); f.write('\n')
print(json.dumps(receipt))
sys.exit(code if code is not None else 1)
