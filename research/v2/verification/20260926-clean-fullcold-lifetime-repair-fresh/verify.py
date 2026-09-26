import ast, hashlib, importlib.util, json, os, pathlib, signal, subprocess, sys, time
sys.dont_write_bytecode=True
P=pathlib.Path('/home/poojith-devan/clean-fullcold-lifetime-fresh-TOCGAp')
O=pathlib.Path(__file__).parent
R=pathlib.Path('research/v2/proposals/20260926-confirmed36-clean-cold-full-trn22-r2')
def check(v,m):
    if not v: raise RuntimeError(m)
def waitfor(p):
    end=time.monotonic()+5
    while not p.exists():
        check(time.monotonic()<end,'missing '+str(p));time.sleep(.01)
def occupied(lock):
    return subprocess.run(['/usr/bin/flock','-n',str(lock),'true']).returncode==1
def worker(case,d):
    spec=importlib.util.spec_from_file_location('review_guard',P/'new'/R/'lifetime_guard.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    lock=d/'lock';s=lock.stat()
    fds=[]
    for f in pathlib.Path('/proc/self/fd').iterdir():
        try:
            if os.fstat(int(f.name)).st_ino==s.st_ino:fds.append(int(f.name))
        except OSError:pass
    check(len(fds)==1,'fd count')
    g=mod.Guardian(fds[0],lock,(s.st_dev,s.st_ino),d/'events',grace=.15,kill_wait=.5)
    child="import os,time,pathlib,signal; os.setsid(); signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path('child').write_text(str(os.getpid())); time.sleep(2); pathlib.Path('natural').touch()"
    exact=next(n for n in ast.walk(ast.parse((P/'protected/research/scratch/submission_rehearsal.py').read_text())) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='Popen')
    statement=ast.unparse(exact)
    harness="import subprocess,sys,os,time,pathlib; command=[sys.executable,'-c',"+repr(child)+"]; sandbox=pathlib.Path('.'); env=dict(os.environ); "+statement+"; "
    harness+= "time.sleep(.2); os._exit(7)" if case in ('crash','old') else "process.wait()"
    command=[sys.executable,'-c',harness]
    if case=='old':
        subprocess.Popen(command,cwd=d,env=dict(os.environ),pass_fds=(fds[0],));time.sleep(.35);return
    if case=='timeout':command=['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=.1s','.3s',*command]
    try:
        if case=='helper':
            try: subprocess.run(command,cwd=d,capture_output=True,text=True,timeout=.3)
            except subprocess.TimeoutExpired:pass
            raise ValueError('helper-failed')
        if case=='enospc':
            real=g.receipt
            class Full:
                def write(self,*a):
                    if (d/'child').exists():raise OSError(28,'synthetic ENOSPC')
                    return real.write(*a)
                def flush(self):real.flush()
            g.receipt=Full()
        rc=g.run(command,cwd=d,env=dict(os.environ),out=subprocess.DEVNULL)
        (d/'result').write_text(str(rc))
        (d/'gap').touch();time.sleep(.2)
    except BaseException as e:(d/'error').write_text(repr(e))
    finally:
        def post():
            (d/'post').touch();time.sleep(.2)
            if case=='postfail':raise ValueError('post-failed')
        g.finish(post)
    (d/'done').touch()
if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],pathlib.Path(sys.argv[3]));sys.exit()
suffix='optimized' if sys.flags.optimize else 'normal'
results={}
pins={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file() and 'author-receipts' not in str(p)}
check(pins['science/workflow/candidates/confirmed-clock36-116188/train.py']=='e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43','science')
check(pins['protected/research/scratch/submission_rehearsal.py']=='7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b','harness')
old=ast.parse((P/'old'/R/'host_driver.py').read_text());new=ast.parse((P/'new'/R/'host_driver.py').read_text())
def constants(tree):return {n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('TRAIN_COMMAND','EVAL_COMMAND','TRAIN_ENV','EVAL_ENV')}
check(constants(old)==constants(new),'commands/environments changed')
b=json.loads((P/'new'/R/'NATIVE_BINDING.disabled.json').read_text())
for filename,key in [('host_driver.py','host_driver_sha256'),('lifetime_guard.py','lifetime_guard_sha256')]:
    if key in b:check(pins[str(pathlib.Path('new')/R/filename)]==b[key],key)
unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
try:
    for case in ('old','normal','crash','timeout','signal','helper','postfail','enospc'):
        d=O/(suffix+'-'+case);d.mkdir();(d/'lock').touch()
        with (d/'worker.log').open('w') as log:
            proc=subprocess.Popen(['/usr/bin/flock','-n',str(d/'lock'),sys.executable,*(['-O'] if sys.flags.optimize else []),__file__,'worker',case,str(d)],stdout=log,stderr=log)
            waitfor(d/'child');pid=int((d/'child').read_text())
            if case=='old':
                proc.wait(timeout=4);check(not occupied(d/'lock'),'old lock retained unexpectedly');check(pathlib.Path('/proc/'+str(pid)).exists(),'old child absent');waitfor(d/'natural');results[case]='reproduced: child alive after lock released';continue
            check(occupied(d/'lock'),'not excluded while child active')
            if case=='signal':
                events=[json.loads(x) for x in (d/'events').read_text().splitlines()];guardian_pid=events[0]['identity']['pid'];os.kill(guardian_pid,signal.SIGTERM)
            end=time.monotonic()+6
            checks=0
            while proc.poll() is None and time.monotonic()<end:
                if (d/'gap').exists() or (d/'post').exists():
                    check(occupied(d/'lock'),'released during gap/post');checks+=1
                text=(d/'events').read_text()
                if 'FAILED_LOCK_RETAINED' in text or (case=='enospc' and 'FAILED_LOCK_RETAINED' in (d/'worker.log').read_text()):break
                time.sleep(.02)
            parked=proc.poll() is None
            check(not pathlib.Path('/proc/'+str(pid)).exists(),'child not reaped '+case)
            check(unrelated.poll() is None,'unrelated killed')
            if case in ('postfail','enospc'):
                check(parked and occupied(d/'lock'),'failure did not retain lock')
                events=[json.loads(x) for x in (d/'events').read_text().splitlines()];gp=events[0]['identity']['pid']
                # Test-only disposal AFTER proved no child remains, not a production remedy.
                os.kill(gp,signal.SIGKILL);proc.wait(timeout=3)
            else:
                check(not parked,'unexpected hang');check(proc.returncode==0,'worker failed');check(not occupied(d/'lock'),'lock never released')
            results[case]={'pass':True,'parked':parked,'gap_post_exclusion_probes':checks}
finally:
    unrelated.terminate();unrelated.wait()
(O/(suffix+'-results.json')).write_text(json.dumps(results,indent=2)+'\n')
(O/'reviewed-pins.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
print(json.dumps(results,indent=2))
