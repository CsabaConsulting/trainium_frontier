"""Real local CPU flock experiment around the exact supplied idle_guard code."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

directory = Path(__file__).resolve().parent
raw_parent = Path('/proc/' + str(os.getppid()) + '/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
lock = Path(raw_parent[2])
spec = importlib.util.spec_from_file_location('pinned_guard', '/home/poojith-devan/clean-fullcold-review-DrcoWi/idle_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
guard_fds=[]
codes=[]
children=[]
identity=None
for phase in ('train','eval'):
    identity=guard.inherited_lock(lock, Path(__file__).resolve())
    fd=identity['fd']
    guard_fds.append(fd)
    codes.append(subprocess.run(['/usr/bin/flock','-n',str(lock),'/usr/bin/true'], capture_output=True).returncode)
    child=subprocess.run(['/usr/bin/python3','-I','-B',str(directory/'phase_probe.py'),str(fd),str(lock)],
        pass_fds=(fd,),capture_output=True,text=True,timeout=5)
    if child.returncode:
        raise RuntimeError(child.stderr)
    result=json.loads(child.stdout)
    children.append(result['fd_matches'])
    codes.append(result['contender_rc'])
print(json.dumps({'guard_fds':guard_fds,'contender_codes':codes,'child_fds_ok':children,
                  'device':identity['device'],'inode':identity['inode']}))
