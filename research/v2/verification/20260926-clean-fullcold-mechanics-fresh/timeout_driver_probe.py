"""CPU reproduction of exact foreground-timeout/lock inheritance pattern."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

parent = Path('/proc/' + str(os.getppid()) + '/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
lock = Path(parent[2])
directory = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pinned_guard', '/home/poojith-devan/clean-fullcold-review-DrcoWi/idle_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
fd = guard.inherited_lock(lock, Path(__file__).resolve())['fd']
rc = subprocess.call(['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=0.1s','0.3s',
    '/usr/bin/python3','-I','-B',str(directory/'timeout_harness_probe.py'),str(fd),str(lock)],
    cwd=directory,env={'PATH':'/usr/bin:/bin'},pass_fds=(fd,))
sys.exit(rc)
