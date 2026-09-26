"""Tiny local-only owned-process fault fixtures; never part of native command."""
import os
import signal
import sys
import time

mode = sys.argv[1]
if mode == 'output':
    os.write(1, b'x' * (3 * 1024**2))
    raise SystemExit(0)
if mode == 'loss':
    raise SystemExit(7)
if mode == 'double_fork':
    if os.fork():
        os._exit(0)
    os.setsid()
    if os.fork():
        os._exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print('ready', flush=True)
while True:
    time.sleep(0.1)
