"""Tiny CPU-only real-process peer fixture; never in the native launch argv."""
import os
from pathlib import Path
import socket
import sys
from common import identity, send, receive

root=Path(sys.argv[1])
sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
sock.settimeout(5)
sock.connect(str(root/'peers.sock'))
own=identity(os.getpid())
send(sock,dict(rank=int(os.environ['RANK']),world=4,pid=own['pid'],start=own['start'],pins={'fixture':'pins'}))
receive(sock)
sock.close()
