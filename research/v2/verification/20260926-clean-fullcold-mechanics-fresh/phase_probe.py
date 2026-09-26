"""No native code: prove inherited FD and competing-lock exclusion in child."""
import json
import os
from pathlib import Path
import subprocess
import sys

fd=int(sys.argv[1])
lock=Path(sys.argv[2])
fst=os.fstat(fd)
st=lock.stat()
rc=subprocess.run(['/usr/bin/flock','-n',str(lock),'/usr/bin/true'],capture_output=True).returncode
print(json.dumps({'fd_matches':(fst.st_dev,fst.st_ino)==(st.st_dev,st.st_ino),'contender_rc':rc}))
