"""Short-lived CPU descendant; writes only its synthetic test receipt."""
import json
import os
from pathlib import Path
import sys
import time

lock=Path(sys.argv[2])
fd=int(sys.argv[1])
st=lock.stat()
try:
    fst=os.fstat(fd)
    owns_lock_fd=(fst.st_dev,fst.st_ino)==(st.st_dev,st.st_ino)
except OSError:
    owns_lock_fd=False
Path(str(lock)+'.child.json').write_text(json.dumps({'pid':os.getpid(),'owns_lock_fd':owns_lock_fd}))
time.sleep(1.5)
Path(str(lock)+'.child.done').write_text('CPU child finished\n')
