"""Exact six now-retired private caches only; preserve checkpoints and logs."""
import hashlib
from pathlib import Path
here=Path(__file__).resolve().parent
assignment=Path('/home/poojith-devan/Desktop/Trainium/research/v2/assignments/20260926-retire-superseded-recursive-cache.md').read_bytes()
root='fs-20260926-ct-clock36-recursive-cold-trn22-r1-packet'
assert root.encode() in assignment and b'INSUFFICIENT' in assignment
old=here.parent/'20260926-final-sprint-disk/cleanup_3e79_cache.py'
raw=old.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='3e15815a8dc0759835ae9e408bd2bd225c679582573ac985dc5bf46f1fde3005'
text=raw.decode()
text=text.replace('trn22-3e79-cache-inventory.json','trn22-4bfaec-cache-inventory.json')
text=text.replace('a96c8d8665981f1c7faaf7d5cc1f609f86a373a43810c011493e0a4df855fe68','808516f85b45a82e21d3cb68763dba2ea095cbb3429d1c796dd40254d2f1df83')
text=text.replace('fs-20260926-ct-arch3e79-cold-trn22-r1-packet',root)
text=text.replace("HERE/'status_once.py'","Path("+repr(str(old.parent/'status_once.py'))+")")
text=text.replace('trn22-3e79-cache-cleanup','trn22-4bfaec-cache-cleanup')
text=text.replace("'authorization_sha256':hashlib.sha256(auth).hexdigest()","'authorization_sha256':hashlib.sha256(auth).hexdigest(),'retirement_assignment_sha256':"+repr(hashlib.sha256(assignment).hexdigest()))
# Same descriptor-relative, no-symlink census/unlink primitive, exact new
# inventory and retirement assignment; no scientific verdict relabeling.
exec(compile(text,str(old),'exec'),{'__file__':__file__,'__name__':'__main__'})
