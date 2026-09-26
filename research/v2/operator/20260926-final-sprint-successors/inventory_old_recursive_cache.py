"""Read-only exact old Recursive cache census; never deletes or changes it."""
import hashlib
from pathlib import Path
here=Path(__file__).resolve().parent
old=here.parent/'20260926-final-sprint-disk/inventory_3e79_cache.py'
raw=old.read_bytes();assert hashlib.sha256(raw).hexdigest()=='300ffa8e180886279e7f4361ec4eed8e3297744a280d1d12a5f01067b365120b'
text=raw.decode()
before="labels=['fs-20260926-arch3e79-primed-trn22-r2-full','fs-20260926-arch3e79-lr012-trn22-r1-full']"
assert text.count(before)==1
text=text.replace(before,"labels=['fs-20260926-clock36-recursive-trn22-r1-full']")
text=text.replace('fs-20260926-ct-arch3e79-cold-trn22-r1-packet','fs-20260926-ct-clock36-recursive-cold-trn22-r1-packet')
text=text.replace("HERE/'status_once.py'","Path("+repr(str(old.parent/'status_once.py'))+")")
text=text.replace('trn22-3e79-cache-inventory.json','trn22-4bfaec-cache-inventory.json')
exec(compile(text,str(old),'exec'),{'__file__':__file__,'__name__':'__main__'})
