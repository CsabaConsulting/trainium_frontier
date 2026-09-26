"""Existing bounded observer for the protected control and batch.021 jobs."""
import ast,hashlib,re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
old=HERE.parent/'20260926-final-sprint-successor-handoff/observe_once.py'
raw=old.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='b56df7a4ca3c6e9a2db951f245ae41e8cb31d5668687c1d2160440014a6a1de2'
assert len(sys.argv)==3 and re.fullmatch('[a-z0-9-]+',sys.argv[2])
text=raw.decode()
before="{'trn21':'trn21-batch-lr14','trn22':'trn22-e1e2-control'}"
assert text.count(before)==1
text=text.replace(before,"{'trn21':'trn21-e1e2-control','trn22':'trn22-batch-lr14'}")
# One unique local receipt per meaningful boundary, no loop or host mutation.
old_expr='"out=Path("+repr(str(here))+")/(host+\'-live.json\');assert not out.exists()"'
new_expr='"out=Path("+repr(str(here))+")/(host+\'-'+sys.argv[2]+'-live.json\');assert not out.exists()"'
assert text.count(old_expr)==1;text=text.replace(old_expr,new_expr)
ast.parse(text)
exec(compile(text,str(old),'exec'),{'__file__':__file__,'__name__':'__main__'})
