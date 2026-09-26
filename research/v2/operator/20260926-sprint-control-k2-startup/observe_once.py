"""Existing bounded observer, rebound to the two protected current jobs."""
import hashlib
from pathlib import Path

original=Path(__file__).resolve().parent.parent/'20260926-final-sprint-successor-handoff/observe_once.py'
raw=original.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='b56df7a4ca3c6e9a2db951f245ae41e8cb31d5668687c1d2160440014a6a1de2'
text=raw.decode()
old="{'trn21':'trn21-batch-lr14','trn22':'trn22-e1e2-control'}"
assert text.count(old)==1
text=text.replace(old,"{'trn21':'trn21-e1e2-control','trn22':'trn22-on-parentcontrol'}")
exec(compile(text,str(original),'exec'),{'__file__':__file__,'__name__':'__main__'})
