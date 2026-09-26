"""Existing read-only handoff observer for the repaired K2's unique label."""
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
old=HERE.parent/'20260926-final-sprint-successor-handoff/observe_once.py';raw=old.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='b56df7a4ca3c6e9a2db951f245ae41e8cb31d5668687c1d2160440014a6a1de2'
text=raw.decode();needle="{'trn21':'trn21-batch-lr14','trn22':'trn22-e1e2-control'}"
assert text.count(needle)==1;text=text.replace(needle,"{'trn21':'trn21-e1e2-control','trn22':'trn22-indexrepair'}")
exec(compile(text,str(old),'exec'),{'__file__':__file__,'__name__':'__main__'})
