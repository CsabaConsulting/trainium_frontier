"""Literal choice-only rebind of the all-outcome K2 receipt retriever."""
import ast,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
old=HERE/'retrieve_k2_parentcontrol.py';raw=old.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='b2e32b55f5b98a34f7a06fc813f8dfe24d160f5f6e78680a6b8f2a619cb19fa9'
text=raw.decode().replace('trn22-on-parentcontrol','trn22-indexrepair').replace('20260926-sprint-control-k2-*','20260926-k2-indexrepair-*')
ast.parse(text)
exec(compile(text,str(old),'exec'),{'__file__':str(old),'__name__':'__main__'})
