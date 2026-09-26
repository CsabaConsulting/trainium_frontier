"""Bounded sample of one actually observed post-SDK-lock-timeout compiler."""
import ast,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
raw=(HERE/'inspect_live_compiler.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='de524a7c6d2016981f260f1aa3a599513c1fe09546cce3d46290c3059f17315e'
text=raw.decode()
for a,b in [('trn21-first-steps-live.json','trn21-boundary-1025-live.json'),
 ('trn21-live-compiler-progress.json','trn21-compiler-progress-1025.json'),
 ('assert len(rows)==1',"assert len(rows)==3\nrows=[p for p in rows if p['identity_before']['pid']==3282560]\nassert len(rows)==1")]:
 assert text.count(a)==1
 text=text.replace(a,b)
ast.parse(text)
exec(compile(text,str(HERE/'inspect_live_compiler.py'),'exec'),{'__file__':__file__,'__name__':'__main__'})
