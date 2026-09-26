"""Existing bounded read-only observer, rebound to the direct P3 diagnostic."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE.parent/'20260926-final-sprint-p0/observe_once.py'
raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='acc5067b73fa106b6b5f2d40b7a1f9f422a930b66e73d246f6731ba53f51418d'
text=raw.decode()
text=text.replace("'train/rehearsal.json'):","'train/rehearsal.json','diagnostic-launch.json','diagnostic.log','diagnostic-exit-code.txt'):")
text=text.replace("train_source=root/'train/train.py'","train_source=root/'package/train.py'")
text=text.replace("with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(one,('trn21','trn22')))","one('trn22')")
ast.parse(text)
exec(compile(text,str(p),'exec'),{'__file__':str(HERE/'observe_once.py'),'__name__':'__main__'})
