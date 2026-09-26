"""Existing bounded read-only observer, rebound to the direct P3 diagnostic."""
import ast,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE.parent/'20260926-final-sprint-p0/observe_once.py'
raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='acc5067b73fa106b6b5f2d40b7a1f9f422a930b66e73d246f6731ba53f51418d'
text=raw.decode()
text=text.replace("'train/rehearsal.json'):","'train/rehearsal.json','component-diagnostic-launch.json','component-diagnostic.log','run-exit-code.txt','diagnostic-output/p3-rank0.json','diagnostic-output/p3-rank1.json','diagnostic-output/p3-rank2.json','diagnostic-output/p3-rank3.json'):")
if len(sys.argv)>1:
 assert len(sys.argv)==2 and sys.argv[1].startswith('trn22-progress-') and sys.argv[1].endswith('.json') and '/' not in sys.argv[1]
 text=text.replace("out=HERE/(host+'-live.json')", "out=HERE/"+repr(sys.argv[1]))
text=text.replace("train_source=root/'train/train.py'","train_source=root/'package/train.py'")
text=text.replace("with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(one,('trn21','trn22')))","one('trn22')")
ast.parse(text)
exec(compile(text,str(p),'exec'),{'__file__':str(HERE/'observe_once.py'),'__name__':'__main__'})
