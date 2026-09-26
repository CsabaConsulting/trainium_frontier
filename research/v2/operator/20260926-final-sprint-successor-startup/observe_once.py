"""One unchanged bounded observer invocation per protected successor host."""
import ast
import hashlib
from pathlib import Path

here=Path(__file__).resolve().parent
original=here.parent/'20260926-final-sprint-p0/observe_once.py'
raw=original.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='acc5067b73fa106b6b5f2d40b7a1f9f422a930b66e73d246f6731ba53f51418d'
text=raw.decode()
def replace_once(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace_once("out=HERE/(host+'-live.json');assert not out.exists()",
             "out=Path("+repr(str(here))+")/(host+'-live.json');assert not out.exists()")
replace_once("b=json.loads((HERE/(host+'-bound-packet.json')).read_text())['binding']",
             "b=json.loads((HERE.parent/'20260926-final-sprint-successors'/({'trn21':'trn21-batch-lr14','trn22':'trn22-e1e2-control'}[host]+'-bound-packet.json')).read_text())['binding']")
replace_once("'train/rehearsal.json'):",
             "'train/rehearsal.json','fresh-public20m-launch.json','fresh-public20m.log','eval/rehearsal.json','eval-exit-code.txt','checkpoint-after-postflight.json'):")
ast.parse(text)
exec(compile(text,str(original),'exec'),{'__file__':str(original),'__name__':'__main__'})
