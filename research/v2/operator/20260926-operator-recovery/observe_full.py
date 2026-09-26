"""Existing bounded process/source/lock observer for the two protected current jobs."""
import ast
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
assert len(sys.argv) == 3 and sys.argv[1] in ('trn21', 'trn22')
host, tag = sys.argv[1:]
assert re.fullmatch('[a-z0-9-]+', tag)
old = HERE.parent / '20260926-final-sprint-p0/observe_once.py'
raw = old.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'acc5067b73fa106b6b5f2d40b7a1f9f422a930b66e73d246f6731ba53f51418d'
text = raw.decode()
replacements = {
    "out=HERE/(host+'-live.json');assert not out.exists()": "out=HERE/(host+'-full-" + tag + ".json');assert not out.exists()",
    "b=json.loads((HERE/(host+'-bound-packet.json')).read_text())['binding']": "b=json.loads((HERE.parent/'20260926-final-sprint-successors'/({'trn21':'trn21-batch-recursive-stack','trn22':'trn22-batch-lr02625'}[host]+'-bound-packet.json')).read_text())['binding']",
    "'train/rehearsal.json'):": "'train/rehearsal.json','run-exit-code.txt','checkpoint.json','fresh-public20m-launch.json','fresh-public20m.log','eval/rehearsal.json','eval-exit-code.txt','eval-under-lock-idle.json','eval-preflight-fuser.json','postflight-under-lock-idle.json','postflight-fuser.json','checkpoint-after-postflight.json','independent-release-idle.json'):",
    "if s.st_size>65536:f.seek(-65536,2)": "if s.st_size>2*1024*1024:f.seek(-2*1024*1024,2)",
    "raw=f.read(65536)": "raw=f.read(2*1024*1024)",
    "'tail_only':s.st_size>65536": "'tail_only':s.st_size>2*1024*1024",
    "train_source=root/'train/train.py'": "assert sum(v.get('captured_bytes',0) for v in files.values())<=16*1024*1024,'CAPTURE_TOTAL_BOUND'\ntrain_source=root/'train/train.py'",
    "with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(one,('trn21','trn22')))": 'one(' + repr(host) + ')',
}
for before, after in replacements.items():
    assert text.count(before) == 1, before
    text = text.replace(before, after)
ast.parse(text)
exec(compile(text, str(old), 'exec'), {'__file__': __file__, '__name__': '__main__'})
