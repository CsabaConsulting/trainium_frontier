"""One protected P0 train/eval handoff observation, no remote mutation."""
import hashlib
from pathlib import Path
here=Path(__file__).resolve().parent
original=here.parent/'20260926-final-sprint-p0/observe_once.py'
raw=original.read_bytes()
if hashlib.sha256(raw).hexdigest()!='acc5067b73fa106b6b5f2d40b7a1f9f422a930b66e73d246f6731ba53f51418d':raise RuntimeError('OBSERVER_DRIFT')
text=raw.decode()
replacements={
 "out=HERE/(host+'-live.json');assert not out.exists()":"out=Path("+repr(str(here))+")/(host+'-live.json');assert not out.exists()",
 "'train/rehearsal.json'):":"'train/rehearsal.json','run-exit-code.txt','checkpoint.json','fresh-public20m-launch.json','fresh-public20m.log','eval/rehearsal.json','eval-exit-code.txt','eval-under-lock-idle.json','eval-preflight-fuser.json','postflight-under-lock-idle.json','postflight-fuser.json','checkpoint-after-postflight.json','independent-release-idle.json'):",
 "if s.st_size>65536:f.seek(-65536,2)":"if s.st_size>2*1024*1024:f.seek(-2*1024*1024,2)",
 "raw=f.read(65536)":"raw=f.read(2*1024*1024)",
 "'tail_only':s.st_size>65536":"'tail_only':s.st_size>2*1024*1024",
 "train_source=root/'train/train.py'":"assert sum(v.get('captured_bytes',0) for v in files.values())<=16*1024*1024,'CAPTURE_TOTAL_BOUND'\ntrain_source=root/'train/train.py'"
}
for before,after in replacements.items():
 if text.count(before)!=1:raise RuntimeError('OBSERVATION_REBIND_NOT_UNIQUE:'+before)
 text=text.replace(before,after)
# Full bounded terminal rehearsal/closed training logs retain all timestamps;
# caches/checkpoints are neither traversed nor transferred. Existing tiny files
# may still be growing, so absence/partial state never triggers a restart.
exec(compile(text,str(original),'exec'),{'__file__':str(original),'__name__':'__main__'})
