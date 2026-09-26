"""One read-only live milestone; never signals, stages or restarts a job."""
import hashlib
from pathlib import Path

here = Path(__file__).resolve().parent
original = here.parent / '20260926-final-sprint-p0' / 'observe_once.py'
raw = original.read_bytes()
if hashlib.sha256(raw).hexdigest() != 'acc5067b73fa106b6b5f2d40b7a1f9f422a930b66e73d246f6731ba53f51418d':
    raise RuntimeError('ORIGINAL_OBSERVER_DRIFT')
text = raw.decode()
replacements = {
    "out=HERE/(host+'-live.json');assert not out.exists()": "out=Path(" + repr(str(here)) + ")/(host+'-live.json');assert not out.exists()",
    "'NEURON_COMPETITION_R1_SEED'}": "'NEURON_COMPETITION_R1_SEED','NEURON_COMPETITION_R1_STEP_LOG'}",
    "train_source=root/'train/train.py'": """# Existing small observer/partial artifacts only, discovered in the exact
# active training sandbox. No recursive directory/cache scan or file writes.
observer_files={};observed_entries=[]
sandbox=root/'train'
if sandbox.is_dir():
 for p in sorted(sandbox.iterdir()):
  s=p.lstat();observed_entries.append({'name':p.name,'size':s.st_size,'mode':s.st_mode})
  if len(observed_entries)>40:raise RuntimeError('SANDBOX_ENTRY_BOUND')
  if len(observer_files)>=4:continue
  if not stat.S_ISREG(s.st_mode) or s.st_size>65536:continue
  if p.suffix=='.py' or not any(term in p.name.lower() for term in ('step','rehearsal','observ')):continue
  raw=p.read_bytes();assert len(raw)<=65536
  observer_files[p.name]={'bytes_on_host':s.st_size,'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode(errors='replace')}
files['existing-observer-artifacts']={'entries':observed_entries,'files':observer_files}
train_source=root/'train/train.py'"""
}
for old, new in replacements.items():
    if text.count(old) != 1: raise RuntimeError('OBSERVER_REBIND_NOT_UNIQUE')
    text = text.replace(old, new)
exec(compile(text, str(original), 'exec'), {'__file__': str(original), '__name__': '__main__'})
