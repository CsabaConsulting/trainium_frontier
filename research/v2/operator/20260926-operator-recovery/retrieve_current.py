"""Exact prior bounded terminal collector; current-job paths and observed IDs only."""
import ast
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
assert len(sys.argv) == 3 and sys.argv[1] in ('trn21', 'trn22')
assert sys.argv[2:] in (['--local-check'], ['--execute'])
old = HERE / 'proven_terminal_template.py'
raw = old.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '86c96c314317f591939ae2c84f891027580963bc46d433bd08f029000268ac4c'
text = raw.decode()


def replace_once(before, after):
    global text
    assert text.count(before) == 1, before
    text = text.replace(before, after)


replace_once("dest=HERE/(host+'-p0-completed');assert not dest.exists()", "dest=HERE/(host+'-current-completed');assert not dest.exists()")
replace_once("old=HERE.parent/'20260926-trn22-confirmed36-clean-cold-five-r2/retrieve_final.py'",
             "old=HERE/'proven_terminal_body.py'")
replace_once("packet=json.loads((HERE.parent/'20260926-final-sprint-p0'/(host+'-bound-packet.json')).read_text());b=packet['binding']",
             "packet=json.loads((HERE.parent/'20260926-final-sprint-successors'/({'trn21':'trn21-batch-recursive-stack','trn22':'trn22-batch-lr02625'}[host]+'-bound-packet.json')).read_text());b=packet['binding']")
begin = text.index("for dirname in ('20260926-final-sprint-p0'")
end = text.index("assert len(paths)>=5 and len(identity_map)>=13", begin)
replacement = '''first=HERE.parent/({'trn21':'20260926-batch-recursive-stack/trn21-first-live-live.json','trn22':'20260926-adopted-batch-handoff/trn22-first-live-live.json'}[host])
for path in [first,*sorted(HERE.glob(host+'-full-*.json'))]:
 if not path.is_file():continue
 parsed=json.loads(path.read_bytes());assert parsed['returncode']==0
 observed=parsed['remote'];assert observed['label']==b['label'],'WRONG_JOB_OBSERVATION'
 paths.append(path)
 for proc in observed['processes']:
  i=proc['identity_before'];pid=i['pid'];v={'pid':pid,'starttime':i['start_ticks'],'ppid':i['ppid'],'pgid':i['pgid'],'sid':i['sid'],'exists':True,'argv':proc['argv']}
  if pid in identity_map:assert identity_map[pid]['starttime']==v['starttime'],'OBSERVED_PID_REUSE'
  identity_map[pid]=v
 item=observed.get('files',{}).get('lifetime-events.jsonl',{})
 for line in item.get('text','').splitlines():
  event=json.loads(line);i=event.get('identity')
  if not i:continue
  pid=i['pid'];v={'pid':pid,'starttime':i['start_ticks'],'ppid':i['ppid'],'pgid':i['pgid'],'sid':i['sid'],'exists':True,'argv':[],'identity_source':'actual retained guardian lifetime-events'}
  if pid in identity_map:assert identity_map[pid]['starttime']==v['starttime'],'GUARDIAN_OBSERVED_PID_REUSE'
  else:identity_map[pid]=v
'''
text = text[:begin] + replacement + text[end:]
replace_once('assert len(paths)>=5 and len(identity_map)>=13', 'assert len(paths)>=2 and len(identity_map)>=9')
ast.parse(text)
exec(compile(text, str(old), 'exec'), {'__file__': __file__, '__name__': '__main__'})
