"""Existing final retrieval for exact current full jobs, all actual-ID evidence."""
import ast,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
assert sys.argv[1] in ('trn21-e1e2-control','trn22-batch-lr14')
old=HERE/'retrieve_successor.py';raw=old.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='32756b73f752cfa9eaf9dbd1545725a8b01e79c18f98e34534a1fe6f108a7029'
text=raw.decode().replace("{'trn21-batch-lr14':'trn21','trn22-e1e2-control':'trn22'}","{'trn21-e1e2-control':'trn21','trn22-batch-lr14':'trn22'}")
text=text.replace("20260926-final-sprint-successor-*/", "20260926-sprint-current-*/")
text=text.replace('+host+"-live.json', '+host+"-*-live.json')
extra="""
extra_path=HERE.parent/'20260926-sprint-control-k2-startup'/(host+'-live.json')
if extra_path.is_file():
 observed=json.loads(extra_path.read_text())['remote']
 if observed['label']==b['label']:
  paths.append(extra_path)
  for proc in observed['processes']:
   i=proc['identity_before'];pid=i['pid'];v={'pid':pid,'starttime':i['start_ticks'],'ppid':i['ppid'],'pgid':i['pgid'],'sid':i['sid'],'exists':True,'argv':proc['argv']}
   if pid in identity_map:assert identity_map[pid]['starttime']==v['starttime'],'OBSERVED_PID_REUSE'
   identity_map[pid]=v
for path in paths:
 observed=json.loads(path.read_text())['remote']
 assert observed['label']==b['label'],'WRONG_JOB_OBSERVATION'
 item=observed.get('files',{}).get('lifetime-events.jsonl',{})
 for line in item.get('text','').splitlines():
  event=json.loads(line);i=event.get('identity')
  if not i:continue
  pid=i['pid'];v={'pid':pid,'starttime':i['start_ticks'],'ppid':i['ppid'],'pgid':i['pgid'],'sid':i['sid'],'exists':True,'argv':[],'identity_source':'actual retained guardian lifetime-events'}
  if pid in identity_map:assert identity_map[pid]['starttime']==v['starttime'],'GUARDIAN_OBSERVED_PID_REUSE'
  else:identity_map[pid]=v
"""
needle='ast.parse(text)\nexec('
replace='replace_once("assert len(paths)>=2 and len(identity_map)>=9", '+repr(extra+'\nassert len(paths)>=2 and len(identity_map)>=9')+')\nast.parse(text)\nexec('
assert text.count(needle)==1;text=text.replace(needle,replace)
ast.parse(text)
exec(compile(text,str(old),'exec'),{'__file__':str(old),'__name__':'__main__'})
