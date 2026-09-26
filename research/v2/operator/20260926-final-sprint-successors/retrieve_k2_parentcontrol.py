"""Existing all-outcome retrieval, exact K2 label and actual guardian identity union."""
import ast
import hashlib
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
raw=(HERE/'retrieve_successor.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='32756b73f752cfa9eaf9dbd1545725a8b01e79c18f98e34534a1fe6f108a7029'
assert sys.argv[1]=='trn22-on-parentcontrol'
assert sys.argv[2:] in (['--local-check'],['--execute'])
source=raw.decode()
old="host={'trn21-batch-lr14':'trn21','trn22-e1e2-control':'trn22'}[choice]"
assert source.count(old)==1
source=source.replace(old,"host={'trn22-on-parentcontrol':'trn22'}[choice]")
assert source.count('20260926-final-sprint-successor-*/')==1
source=source.replace('20260926-final-sprint-successor-*/','20260926-sprint-control-k2-*/')
extra=r'''
for path in paths:
 observed=json.loads(path.read_text())['remote']
 item=observed.get('files',{}).get('lifetime-events.jsonl',{})
 for line in item.get('text','').splitlines():
  event=json.loads(line);ident=event.get('identity')
  if not ident:continue
  pid=ident['pid'];value={'pid':pid,'starttime':ident['start_ticks'],'ppid':ident['ppid'],'pgid':ident['pgid'],'sid':ident['sid'],'exists':True,'argv':[],'identity_source':'actual retained guardian lifetime-events'}
  if pid in identity_map:
   assert identity_map[pid]['starttime']==value['starttime'],'GUARDIAN_OBSERVED_PID_REUSE'
  else:identity_map[pid]=value
'''
needle='ast.parse(text)\nexec('
replacement='replace_once("assert len(paths)>=2 and len(identity_map)>=9", '+repr(extra+'\nassert len(paths)>=2 and len(identity_map)>=9')+')\nast.parse(text)\nexec('
assert source.count(needle)==1
source=source.replace(needle,replacement)
ast.parse(source)
exec(compile(source,str(HERE/'retrieve_successor.py'),'exec'),{'__file__':str(HERE/'retrieve_successor.py'),'__name__':'__main__'})
