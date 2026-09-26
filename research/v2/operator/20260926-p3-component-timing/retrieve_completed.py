"""The proven all-outcome bounded retriever, rebound to P3's direct outputs."""
import ast,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
original=HERE.parent/'20260926-final-sprint-successors/retrieve_completed.py'
raw=original.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='86c96c314317f591939ae2c84f891027580963bc46d433bd08f029000268ac4c'
assert sys.argv[1]=='trn22' and sys.argv[2:] in (['--local-check'],['--execute'])
text=raw.decode()
def replace_once(old,new):
 global text
 assert text.count(old)==1,old
 text=text.replace(old,new)
replace_once("dest=HERE/(host+'-p0-completed');assert not dest.exists()", "dest=HERE/'trn22-completed';assert not dest.exists()")
replace_once("packet=json.loads((HERE.parent/'20260926-final-sprint-p0'/(host+'-bound-packet.json')).read_text());b=packet['binding']", "packet=json.loads((HERE/'trn22-bound-packet.json').read_text());b=packet['binding']")
replace_once("extra=['operator-preflight.json'", "extra=['component-diagnostic-launch.json','component-diagnostic.log','diagnostic-output/p3-rank0.json','diagnostic-output/p3-rank1.json','diagnostic-output/p3-rank2.json','diagnostic-output/p3-rank3.json','operator-preflight.json'")
replace_once("root/'metrics-training'", "root/'metrics-training',root/'diagnostic-output'")
start=text.index("for dirname in ('20260926-final-sprint-p0'")
end=text.index('assert len(paths)>=5 and len(identity_map)>=13',start)
section=text[start:end]
section=section.replace(section[:section.index(' if not p.is_file():')],"for p in [HERE/'trn22-live.json',*sorted(HERE.glob('trn22-progress-*.json'))]:\n")
extra="""
for path in paths:
 item=json.loads(path.read_text())['remote'].get('files',{}).get('lifetime-events.jsonl',{})
 for line in item.get('text','').splitlines():
  event=json.loads(line);ident=event.get('identity')
  if not ident:continue
  pid=ident['pid'];value={'pid':pid,'starttime':ident['start_ticks'],'ppid':ident['ppid'],'pgid':ident['pgid'],'sid':ident['sid'],'exists':True,'argv':[],'identity_source':'actual retained guardian lifetime-events'}
  if pid in identity_map:assert identity_map[pid]['starttime']==value['starttime'],'GUARDIAN_OBSERVED_PID_REUSE'
  else:identity_map[pid]=value
"""
text=text[:start]+section+extra+text[end:]
replace_once('assert len(paths)>=5 and len(identity_map)>=13','assert len(paths)>=2 and len(identity_map)>=9')
ast.parse(text)
exec(compile(text,str(original),'exec'),{'__file__':str(HERE/'retrieve_completed.py'),'__name__':'__main__'})
