"""Existing bounded final retrieval, bound to exact two protected successor jobs."""
import ast
import hashlib
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
choice=sys.argv[1]
host={'trn21-batch-lr14':'trn21','trn22-e1e2-control':'trn22'}[choice]
original=HERE/'retrieve_completed.py'
raw=original.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='86c96c314317f591939ae2c84f891027580963bc46d433bd08f029000268ac4c'
text=raw.decode()
def replace_once(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace_once("host=sys.argv[1];assert host in ('trn21','trn22')", "host="+repr(host))
replace_once("dest=HERE/(host+'-p0-completed');assert not dest.exists()", "dest=HERE/"+repr(choice+'-completed')+";assert not dest.exists()")
replace_once("packet=json.loads((HERE.parent/'20260926-final-sprint-p0'/(host+'-bound-packet.json')).read_text());b=packet['binding']",
             "packet=json.loads((HERE/"+repr(choice+'-bound-packet.json')+").read_text());b=packet['binding']")
start=text.index("for dirname in ('20260926-final-sprint-p0'")
end=text.index("assert len(paths)>=5 and len(identity_map)>=13",start)
section=text[start:end]
section=section.replace(section[:section.index(' if not p.is_file():')],
    "for p in [HERE/"+repr(choice+'-live.json')+"] + sorted(HERE.parent.glob('20260926-final-sprint-successor-*/"+host+"-live.json')):\n")
text=text[:start]+section+text[end:]
replace_once('assert len(paths)>=5 and len(identity_map)>=13','assert len(paths)>=2 and len(identity_map)>=9')
replace_once("(HERE/(host+'-retrieval-error.json')).open('x')", "(HERE/"+repr(choice+'-retrieval-error.json')+").open('x')")
# The original actual-PID/starttime/PPID/PGID/SID union and remote postflight
# retain their complete behavior; the names above only select these successors.
ast.parse(text)
exec(compile(text,str(original),'exec'),{'__file__':str(original),'__name__':'__main__'})
