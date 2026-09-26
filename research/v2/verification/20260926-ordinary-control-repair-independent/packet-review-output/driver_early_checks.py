from pathlib import Path
import importlib.util,json
from unittest.mock import patch
from types import SimpleNamespace
out=Path(__file__).resolve().parent;src=out.parent/'packet-input/host_driver.py'
s=importlib.util.spec_from_file_location('host_driver',src);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
results={}
for scenario in ('wrong_root','wrong_host','stop','disk','support_hash'):
 records=[]
 def forbidden(*a,**k):raise AssertionError('CHILD_OR_MODULE_MUST_NOT_RUN')
 with patch.object(h,'record',lambda *x:records.append(x)),patch.object(h.subprocess,'call',forbidden),patch.object(h,'load_module',forbidden),patch.object(h,'__file__',str(src if scenario=='wrong_root' else h.ROOT/'host_driver.py')),patch.object(h.socket,'gethostname',lambda:'wrong' if scenario=='wrong_host' else h.EXPECTED_HOSTNAME),patch.object(Path,'glob',lambda *a:[]),patch.object(h.os.path,'lexists',lambda *a:scenario=='stop'),patch.object(h.os,'statvfs',lambda *a:SimpleNamespace(f_bavail=0 if scenario=='disk' else 20*1024**3,f_frsize=1)),patch.object(h,'sha',lambda *a:'0'*64):
  try:h.main()
  except RuntimeError as e:results[scenario]={'error':str(e),'records':records}
  else:raise RuntimeError('FALSE_ACCEPT:'+scenario)
 expected={'wrong_root':'WRONG_PACKET_ROOT','wrong_host':'WRONG_HOST','stop':'STOP:','disk':'DISK_HEADROOM','support_hash':'SUPPORT_SHA:'}[scenario]
 if not results[scenario]['error'].startswith(expected):raise RuntimeError('WRONG_REFUSAL:'+scenario)
(out/'driver-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
