import importlib.util, json, hashlib, ast, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
torch.set_num_threads(1)
s=importlib.util.spec_from_file_location('under_review',ROOT/'compare_checkpoints.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
results=[]
class Fixture(torch.nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.p=torch.nn.Parameter(torch.ones(1));self.register_buffer('b',torch.zeros(1),persistent=False);self.calls=0;self.scratch=torch.zeros(1,8,3)
 def forward(self,x):
  self.calls+=1
  y=x.float().unsqueeze(-1).expand(-1,-1,3).clone()
  if self.kind=='future': y+=x[:,-1:].float().unsqueeze(-1)
  if self.kind=='parameter': self.p.add_(1)
  if self.kind=='buffer': self.b.add_(1)
  if self.kind=='input': x.add_(1)
  if self.kind=='alias': self.scratch.fill_(self.calls);y=self.scratch
  if self.kind=='signedzero': y=torch.zeros_like(y) if self.calls==1 else -torch.zeros_like(y)
  if self.kind=='nonfinite': y.fill_(float('nan'))
  return y
expected={'reference':None,'future':'SYNTHETIC_PREFIX_CAUSALITY','parameter':'MODEL_STATE_CHANGED','buffer':'MODEL_STATE_CHANGED','input':'INPUT_CHANGED','alias':'REPEATED_LOGITS_BYTES','signedzero':'REPEATED_LOGITS_BYTES','nonfinite':'NONFINITE_TENSOR'}
for kind,want in expected.items():
 try:
  r=m.verify_trained_model(Fixture(kind).eval(),torch.arange(1,9).reshape(1,8),3);got=None
 except RuntimeError as e: got=str(e)
 results.append({'test':kind,'expected':want,'observed':got,'passed':got==want})
p=json.loads((ROOT/'CPU_PLAN.disabled.json').read_text())
def check(name,value):results.append({'test':name,'passed':bool(value)})
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
check('disabled',p['launch_authorized'] is False and p['native_launch_authorized'] is False and p['host_reserved'] is False and p['results'] is None)
for probe in p['probes']:
 name=probe['name']; f=ROOT/name/'request.json';q=json.loads(f.read_text());cmd=q['command'];stage=p['stage_root']+'/'+name
 check(name+'_bindings',sha(f)==probe['request_sha256'] and cmd==probe['command'] and hashlib.sha256(json.dumps(cmd,separators=(',',':')).encode()).hexdigest()==probe['command_sha256'] and q['cwd']==stage and cmd[cmd.index('--train-py')+1]==stage+'/train.py' and q['source_sha256']==p['sources'][name]['sha256']==m.EXPECTED_SOURCES[name] and q['g2_tool_sha256']==p['sources']['g2']['sha256'] and q['prepare_sha256']==p['sources']['prepare']['sha256'] and cmd[cmd.index('--out')+1]==q['receipt']==stage+'/g2.json' and q['temporary_root']==stage+'/tmp')
check('driver_pin',sha(ROOT/'host_driver.py')==p['driver_sha256'])
for f,h in p['support_sha256'].items():check('support_'+f,sha(ROOT/f)==h)
for field,h in [('expected_outer_command','outer_command_sha256'),('detached_dispatch_command','detached_command_sha256')]:check(field,hashlib.sha256(json.dumps(p[field],separators=(',',':')).encode()).hexdigest()==p[h])
manifest=json.loads((ROOT/'MANIFEST.json').read_text())
for item in manifest['files']:
 if item['path']=='README.md':
  results.append({'test':'manifest_README.md','passed':None,'excluded':'Parent deliberately withheld narrative notes; Supervisor must authenticate separately.'});continue
 f=ROOT/item['path'];check('manifest_'+item['path'],f.exists() and sha(f)==item['sha256'])
for f,key in [('idle_guard.py','idle_guard'),('dispatch_driver.py','dispatch'),('preflight_dryrun.py','g2')]:check('dependency_'+f,sha(ROOT/f)==p['sources'][key]['sha256'])
diff=(ROOT/'full-context.diff').read_text();section=None;parts={}
for line in diff.splitlines(keepends=True):
 if line.startswith('+++ b/'):section=line.strip().split('/')[-1];parts[section]=[]
 elif section and line.startswith('+') and not line.startswith('+++'):parts[section].append(line[1:])
for name in ('compare_checkpoints.py','host_driver.py','validate_receipts.py'):
 if name in parts:check('diff_matches_'+name,''.join(parts[name])==(ROOT/name).read_text())
print(json.dumps({'torch':torch.__version__,'threads':torch.get_num_threads(),'results':results},indent=2))
sys.exit(0 if all(r['passed'] is not False for r in results) else 1)
