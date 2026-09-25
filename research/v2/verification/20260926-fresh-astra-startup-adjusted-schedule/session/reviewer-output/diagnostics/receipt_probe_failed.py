import json
from pathlib import Path
for file in ['/packet/raw/tiny-on1000.json','/packet/raw/attempt2-stub2/tiny-on1000.json']:
 d=json.loads(Path(file).read_text()); print(file)
 for r in d['ranks']:
  c=r['cache_observations'];print('rank',r['rank'],'losses',r['losses'],'checkpointfile',c['checkpoint_file_sha256'],'checkpoint present',c['checkpoint'] is not None)
  print('forward_noncache',{k:v for k,v in c['forwards'][0].items() if k!='caches'})
  print('schedules',[(x[-1]['lr'],x[-1]['weight_decay']) for x in r['schedule_observations']['optimizer_groups']])
  print('grads missing',[[k for k,v in s['gradients_before_clip'].items() if v is None] for s in c['steps']])
  print('state wrapper',c['steps'][0]['optimizer_state']['optimizer'].keys())
  print('state sample',str(c['steps'][0]['optimizer_state']['optimizer']['state'])[:350])
 print('observer',d['schedule_observer_sha256'])
