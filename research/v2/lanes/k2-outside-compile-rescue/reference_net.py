"""Freeze a same-batch, same-host timing comparator before native rescue runs."""
import hashlib
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
PINS={'trn21':'30826ff24d91bf6fac6747b6dac618ceee543993e87575ad0effce1617bdc962',
      'trn22':'c05eaa57272719cb7ba3e2d24dae67f11ba6ca6e2de170ba7d0fbffe163bd91f'}
rows=[]
for host,expected in PINS.items():
    path=REPO/'research/v2/operator/20260926-final-sprint-successors'/(host+'-batch-lr14-completed/results-bundle.json')
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected
    bundle=json.loads(raw)
    for name,item in bundle['files'].items():
        assert hashlib.sha256(item['text'].encode()).hexdigest()==item['sha256'],name
    binding=json.loads(bundle['files']['NATIVE_BINDING.json']['text'])
    assert binding['source_sha256']=='8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee'
    command=bundle['files']['package/launch-command.txt']['text']
    assert '--total-batch-size 262144 --matrix-lr 0.021' in command
    rehearsal=json.loads(bundle['files']['train/rehearsal.json']['text'])
    obs={o['step']:o['observed_seconds'] for o in rehearsal['observations']}
    seconds=obs[200]-obs[9]
    rate=(200-9)*262144/seconds
    # Zero startup charge makes a conservative stronger baseline. The archive
    # is warm: never call its startup a cold measurement or submission evidence.
    upper_net=1800*rate
    losses=[]
    for line in bundle['files']['training.log']['text'].splitlines():
        match=re.match(r'step (\d+) \| loss ([\d.]+)',line)
        if match and int(match[1])<=200:losses.append({'step':int(match[1]),'smoothed_logged_loss':float(match[2])})
    rows.append({'host':host,'label':binding['label'],'archive':str(path.relative_to(REPO)),
                 'archive_sha256':expected,'window_boundaries':[9,200],
                 'measured_window_seconds':seconds,'measured_window_tokens':191*262144,
                 'steady_aggregate_tok_s':rate,'zero_charge_base_upper_net_tokens':upper_net,
                 'required_candidate_net_tokens':1.03*upper_net,'matched_launch_command':command.strip(),
                 'cold_base_startup_measured':False,'assumed_base_charge_for_conservative_screen':0,
                 'smoothed_loss_samples':losses})
receipt={'purpose':'Prospective K2 native201 comparator on current batch262144 base, not a native result',
         'candidate_formula':'max(0,1800-max(0,measured_cold_startup-900))*same_window_steady_tok_s',
         'qualification':'At least1.03 times same-host zero-charge upper-bound net; all correctness/budget checks stay',
         'old_479791343_reference':'Historical batch131072; not matched to this rescue batch262144',
         'rows':rows}
(HERE/'NATIVE_TIMING_REFERENCE.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ('host','steady_aggregate_tok_s','zero_charge_base_upper_net_tokens','required_candidate_net_tokens')} for r in rows]))
