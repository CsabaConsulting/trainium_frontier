"""Authenticate only the already captured live evidence; no host calls."""
import hashlib
import json
import math
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
PREP=HERE.parent/'20260926-final-sprint-successors'
def sha(x):return hashlib.sha256(x).hexdigest()
expected={'trn21':('trn21-batch-lr14',3231387,list(range(3231435,3231439)),596823),
          'trn22':('trn22-e1e2-control',3345115,list(range(3345162,3345166)),832589)}
rows=[]
for host,(choice,guardian,workers,inode) in expected.items():
    raw=(HERE/(host+'-live.json')).read_bytes();receipt=json.loads(raw)
    assert receipt['returncode']==0 and not receipt['stderr']
    current=receipt['remote']
    initial=json.loads((PREP/(choice+'-live.json')).read_text())['remote']
    binding=json.loads((PREP/(choice+'-bound-packet.json')).read_text())['binding']
    before={p['identity_before']['pid']:p for p in initial['processes']}
    after={p['identity_before']['pid']:p for p in current['processes']}
    assert current['label']==binding['label']
    assert current['actual_train_source']['sha256']==binding['source_sha256']
    assert current['original_lock']=={'device':66305,'inode':inode,'held':True}
    assert current['neuron_ls']['returncode']==0
    for pid in [guardian]+workers:
        assert pid in after and pid in before
        for key in ('pid','ppid','pgid','sid','start_ticks'):
            assert after[pid]['identity_before'][key]==before[pid]['identity_before'][key]
            assert after[pid]['identity_before'][key]==after[pid]['identity_after'][key]
        assert after[pid]['identity_after']['state']!='Z'
    assert after[guardian]['original_lock_fds']
    for rank,pid in enumerate(workers):
        env=after[pid]['environment_whitelist']
        assert env['RANK']==env['LOCAL_RANK']==str(rank) and env['WORLD_SIZE']=='4'
        for key in ('NEURON_LOGICAL_NC_CONFIG','NEURON_CC_FLAGS','NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS','NEURON_COMPETITION_R1_SEED','NEURON_COMPILE_CACHE_URL','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'):
            assert env[key]==binding['train_environment'][key]
    for name,entry in current['files'].items():
        if 'text' in entry:assert sha(entry['text'].encode())==entry['sha256'],name
    assert current['files']['terminal.json'].get('absent') is True
    log=current['files']['training.log']['text']
    fatal=[line for line in log.splitlines() if any(term in line for term in ('Traceback (most recent call last)','ChildFailedError','RuntimeError:','TRAINING_ERROR','AssertionError:'))]
    step_lines=[line for line in log.splitlines() if line.startswith('step ')]
    assert len(step_lines)>=5
    parsed=[]
    for line in step_lines:
        m=re.search(r'^step (\d+) \| loss ([^ |]+).*\| dt ([0-9.]+)s \| tok/s ([0-9,]+).*\| remaining_time ([0-9.]+)s',line)
        assert m,line
        loss=float(m[2]);assert math.isfinite(loss)
        parsed.append({'step':int(m[1]),'loss':loss,'logged_dt_seconds':float(m[3]),'tokens_per_second':int(m[4].replace(',','')),'source_remaining_seconds':float(m[5])})
    assert [x['step'] for x in parsed[:5]]==list(range(5))
    compiler=[p['identity_before'] for p in current['processes'] if any('neuronx-cc' in x or 'walrus_driver' in x for x in p['argv'])]
    rows.append({'host':host,'label':binding['label'],'observed_utc':current['utc'],'phase':'TRAINING',
        'guardian_pid':guardian,'worker_pids':workers,'source_sha256':binding['source_sha256'],
        'original_lock':current['original_lock'],'first_five_logged_steps':parsed[:5],
        'latest_logged_step':parsed[-1],'all_captured_logged_losses_finite':True,'fatal_lines':fatal,
        'active_compiler_identities_observed':compiler,'free_bytes':current['free_bytes'],
        'actual_uncapped_fifth_boundary':None,'exact_startup_note':'First five logged, but full observer timestamps exist only after natural terminal. Logged dt is not process-start wall time.',
        'cache_provenance':'Retained research namespace, neither cold nor hit proven.',
        'receipt_sha256':sha(raw)})
assert rows[1]['latest_logged_step']['step']==20
out={'status':'ACTUAL_LIVE_SOURCE_RANK_LOCK_ENV_AND_CAPTURE_HASHES_AUTHENTICATED',
     'hosts':rows,'k2_window_precommit':'d3da2359, before captured parent step20 and before any t44 timing observation',
     'no_host_mutations':True,'no_native_or_quality_adoption_claim':True,
     'next_boundary':'Natural training-to-eval around08:45UTC trn21 and08:50UTC trn22; estimates, not deadlines.'}
with (HERE/'AUTHENTICATED.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2))
