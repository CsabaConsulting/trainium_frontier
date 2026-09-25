"""Read-only, stdlib authentication of immutable batch result and seed43 launch."""
import copy
from datetime import datetime
from decimal import Decimal, getcontext
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess

REPO = Path('/home/poojith-devan/Desktop/Trainium')
BREF = '6c00343d4938349c55c04e9b0e4a55a429983d1d'
SREF = '746b53df4745e5521868034128ffc3d296cba9b8'
B = 'research/v2/operator/20260926-trn21-batch262144-timing/'
S = 'research/v2/operator/20260926-trn21-recursive-seed43-guarded/'
checks, hashes = [], {}
def blob(ref, path):
    data = subprocess.check_output(['git','-C',str(REPO),'show',ref + ':' + path])
    hashes[ref[:8] + ':' + path] = hashlib.sha256(data).hexdigest()
    return data
def load(ref, path): return json.loads(blob(ref, path))
def check(name, condition):
    if not condition: raise RuntimeError('FAILED: ' + name)
    checks.append(name)
def digest(data): return hashlib.sha256(data).hexdigest()
def instant(s): return datetime.fromisoformat(s)

retrieval = load(BREF, B+'completed/retrieval.json')
for name, pin in retrieval['hashes'].items():
    raw = blob(BREF, B+'completed/'+name)
    check('batch retrieved '+name, len(raw) == pin['bytes'] and digest(raw) == pin['sha256'])
bp = load(BREF,B+'EXPERIMENT_PROPOSAL.json')
bb = load(BREF,B+'BATCH_BINDING.disabled.json')
check('precommitted proposal unchanged before result', blob('a4b4838f',B+'EXPERIMENT_PROPOSAL.json') == blob(BREF,B+'EXPERIMENT_PROPOSAL.json'))
log = blob(BREF,B+'completed/training.log').decode()
control_blob = blob(BREF,bp['control']['log_path'])
check('control raw log pinned', digest(control_blob) == bp['control']['log_sha256'])
def rows(text):
    answer = {}
    for line in text.splitlines():
        if not line.startswith('step '): continue
        parts = line.split(' | ')
        step = int(parts[0].split()[1])
        if step in answer: raise RuntimeError('duplicate logged step')
        loss, lrm = Decimal(parts[1].split()[1]), Decimal(parts[2].split()[1])
        if not loss.is_finite() or not lrm.is_finite(): raise RuntimeError('nonfinite log')
        answer[step] = {'loss':loss,'lrm':lrm,'tok_s':int(parts[4].split()[1].replace(',',''))}
    return answer
crows, brows = rows(control_blob.decode()), rows(log)
window = bp['control']['window_steps']
check('predeclared window', window == list(range(10,101,10)))
control_median = Decimal(str(statistics.median(crows[s]['tok_s'] for s in window)))
candidate_median = Decimal(str(statistics.median(brows[s]['tok_s'] for s in window)))
getcontext().prec = 35
threshold = control_median * Decimal('1.05')
gain_pct = (candidate_median / control_median - 1) * 100
check('raw control median268831', control_median == Decimal('268831'))
check('raw candidate median282266.5', candidate_median == Decimal('282266.5'))
check('unrounded threshold decision remains BELOW', candidate_median < threshold and threshold == Decimal('282272.55'))
check('gap6.05 tokens per second', threshold-candidate_median == Decimal('6.05'))
check('same LR window', all(crows[s]['lrm'] == brows[s]['lrm'] == 1 for s in window))
check('actual full-chip accumulation2', 'device=neuron dtype=torch.bfloat16 world_size=4' in log and 'micro_tokens=131,072 grad_accum_steps=2 pack_factor=32' in log)
iso = json.loads(next(line.removeprefix('ISOLATED_SUBMISSION ') for line in log.splitlines() if line.startswith('ISOLATED_SUBMISSION ')))
command = json.loads(next(line.removeprefix('REHEARSAL_COMMAND ') for line in log.splitlines() if line.startswith('REHEARSAL_COMMAND ')))
check('batch actual source/prepare', iso['train_sha256'] == bb['source_sha256'] and iso['prepare_sha256'] == bb['prepare_sha256'])
check('batch actual argv', command == bb['training_command'] == bp['argv'])
prov = load(BREF,B+'completed/package/provenance.json')
launch_text = blob(BREF,B+'completed/package/launch-command.txt').decode().strip()
check('batch packaged provenance command and source', shlex.split(launch_text) == command and prov['launch_command'] == launch_text and prov['train_sha256'] == bb['source_sha256'])
check('batch Git source pin', digest(blob(bb['source_commit'],bb['source_git_path'])) == bb['source_sha256'])
rehearsal = load(BREF,B+'completed/train/rehearsal.json')
check('all105 raw step indices', [int(x) for x in blob(BREF,B+'completed/train/steps.jsonl').decode().splitlines()] == list(range(105)))
obs = rehearsal['observations']
check('all105 observer indices', rehearsal['step_boundaries'] == len(obs) == 105 and [x['step'] for x in obs] == list(range(105)))
times = [x['observed_seconds'] for x in obs]
check('observer monotonic', all(0 <= a <= z for a,z in zip(times,times[1:])))
allowance = min(times[4],900)
charged = rehearsal['wall_seconds']-allowance
steady_seconds = times[104]-times[4]
aggregate = 100*262144/steady_seconds
check('actual observer budget arithmetic', allowance == rehearsal['startup_allowance_observed'] and charged == rehearsal['charged_through_process_exit'] and charged <= 1800 and rehearsal['charged_through_process_exit_within_budget'] is True)
check('checkpoint produced at exit', rehearsal['checkpoint_bytes'] == 469834849 and rehearsal['checkpoint_sha256'] == '11db02c3b83cf2c8502c088b54046f9d52567c1f99b002bcf011e7d6d673e5f6')
terminal = load(BREF,B+'completed/terminal.json')
outer = load(BREF,B+'completed/outer-flock-exit.json')
check('all batch exits zero', rehearsal['returncode'] == terminal['returncode'] == outer['returncode'] == int(blob(BREF,B+'completed/run-exit-code.txt')) == 0)
screen = load(BREF,B+'completed/speed-screen.json')
check('screen independently matches arithmetic', screen['candidate_median_tok_s'] == float(candidate_median) and screen['candidate_observed_100step_seconds'] == steady_seconds and screen['candidate_observed_100step_tok_s'] == aggregate and screen['status'] == 'SPEED_SCREEN_BELOW_THRESHOLD')
check('no quality or automatic full', screen['quality_evidence'] is False and screen['automatic_full_run'] is False and terminal['no_eval'] is True and terminal['no_automatic_full'] is True)
check('warm not cold evidence', screen['cold_proven'] is False and rehearsal['fresh_empty_sdk_caches'] == {})
check('batch postflight quiescent', retrieval['job_lines'] == [] and retrieval['process_details'] == [] and retrieval['lock_competitor_rc'] == 0 and retrieval['neuron_ls_rc'] == 0 and json.loads(retrieval['neuron_ls'])[0]['neuron_processes'] == [])
analysis_ns = {'__name__':'independent_replay'}
exec(compile(blob(BREF,B+'analyze_timing.py'),'archived-screen','exec'),analysis_ns)
check('exact archived analyzer replay', analysis_ns['analyze'](log,rehearsal) == screen)
for label, mutated in [('missing boundary',dict(rehearsal,observations=obs[:-1])),('budget failure',dict(rehearsal,charged_through_process_exit_within_budget=False)),('exit failure',dict(rehearsal,returncode=124))]:
    refused = False
    try: analysis_ns['analyze'](log,mutated)
    except ValueError: refused=True
    check('analyzer refuses '+label,refused)

sb = load('90e942ea',S+'SEED_BINDING.disabled.json')
sl = load(SREF,S+'launch/training-launch.json')
sp = load(SREF,S+'launch/launch-preflight.json')
live = load(SREF,S+'launch/live-process-receipt.json')
sg = load(SREF,S+'launch/under-lock-idle.json')
bg = load(BREF,B+'completed/under-lock-idle.json')
for name in ('under-lock-idle.json','training-launch.json','launch-preflight.json','operator-launch.json'):
    raw = blob(SREF,S+'launch/'+name); pin=live['hashes'][name]
    check('seed retrieved '+name,len(raw)==pin['bytes'] and digest(raw)==pin['sha256'])
staged = load(SREF,S+'staging-receipt.json')
raw = (json.dumps(staged['preflight'],indent=2)+'\n').encode()
check('seed earlier preflight reconstructs exact raw pin', len(raw)==live['hashes']['preflight.json']['bytes'] and digest(raw)==live['hashes']['preflight.json']['sha256'])
check('seed package launch hash independently constructed', digest((shlex.join(sb['training_command'])+'\n').encode()) == sp['staged_hashes']['package/launch-command.txt'])
for name,key in [('EXPERIMENT_PROPOSAL.json',None),('SEED_BINDING.disabled.json',None),('host_driver.py','driver_sha256')]:
    raw=blob('90e942ea',S+name)
    check('seed staged exact '+name,digest(raw)==sp['staged_hashes'][name] and (key is None or digest(raw)==sb[key]))
for name,key in [('package/train.py','source_sha256'),('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
    check('seed staged '+name,sp['staged_hashes'][name]==sb[key])
check('seed Git source pin',digest(blob(sb['source_commit'],sb['source_git_path'])) == sb['source_sha256'])
check('seed actual launch command/environment',sl['command']==sb['command'] and sl['environment']==sb['training_environment'])
check('seed unchanged prepare/harness/source in actual launch',all(sl[k]==sb[k] for k in ('source_sha256','prepare_sha256','harness_sha256')))
check('seed staging source/support checked before batch completion not launch', staged['staged_only'] is True and staged['native_process_started'] is False)
check('batch fully exited before fresh seed guard',instant(outer['ended_utc']) < instant(sg['started_utc']) < instant(sg['ended_utc']) <= instant(sl['utc']) < instant(live['utc']))
for label,guard in [('batch',bg),('seed',sg)]:
    check(label+' shared exclusive original lock',guard['lock']['path']==sb['device_lock'] and guard['lock']['inode']==596823 and guard['lock']['device']==66305 and guard['lock']['verified_existing_exclusive_lock'] is True)
    check(label+' fresh ps and neuron idle under lock',guard['status']=='PASS' and guard['ps']['returncode']==guard['neuron_ls']['returncode']==0 and guard['ps']['busy']==[] and guard['neuron_ls']['process_pids']==[] and guard['ps']['stderr']==guard['neuron_ls']['stderr']=='')
check('seed live lock contention same inode',live['lock_competitor_rc']==1 and live['lock_inode']==596823 and live['lock_device']==66305)
workers=sorted([x for x in live['process_details'] if x['rank'] is not None],key=lambda x:int(x['rank']))
check('four seed live workers',len(workers)==4 and [int(x['rank']) for x in workers]==[0,1,2,3] and all(x['state']!='Z' for x in workers))
for worker in workers:
    check('seed actual rank'+worker['rank']+' argv and cwd',worker['argv'][2:]==sb['training_command'][3:] and worker['cwd']==sb['sandbox'] and worker['ppid']==3015306)
device=json.loads(live['neuron_ls'])[0]
check('same four workers occupy device',live['neuron_ls_rc']==0 and sorted(x['pid'] for x in device['neuron_processes'])==[x['pid'] for x in workers] and device['neuroncore_ids']==[0,1,2,3])
check('old runner absent at launch',live['old_queue_command_lines']==[] and all(x['exists'] is False for x in live['old_queue_pid_checks']))
parent=next(x for x in live['process_details'] if x['pid']==sg['lock']['parent_pid'])
driver=next(x for x in live['process_details'] if x['pid']==sg['lock']['driver_pid'])
check('actual outer parent and driver identity',parent['argv']==sb['outer_command'] and driver['ppid']==parent['pid'])
print(json.dumps({'verdict':'AUTHENTICATED_BATCH_BELOW_THRESHOLD_AND_SEED43_LAUNCH_ONLY','checks_passed':len(checks),'checks':checks,'batch':{'commit':BREF,'control_median_tok_s':str(control_median),'candidate_median_tok_s':str(candidate_median),'threshold_tok_s':str(threshold),'gain_percent_decimal':str(gain_pct),'shortfall_tok_s':str(threshold-candidate_median),'observer_wall_seconds':rehearsal['wall_seconds'],'observed_fifth_seconds':times[4],'charged_through_exit_seconds':charged,'steady100_seconds':steady_seconds,'steady100_tok_s':aggregate,'checkpoint_bytes':rehearsal['checkpoint_bytes'],'checkpoint_sha256':rehearsal['checkpoint_sha256'],'status':'SPEED_SCREEN_BELOW_THRESHOLD','quality_evidence':False,'cold_proven':False},'seed':{'commit':SREF,'training_launch_utc':sl['utc'],'live_observation_utc':live['utc'],'driver_pid':driver['pid'],'worker_pids':[x['pid'] for x in workers],'lock_inode':596823,'rank_worker_and_device_identity_pass':True,'completed':False,'retrieved_hashed_files':4,'reconstructed_hashed_earlier_preflight':1,'empty_outer_and_dispatch_logs':'remote hash/size reports only, not retrieved in this commit','quality_result':None},'input_sha256':hashes,'host_actions':False,'tensor_execution':False},indent=2))
