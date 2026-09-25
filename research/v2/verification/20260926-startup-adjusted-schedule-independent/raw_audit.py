"""Independent authentication/comparison of frozen reduced CPU receipts only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

REPO=Path(__file__).resolve().parents[4]
COMMIT='225f27d13c25c9160f5fd2758cf016fb4c70bf0f'
PACKET='research/v2/lanes/startup-adjusted-schedule/'
checks=[];pins={}
def check(name,value):
    if not value:raise RuntimeError(name)
    checks.append(name)
def read(name):
    raw=subprocess.check_output(['git','show',COMMIT+':'+PACKET+name],cwd=REPO)
    pins[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    return json.loads(raw)
arms={name:read('tiny-'+name+'.json') for name in ('base','default','off','on779','on790','on1000')}
for name,report in arms.items():
    steps=3 if name in ('base','default','off') else 6
    check(name+' raw terminal/domain',report['torchrun_rc']==0 and report['timed_out'] is False and report['verdict']=='PASS' and report['nproc']==4 and report['steps']==steps and report['micro_batch']==[2,128])
    check(name+' source pin',report['train_py_sha256']==('3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c' if name=='base' else '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'))
    check(name+' rank inventory',[r['rank'] for r in report['ranks']]==list(range(4)))
    for rank in report['ranks']:
        label=name+'/'+str(rank['rank']);cache=rank['cache_observations']
        check(label+' rank completed',rank['status']=='PASS' and not rank.get('error'))
        check(label+' actual five-boundary override',rank['schedule_observations']['startup_steps_excluded']==5)
        check(label+' observations',len(rank['losses'])==len(cache['steps'])==len(rank['schedule_observations']['optimizer_groups'])==steps and rank['n_params']==rank['n_changed']==65 and rank['cross_rank_mismatches']==[])
        check(label+' finite reported losses',all(math.isfinite(x) for x in rank['losses']))
        if rank['rank']==0:
            check(label+' reported rank0 reload',rank['checkpoint_check']['ok'] is rank['checkpoint_check']['deterministic'] is rank['checkpoint_check']['path_exists'] is True)
            check(label+' reduced config',cache['checkpoint']['model_config']['n_layer']==8 and cache['checkpoint']['model_config']['n_embd']==128 and cache['checkpoint']['model_config']['vocab_size']==8192)
        for i,step in enumerate(cache['steps']):
            for field in ('gradients_before_clip','gradients_after_clip','parameters_after_update'):
                records=step[field]
                check(label+'/'+str(i)+'/'+field,len(records)==65 and all(isinstance(v,dict) and v['dtype']=='torch.float32' and re.fullmatch('[0-9a-f]{64}',v['sha256']) for v in records.values()))
    first=report['ranks'][0]
    for rank in report['ranks'][1:]:
        check(name+' cross rank final '+str(rank['rank']),rank['cache_observations']['steps'][-1]['parameters_after_update']==first['cache_observations']['steps'][-1]['parameters_after_update'])

for name in ('default','off'):
    for a,b in zip(arms['base']['ranks'],arms[name]['ranks']):
        for field in ('losses','plan_sha256','plan_summary','schedule_observations')+(('final_digests',) if a['rank']==0 else ()):
            check(name+'/'+str(a['rank'])+' identical '+field,a[field]==b[field])
        for field in ('steps','cache_refreshes','forwards','mlp_custom_backward_calls')+(('checkpoint','checkpoint_file_sha256') if a['rank']==0 else ()):
            check(name+'/'+str(a['rank'])+' identical cache '+field,a['cache_observations'][field]==b['cache_observations'][field])

# Same six-step ON trajectory under the two below-cap fake delays.
for a,b in zip(arms['on779']['ranks'],arms['on790']['ranks']):
    for field in ('losses','plan_sha256','plan_summary')+(('final_digests',) if a['rank']==0 else ()):
        check('below-cap ON exact '+str(a['rank'])+'/'+field,a[field]==b[field])
    for field in ('steps','cache_refreshes','forwards','mlp_custom_backward_calls')+(('checkpoint','checkpoint_file_sha256') if a['rank']==0 else ()):
        check('below-cap ON cache exact '+str(a['rank'])+'/'+field,a['cache_observations'][field]==b['cache_observations'][field])
    check('below-cap scalar histories '+str(a['rank']),a['schedule_observations']['optimizer_groups']==b['schedule_observations']['optimizer_groups'])

scalars={}
for name in ('on779','on790','on1000'):
    rank=arms[name]['ranks'][0];history=rank['schedule_observations']['optimizer_groups'];rows=[]
    for step,groups in enumerate(history):
        muon=[g for g in groups if g['kind']=='muon']
        rows.append({'step':step,'muon_lr':muon[0]['lr'],'muon_weight_decay':muon[0]['weight_decay']})
        check(name+' identical Muon group progress '+str(step),all(g['lr']==muon[0]['lr'] and g['weight_decay']==muon[0]['weight_decay'] for g in muon))
        if step<5 and name!='on1000':check(name+' excluded startup scalar '+str(step),muon[0]['lr']==.015 and muon[0]['weight_decay']==.2)
        if name=='on1000' and step>=1:check('capped excess is charged '+str(step),0<muon[0]['weight_decay']<.2)
    scalars[name]=rows
check('above-cap ON differs intentionally',arms['on1000']['ranks'][0]['final_digests']!=arms['on790']['ranks'][0]['final_digests'])
check('ON has intentional different trajectory',arms['base']['ranks'][0]['losses']!=arms['on790']['ranks'][0]['losses'][:3])

failed=read('failed-attempt1-base.json')
check('original observer failure retained',failed['verdict']=='FAIL' and failed['reason']=='AttributeError in <lambda>:4724' and len(failed['ranks'])==4 and all(r['status']=='FAIL' and r['error']=="AttributeError: 'OwnedOptimizer' object has no attribute 'param_groups'" for r in failed['ranks']))
check('caught failure exit is not acceptance',failed['torchrun_rc']==0 and failed['plan_sha256'] is None)
for name in arms:
    prior=read('attempt2-stub2/tiny-'+name+'.json')
    check('prior2-boundary raw retained '+name,prior['torchrun_rc']==0 and len(prior['ranks'])==4)
limitation=read('attempt2-stub2/LIMITATION.json')
result={'status':'PASS_REDUCED_RECORDED_EVIDENCE_ONLY','raw_commit':COMMIT,'checks':len(checks),'pins':pins,'scalars_rank0':scalars,'limits':['Raw hashes/recorded comparisons authenticated; no hidden observer replay or independently loaded checkpoint tensors.','Reduced8x128 CPU BF16, synthetic sliced2x128 per rank and fake time; recorded production token counters do not establish131072 actually forwarded tokens.','Final six arms explicitly override imported startup count5; prior two-boundary evidence and comparison failure preserved.','Three-step default/OFF numerical identity does not itself exercise the fifth reset; source fake-clock tests and six-step ON provide separate evidence.','Gradient/update/state digest equality is not independent native finiteness or compiler evidence.','No full113M, native/cold/quality/causal BPB or adoption approval.'],'no_tensors':True,'no_hosts':True,'fresh_review_pending':True,'native_acceptance':False,'prior_harness_limitation':limitation,'checks_passed':checks}
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
if args.output:args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('pins','scalars_rank0','checks_passed','prior_harness_limitation')},indent=2))
