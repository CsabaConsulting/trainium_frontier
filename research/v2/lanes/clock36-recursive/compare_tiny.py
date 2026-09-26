"""Fail-closed Implementer self-comparison; not independent/native acceptance."""
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
NAMES=[f'clock{clock}-{arm}' for clock in (0,1) for arm in ('base','default','off','on','reference')]


def require(ok,text):
    if not ok:
        raise RuntimeError(text)


def observables(rank, old_compat=False):
    c=rank['cache_observations']
    checkpoint=copy.deepcopy(c.get('checkpoint'))
    if old_compat and checkpoint:
        require(checkpoint['toggles'].get('RECURSIVE_SCHEDULE',False) is False,'non-OFF metadata')
        checkpoint['toggles'].pop('RECURSIVE_SCHEDULE',None)
    s=rank['recursive_observations']
    return dict(losses=rank['losses'],final=rank.get('final_digests'),steps=c['steps'],forwards=c['forwards'],
                checkpoint=checkpoint,groups=s['optimizer_groups'],lr_calls=s['lr_calls'],beta2=s['beta2_routes'])


def check(receipts):
    require(set(receipts)==set(NAMES),'missing arm')
    plan=receipts['clock0-base']['plan_sha256']
    for name,r in receipts.items():
        require(r['verdict']=='PASS' and r['torchrun_rc']==0,'failed G2:'+name)
        require(r['steps']==6 and r['nproc']==4 and r['micro_batch']==[2,128],'shape:'+name)
        require(r['plan_sha256']==plan,'ownership:'+name)
        require({q['rank'] for q in r['ranks']}=={0,1,2,3},'rankset:'+name)
        require(r['observer_sha256']==hashlib.sha256((HERE/'g2_observe.py').read_bytes()).hexdigest(),'observer:'+name)
        for q in r['ranks']:
            require(q['status']=='PASS','rank status:'+name)
            c=q['cache_observations']; s=q['recursive_observations']
            require(s['startup_steps_excluded']==5 and s['first_step_fake_delay_s']==1000.,'clock fixture:'+name)
            require(len(c['steps'])==6 and c['mlp_custom_backward_calls']==48,'six real updates/MLP:'+name)
            for step in c['steps']:
                require(len(step['parameters_after_update'])==65,'all65 params:'+name)
                require(set(step['gradients_before_clip'])==set(step['parameters_after_update']),'all gradient/missing statuses:'+name)
                require(set(step['gradients_after_clip'])==set(step['parameters_after_update']),'all clipped gradients:'+name)
            if q['rank']==0:
                require(q['checkpoint_check']['ok'] is True and q['checkpoint_check']['deterministic'] is True,'reload:'+name)
            on=name.endswith(('-on','-reference'))
            require(len(s['beta2_routes'])>0,'no actual optimizer beta2 route:'+name)
            require(all(z['is_tensor']==on for z in s['beta2_routes']),'beta2 actual route:'+name)
            if on:
                calls=s['lr_calls']
                require(len(calls)==12,'two LR calls per update:'+name)
                for i in range(6):
                    mu,adam=calls[2*i:2*i+2]
                    require(mu['step']==adam['step']==i and mu['kind']=='muon' and adam['kind']=='adamw','LR call order:'+name)
                    require(mu['elapsed']==adam['elapsed'],'shared LR clock:'+name)
                    if name.startswith('clock1') and i in (1,4,5):
                        require(mu['elapsed']>130.,'capped excess survives fifth boundary:'+name)
    for clock in (0,1):
        for arm in ('default','off'):
            require([observables(q,True) for q in receipts[f'clock{clock}-{arm}']['ranks']]==
                    [observables(q,True) for q in receipts[f'clock{clock}-base']['ranks']],f'clock{clock} {arm} identity')
        require([observables(q) for q in receipts[f'clock{clock}-on']['ranks']]==
                [observables(q) for q in receipts[f'clock{clock}-reference']['ranks']],f'clock{clock} independent formula update identity')
    require(receipts['clock0-on']['ranks'][0]['final_digests']!=receipts['clock1-on']['ranks'][0]['final_digests'],'clock scientifically observable')
    return True


def main():
    receipts={n:json.loads((HERE/'raw'/f'{n}.json').read_text()) for n in NAMES}
    check(receipts)
    negatives=[]
    for name,mutate in (
        ('missing_arm',lambda r:r.pop('clock0-base')),
        ('missing_rank',lambda r:r['clock1-on']['ranks'].pop()),
        ('missing_update',lambda r:r['clock0-on']['ranks'][0]['cache_observations']['steps'].pop()),
        ('wrong_lr_clock',lambda r:r['clock1-on']['ranks'][0]['recursive_observations']['lr_calls'][3].update(elapsed=1000.)),
        ('scalar_beta2',lambda r:r['clock1-on']['ranks'][0]['recursive_observations']['beta2_routes'][0].update(is_tensor=False)),
        ('changed_gradient',lambda r:next(iter(r['clock0-off']['ranks'][0]['cache_observations']['steps'][0]['gradients_after_clip'].values())).update(sha256='0'*64)),
    ):
        altered=copy.deepcopy(receipts); mutate(altered)
        try:
            check(altered)
        except RuntimeError as e:
            negatives.append(dict(name=name,refused=str(e)))
        else:
            raise RuntimeError('negative falsely accepted:'+name)
    out=dict(status='IMPLEMENTER_SELF_CHECKS_ONLY',candidate_sha256='4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea',
             arms=10,ranks_per_arm=4,steps=6,parameter_tensors=65,parameters=sum(receipts['clock0-base']['ranks'][0]['plan_summary']['elements_by_owner']),
             default_and_explicit_off_identity_both_clock_modes=True,recursive_updates_equal_independent_formulas_both_clock_modes=True,
             bytewise_checkpoint_model_content=True,new_checkpoint_toggle_only_for_off=True,serialized_checkpoint_file_identity_claimed=False,
             actual_tensor_beta2_route_on=True,fullsize_or_native=False,negative_controls=negatives,
             raw_sha256={n:hashlib.sha256((HERE/'raw'/f'{n}.json').read_bytes()).hexdigest() for n in NAMES})
    print(json.dumps(out,indent=2,sort_keys=True))
    if len(sys.argv)==2:
        Path(sys.argv[1]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')


if __name__=='__main__': main()
