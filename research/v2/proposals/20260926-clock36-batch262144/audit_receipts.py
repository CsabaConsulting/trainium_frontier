"""Author receipt consistency checks; never independent approval."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
OUT=HERE/'observed-receipts'
checks=[]
def check(label,value):
    if not value:raise RuntimeError(label)
    checks.append(label)
for mode in ('normal','optimized'):
    arms={arm:json.loads((OUT/(arm+'-'+mode+'.json')).read_text()) for arm in ('parent','default','batch262144')}
    for arm,r in arms.items():
        check(mode+arm+' pass',r['verdict']=='PASS' and r['torchrun_rc']==0 and not r['timed_out'])
        for rank in r['ranks']:
            o=rank['cache_observations']
            check(mode+arm+str(rank['rank'])+' three updates',len(o['steps'])==3)
            micros=6 if arm=='batch262144' else 3
            check(mode+arm+str(rank['rank'])+' actual forwards',len(o['forwards'])==micros)
            check(mode+arm+str(rank['rank'])+' all65 contributions',len(o['micro_gradients'])==65 and all(len(g)==micros for g in o['micro_gradients'].values()))
            check(mode+arm+str(rank['rank'])+' cache4refresh',len(o['cache_refreshes'])==4*len(o['forwards'][0]['caches']) and all(x['grad_enabled'] for x in o['cache_refreshes']))
            check(mode+arm+str(rank['rank'])+' differentiable caches',all(x is not None and x['requires_grad'] and x['grad_fn']!='NoneType' for f in o['forwards'] for x in f['caches'].values()))
            check(mode+arm+str(rank['rank'])+' full step snapshots',all(len(s['gradients_before_clip'])==65 and len(s['gradients_after_clip'])==65 and len(s['parameters_after_update'])==65 for s in o['steps']))
            if arm=='batch262144':
                check(mode+str(rank['rank'])+' second micro nonzero',all(any(g[i]['nonzero'] for g in o['micro_gradients'].values()) for i in (1,3,5)))
                check(mode+str(rank['rank'])+' distinct micro gradients',all(any(g[i]['gradient']!=g[i-1]['gradient'] for g in o['micro_gradients'].values()) for i in (1,3,5)))
            if arm=='default':
                ref=arms['parent']['ranks'][rank['rank']]
                check(mode+str(rank['rank'])+' exact losses',rank['losses']==ref['losses'])
                check(mode+str(rank['rank'])+' exact gradients updates optimizer',o['steps']==ref['cache_observations']['steps'])
                check(mode+str(rank['rank'])+' exact micro gradients',o['micro_gradients']==ref['cache_observations']['micro_gradients'])
    check(mode+' standalone normalization',json.loads((OUT/('cache-'+mode+'.stdout')).read_text())['verdict']=='PASS')
    check(mode+' no failed commands',all(x['exit_code']==0 for x in json.loads((OUT/'commands.json').read_text())))
print(json.dumps({'scope':'author tiny CPU checks only; fresh independent review/full-size/native pending','count':len(checks),'checks':checks},indent=2))
