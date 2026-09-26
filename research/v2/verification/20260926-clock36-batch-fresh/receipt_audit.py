import json,pathlib
p=pathlib.Path('/home/poojith-devan/clock36-batch-fresh-review-QFHzvD/receipts')
checks=[]
def check(ok,name):
    if not ok: raise RuntimeError(name)
    checks.append(name)
for mode in ['normal','optimized']:
    parent,default,arm=[json.loads((p/(name+'-'+mode+'.json')).read_text()) for name in ['parent','default','batch262144']]
    for name,data in [('parent',parent),('default',default),('arm',arm)]:
        check(data['torchrun_rc']==0 and not data['timed_out'],mode+name+' receipt exit')
        check(len(data['ranks'])==4 and data['micro_batch']==[2,128],mode+name+' tiny world4 2x128 scope')
        for r in data['ranks']:
            c=r['cache_observations']; prefix=mode+name+str(r['rank'])
            check(r['n_params']==65 and all(len(s['parameters_after_update'])==65 for s in c['steps']),prefix+' 65 parameter records')
            check(len(c['steps'])==3,prefix+' three update snapshots')
            check(len(c['forwards'])==(6 if name=='arm' else 3),prefix+' forward count')
            check(all(v['grad_enabled'] for v in c['cache_refreshes']),prefix+' grad-enabled refresh')
    for a,b in zip(parent['ranks'],default['ranks']):
        for key in ['losses','plan_summary'] + (['final_digests'] if a['rank']==0 else []):
            check(a[key]==b[key],mode+str(a['rank'])+' parent/default '+key)
        for key in ['steps','micro_gradients']:
            check(a['cache_observations'][key]==b['cache_observations'][key],mode+str(a['rank'])+' parent/default '+key)
print(json.dumps({'verdict':'PASS_RECEIPT_CONSISTENCY_ONLY','scope':'Independent comparison of supplied raw records; author observer not inspected or rerun; not independent full-model execution','checks':checks},indent=2))
