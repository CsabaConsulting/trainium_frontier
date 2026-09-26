"""Authenticate a failed ordinary-only native diagnostic, never promote it."""
import base64, hashlib, json
from pathlib import Path

repo=Path(__file__).resolve().parents[4]
directory=repo/'research/v2/operator/20260926-trn21-ordinary-subnormal/completed'
def require(ok, message):
    if not ok:
        raise RuntimeError(message)
def pinned(name, pin):
    raw=(directory/name).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==pin, 'SHA:'+name)
    return json.loads(raw)
b=pinned('results-bundle.json','b91f230736f522d20174749b2792da1e3b34c9c842c3cf82889c41ec22152c0f')
raw={}
for name,record in b['files'].items():
    value=record['text'].encode() if 'text' in record else base64.b64decode(record['base64'],validate=True)
    require(len(value)==record['bytes'] and hashlib.sha256(value).hexdigest()==record['sha256'],'ARTIFACT:'+name)
    raw[name]=value
require(len(raw)==31 and sum(map(len,raw.values()))==7295580,'ARTIFACT_COUNTS')
j=lambda name:json.loads(raw[name])
packet=repo/'research/v2/proposals/20260926-ordinary-embedding-subnormal-control'
require(raw['NATIVE_BINDING.disabled.json']==(packet/'NATIVE_BINDING.disabled.json').read_bytes(),'BINDING_BYTES')
binding=j('NATIVE_BINDING.disabled.json')
launch=j('diagnostic-launch.json');term=j('terminal.json');child=j('child-exit.json')
require(launch['support_sha256']==binding['support_sha256'],'SUPPORT_PINS')
require(term['returncode']==1 and term['status']=='DIAGNOSTIC_CHILD_FAILED' and term['complete'] is False,'TERMINAL')
require(term['no_automatic_next_job'] and term['no_eval'] and child['returncode']==1,'NO_FOLLOW_ON')
ranks=[]
for rank in range(4):
    r=j('receipts/rank'+str(rank)+'.json')
    require(r['rank']==rank and r['world_size']==4 and r['seed']==42,'RANK_BINDING')
    require(r['checker_sha256']=='1cc6ee395f88125b9c9d280562e918d002a1e1a10dc88d50501d1fb4a09a551c','CHECKER_BYTES')
    require(r['validation_error']=='ValueError: ROUTE_FAILED' and r['error'] is None and r['returncode']==1,'RANK_FAILURE')
    require(r['cleanup']=='DESTROYED' and r['compiled_callable_count']==1 and r['native_qualification'] is False,'SCOPE')
    require(len(r['routes'])==8,'EIGHT_ROUTES')
    keys={(x['case'],x['route']) for x in r['routes']}
    require(keys=={(c,t) for c in ('sentinel','subnormal','normal','negated') for t in ('ordinary_eager','ordinary_compiled')},'EXACT_ROUTE_SET')
    result=[]
    for x in r['routes']:
        require(x['error'] is None and x['output']['exact'] is True and all(v is True for v in x['checks'].values()),'INPUT_FORWARD:'+str(rank))
        g=x['gradient'];sub=x['case'] in ('subnormal','negated')
        require(g['exact'] is (not sub) and x['status']==('MISMATCH' if sub else 'EXACT'),'GRAD_STATUS')
        if sub:
            require(g['actual']['nonzero']==0 and g['expected']['nonzero']==1408,'SUBNORMAL_COUNTS')
            require(g['max_abs']==1.2611686178923354e-44,'SUBNORMAL_ERROR')
            require(g['actual']['sha256']=='fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471','ZERO_BYTES')
        else:
            require(g['max_abs']==0 and g['actual']['sha256']==g['expected']['sha256'],'NORMAL_EXACT')
        result.append({'case':x['case'],'route':x['route'],'status':x['status'],'gradient_max_abs':g['max_abs'],'actual_nonzero':g['actual']['nonzero'],'expected_nonzero':g['expected']['nonzero']})
    ranks.append({'rank':rank,'pid':r['pid'],'routes':result})
post=pinned('explicit-process-postflight.json','ca4c80678f3a4fa049cbcbaffbddaaa0103317c2685253dd010b41033585c8a5')
require(post['all_observed_pids_absent'] is True and post['neuron_device_idle'] is True,'QUIESCENCE')
require(len(post['observed_pids'])==11 and set(map(int,post['observed_starttime_records']))==set(post['observed_pids']),'OBSERVED_IDENTITIES')
require(post['related_rows']==[] and post['fuser_rc']==1 and post['ps_rc']==0 and post['lock_competitor_rc']==0,'POSTFLIGHT')
require(post['lock_device']==66305 and post['lock_inode']==596823,'ORIGINAL_LOCK')
print(json.dumps({'status':'FAILED_DIAGNOSTIC_AUTHENTICATED_NOT_PROMOTED','artifact_count':31,'raw_bytes':7295580,'terminal':term,'wall_seconds':child['timeout_wrapped_child_wall_seconds'],'ranks':ranks,'postflight_utc':post['utc'],'all_observed_pids_absent':True,'interpretation':'Both ordinary eager and compiled native routes zero both-sign subnormal gradients; mismatch is not exclusive to compiled execution or K2. Exact mechanism remains unknown. Original K2 gate stays FAILED; no oracle waiver, retry or throughput claim.','native_qualification':False,'retry_authorized':False},indent=2))
