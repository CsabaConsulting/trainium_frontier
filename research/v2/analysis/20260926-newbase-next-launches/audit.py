"""Read-only authentication of the two 08:xx launch archives; no host/model actions."""
import hashlib,json
from pathlib import Path
R=Path('/home/poojith-devan/Desktop/Trainium')
out=[]
def need(x,why):
    if not x: raise RuntimeError('LAUNCH_AUDIT_REFUSE:'+why)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for host,folder,binding,livefile,source,inode in [
 ('trn21','20260926-trn21-clock36-lr012','20260926-clock36-lr012-trn21','first-training-observation.json','36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb',596823),
 ('trn22','20260926-trn22-clock36-recursive-cold-five','20260926-clock36-recursive-cold-five-trn22','first-four-rank-evidence.json','4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea',832589)]:
    path=R/'research/v2/operator'/folder/'launch'/livefile
    b=json.loads(path.read_text())
    bindpath=R/'research/v2/proposals'/binding/('NATIVE_BINDING.disabled.json' if host=='trn21' else 'CT_BINDING.disabled.json')
    bind=json.loads(bindpath.read_text())
    for name,f in b['files'].items():
        data=f['text'].encode()
        need(len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256'],host+':'+name)
    get=lambda n:json.loads(b['files'][n]['text'])
    g=get('under-lock-idle.json');p=get('preflight.json');l=get('training-launch.json')
    need(g['status']=='PASS' and g['lock']['device']==66305 and g['lock']['inode']==inode and g['lock']['verified_existing_exclusive_lock'],host+':guard_lock')
    need(g['ps']['returncode']==g['neuron_ls']['returncode']==0 and g['ps']['busy']==g['neuron_ls']['process_pids']==[],host+':idle')
    need(l['source_sha256']==source and l['command']==bind['command'] and l['environment']==bind['training_environment'],host+':launch_pins')
    need(l['harness_sha256']==bind['harness_sha256'] and l['prepare_sha256']==bind['prepare_sha256'],host+':support')
    live=json.loads((path.parent/'first-worker-descendants.json').read_text()) if host=='trn21' else b
    rows=live['processes'] if host=='trn21' else b['observations'][0]['processes']
    torchruns=[x for x in rows if x['comm']=='pt_elastic']
    need(len(torchruns)==1,host+':torchrun')
    workers=[x for x in rows if x['ppid']==torchruns[0]['pid']]
    need(len(workers)==4 and all(x['starttime_ticks']>0 for x in workers),host+':four_workers')
    devices=live['neuron_ls']
    if isinstance(devices,str):devices=json.loads(devices)
    need({x['pid'] for x in devices[0]['neuron_processes']}=={x['pid'] for x in workers},host+':device_owners')
    need(live['lock_device']==66305 and live['lock_inode']==inode and live['lock_competitor_rc']==1,host+':held')
    need(live['ps_rc']==live['neuron_ls_rc']==0,host+':process_status')
    need(not any(k in b['files'] for k in ('terminal.json','outer-flock-exit.json')),host+':not_terminal')
    log=b['files']['training.log']['text']
    need('Traceback (most recent call last)' not in log and 'num_params=113,256,724' in log,host+':init')
    if host=='trn22':
        need('--fresh-caches' in l['command'],host+':fresh_cache_request')
    out.append({'host':host,'status':'ARCHIVED_START_ACCEPTED_NOT_FINAL','source':source,'bundle_sha256':sha(path),'embedded_files':len(b['files']),'observed_utc':live['utc'],'worker_pids':[x['pid'] for x in workers],'original_inode':inode,'no_quality_result':True})
print(json.dumps({'status':'PASS','scope':'Archived launch evidence only; no current live check or final quality claim','runs':out},indent=2))

