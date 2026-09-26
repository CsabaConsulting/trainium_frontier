"""Read-only exact archived training boundary; never a final quality result."""
import hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def blob(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
def sha(b):return hashlib.sha256(b).hexdigest()
helper=blob('ae725297','research/v2/analysis/20260926-basebcbd-final-audit-preparation/audit.py')
if sha(helper)!='1ec138987f32d050448d2318702f06632f6ca8e1d531c9c3e1142812730dc770':raise RuntimeError('HELPER_PIN')
ns={'__name__':'immutable_archive_helper'}
exec(compile(helper,'immutable_helper','exec'),ns)
require,parse,timing,tagged=(ns[x] for x in ('require','parse','timing','tagged'))
raw=blob('7d58fb4886ae0ac189153427cf20cb173688abf5','research/v2/operator/20260926-trn21-lr-train-eval-boundary-0326/trn21.json')
require(sha(raw)=='fc48525fee3d026bd2fdda02daeb5ca8d1ef78369d321158711b978b30d67ccf','raw_pin')
o=parse(raw);require(o['rc']==0 and not o['stderr'],'read_success');d=o['remote']
f={}
for name,item in d['files'].items():
    if 'text' not in item:continue
    v=item['text'].encode();require(sha(v)==item['sha256_captured'] and len(v)==item['captured_bytes'],'file:'+name);f[name]=v
require(sum(map(len,f.values()))==d['captured_file_bytes'],'capture_total')
b=parse(blob('d0963aef0712adc7a00bd0d202d8de82f5fed7ef','research/v2/proposals/20260926-clock36-lr012-trn21/NATIVE_BINDING.disabled.json'))
require(d['root']==b['root'] and d['label']==b['label'],'exact_run')
require(b['source_sha256']=='36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb','source')
argv=b['training_command'];require(argv[argv.index('--matrix-lr')+1]=='0.012' and '--recursive-schedule' not in argv,'only_lr')
for name,cmd,env in [('training-launch.json','command','training_environment'),('fresh-public20m-launch.json','evaluation_command','evaluation_environment')]:
    a=parse(f[name]);require(a['command']==b[cmd] and a['environment']==b[env] and a['source_sha256']==b['source_sha256'],'actual_launch:'+name)
tr=f['training.log'].decode();h=parse(f['train/rehearsal.json']);facts=timing(h)
require(tagged(tr,'REHEARSAL_COMMAND ')==argv,'actual_train_argv')
require(tagged(tr,'SUBMISSION_REHEARSAL ')=={k:v for k,v in h.items() if k!='observations'},'exact_train_summary')
require([int(x) for x in f['train/steps.jsonl'].splitlines()]==list(range(facts['steps'])),'all_boundaries')
require(f['run-exit-code.txt'].strip()==b'0' and re.search(r'^over_budget:\s*False$',tr,re.M),'source_budget_exit')
ck=parse(f['checkpoint.json']);require(ck['sha256']==h['checkpoint_sha256'] and ck['bytes']==h['checkpoint_bytes']>0,'checkpoint_record_not_new_hash')
p={x['pid']:x for x in d['processes']};ev=p[3175146]
expected=[b['evaluation_command'][5],'prepare.py','eval-public','--train-py','train.py','--checkpoint',b['sandbox']+'/out/final.pt','--device','neuron','--batch-size','1','--seq-len','1024','--eval-tokens','20971520','--timeout-seconds','1200']
require(ev['argv']==expected and ev['ppid']==3175132 and p[3175132]['ppid']==3175131,'actual_eval_chain')
for pid in (3175131,3175132,3175146):
    require(p[pid]['exists'] and p[pid]['identity_stable'] and p[pid]['pgid']==p[pid]['sid']==3152603,'live_eval_identity')
require(d['lock']['held'] and d['lock']['device']==66305 and d['lock']['inode']==596823,'original_lock_held')
require(d['neuron_ls_rc']==0 and d['ps_rc']==0,'read_success')
require(any(x['pid']==3175146 for dev in d['neuron_ls'] for x in dev['neuron_processes']),'actual_device_eval')
require('terminal.json' not in f and 'eval/rehearsal.json' not in f,'no_final_result_claim')
out=dict(status='TRAINING_AUTHENTICATED_FRESH20M_STILL_LIVE',checks=len(ns['checks']),raw_sha256=sha(raw),observed_utc=d['utc'],facts=facts,training_end_utc=re.search(r'^TRAINING_END_UTC=(.+)$',tr,re.M).group(1),checkpoint_record=ck,posteval_checkpoint_hash_pending=True,eval_pids=[3175131,3175132,3175146],final_bpb=None,adoption=False,submission_ready=False,source_modified=False)
print(json.dumps(out,indent=2,allow_nan=False))
