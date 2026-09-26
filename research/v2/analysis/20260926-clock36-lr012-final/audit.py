"""Exact LR012 final archive analysis, no hosts, source edits or ledger writes."""
import hashlib,json,math,re,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
RAW='bfe0e3503783018fc5abaaf9484b6271344179f5'
OP='research/v2/operator/20260926-trn21-clock36-lr012/completed/'
P='research/v2/proposals/20260926-clock36-lr012-trn21/'
PACKET='d0963aef0712adc7a00bd0d202d8de82f5fed7ef'
SOURCE='36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'
def blob(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
def sha(x):return hashlib.sha256(x).hexdigest()
helper=blob('ae725297','research/v2/analysis/20260926-basebcbd-final-audit-preparation/audit.py')
if sha(helper)!='1ec138987f32d050448d2318702f06632f6ca8e1d531c9c3e1142812730dc770':raise RuntimeError('HELPER_PIN')
ns={'__name__':'archive_only'};exec(compile(helper,'immutable-helper','exec'),ns)
need,parse,timing,evaluation,tagged=[ns[k] for k in ('require','parse','timing','evaluation','tagged')]
raw=blob(RAW,OP+'results-bundle.json');postraw=blob(RAW,OP+'explicit-process-postflight.json')
need(sha(raw)=='60b2f0449482a1a8c35cd097d06fbd52f4b03e849fef91ac34c1c3db5efa1655','raw pin')
need(sha(postraw)=='b731af45e2303166f58c73d8c939feb6f82a71c0b3da53a6aa707099d0efbcfb','post pin')
r=parse(raw);post=parse(postraw);f={}
for name,item in r['files'].items():
    v=item['text'].encode();need(item['capture']=='complete_at_observation' and len(v)==item['captured_bytes']==item['bytes_at_stat'] and sha(v)==item['sha256_captured'],'complete file '+name);f[name]=v
need(sum(map(len,f.values()))==r['captured_file_bytes'],'total bytes')
braw=blob(PACKET,P+'NATIVE_BINDING.disabled.json');b=parse(braw)
need(braw==f['NATIVE_BINDING.disabled.json'] and r['root']==b['root'] and r['label']==b['label'],'exact bound run')
need(sha(blob(b['source_commit'],b['source_git_path']))==b['source_sha256']==SOURCE,'frozen source')
need(sha(f['host_driver.py'])==b['driver_sha256'],'driver pin')
need(sha(blob(PACKET,'prepare.py'))==b['prepare_sha256'] and sha(blob('00f2ca7f','research/scratch/submission_rehearsal.py'))==b['harness_sha256'],'protected prepare harness')
for name,cmd,env in [('training-launch.json','command','training_environment'),('fresh-public20m-launch.json','evaluation_command','evaluation_environment')]:
    q=parse(f[name]);need(q['command']==b[cmd] and q['environment']==b[env] and q['source_sha256']==SOURCE,'actual bound '+name)
argv=b['training_command'];adopt=parse(blob('227a5c5d','research/v2/adoptions/20260926-clock36-confirmed.json'))
baseline=adopt['argv'];actual=argv[4:];changed=[i for i,(x,y) in enumerate(zip(baseline,actual)) if x!=y]
need(len(actual)==len(baseline) and changed==[baseline.index('--matrix-lr')+1] and actual[changed[0]]=='0.012','sole scientific argv LR value')
tr=f['training.log'].decode();el=f['fresh-public20m.log'].decode();h=parse(f['train/rehearsal.json']);facts=timing(h)
for log,mode,sandbox in [(tr,'train',b['sandbox']),(el,'eval',b['eval_sandbox'])]:
    q=tagged(log,'ISOLATED_SUBMISSION ');need(q['train_sha256']==SOURCE and q['prepare_sha256']==b['prepare_sha256'] and q['files']==['prepare.py','train.py'] and q['sandbox']==sandbox and q['mode']==mode and shlex.split(q['launch_command'])==argv,'isolated '+mode)
need(tagged(tr,'REHEARSAL_COMMAND ')==argv and tagged(tr,'SUBMISSION_REHEARSAL ')=={k:v for k,v in h.items() if k!='observations'},'actual train command summary')
need([int(x) for x in f['train/steps.jsonl'].splitlines()]==list(range(facts['steps'])),'all step boundaries')
need(f['run-exit-code.txt'].strip()==b'0' and re.search(r'^over_budget:\s*False$',tr,re.M),'source budget exit')
losses=re.findall(r'^step \d+ \| loss (\S+) \|',tr,re.M);need(losses and all(math.isfinite(float(x)) for x in losses),'finite printed losses')
ck=parse(f['checkpoint.json']);after=r['checkpoint_after_terminal']
need(after['stable'] and after['sha256']==ck['sha256']==h['checkpoint_sha256'] and after['bytes']==ck['bytes']==h['checkpoint_bytes']>0,'actual unchanged retained checkpoint')
expected=[b['evaluation_command'][5],'prepare.py','eval-public','--train-py','train.py','--checkpoint',b['sandbox']+'/out/final.pt','--device','neuron','--batch-size','1','--seq-len','1024','--eval-tokens','20971520','--timeout-seconds','1200']
need(tagged(el,'REHEARSAL_COMMAND ')==expected,'actual B1 context1024 fresh20M')
ev=evaluation(el);eh=parse(f['eval/rehearsal.json']);terminal=parse(f['terminal.json']);outer=parse(f['outer-flock-exit.json'])
need(eh==tagged(el,'SUBMISSION_REHEARSAL ') and eh['returncode']==terminal['eval_exit_code']==outer['returncode']==0 and f['eval-exit-code.txt'].strip()==b'0','final exits')
need(terminal['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and terminal['label']==b['label'] and outer['command']==b['outer_command'],'terminal exact command')
for n in ('under-lock-idle.json','eval-under-lock-idle.json'):
    g=parse(f[n]);need(g['status']=='PASS' and g['lock']['device']==66305 and g['lock']['inode']==596823 and g['lock']['verified_existing_exclusive_lock'] and g['ps']['returncode']==g['neuron_ls']['returncode']==0 and not g['ps']['busy'] and not g['neuron_ls']['process_pids'],'actual locked guard '+n)
ids={3152603,3152604,3152605,3152621,3152622,3152637,3152652,3152653,3152654,3152655,3153251,3153259,3153290,3175131,3175132,3175146}
need(set(post['observed_pids'])==ids=={x['pid'] for x in post['pid_rows']} and all(not x['exists'] for x in post['pid_rows']),'all16 observed IDs absent')
need(post['all_observed_pids_absent'] and post['related_rows']==post['busy_job_rows']==[] and post['ps_rc']==0,'related process census')
need(post['neuron_device_idle'] and post['neuron_ls_rc']==0 and all(not d['neuron_processes'] for d in post['neuron_ls']) and post['fuser_rc']==1 and not post['fuser_stdout'],'device idle')
need(post['lock_competitor_rc']==0 and (post['lock_device'],post['lock_inode'])==(66305,596823),'original lock free')
row=dict(schema='trn-run-record-v2',label=b['label'],arm_id='clock36-matrix-lr012',kind='lr_arm',run_purpose='quality',host='trn21',timestamp_utc=terminal['utc'],base_sha256=SOURCE,train_py_sha256=SOURCE,source_commit=b['source_commit'],status='evaluated',git_dirty=False,git_dirty_scope='Authenticated exact scientific bytes, not globally clean host checkout',argv=actual,env=b['training_environment'],training_budget_seconds=1800,over_budget=False,steps=facts['steps'],charged_through_exit_seconds=facts['charged_through_exit_seconds'],startup_measured_seconds=facts['fifth_seconds'],cold_startup_proven=False,causality_passed=None,eval_tokens=20971520,eval_seq_len=1024,public_20m=ev['val_bpb'],checkpoint_sha256=ck['sha256'],raw_archive_commit=RAW,raw_archive_sha256=sha(raw),submission_eligible=False,result_authentication='research/v2/analysis/20260926-clock36-lr012-final/result.json',notes='Exact36 LR.012 only. Historical same-host architecture result is not a kind=control row. Small observed delta is below adoption threshold; no cold-full or cache-hit claim.')
selector=blob('227a5c5d','workflow/tools/decide.py');need(sha(selector)=='fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac','unchanged selector')
ss={'__name__':'unchanged_selector'};exec(compile(selector,'immutable-decide','exec'),ss)
ledger=[parse(x) for x in blob('227a5c5d','research/v2/ledger.jsonl').splitlines() if x.strip()]
need(ss['invalid_reason'](row) is None,'valid candidate accounting')
out=dict(status='AUTHENTICATED_LR_RESULT_NOT_ADOPTED',checks=len(ns['checks']),facts=facts,evaluation=ev,checkpoint=ck,historical_same_host_bpb=0.9882310747242088,observed_delta=ev['val_bpb']-0.9882310747242088,formal_selector=ss['decide_arm'](ledger+[row],row['arm_id']),ledger_candidate=row,ledger_written=False,source_changed=False,submission_ready=False)
print(json.dumps(out,indent=2,allow_nan=False))
