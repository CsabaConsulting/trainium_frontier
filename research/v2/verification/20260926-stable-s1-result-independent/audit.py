"""Read-only immutable raw-result authentication; no host/data/model execution."""
import argparse
import copy
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shlex
import stat
import subprocess

REPO=Path('/home/poojith-devan/Desktop/Trainium')
RAW='190ccf4a';SHARDS='4b2059af';PACKET='3edbc186353891983ed8d02005fa16d4583830c6'
BASE='research/v2/operator/20260926-trn22-stable-source-pair/completed/'
SP='research/v2/operator/20260926-training-shard-provenance/'
PP='research/v2/proposals/20260926-architecture-stable-source-pair/'
checks=[]
def check(name,value):
    if not value:raise RuntimeError(name)
    checks.append(name)
def raw(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=REPO)
def sha(data):return hashlib.sha256(data).hexdigest()
def jsonraw(ref,path):return json.loads(raw(ref,path))
def date(s):return dt.datetime.fromisoformat(s)
bundle_bytes=raw(RAW,BASE+'results-bundle.json');bundle=json.loads(bundle_bytes);files=bundle['files']
check('exact count/bytes',len(files)==bundle['file_count']==26 and sum(x['bytes'] for x in files.values())==bundle['raw_bytes']==90280)
for name,entry in files.items():check('raw '+name,len(entry['text'].encode())==entry['bytes'] and sha(entry['text'].encode())==entry['sha256'])
def doc(name):return json.loads(files[name]['text'])
for name in ('PAIR_BINDING.disabled.json','MANIFEST.sha256.json','EXPERIMENT_PROPOSAL.json'):
    check('frozen packet '+name,files[name]['text'].encode()==raw(PACKET,PP+name))
binding=doc('PAIR_BINDING.disabled.json');phase=binding['phases'][0];pre=doc('preflight.json');launch=doc('S1-launch.json')
check('plan and source pin',files['PAIR_BINDING.disabled.json']['sha256']=='518c039d570d02b5ac34a4649b866465e18df834061a992949841ca0b8981cca' and sha(raw(binding['source_commit'],binding['source_git_path']))==binding['source_sha256']=='3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c')
for name in ('host_driver.py','identity_checks.py','s1/package/launch-command.txt','s1/package/provenance.json','s2/package/launch-command.txt','s2/package/provenance.json'):
    check('staged frozen '+name,sha(raw(PACKET,PP+name))==pre['staged_hashes'][name])
check('source preflight',pre['stable_source_sha256']==binding['source_sha256'] and pre['stable_prepare_sha256']==binding['prepare_sha256'])
check('exact S1 command/env',all(launch[k]==phase[k] for k in ('label','command','training_command','environment')))
check('unchanged science and bounded5',launch['training_command'][launch['training_command'].index('--num-steps')+1]=='5' and launch['training_command'][launch['training_command'].index('--max-train-seconds')+1]=='1800' and '--no-eval-public' in launch['training_command'] and '--recursive-schedule' not in launch['training_command'] and '--startup-adjusted-schedule' not in launch['training_command'])
before=doc('S1-pre-source-identity.json');after=doc('S1-post-source-identity.json')
check('all source identities unchanged',before==after)
for k in ('executed_train','package_train'):check('source identity '+k,before[k]['sha256']==binding['source_sha256'] and stat.S_IMODE(before[k]['st_mode'])==0o444 and before[k]['st_nlink']==1)
for k in ('expected_imported_prepare','protected_prepare'):check('prepare identity '+k,before[k]['sha256']==binding['prepare_sha256'])
check('stable absolute source path',before['executed_train']['path']==launch['training_command'][3] and before['expected_imported_prepare']['path']==launch['expected_prepare_path'])
check('actual import origin remains unobserved',launch['actual_import_origin_observed'] is False)
check('harness identity',before['harness']['sha256']==binding['harness_sha256']=='7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b')
check('package launch identity',before['launch']['sha256']==phase['launch_sha256'] and before['provenance']['sha256']==phase['provenance_sha256'])
log=files['s1/training.log']['text'];lines=log.splitlines()
iso=json.loads(next(x.removeprefix('ISOLATED_SUBMISSION ') for x in lines if x.startswith('ISOLATED_SUBMISSION ')))
check('harness source and launch confirmation',iso['train_sha256']==binding['source_sha256'] and iso['prepare_sha256']==binding['prepare_sha256'] and shlex.split(iso['launch_command'])==launch['training_command'])
check('actual torchrun command',json.loads(next(x.removeprefix('REHEARSAL_COMMAND ') for x in lines if x.startswith('REHEARSAL_COMMAND ')))==launch['training_command'])
step_rows=re.findall(r'^step (\d+) \| loss (\S+) \| lrm (\S+) \| dt (\S+)s \|',log,re.M)
check('five finite printed rows',[int(x[0]) for x in step_rows]==list(range(5)) and all(math.isfinite(float(v)) for row in step_rows for v in row[1:]))
losses=[float(row[1]) for row in step_rows]
check('official step file',[json.loads(x) for x in files['s1/train/steps.jsonl']['text'].splitlines()]==list(range(5)))
rehearsal=doc('s1/train/rehearsal.json');obs=rehearsal['observations'];fifth=obs[4]['observed_seconds'];wall=rehearsal['wall_seconds'];charge=wall-min(fifth,900)
check('five ordered observations',[x['step'] for x in obs]==list(range(5)) and all(math.isfinite(x['observed_seconds']) for x in obs) and all(a['observed_seconds']<b['observed_seconds'] for a,b in zip(obs,obs[1:])))
check('S1 exits and mode',rehearsal['returncode']==doc('S1-exit.json')['returncode']==0 and rehearsal['mode']=='train' and rehearsal['step_boundaries']==5)
check('measured startup gate',fifth==845.5936225099722 and fifth<=900 and rehearsal['startup_allowance_observed']==min(fifth,900))
check('measured through-exit charge',wall==849.8144521720242 and abs(charge-4.220829662051983)<1e-10 and rehearsal['charged_through_process_exit']==charge and rehearsal['charged_through_process_exit_within_budget'] is True and charge<=1800)
check('retained checkpoint receipt',rehearsal['checkpoint_sha256']=='a4e11a3d781dd85667e69abbc5ceba9a23cbf57eeb5762a38dff5c3984ba0a07' and rehearsal['checkpoint_bytes']==453048594 and bundle['no_checkpoint_copy'] is True)
check('cold cache paths',len(rehearsal['fresh_empty_sdk_caches'])==6 and len(set(rehearsal['fresh_empty_sdk_caches'].values()))==6 and rehearsal['fresh_empty_sdk_caches']==iso['fresh_empty_sdk_caches'] and all(launch['environment'][k]==v and v.startswith(phase['sandbox']+'/caches/') for k,v in rehearsal['fresh_empty_sdk_caches'].items()))
check('fresh flag and URL DIR','--fresh-caches' in launch['command'] and launch['environment']['NEURON_COMPILE_CACHE_URL']==launch['environment']['NEURON_COMPILE_CACHE_DIR'])
owners=[json.loads(x.removeprefix('SUBMISSION_OPTIMIZER_INTEGRATION ')) for x in lines if x.startswith('SUBMISSION_OPTIMIZER_INTEGRATION ')]
check('four training ranks present',{x['rank'] for x in owners}==set(range(4)) and len(owners)==4 and 'num_params=113,256,724' in log and 'over_budget:       False' in log)
for name in ('under-lock-idle.json','S1-pre-under-lock-idle.json','S1-post-under-lock-idle.json'):
    g=doc(name);check('same lock/idle '+name,g['status']=='PASS' and g['lock']['verified_existing_exclusive_lock'] is True and g['lock']['inode']==pre['lock_inode']==bundle['lock_inode']==832589 and g['lock']['device']==pre['lock_device']==bundle['lock_device']==66305 and g['lock']['path']==binding['device_lock'] and g['ps']['returncode']==g['neuron_ls']['returncode']==0 and g['ps']['busy']==g['neuron_ls']['process_pids']==[] and g['ps']['stderr']==g['neuron_ls']['stderr']=='')
terminal=doc('terminal.json');outer=doc('outer-flock-exit.json')
check('pair remains refused',terminal['status']=='PAIR_REFUSED_OR_INCOMPLETE' and terminal['error']=='MATCHED_PRINCIPAL_COMPILER_BYTES_MISSING' and all(terminal[k] is True for k in ('no_S2_fallback','no_automatic_full','no_eval','no_retry')) and outer['returncode']==1 and outer['command']==binding['outer_command'])
check('no S2 execution artifacts',not any(name.startswith(('S2-','s2/')) for name in files))
compiler=doc('S1-compiler-evidence.json');graph_summary={}
check('cache proof unresolved',compiler['exact_input_bytes_complete'] is compiler['cache_hit'] is compiler['zero_compile_us_is_hit_proof'] is False and compiler['native_neff_reuse']=='UNPROVEN' and compiler['authenticated_raw_inputs']=={} and compiler['missing_keys']==compiler['principal_keys'] and len(compiler['missing_keys'])==2)
for rank in range(4):
    name=f's1/compiler-evidence/metric_table_graph_stats_rank{rank}.csv';rows=list(csv.DictReader(io.StringIO(files[name]['text'])));record=compiler['graphs'][str(rank)]
    check('graph raw matches summary '+str(rank),record['rows']==rows and record['file']['sha256']==files[name]['sha256'] and record['file']['st_size']==files[name]['bytes'])
    graph_summary[str(rank)]={'rows':len(rows),'compile_us':sum(int(x['torch_neuronx_compile_us']) for x in rows),'principal_keys':[x['graph_id'].split(':')[0] for x in rows if '_forward_' in x['graph_name'] or '_backward_' in x['graph_name']]}
    check('same principal keys '+str(rank),graph_summary[str(rank)]['principal_keys']==compiler['principal_keys'])
post=jsonraw(RAW,BASE+'explicit-process-postflight.json');live=jsonraw(RAW,BASE+'live-compiler-observations.json');pids=set(post['observed_pids'])
check('26 PID absence records',len(pids)==len(post['observed_pids'])==len(post['pid_rows'])==26 and {x['pid'] for x in post['pid_rows']}==pids and all(x['exists'] is False for x in post['pid_rows']))
check('explicit postflight quiet',post['all_observed_pids_absent'] is post['neuron_device_idle'] is post['no_kills_or_changes'] is True and post['related_rows']==[] and post['ps_rc']==post['lock_competitor_rc']==0 and post['ps_stderr']==post['neuron_ls_stderr']=='' and all(not x['neuron_processes'] for x in post['neuron_ls']))
for entry in live:
    check('observed active PIDs included '+entry['utc'],set(entry['worker_pids'])|{int(x.split()[0]) for x in entry['compiler_process_rows']}<=pids)
check('missing live MLIR evidence remains explicit',all(x['no_module_mlir_bytes_or_directory_inventory_collected'] is True for x in live))
check('postflight follows terminal',date(post['utc'])>date(outer['ended_utc'])>date(terminal['utc'])>date(doc('S1-exit.json')['utc'])>date(launch['utc']))
check('closed retrieval quiet',bundle['job_lines']==bundle['workers']==[] and bundle['neuron_ls_rc']==bundle['lock_competitor_rc']==0 and all(not x['neuron_processes'] for x in json.loads(bundle['neuron_ls'])) and all(x['exists'] is False for x in bundle['old_queue_pids']))

# Decode original transport stdout with Python integers. Convenience JSON copies
# rounded nanosecond-sized values; they are NOT our exact stat source.
shards={};rounded=[];shard_pins={}
for host in ('trn21','trn22'):
    transport_bytes=raw(SHARDS,SP+host+'-transport.json');transport=json.loads(transport_bytes)
    if host=='trn22':
        check('outer trn22 transport exit',transport['exit_code']==0);transport=json.loads(transport['output'])
    check('transport success '+host,transport['rc']==0 and transport['stderr']=='')
    exact=json.loads(transport['stdout']);summary=jsonraw(SHARDS,SP+host+'.json');shards[host]=exact;shard_pins[host]={'transport_sha256':sha(transport_bytes),'raw_stdout_sha256':sha(transport['stdout'].encode())}
    expected_paths=[f'/home/ubuntu/.cache/neuron_competition_r1/data/shard_{i:05d}.parquet' for i in range(5)]
    check('only five current files '+host,exact['host']==host and exact['status']=='COMPLETE_SHA256_ONLY' and [x['path'] for x in exact['files']]==expected_paths and exact['aggregate_bytes']==sum(x['before']['st_size'] for x in exact['files'])==459442478)
    check('read-only hash idle receipt '+host,exact['lock_fds']==[3] and exact['ps_busy']==[] and all(not x['neuron_processes'] for x in exact['neuron_ls']) and exact['buffer_bytes']==65536)
    for a,b in zip(exact['files'],summary['files']):
        check('exact before/after '+host+'/'+a['path'],a['before']==a['after'] and stat.S_ISREG(a['before']['st_mode']) and a['before']['st_nlink']==1 and re.fullmatch('[0-9a-f]{64}',a['sha256']) is not None)
        check('summary hash/size '+host+'/'+a['path'],a['path']==b['path'] and a['sha256']==b['sha256'] and a['before']['st_size']==b['before']['st_size'])
        for moment in ('before','after'):
            for key,value in a[moment].items():
                if b[moment][key]!=value:
                    check('only ns summary rounding',key in ('st_mtime_ns','st_ctime_ns') and abs(b[moment][key]-value)<=256)
                    rounded.append({'host':host,'path':a['path'],'moment':moment,'field':key,'exact_transport_value':value,'summary_value':b[moment][key]})
comparison=[]
for a,b in zip(shards['trn21']['files'],shards['trn22']['files']):
    check('cross-host current bytes '+a['path'],a['path']==b['path'] and a['sha256']==b['sha256'] and a['before']['st_size']==b['before']['st_size'])
    comparison.append({'path':a['path'],'sha256':a['sha256'],'bytes':a['before']['st_size']})
check('shard scope chronology',date(shards['trn22']['started_utc'])>date(post['utc']) and shards['trn22']['lock_inode']==832589 and shards['trn21']['lock_inode']==596823)

result={'status':'AUTHENTICATED_S1_COLD_MEASUREMENT_ONLY_PAIR_REFUSAL_PRESERVED','checks':len(checks),'raw_commit':RAW,'shard_commit':SHARDS,'raw_bundle_sha256':sha(bundle_bytes),'raw_files':26,'raw_bytes':90280,'S1':{'source_sha256':binding['source_sha256'],'printed_losses':losses,'uncapped_fifth_seconds':fifth,'torchrun_process_wall_seconds':wall,'startup_allowance_seconds':min(fifth,900),'charged_through_exit_seconds':charge,'checkpoint_sha256':rehearsal['checkpoint_sha256'],'checkpoint_bytes':rehearsal['checkpoint_bytes'],'actual_import_origin_observed':False},'pair_status':terminal['status'],'pair_error':terminal['error'],'S2_launched':False,'native_cache_reuse':'UNRESOLVED','graph_summary':graph_summary,'explicit_postflight_pid_count':26,'training_shards':{'status':'FIVE_CURRENT_FILES_SAME_HASH_AND_SIZE','aggregate_bytes':459442478,'files':comparison,'transport_pins':shard_pins,'summary_ns_rounding':rounded,'historical_runtime_identity_proven':False,'loader_order_proven':False},'quality_or_control_claim':False,'no_hosts':True,'no_tensors':True,'no_source_or_ledger_edits':True,'checks_passed':checks}
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args();encoded=json.dumps(result,indent=2)+'\n'
if args.output:args.output.write_text(encoded)
print(json.dumps({k:v for k,v in result.items() if k not in ('checks_passed','training_shards')},indent=2))
