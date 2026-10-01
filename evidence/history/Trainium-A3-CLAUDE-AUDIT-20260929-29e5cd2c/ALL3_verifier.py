"""Read-only stdlib ALL3 qualification; no export, host calls or model imports."""
import argparse, hashlib, importlib.util, json, math, re, sys
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;R=HERE.parents[3];OP=R/'research/v2/operator'
SOURCE='c5ef8414e2df998a9b52eeb2e249b66c0110a3c37869e676719e93975a3432b4'
COMMAND='NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0'
RUNS=[('first','trn21','20260929-r71-a3-full-trn21-preparation','trn21-r71-a3-full-r1','d3b6085b37bbf9fe1f92228e1bf66f7168ddf30a331665fc1d7d9750cf44805d','d38a9edd82b3c451a2f01825a404281a87c3c1fd9a3c03a1b73b4b9656c55e4b'),('repeat21','trn21','20260929-r72-a3-best-single-filler-full-trn21-preparation','trn21-r72-a3-best-single-filler-full-r1','f2acb653b619ac9a665a459c1dc6c2921d0e603ee320d192aaf5746c7e631d20',None),('repeat22','trn22','20260929-r72-a3-filler22-retry2-full-trn22-preparation','trn22-r72-a3-filler22-retry2-full-r1','fa9e809ebfdc5fbaf264552119955123b08924ed62fa41b15473ebc7b4f250d0',None)]
def sha(b):return hashlib.sha256(b).hexdigest()
def checked(p,pin):
 b=Path(p).read_bytes();assert sha(b)==pin,'PIN:'+str(p);return b
def load(relative,pin,name):
 p=OP/relative;checked(p,pin);s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def helpers():
 return (load('20260929-r62-d2-winner-confirmation-preparation/common.py','ecbf0cc34453d8a2fdb75e32fe0b2218c5cb5b27fa814cefa630860c80b154ef','a3_all3_common'),load('20260929-r62-d2-continue-full-trn21-preparation/evidence.py','8274c8824caa70f2f527c0bdfa0ea5e0943ed25e9eac30abfb7ce78e2a22c2b9','a3_all3_score'))
def authenticate(run,pin):
 slot,host,folder,key,packetpin,_=run;path=OP/folder/'artifacts/results-bundle.json';packet=OP/folder/'artifacts/packet.json'
 common,evidence=helpers();rel=lambda p:str(p.relative_to(R))
 inputs=dict(actual_predecessor_key=key,actual_predecessor_packet_path=rel(packet),actual_predecessor_packet_sha256=packetpin,actual_predecessor_archive_path=rel(path),actual_predecessor_archive_sha256=pin)
 proof=common.original_support(OP/folder,host).predecessor(inputs)
 assert proof['terminal_status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and proof['original_guardian_released'] is True
 a=evidence.archive(rel(path),pin,SimpleNamespace(checked=checked,sha=sha));f=a['files'];get=lambda n:json.loads(f[n]['text'])
 b=get('NATIVE_BINDING.json');assert b['source_sha256']==SOURCE and b['launch_command']==COMMAND and b['steps'] is None and b['no_eval'] is False
 assert sha(f['package/train.py']['text'].encode())==SOURCE and f['package/launch-command.txt']['text'].strip()==COMMAND
 cfg=b['scientific_configuration'];assert cfg['tbl_freeze_tokens']==0 and cfg['tbl_every']==2 and cfg['table_rows']==262144 and cfg['table_width']==256 and cfg['table_gradient_sink']=='FP32' and cfg['layer7_projection'] is False
 assert b['train_environment']['NEURON_COMPETITION_R1_SEED']=='42';assert b['budget']['source_seconds']==1800 and b['budget']['startup_cap_seconds']==900
 log=f['training.log']['text'];recorded,_=json.JSONDecoder().raw_decode(log.split('model_config=',1)[1]);assert recorded['tbl_freeze_tokens']==0 and recorded['tbl_every']==2 and recorded['d2_two_site'] is True and recorded['n_embd']==1024 and recorded['n_head']==8 and recorded['sequence_len']==1024
 runlog=get('train/rehearsal.json');assert runlog['returncode']==0 and runlog['charged_through_process_exit_within_budget'] is True
 fifth=runlog['observations'][4];assert fifth['step']==4;cold=fifth['observed_seconds'];charged=runlog['charged_through_process_exit'];assert math.isfinite(cold) and 0<cold<=900;assert math.isfinite(charged) and 0<charged<=1800
 caches=get('operator-preflight.json')['fresh_cache_absent_then_created_empty'];assert len(caches)==6 and all(x['created_empty'] is True for x in caches) and {x['path'] for x in caches}==set(b['fresh_cache_directories'])
 cp=get('checkpoint.json');after=get('checkpoint-after-postflight.json');stable=a['checkpoint_after_terminal']
 assert cp['bytes']>0 and stable['stable'] is True
 assert all((v['sha256'],v['bytes'])==(cp['sha256'],cp['bytes']) for v in (after,stable))
 assert (runlog['checkpoint_sha256'],runlog['checkpoint_bytes'])==(cp['sha256'],cp['bytes'])
 score=evidence.public_score(a,SOURCE)
 cmds=[json.loads(line.split(' ',1)[1]) for line in f['fresh-public20m.log']['text'].splitlines() if line.startswith('REHEARSAL_COMMAND ')];assert len(cmds)==1
 for k,v in [('--seq-len','1024'),('--eval-tokens','20971520'),('--device','neuron'),('--checkpoint',a['root']+'/train/out/final.pt')]:assert cmds[0][cmds[0].index(k)+1]==v
 rank=re.findall(r'^D2_TABLE_RANK_CONSISTENCY world_size=(\d+) max_abs_diff_vs_rank0=(\S+) method=(\S+)',log,re.M)
 assert len(rank)==1 and rank[0][0]=='4' and math.isfinite(float(rank[0][1])) and float(rank[0][1])==0 and rank[0][2]=='cloned_rank0_broadcast_host_fp32_max_scalar_allreduce'
 return dict(slot=slot,status='PASS',host=host,key=key,archive_sha256=pin,source_sha256=SOURCE,seed=42,tbl_freeze_tokens=0,val_bpb=score,raw_cold_seconds=cold,charged_seconds=charged,checkpoint_sha256=cp['sha256'],checkpoint_bytes=cp['bytes'],fresh_tokens=20971520,context=1024,six_empty_caches=True,original_guardian_released=True,finite_exact_zero_four_rank_table_diagnostic=True,authenticated_members=len(f))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repeat21-sha');ap.add_argument('--repeat22-sha');args=ap.parse_args()
 q=load('20260929-r71-table-full-preparation/evidence.py','89bca3b7d308d998d804c3603d886d4a4101fc4ea2f0c4d6f003e4df3d918515','a3_q2');review,_=q.review('a3',q.independent('a3'))
 out=[]
 for run,pin in zip(RUNS,[RUNS[0][-1],args.repeat21_sha,args.repeat22_sha]):
  p=OP/run[2]/'artifacts/results-bundle.json'
  if pin is None or not p.exists():out.append(dict(slot=run[0],status='PENDING',archive_exists=p.exists(),reason='Actual archive AND explicit Root SHA required'));continue
  try:out.append(authenticate(run,pin))
  except (AssertionError,KeyError,ValueError) as e:out.append(dict(slot=run[0],status='FAIL',reason=type(e).__name__+': '+str(e)))
 complete=all(x['status']=='PASS' for x in out);mean=sum(x['val_bpb'] for x in out)/3 if complete else None
 if complete:assert len({x['archive_sha256'] for x in out})==3
 status='PASS' if complete and mean<=.97225 else ('FAIL' if any(x['status']=='FAIL' for x in out) or complete else 'PENDING')
 q2={k:review[k] for k in ('status','path','sha256','original_review_status','original_review_bytes_unchanged','authority','original_source_gate_pass','tolerance_waiver')}
 print(json.dumps(dict(status=status,runs=out,all3_mean=mean,delta_vs_D2_mean=(mean-.9725512952419128) if complete else None,adoption_threshold=.97225,D2_control_mean=.9725512952419128,Q2_effective_resolution=q2,original_Q2_failure_reclassified=False,submission_tolerance_waived=False,scope='Archive qualification only; same-seed repeats, not independent seeds',ready_package=False,export_performed=False,launch_authorized=False,upload_authorized=False),indent=2))
if __name__=='__main__':main()
