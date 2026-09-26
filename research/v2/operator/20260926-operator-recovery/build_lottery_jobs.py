"""Two human-selected non-kernel arms; mechanical packaging, no host calls."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
JOBS=HERE/'manual-jobs'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def enc(obj): return (json.dumps(obj,indent=2)+'\n').encode()
def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:
            assert not any((path.parent/n).exists() for n in ('attempt.json','dispatch.json','results-bundle.json')),'CONSUMED_ARTIFACT_IMMUTABLE'
            path.write_bytes(data)
    else: path.write_bytes(data)
manifest=json.loads((JOBS/'manifest.json').read_bytes())
clean=REPO/'workflow/candidates/final-batch262144-lr021'
cache_keys={'NEURON_COMPILE_CACHE_DIR':'compiler','TORCH_NEURONX_NEFF_CACHE_DIR':'neff',
 'TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR':'local','TORCH_NEURONX_HLO_CACHE_DIR':'hlo',
 'TORCHINDUCTOR_CACHE_DIR':'inductor','TORCH_NEURONX_DEBUG_DIR':'debug'}
for host,arm,parent_key in [('trn22','compilerdefault5','trn22-muon150'),('trn21','a1framework','trn21-finalcold')]:
    spec=REPO/'research/v2/proposals'/('20260926-compiler-default' if arm=='compilerdefault5' else '20260926-a1-framework-lottery')
    source=(clean/'train.py' if arm=='compilerdefault5' else spec/'train.py').read_bytes()
    expected=('cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d' if arm=='compilerdefault5'
              else 'b11c9fc54b61e003a2bfee00c829d642e75a0803da2af445c0ad73ce9b0dfe72')
    assert sha(source)==expected
    cpu=(clean/'CPU_RECEIPT.json' if arm=='compilerdefault5' else spec/'CPU_RECEIPT.json').read_bytes()
    receipt=json.loads(cpu)
    assert receipt.get('candidate_sha256',receipt.get('source_sha256'))==expected
    assert receipt['status'] in ('PASS','PASS_REDUCED_CPU')
    parent_raw=(JOBS/manifest[parent_key]['packet']).read_bytes()
    assert sha(parent_raw)==manifest[parent_key]['packet_sha256']
    parent=json.loads(parent_raw);old=parent['binding']
    originals={k:base64.b64decode(v['b64'],validate=True) for k,v in parent['files'].items()}
    assert all(sha(v)==parent['files'][k]['sha256'] for k,v in originals.items())
    key=host+'-'+arm
    label='fs-20260926-batch-lr021-'+arm+'-'+host+'-r1'
    root=str(Path(old['root']).parent/(label+'-packet'))
    command=(clean/'launch-command.txt').read_text().strip()
    if arm=='compilerdefault5': command+=' --num-steps 5'
    else: command=command.replace('--no-ngram-ve','--ngram-ve')+' --no-k2-embedding'
    command_raw=(command+'\n').encode()
    assignment=(spec/'SPEC.md').read_bytes()
    proposal=copy.deepcopy(json.loads(originals['EXPERIMENT_PROPOSAL.json']))
    proposal.update(proposal_id=label,arm_id=arm,kind='compiler_timing' if arm=='compilerdefault5' else 'architecture_arm',
      status='HUMAN_SELECTED_FLAG_SCALAR_VARIANT_CPU_PASS_PENDING_MACHINE',
      authority='Latest direct Poojith: A compiler default cold5 next free host; B standard A1 other host',
      single_causal_change='Training compiler optlevel1 override removed' if arm=='compilerdefault5' else 'Last-layer FP32 b7/t7 tables, standard framework backward',
      hypothesis='Measure net-token benefit of compiler default' if arm=='compilerdefault5' else 'Measure fixed-budget BPB from last-layer lexical memory on eight-layer batch/LR',
      review={'scope':'Unchanged reviewed functions; human-selected constants/CLI/environment only; no fresh independent review claimed'},
      submission_authorized=False)
    proposal['candidate'].update(source_sha256=expected,launch_command=command,launch_command_sha256=sha(command_raw),
      launch_string_sha256=sha(command.encode()),argv=command.split('train.py ',1)[1].split(),
      batch_size=262144,matrix_lr=.021,warmdown_ratio=.75,muon_momentum_warmup_steps=300)
    for name in ('verification','implementation','source_review','cpu_gate','g2'):
        if name in proposal: proposal[name]={'inherited_claim_removed':True,'scope':'Exact CPU_RECEIPT plus unchanged-function evidence; native pending'}
    proposal['dispatch'].update(selected_host=host,selected_label=label,ready=True,stage_authorized=True,launch_authorized=True,
      operator='/root sole Operator',order='After actual predecessor terminal/idle; preserve current jobs')
    proposal['control']['comparison']='Same-host actual batch262144/LR.021/ordinary.75/momentum300; keep historical ledger roles'
    proposal['accounting']={'budget':1800,'startup_cap':900,'cold':True,'screen_steps':5 if arm=='compilerdefault5' else None}
    proposal['qualification']='cold startup<=950s or measured steady gain outweighs startup charge; then distinct full run' if arm=='compilerdefault5' else 'Final fixed1800s freshpublic20M/context1024; no step-count/speed gate'
    proposal_raw=enc(proposal)
    provenance={'label':label,'train_sha256':expected,'source_sha256':expected,'launch_command':command,
      'base_sha256':old['base_sha256'],'submission_authorized':False,'human_selected_flag_arm':True}
    prov_raw=enc(provenance)
    replacements=[(old['root'],root),(old['label'],label),(old['source_sha256'],expected),
      (sha(originals['package/launch-command.txt']),sha(command_raw)),(sha(originals['package/provenance.json']),sha(prov_raw))]
    replacements=[(a,b) for a,b in replacements if a!=b]
    def transform(value):
        if isinstance(value,str):
            for before,after in replacements: value=value.replace(before,after)
            return value
        if isinstance(value,dict): return {k:transform(v) for k,v in value.items()}
        if isinstance(value,list): return [transform(v) for v in value]
        return value
    driver=transform(originals['host_driver.py'].decode())
    reverse=driver
    for before,after in reversed(replacements): reverse=reverse.replace(after,before)
    assert reverse==originals['host_driver.py'].decode()
    binding=transform(copy.deepcopy(old))
    env=binding['train_environment']
    for k,suffix in cache_keys.items(): env[k]=root+'/cold-training-caches/'+suffix
    env['NEURON_COMPILE_CACHE_URL']=env['NEURON_COMPILE_CACHE_DIR']
    if arm=='compilerdefault5': env['NEURON_CC_FLAGS']='--auto-cast matmult --auto-cast-type bf16'
    binding['fresh_cache_directories']=[env[k] for k in cache_keys]
    # Change ONLY literal TRAIN_ENV, not evaluator settings or driver guards.
    lines=driver.splitlines(keepends=True)
    nodes=[n for n in ast.parse(driver).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRAIN_ENV' for t in n.targets)]
    assert len(nodes)==1 and nodes[0].lineno==nodes[0].end_lineno
    lines[nodes[0].lineno-1]='TRAIN_ENV = '+repr(env)+'\n'
    driver=''.join(lines)
    if arm=='compilerdefault5':
        begin=driver.index('    eval_fd=guard.require_idle_under_inherited_lock(')
        end=driver.index('except BaseException as exc:',begin)
        assert "rc=phase('fresh-public20m',EVAL_COMMAND,EVAL_ENV,guardian)" in driver[begin:end]
        driver=driver[:begin]+"    record('terminal.json',{'utc':utc(),'label':LABEL,'status':'TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS','train_exit_code':rc,'no_eval':True,'submission_authorized':False})\n"+driver[end:]
    ast.parse(driver)
    # Decoded driver values must equal the exact runtime binding.
    train_node=next(n for n in ast.parse(driver).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRAIN_ENV' for t in n.targets))
    assert ast.literal_eval(train_node.value)==env
    binding.update(source_commit=None,source_sha256=expected,launch_command=command,
      driver_sha256=sha(driver.encode()),kind=proposal['kind'],arm_id=arm,
      completed_proposal_path=str((JOBS/key/'EXPERIMENT_PROPOSAL.json').relative_to(REPO)),
      completed_proposal_sha256=sha(proposal_raw),operator_assignment_sha256=sha(assignment),
      independent_review_sha256=None,stage_authorized=True,launch_authorized=True,submission_authorized=False,
      scientific_patch=proposal['single_causal_change'],human_selected_flag_scalar_variant=True,
      no_eval=arm=='compilerdefault5',steps=5 if arm=='compilerdefault5' else None)
    for field in ('first_source_native_evidence','first_source_native_evidence_sha256',
                  'scalar_scope_sha256','source_adoption','native_short_timing','no_new_logic_human_scalar_authorization'):
        binding.pop(field,None)
    binding.update(bound_driver_sha256=sha(driver.encode()),first_source_native_timing_pending=True,
      run_purpose='cold_startup5' if arm=='compilerdefault5' else 'full_quality',
      cache_mode='cold_private',cache_owner_label=label,cache_owner_root=root,
      selected_cache_policy='Six private empty directories; no warm startup claim',release_file_created=True)
    files=dict(originals)
    scope=b'Human-selected flag/scalar variant; no fresh independent review. Exact CPU evidence attached. Original lifetime guards unchanged. Native unknown.\n'
    for k in ('RELEASE.json','NATIVE_BINDING.json','INDEPENDENT_REVIEW_RECEIPT.json','PARENT_WARMDOWN_REVIEW_ONLY.md'):
        files.pop(k,None)
    files.update({'host_driver.py':driver.encode(),'package/train.py':source,'package/launch-command.txt':command_raw,
      'package/provenance.json':prov_raw,'EXPERIMENT_PROPOSAL.json':proposal_raw,'CPU_RECEIPT.json':cpu,
      'EXACT_SOURCE_REVIEW.md':scope,'SUPERVISOR_ASSIGNMENT.md':assignment,'NATIVE_BINDING.json':enc(binding),
      'RELEASE.json':enc({'label':label,'source_sha256':expected,'launch_authorized':True,'submission_authorized':False,
      'assignment_sha256':sha(assignment),'assignment_path':str((spec/'SPEC.md').relative_to(REPO)),
      'guardian_sha256':sha(files['lifetime_guard.py']),'independent_review_sha256':None,'human_flag_variant':True,'no_automatic_next_job':True})})
    for name in ('lifetime_guard.py','idle_guard.py','dispatch_driver.py'): assert files[name]==originals[name]
    packet=enc({'binding':binding,'files':{k:{'sha256':sha(v),'b64':base64.b64encode(v).decode()} for k,v in files.items()},
      'derivation':{'parent_packet_sha256':sha(parent_raw),'guards_unchanged':True,'host_calls':0,
      'driver_changes':'Literal source/label/hash/TRAIN_ENV rebind; timing variant omits eval block only'}})
    write_once(JOBS/key/'packet.json',packet);write_once(JOBS/key/'EXPERIMENT_PROPOSAL.json',proposal_raw)
    manifest[key]={'host':host,'order':8,'predecessor':parent_key,'label':label,'packet':key+'/packet.json',
      'packet_sha256':sha(packet),'source_sha256':expected,'command':command,'source_review':'UNCHANGED_FUNCTIONS_HUMAN_FLAG_VARIANT_NOT_FRESH_REVIEW',
      'machine_admission':'PENDING','status':'READY_PENDING_PREDECESSOR_AND_FRESH_MACHINE'}
(JOBS/'manifest.json').write_bytes(enc(manifest))
print(json.dumps({'prepared':['trn22-compilerdefault5','trn21-a1framework'],'host_calls':0}))
