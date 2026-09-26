"""Disabled direct P3 diagnostic bindings; existing guardian, no remote calls."""
import ast, base64, hashlib, json
from pathlib import Path
here = Path(__file__).resolve().parent
repo = Path('/home/poojith-devan/Desktop/Trainium')
source = (repo/'research/v2/lanes/p3-component-timing/train.py').read_bytes()
spec = (repo/'research/v2/lanes/p3-component-timing/SPEC.md').read_bytes()
review = (repo/'research/v2/verification/20260926-p3-sprint-independent/REVIEW.md').read_bytes()
prepare = (repo/'prepare.py').read_bytes()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def require(value, reason):
    if not value: raise RuntimeError(reason)
require(sha(source)=='3637166b43a0b996e7eb7c71bb3b5d56e7f8119241b0278107a72787044dcc99','SOURCE')
require(sha(prepare)=='5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb','PREPARE')
require(sha(review)=='826268d46a4dc71ba893d4b382e5b62e56b78a7f5f095d9af53e1685320bb6da','REVIEW')
require(review.startswith(b'# PASS_FOR_NATIVE_DIAGNOSTIC'),'SOURCE_REVIEW_SCOPE')
rows=[]
for host in ('trn21','trn22'):
    original_raw=(here.parent/'20260926-final-sprint-p0'/(host+'-bound-packet.json')).read_bytes()
    original=json.loads(original_raw);ob=original['binding']
    old={name:base64.b64decode(entry['b64'],validate=True) for name,entry in original['files'].items()}
    for name,raw in old.items():require(sha(raw)==original['files'][name]['sha256'],'PARENT_PIN:'+name)
    label='fs-20260926-p3-component-'+host+'-r1'
    root=str(Path(ob['root']).parent/(label+'-packet'))
    output=root+'/diagnostic-output'
    argv=['--standalone','--nproc_per_node=4','train.py','--p3-component-timing','--device-type','neuron','--seed','42','--p3-max-seconds','900','--p3-blocks','4','--p3-repeats','3','--out-dir',output]
    launch='NEURON_LOGICAL_NC_CONFIG=2 torchrun '+' '.join(argv)
    launch_raw=(launch+'\n').encode()
    command=['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=15s','930s','/home/ubuntu/workspace/native_venv/bin/torchrun']+argv
    provenance={'label':label,'train_sha256':sha(source),'launch_command':launch,'diagnostic_only':True,'no_checkpoint':True,'no_evaluation':True,'no_training_steps_or_budget_claim':True,'submission_authorized':False}
    provenance_raw=(json.dumps(provenance,indent=2)+'\n').encode()
    text=old['host_driver.py'].decode().replace(ob['root'],root).replace(ob['label'],label).replace(ob['source_sha256'],sha(source))
    text=text.replace(sha(old['package/launch-command.txt']),sha(launch_raw)).replace(sha(old['package/provenance.json']),sha(provenance_raw))
    env={key:value.replace(ob['root'],root).replace(ob['label'],label) for key,value in ob['train_environment'].items()}
    env.pop('NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS',None)
    env.pop('TORCH_NEURONX_ENABLED_METRIC_TABLES',None)
    env.pop('TORCH_NEURONX_METRICS_DIR',None)
    cache_keys={'NEURON_COMPILE_CACHE_URL':'compiler','NEURON_COMPILE_CACHE_DIR':'compiler','TORCH_NEURONX_NEFF_CACHE_DIR':'neff','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR':'local','TORCH_NEURONX_HLO_CACHE_DIR':'hlo','TORCHINDUCTOR_CACHE_DIR':'inductor','TORCH_NEURONX_DEBUG_DIR':'debug'}
    for key,leaf in cache_keys.items():env[key]=root+'/cold-diagnostic-caches/'+leaf
    tree=ast.parse(text);lines=text.splitlines(keepends=True)
    replacements={'TRAIN_COMMAND':repr(command),'TRAIN_ENV':repr(env),'EVAL_COMMAND':'None','EVAL_ENV':'None'}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in replacements:
                    require(node.lineno==node.end_lineno,'ONE_LINE_BINDING')
                    lines[node.lineno-1]=target.id+' = '+replacements[target.id]+'\n'
    text=''.join(lines)
    text=text.replace('"""Disabled exact Recursive-on-confirmed36 trn21 repeat with attached fresh20M."""','"""Disabled direct P3 component diagnostic; no scorer or training harness."""')
    # This is a direct invocation of the reviewed diagnostic source, not a
    # synthetic training run. Keep the existing process-tree guardian unchanged.
    text=text.replace("    assert sha(HARNESS)==HARNESS_SHA,'HARNESS_SHA'", "    assert sha(ROOT/'package/prepare.py')==PREP_SHA,'UNCHANGED_IMPORT_PREPARE_SHA'")
    text=text.replace("    phase_root = ROOT/('train' if env is TRAIN_ENV else 'eval')", "    phase_root = ROOT/'diagnostic-output'")
    text=text.replace("guardian.run(command,cwd=ROOT,env=env,out=out)","guardian.run(command,cwd=ROOT/'package',env=env,out=out)")
    text=text.replace("'harness_sha256':HARNESS_SHA", "'invocation_mode':'DIRECT_COMPONENT_DIAGNOSTIC_NO_HARNESS'")
    text=text.replace("assert not os.path.lexists(ROOT/'train'),'TRAIN_SANDBOX_ALREADY_EXISTS'", "assert not os.path.lexists(ROOT/'diagnostic-output'),'DIAGNOSTIC_OUTPUT_ALREADY_EXISTS'")
    text=text.replace("    assert not os.path.lexists(ROOT/'eval'),'EVAL_SANDBOX_ALREADY_EXISTS'\n",'')
    text=text.replace("ROOT/'lifetime-events.jsonl')", "ROOT/'lifetime-events.jsonl',grace=15.0,kill_wait=10.0)")
    begin=text.index("    checkpoint=ROOT/'train/out/final.pt'\n")
    end=text.index('\ndef independent_release_idle():',begin)
    text=text[:begin]+text[end:]
    begin=text.index("    rc=phase('training',TRAIN_COMMAND,TRAIN_ENV,guardian)")
    end=text.index('except BaseException as exc:',begin)
    phase="""    rc=phase('component-diagnostic',TRAIN_COMMAND,TRAIN_ENV,guardian)
    exit_code=rc if rc>=0 else 128-rc
    record('terminal.json',{'utc':utc(),'label':LABEL,'status':'DIRECT_DIAGNOSTIC_EXIT_PENDING_RAW_INTERPRETATION' if rc==0 else 'DIRECT_DIAGNOSTIC_FAILED','diagnostic_exit_code':rc,'no_checkpoint':True,'no_evaluation':True,'no_automatic_next_job':True,'submission_authorized':False})
"""
    text=text[:begin]+phase+text[end:]
    text=text.replace("'FULL_DRIVER_FAILED'","'DIRECT_DIAGNOSTIC_DRIVER_FAILED'")
    # Phase already writes its literal diagnostic name in logs; use an exit file
    # with the same name rather than the inherited unused eval filename.
    text=text.replace("('run-exit-code.txt' if name=='training' else 'eval-exit-code.txt')", "(name+'-exit-code.txt')")
    compile(ast.parse(text),label+'-driver','exec')
    binding={'schema':'disabled-p3-direct-component-diagnostic-v1','host':host,'hostname':ob['hostname'],'label':label,'root':root,'source_sha256':sha(source),'prepare_source_path':ob['prepare_path'],'prepare_sha256':sha(prepare),'spec_sha256':sha(spec),'independent_review_sha256':sha(review),'driver_sha256':sha(text.encode()),'guardian_sha256':sha(old['lifetime_guard.py']),'idle_guard_sha256':sha(old['idle_guard.py']),'dispatch_sha256':sha(old['dispatch_driver.py']),'lock_path':ob['lock_path'],'lock_identity':ob['lock_identity'],'outer_command':[value.replace(ob['root'],root) for value in ob['outer_command']],'diagnostic_command':command,'cwd':root+'/package','environment':env,'prospective_launch_command':launch,'output_directory':output,'fresh_cache_directories':sorted(set(env[key] for key in cache_keys)),'fresh_private_cache_absent_then_created_empty_required':True,'source_max_seconds':900,'outer_timeout_seconds':930,'outer_kill_after_seconds':15,'guardian_term_grace_seconds':15,'guardian_kill_wait_seconds':10,'normal_max_with_cleanup_seconds_estimate':970,'cleanup_unknown_policy':'Retain original lock if exact owned descendants or devices remain; never release on timeout alone.','resource_reserve':ob['resource_reserve'],'mem_available_floor_bytes':64*1024**3,'stage_authorized':False,'launch_authorized':False,'release_file_created':False,'full_proposal_present':False,'submission_authorized':False,'no_harness':True,'no_checkpoint':True,'no_eval':True,'no_ntff':True,'no_automatic_next_job':True,'native_memory_and_compile_feasibility':'UNPROVEN','priority':'AFTER_K2_AND_PROTECTED_CURRENT_P0_EVALS','cold_claim_requires_actual_fresh_empty_cache_and_worker_environment_receipts':True,'interpretation':'Per-rank component snapshots/first-call and steady timing only; incomplete/errors stay incomplete, no whole-step or BPB saving.'}
    files={name:old[name] for name in ('dispatch_driver.py','idle_guard.py','lifetime_guard.py')}
    files.update({'host_driver.py':text.encode(),'package/train.py':source,'package/prepare.py':prepare,'package/launch-command.txt':launch_raw,'package/provenance.json':provenance_raw,'NATIVE_BINDING.disabled.json':(json.dumps(binding,indent=2)+'\n').encode()})
    packet={'binding':binding,'files':{name:{'sha256':sha(raw),'b64':base64.b64encode(raw).decode()} for name,raw in files.items()},'local_only':True,'host_calls':0,'native_started':False,'derivation':{'parent_driver_sha256':sha(old['host_driver.py']),'parent_packet_sha256':sha(original_raw),'shared_guardian_unchanged':True,'source_prepare_unchanged':True,'control_plane_changes':['direct timeout+torchrun phase in package cwd','900s source with930s external timeout and15s kill grace; existing guardian15s TERM+10s KILL/reap','no rehearsal, checkpoint or evaluator phases','literal source/label/command/cache/output bindings and unchanged prepare import copy'],'no_new_containment_framework':True},'missing_before_dispatch':['completed Supervisor proposal','selected eligible free host and exact scope','fresh original-lock/idle/STOP/resource/source and unique-cache checks']}
    path=here/(host+'-prepared.json')
    with path.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
    rows.append({'host':host,'label':label,'packet':path.name,'packet_sha256':sha(path.read_bytes()),'driver_sha256':sha(text.encode()),'source_sha256':sha(source),'status':'DISABLED_LOCAL_DIRECT_DIAGNOSTIC_PREPARED'})
with (here/'PREPARED.json').open('x') as f:json.dump({'alternatives_for_one_selected_host':rows,'source_sha256':sha(source),'no_host_calls':True,'source_edited':False,'new_containment_framework':False,'no_new_review_or_native_test':True,'proposal_pending':True},f,indent=2);f.write('\n')
print(json.dumps({'prepared':rows,'host_calls':0,'native_started':False}))
