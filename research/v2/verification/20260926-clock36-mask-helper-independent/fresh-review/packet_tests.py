import ast, copy, hashlib, importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
base=Path(__file__).resolve().parent.parent
packet=base/'packet'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins={p.name:digest(p) for p in packet.iterdir() if p.is_file()}
def load(name):
    spec=importlib.util.spec_from_file_location(name,packet/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
h=load('host_driver');g=load('idle_guard')
plan=json.loads((packet/'CPU_PLAN.disabled.json').read_text())
inputs=json.loads((packet/'INPUTS.json').read_text())
bundle=json.loads((base/'results-bundle.json').read_text())
cases=[]
def check(name,fn,error=None):
    try: result=fn()
    except Exception as exc:
        if error is None or error not in str(exc):raise
        cases.append({'case':name,'rejected':str(exc)});return
    if error is not None:raise RuntimeError('UNEXPECTED_ACCEPT:'+name)
    if result is False:raise RuntimeError('FAILED:'+name)
    cases.append({'case':name,'passed':True})
check('driver_frozen',lambda:pins['host_driver.py']=='f15220ab890b4219c6eaa26b13f3e73fe6b6de1f169a413b77a420e873ba8f2a')
check('plan_frozen',lambda:pins['CPU_PLAN.disabled.json']=='3aee91719b04f95896639afb674240a23ad10b988a6bf79c11797f5f88a4ba4c')
check('disabled',lambda:all(plan[k] is False for k in ('stage_authorized','launch_authorized','native_launch_authorized','host_reserved')))
check('exact_command',lambda:h.COMMAND==plan['command'] and h.COMMAND[:5]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','600s'])
check('one_thread_no_bytecode_private_tmp',lambda:all(h.ENV[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE')) and h.ENV['TMPDIR']==str(h.ROOT/'tmp'))
for name,pin in h.PINS.items():
    p=base/'new_helper.py' if name=='compare_checkpoints.py' else packet/name
    check('support_pin_'+name,lambda:digest(p)==pin)
check('sixteen_inputs',lambda:inputs==plan['input_pins'] and len(inputs)==16)
for absolute,pin in inputs.items():
    rel=str(Path(absolute).relative_to(h.OLD_ROOT));arm=rel.split('/')[0]
    if rel.endswith('train.py'):
        expected=json.loads(bundle['files'][arm+'/request.json']['text'])['source_sha256']
        check('source_'+arm,lambda:pin['sha256']==expected)
    elif rel.endswith('final.pt'):
        expected=json.loads(bundle['files'][arm+'/checkpoint-identity.json']['text'])
        check('checkpoint_'+arm,lambda:expected['path']==absolute and all(pin[k]==expected[k] for k in ('bytes','sha256')))
    else:
        raw=bundle['files'][rel]
        check('raw_'+rel,lambda:all(pin[k]==raw[k] for k in ('bytes','sha256')))
tree=ast.parse((packet/'host_driver.py').read_text())
spawns=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen']
check('single_sync_child_inherited_fd',lambda:len(spawns)==1 and any(k.arg=='pass_fds' and ast.unparse(k.value)=='(lock_fd,)' for k in spawns[0].keywords))
check('three_idle_observations',lambda:sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='require_idle_under_inherited_lock' for n in ast.walk(tree))==3)
check('postflight_obligation',lambda:all(s in plan['postflight'] for s in ('On every outcome','PID/starttime/PPID/PGID/SID','compiler','Missing/ambiguous/alive=>HOLD/no next job')))
def lock_test(fd_pair,path_pair):
    with patch.object(h.os,'fstat',return_value=NS(st_dev=fd_pair[0],st_ino=fd_pair[1])),patch.object(Path,'lstat',return_value=NS(st_dev=path_pair[0],st_ino=path_pair[1])):
        h.check_original_lock(77)
check('original_lock',lambda:lock_test((66305,832589),(66305,832589)))
check('changed_fd',lambda:lock_test((66305,1),(66305,832589)),'ORIGINAL_LOCK_FD')
check('changed_path',lambda:lock_test((66305,832589),(66305,2)),'ORIGINAL_LOCK_PATH')
def identity_test(size=7,actual='abc',symlink=False,regular=True):
    with patch.object(Path,'is_file',return_value=regular),patch.object(Path,'is_symlink',return_value=symlink),patch.object(Path,'stat',return_value=NS(st_size=size)),patch.object(h,'sha',side_effect=lambda p:h.PREP_SHA if p==h.PREP else actual):
        h.input_identities({'/synthetic/input':{'bytes':7,'sha256':'abc'}})
check('identity_fixture',identity_test)
check('size_mismatch',lambda:identity_test(size=8),'INPUT_BYTES')
check('digest_mismatch',lambda:identity_test(actual='bad'),'INPUT_SHA')
check('symlink_rejected',lambda:identity_test(symlink=True),'INPUT_REGULAR_FILE')
check('missing_rejected',lambda:identity_test(regular=False),'INPUT_REGULAR_FILE')
q=json.loads((base/'fresh-review/normal.json').read_text())['positive']
q['named_parameters_and_all_buffers']={k:v for k,v in q['named_parameters_and_all_buffers'].items() if not k.startswith('parameter:')}
q['named_parameters_and_all_buffers'].update({'parameter:fixture'+str(i):{} for i in range(65)})
result={'status':'FULL_PARAMETER_CPU_CHECKPOINT_AND_LOGIT_IDENTITY','states':65,'parameters':113256724,'native_correctness':False,'quality_evidence':False,'on_off_equivalence_claimed':False,'original_r2_qualification':'FAILED_PRESERVED','checkpoint_only_retry':True,'records':[{'name':n,'trained_model_qualification':copy.deepcopy(q)} for n in ('reference','default','off','on')]}
check('synthetic_result_envelope',lambda:h.validate_result(result))
mutations=[('bad_status',('status',),'bad','RESULT_STATUS'),('missing_arms',('records',),[],'FOUR_ARM_COMPLETION'),('native',('native_correctness',),True,'RESULT_SCOPE'),('original_fail',('original_r2_qualification',),'PASS','ORIGINAL_FAILURE_SCOPE'),('repeat',('records',0,'trained_model_qualification','repeated_bytes_equal'),False,'STATE_REPEAT'),('causality_nan',('records',0,'trained_model_qualification','causality','max_abs_diff'),float('nan'),'PREFIX_CAUSALITY'),('mask_proof',('records',0,'trained_model_qualification','named_parameters_and_all_buffers','buffer:mask_full','mask_semantics_validated'),False,'MASK_PROOF')]
for name,keys,value,error in mutations:
    bad=copy.deepcopy(result);at=bad
    for key in keys[:-1]:at=at[key]
    at[keys[-1]]=value
    check(name,lambda:h.validate_result(bad),error)
device={'neuron_device':0,'instance_type':'trn2.3xlarge','nc_count':4,'logical_neuroncore_config':2,'neuroncore_ids':[0,1,2,3],'neuron_processes':[]}
def idle(ps='10 9 python own\n9 1 flock parent\n',dev=None,code=0):
    def fake(command,**kwargs):
        return NS(returncode=code,stdout=ps if command==g.PS else json.dumps([device if dev is None else dev]),stderr='')
    g.check_idle(fake,10,9,{})
check('idle_fixture',idle)
check('compiler_busy',lambda:idle(ps='10 9 python own\n9 1 flock parent\n15 1 neuronx-cc /opt/neuronx-cc\n'),'PS_NATIVE_JOB_BUSY')
check('device_busy',lambda:idle(dev=dict(device,neuron_processes=[{'pid':15}])),'NEURON_DEVICE_BUSY')
check('bad_ps',lambda:idle(ps='broken'),'PS_PARSE_FAILED')
check('missing_own',lambda:idle(ps='9 1 flock parent\n'),'PS_OWN_LOCK_PROCESSES_MISSING')
check('command_failure',lambda:idle(code=1),'PS_COMMAND_FAILED')
def resource_test(disk=11274289152,ram=67108864,hostname='ip-172-31-32-253',parent=9,stop=False):
    with patch.object(h.os,'getppid',return_value=parent),patch.object(h.socket,'gethostname',return_value=hostname),patch.object(h.os.path,'lexists',return_value=stop),patch.object(h.os,'statvfs',return_value=NS(f_bavail=disk,f_frsize=1)),patch.object(Path,'glob',return_value=[]),patch.object(Path,'read_text',return_value='MemAvailable: '+str(ram)+' kB\n'):
        h.checks(9)
check('resource_exact_floor',resource_test)
check('disk_below_floor',lambda:resource_test(disk=11274289151),'DISK_FLOOR')
check('ram_below_floor',lambda:resource_test(ram=67108863),'AVAILABLE_RAM_FLOOR')
check('wrong_host',lambda:resource_test(hostname='other'),'HOSTNAME')
check('wrong_parent',lambda:resource_test(parent=8),'AUTHENTICATED_PARENT_CHANGED')
check('stop_present',lambda:resource_test(stop=True),'STOP:')
print(json.dumps({'verdict':'PASS_DISABLED_PACKET_ONLY','cases':cases,'count':len(cases),'packet_input_sha256':pins,'test_sha256':digest(Path(__file__)),'helper_sha256':digest(base/'new_helper.py'),'original_bundle_sha256':digest(base/'results-bundle.json'),'optimized':sys.flags.optimize,'host_actions':False,'actual_result':False,'postflight_required_every_outcome':True},indent=2))
