"""Own stdlib-only binding fixtures; no driver main, tensors, SDK or hosts."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
spec=importlib.util.spec_from_file_location('driver_fixture',HERE/'host_driver.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
checks=[]


def require(ok,name):
    if not ok:raise RuntimeError(name)
    checks.append(name)


def refuse(name,fn,prefix):
    try:fn()
    except RuntimeError as error:require(str(error).startswith(prefix),name+':'+str(error))
    else:raise RuntimeError('FALSE_ACCEPT:'+name)


def fixture_result():
    state={'parameter:p'+str(i):{'sha256':'0'*64} for i in range(65)}
    for name in ('mask_full','mask_short'):
        state['buffer:'+name]={'mask_semantics_validated':True,'dtype':'torch.float32','nan_count':0,'positive_infinity_count':0,'shape':[1,1,8,8],'zero_count':36,'negative_infinity_count':28}
    q={'route':'ordinary_unpatched_cpu_load_for_eval','repeated_bytes_equal':True,'model_state_equal_after_each_call':True,
       'outputs':[{'sha256':'0'*64},{'sha256':'0'*64},{'sha256':'1'*64}],
       'causality':{'passed':True,'tolerance':1e-2,'max_abs_diff':0.0},'mask_configuration':{'sequence_len':8},'named_parameters_and_all_buffers':state}
    return {'status':'FULL_PARAMETER_CPU_CHECKPOINT_AND_LOGIT_IDENTITY','states':65,'parameters':113256724,
            'native_correctness':False,'quality_evidence':False,'on_off_equivalence_claimed':False,
            'original_r2_qualification':'FAILED_PRESERVED','checkpoint_only_retry':True,
            'records':[{'name':n,'trained_model_qualification':copy.deepcopy(q)} for n in ('reference','default','off','on')]}


def main():
    p=json.loads((HERE/'CPU_PLAN.disabled.json').read_text())
    proposal=json.loads((HERE/'EXPERIMENT_PROPOSAL.json').read_text())
    pins=json.loads((HERE/'INPUTS.json').read_text())
    require(len(pins)==16 and pins==p['input_pins'],'all16_input_pins')
    require(str(d.ROOT)==p['root']==proposal['host']['unique_new_root'] and str(d.OLD_ROOT)==p['old_readonly_root']==proposal['host']['old_readonly_root'],'exact_old_new_roots')
    require(d.LABEL==p['label']==proposal['candidate']['new_label'],'exact_label')
    require(d.COMMAND==p['command'] and d.COMMAND[:5]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','600s'],'one600_plus60_command')
    require(d.COMMAND[-1]==str(d.OLD_ROOT) and '--train-py' not in d.COMMAND and 'torchrun' not in ' '.join(d.COMMAND),'checkpoint_only_not_g2_training')
    require(d.ENV['TMPDIR']==str(d.ROOT/'tmp') and d.ENV['PYTHONDONTWRITEBYTECODE']=='1' and '-B' in d.COMMAND,'private_tmp_no_old_bytecode_write')
    require(all(d.ENV[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'one_thread')
    require(p['resources']['minimum_available_ram_bytes']==64*1024**3 and p['resources']['minimum_free_disk_bytes']==10.5*1024**3,'resource_floors')
    require(all(p[k] is False for k in ('launch_authorized','stage_authorized','native_launch_authorized','host_reserved')),'disabled_authority')
    require(hashlib.sha256((HERE/'host_driver.py').read_bytes()).hexdigest()==p['driver_sha256'],'driver_pin')
    for name,pin in d.PINS.items():
        path=HERE/name if (HERE/name).exists() else REPO/'research/v2/operator/20260926-common-host-idle-guard'/name
        require(hashlib.sha256(path.read_bytes()).hexdigest()==pin,'support_pin:'+name)
    require(d.PREP_SHA==p['prepare_sha256'] and str(d.PREP)==p['prepare_path'],'prepare_binding')
    for checkpoint in proposal['checkpoints']:
        require(pins[checkpoint['path']]['sha256']==checkpoint['sha256'],'checkpoint_pin:'+checkpoint['arm'])
    require(p['outer_command']==['/usr/bin/flock','-n',d.LOCK,'/usr/bin/python3','-I','-B',str(d.ROOT/'host_driver.py')],'original_outer_flock')
    tree=ast.parse((HERE/'host_driver.py').read_text())
    launches=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen']
    require(len(launches)==1,'sole_child_spawn')
    fd=next(k.value for k in launches[0].keywords if k.arg=='pass_fds')
    require(ast.dump(fd)=="Tuple(elts=[Name(id='lock_fd', ctx=Load())], ctx=Load())",'inherited_fd')
    guards=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='require_idle_under_inherited_lock']
    require(len(guards)==3 and guards[1].lineno<launches[0].lineno<guards[2].lineno,'fresh_guard_before_and_after_child')
    real=SimpleNamespace(st_dev=66305,st_ino=832589)
    with patch.object(d.os,'fstat',return_value=real),patch.object(d.Path,'lstat',return_value=real):
        d.check_original_lock(7);require(True,'original_lock_positive')
    with patch.object(d.os,'fstat',return_value=SimpleNamespace(st_dev=66305,st_ino=832590)),patch.object(d.Path,'lstat',return_value=real):
        refuse('replaced_fd_refused',lambda:d.check_original_lock(7),'ORIGINAL_LOCK_FD')
    with patch.object(d.os,'fstat',return_value=real),patch.object(d.Path,'lstat',return_value=SimpleNamespace(st_dev=66305,st_ino=832590)):
        refuse('replaced_named_lock_refused',lambda:d.check_original_lock(7),'ORIGINAL_LOCK_PATH')
    with tempfile.TemporaryDirectory(prefix='mask-r3-packet-') as tmp:
        path=Path(tmp)/'input';path.write_bytes(b'exact-small-fixture')
        pin={str(path):{'bytes':path.stat().st_size,'sha256':d.sha(path)}}
        with patch.object(d,'PREP',path),patch.object(d,'PREP_SHA',d.sha(path)):
            require(d.input_identities(pin)==pin,'input_pin_positive')
            bad=copy.deepcopy(pin);bad[str(path)]['bytes']+=1
            refuse('input_size_refused',lambda:d.input_identities(bad),'INPUT_BYTES')
            bad=copy.deepcopy(pin);bad[str(path)]['sha256']='0'*64
            refuse('input_hash_refused',lambda:d.input_identities(bad),'INPUT_SHA')
            missing={str(Path(tmp)/'absent'):pin[str(path)]}
            refuse('input_missing_refused',lambda:d.input_identities(missing),'INPUT_REGULAR_FILE')
    good=fixture_result();d.validate_result(good);require(True,'result_fixture_positive_not_actual_result')
    mutations={
        'status':lambda x:x.update(status='PASS'),
        'count':lambda x:x.update(parameters=1),
        'scope':lambda x:x.update(native_correctness=True),
        'old_failure':lambda x:x.update(original_r2_qualification='PASS'),
        'missing_arm':lambda x:x['records'].pop(),
        'repeat':lambda x:x['records'][0]['trained_model_qualification'].update(repeated_bytes_equal=False),
        'causality':lambda x:x['records'][0]['trained_model_qualification']['causality'].update(max_abs_diff=.011),
        'nan_causality':lambda x:x['records'][0]['trained_model_qualification']['causality'].update(max_abs_diff=float('nan')),
        'mask_unvalidated':lambda x:x['records'][0]['trained_model_qualification']['named_parameters_and_all_buffers']['buffer:mask_full'].update(mask_semantics_validated=False),
        'mask_positive_inf':lambda x:x['records'][0]['trained_model_qualification']['named_parameters_and_all_buffers']['buffer:mask_full'].update(positive_infinity_count=1),
        'mask_count':lambda x:x['records'][0]['trained_model_qualification']['named_parameters_and_all_buffers']['buffer:mask_full'].update(zero_count=35),
    }
    for name,change in mutations.items():
        bad=copy.deepcopy(good);change(bad);refuse('result_negative:'+name,lambda:d.validate_result(bad),'')
    print(json.dumps({'status':'IMPLEMENTER_STDLIB_PACKET_FIXTURES_ONLY','checks':len(checks),'names':checks,'host_actions':False,'tensor_allocations':False,'actual_checkpoint_qualification':'NOT_EXECUTED','cold_authorized':False},indent=2))


if __name__=='__main__':main()
