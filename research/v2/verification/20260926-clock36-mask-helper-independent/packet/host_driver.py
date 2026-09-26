"""Disabled one-child retained-checkpoint CPU binding; sole Operator after release."""
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess

ROOT=Path('/home/ubuntu/trainium_clock36_recursive_checkpoint_20260926_trn22_r3')
OLD_ROOT=Path('/home/ubuntu/trainium_clock36_recursive_cpu_20260926_trn22_r2')
LABEL='fs-20260926-clock36-recursive-checkpoint-trn22-r3'
PREP=Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/prepare.py')
PREP_SHA='5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb'
LOCK='/home/ubuntu/.cache/trainium_freshstart/device0.lock'
PINS={'idle_guard.py':'adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737',
      'dispatch_driver.py':'c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a',
      'compare_checkpoints.py':'1b3714e050e473b176a3e1f1fbd59387579867f3d06d14ecf71248a6d8a1714c',
      'INPUTS.json':'b1654bcc207cce36324f7f4e4ad8ca6594445c2bd0cc68414e399d6dde1494de'}
VENV='/home/ubuntu/workspace/native_venv'
COMMAND=['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','600s',
         VENV+'/bin/python','-B',str(ROOT/'compare_checkpoints.py'),str(OLD_ROOT)]
ENV={'HOME':'/home/ubuntu','PATH':VENV+'/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin',
     'PYTHONPATH':str(PREP.parent),'NKI_VENV_PATH':VENV,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
     'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TMPDIR':str(ROOT/'tmp'),
     'NEURON_LOGICAL_NC_CONFIG':'2','NEURON_CC_FLAGS':'--optlevel=1 --auto-cast matmult --auto-cast-type bf16',
     'NEURON_COMPETITION_R1_CACHE_DIR':'/home/ubuntu/.cache/neuron_competition_r1',
     'NEURON_COMPETITION_R1_SEED':'42','NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS':'1800','TRN_EXP_LABEL':LABEL}


def require(ok,message):
    if not ok:raise RuntimeError(message)


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def record(name,value):
    with (ROOT/name).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())


def check_original_lock(fd):
    opened=os.fstat(fd);named=Path(LOCK).lstat()
    require((opened.st_dev,opened.st_ino)==(66305,832589),'ORIGINAL_LOCK_FD')
    require((named.st_dev,named.st_ino)==(66305,832589),'ORIGINAL_LOCK_PATH')


def input_identities(pins):
    observed={}
    for name,pin in pins.items():
        path=Path(name)
        require(path.is_file() and not path.is_symlink(),'INPUT_REGULAR_FILE:'+name)
        require(path.stat().st_size==pin['bytes'],'INPUT_BYTES:'+name)
        digest=sha(path);require(digest==pin['sha256'],'INPUT_SHA:'+name)
        observed[name]={'bytes':path.stat().st_size,'sha256':digest}
    require(sha(PREP)==PREP_SHA,'PREPARE_SHA')
    return observed


def checks(parent):
    require(os.getppid()==parent,'AUTHENTICATED_PARENT_CHANGED')
    require(socket.gethostname()=='ip-172-31-32-253','HOSTNAME')
    for base in (Path('/home/ubuntu'),PREP.parent,ROOT,OLD_ROOT,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
        for name in ('STOP','research/agentic/STOP','research/v2/STOP'):
            require(not os.path.lexists(base/name),'STOP:'+str(base/name))
    stat=os.statvfs('/');free=stat.f_bavail*stat.f_frsize
    require(free>=10.5*1024**3,'DISK_FLOOR')
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    require(available>=64*1024**3,'AVAILABLE_RAM_FLOOR')
    return {'utc':utc(),'free_disk_bytes':free,'available_ram_bytes':available,'hostname':socket.gethostname()}


def validate_result(result):
    require(result['status']=='FULL_PARAMETER_CPU_CHECKPOINT_AND_LOGIT_IDENTITY','RESULT_STATUS')
    require(result['states']==65 and result['parameters']==113256724,'FULL_PARAMETER_COUNT')
    require(result['native_correctness'] is False and result['quality_evidence'] is False and result['on_off_equivalence_claimed'] is False,'RESULT_SCOPE')
    require(result['original_r2_qualification']=='FAILED_PRESERVED' and result['checkpoint_only_retry'] is True,'ORIGINAL_FAILURE_SCOPE')
    require([r['name'] for r in result['records']]==['reference','default','off','on'],'FOUR_ARM_COMPLETION')
    for arm in result['records']:
        q=arm['trained_model_qualification']
        require(q['route']=='ordinary_unpatched_cpu_load_for_eval','ORDINARY_LOADER')
        require(q['repeated_bytes_equal'] is True and q['model_state_equal_after_each_call'] is True,'STATE_REPEAT')
        require(len(q['outputs'])==3 and q['outputs'][0]==q['outputs'][1],'THREE_OUTPUTS_REPEATED_BYTES')
        c=q['causality'];error=c['max_abs_diff']
        require(c['passed'] is True and c['tolerance']==1e-2 and math.isfinite(error) and error<=1e-2,'PREFIX_CAUSALITY')
        state=q['named_parameters_and_all_buffers']
        require(sum(k.startswith('parameter:') for k in state)==65,'ORDINARY_PARAMETER_COUNT')
        for name in ('mask_full','mask_short'):
            mask=state['buffer:'+name]
            require(mask['mask_semantics_validated'] is True and mask['dtype']=='torch.float32','MASK_PROOF')
            require(mask['nan_count']==mask['positive_infinity_count']==0,'MASK_NONFINITE')
            length=q['mask_configuration']['sequence_len']
            require(mask['shape']==[1,1,length,length] and mask['zero_count']+mask['negative_infinity_count']==length*length,'MASK_COUNTS')


def main():
    try:
        for name,pin in PINS.items():require(sha(ROOT/name)==pin,'PACKET_SHA:'+name)
        spec=importlib.util.spec_from_file_location('bound_idle_guard',ROOT/'idle_guard.py')
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        parent=os.getppid()
        lock_fd=guard.require_idle_under_inherited_lock(LOCK,ROOT/'host_driver.py',record)
        check_original_lock(lock_fd)
        record('preflight.json',checks(parent))
        inputs=json.loads((ROOT/'INPUTS.json').read_text())
        before=input_identities(inputs);record('inputs-before.json',before)
        (ROOT/'tmp').mkdir(exist_ok=False)
        # Repeat BOTH fresh idle observations under the same FD immediately before the sole child.
        lock_fd=guard.require_idle_under_inherited_lock(LOCK,ROOT/'host_driver.py',lambda n,v:record('launch-'+n,v))
        check_original_lock(lock_fd)
        require(os.getppid()==parent,'AUTHENTICATED_PARENT_CHANGED')
        record('launch.json',{'utc':utc(),'label':LABEL,'command':COMMAND,'environment':ENV,'cpu_only':True})
        with (ROOT/'checkpoint-result.json').open('x') as out,(ROOT/'checkpoint-result.stderr').open('x') as err:
            child=subprocess.Popen(COMMAND,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,pass_fds=(lock_fd,))
            record('process.json',{'utc':utc(),'pid':child.pid,'parent_pid':os.getpid(),'command':COMMAND})
            code=child.wait()
        record('checkpoint-exit.json',{'utc':utc(),'exit_code':code})
        after=input_identities(inputs);record('inputs-after.json',after)
        require(after==before,'READONLY_INPUTS_CHANGED')
        lock_fd=guard.require_idle_under_inherited_lock(LOCK,ROOT/'host_driver.py',lambda n,v:record('post-'+n,v))
        check_original_lock(lock_fd)
        require(code==0,'CHECKPOINT_HELPER_FAILED:'+str(code))
        result=json.loads((ROOT/'checkpoint-result.json').read_text());validate_result(result)
        record('terminal.json',{'utc':utc(),'exit_code':0,'status':'CPU_CHECKPOINT_ONLY_COMPLETE_PENDING_INDEPENDENT_REVIEW','cpu_only':True,'original_r2_qualification':'FAILED_PRESERVED','native_launch_authorized':False})
    except Exception as error:
        if not (ROOT/'terminal.json').exists():
            record('terminal.json',{'utc':utc(),'exit_code':1,'error':type(error).__name__+': '+str(error),'cpu_only':True,'original_r2_qualification':'FAILED_PRESERVED','native_launch_authorized':False})
        raise


if __name__=='__main__':main()
