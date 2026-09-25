"""Fixed two-phase research-only composition of the unchanged guarded rehearsal."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-stablepath-trn22-r1-pair')
PLAN_SHA = '518c039d570d02b5ac34a4649b866465e18df834061a992949841ca0b8981cca'

def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def require(ok, message):
    if not ok: raise RuntimeError(message)
def record(name, value):
    with (ROOT/name).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def resources(plan):
    require(socket.gethostname()==plan['expected_hostname'],'HOSTNAME')
    for base in (Path('/home/ubuntu'),Path(plan['prepare_path']).parent,ROOT,Path(plan['stable_directory']),
                 *[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
        for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):
            require(not os.path.lexists(base/rel),'STOP:'+str(base/rel))
    space=os.statvfs('/')
    free=space.f_bavail*space.f_frsize
    require(free>=plan['minimum_free_disk_gib']*1024**3,'DISK_BELOW_6_5_GIB')
    return dict(utc=utc(),free_disk_bytes=free)

try:
    require(Path(__file__).resolve()==ROOT/'host_driver.py','EXACT_DRIVER_PATH')
    require(sha(ROOT/'PAIR_BINDING.disabled.json')==PLAN_SHA,'PLAN_SHA')
    plan=json.loads((ROOT/'PAIR_BINDING.disabled.json').read_text())
    require(plan['root']==str(ROOT),'PLAN_ROOT')
    for file,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256'),
                     ('identity_checks.py','helper_sha256')]:
        require(sha(ROOT/file)==plan[key],'SUPPORT_SHA:'+file)
    guard=load('stable_pair_idle_guard',ROOT/'idle_guard.py')
    checks=load('stable_pair_identity_checks',ROOT/'identity_checks.py')
    lock_fd=guard.require_idle_under_inherited_lock(plan['device_lock'],ROOT/'host_driver.py',record)
    results={}
    inputs_by_phase={}
    stable_baseline=None
    for phase in plan['phases']:
        name=phase['name']
        require(name in ('S1','S2'),'UNKNOWN_FIXED_PHASE')
        if name=='S2':
            require(results.get('S1',{}).get('status')=='FIVE_STEP_MECHANICAL_GATES_PASS','S1_NOT_ACCEPTED')
        root=Path(phase['root'])
        require(root.is_dir() and not os.path.lexists(phase['sandbox']),'ABSENT_PHASE_SANDBOX_REQUIRED')
        metrics=checks.no_symlink(phase['environment']['TORCH_NEURONX_METRICS_DIR'])
        require(metrics.is_dir() and not any(metrics.iterdir()),'EMPTY_PRIVATE_METRICS_DIRECTORY_REQUIRED')
        record(name+'-resources.json',resources(plan))
        before=checks.inputs(plan,phase)
        stable={k:before[k] for k in ('executed_train','expected_imported_prepare','stable_directory','protected_prepare','harness')}
        if stable_baseline is None: stable_baseline=stable
        require(stable==stable_baseline,'STABLE_INPUT_CHANGED_BETWEEN_PHASES')
        inputs_by_phase[name]=before
        record(name+'-pre-source-identity.json',before)
        if name=='S2':
            for path in plan['cache_environment'].values():
                require(checks.no_symlink(path).is_dir(),'S1_CACHE_DIRECTORY_MISSING')
        # Recheck under the SAME original lock immediately before each synchronous child.
        fd=guard.require_idle_under_inherited_lock(plan['device_lock'],ROOT/'host_driver.py',
             lambda file,value:record(name+'-pre-'+file,value))
        require(fd==lock_fd,'LOCK_FD_CHANGED')
        record(name+'-launch.json',dict(utc=utc(),label=phase['label'],command=phase['command'],
               environment=phase['environment'],training_command=phase['training_command'],
               actual_import_origin_observed=False,expected_prepare_path=plan['stable_directory']+'/prepare.py'))
        with (root/'training.log').open('x') as out:
            out.write('HARNESS_START_UTC='+utc()+'\n');out.flush()
            rc=subprocess.call(phase['command'],cwd=root,env=phase['environment'],stdin=subprocess.DEVNULL,
                               stdout=out,stderr=subprocess.STDOUT,pass_fds=(lock_fd,))
            out.write('HARNESS_END_UTC='+utc()+'\n');out.flush()
        record(name+'-exit.json',dict(utc=utc(),returncode=rc))
        after=checks.inputs(plan,phase)
        record(name+'-post-source-identity.json',after)
        require(after==before,'SOURCE_OR_PACKAGE_CHANGED_DURING_PHASE')
        fd=guard.require_idle_under_inherited_lock(plan['device_lock'],ROOT/'host_driver.py',
             lambda file,value:record(name+'-post-'+file,value))
        require(fd==lock_fd,'LOCK_FD_CHANGED')
        require(rc==0,'HARNESS_EXIT_NONZERO:'+name)
        observed=checks.validate_phase(plan,phase)
        evidence=checks.capture_inputs(plan,phase)
        record(name+'-compiler-evidence.json',evidence)
        require(evidence['exact_input_bytes_complete'],'MATCHED_PRINCIPAL_COMPILER_BYTES_MISSING')
        results[name]=observed
        record(name+'-mechanical-acceptance.json',observed)
    record('first-five-loss-comparison.json',checks.compare_losses(results['S1'],results['S2']))
    record('terminal.json',dict(utc=utc(),status='PAIR_COMPLETE_PENDING_INDEPENDENT_EVIDENCE_REVIEW',
           native_cache_reuse='UNPROVEN',path_only_causality='UNPROVEN',actual_import_origin_observed=False,
           no_eval=True,no_automatic_full=True,source_adoption=False,submission_authorized=False))
except Exception as error:
    if not (ROOT/'terminal.json').exists():
        record('terminal.json',dict(utc=utc(),status='PAIR_REFUSED_OR_INCOMPLETE',error=str(error),
               no_retry=True,no_S2_fallback=True,no_eval=True,no_automatic_full=True))
    raise
