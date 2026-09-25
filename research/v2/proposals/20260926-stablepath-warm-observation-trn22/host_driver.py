"""One new operational warm five-step observation; original failed pair is not resumed."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-stablepath-warm-trn22-r1')
PLAN_SHA = 'efbaf85470bd1ac51c0e791fe6bc6794d03510551205c62bf7ab1cc338e08796'

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

def classify_target(observed):
    fifth=observed['fifth_boundary_seconds']
    return 'USEFUL_WARM_START_OBSERVED' if fifth <= 120.0 else 'WARM_TARGET_MISSED'

try:
    require(Path(__file__).resolve()==ROOT/'host_driver.py','EXACT_DRIVER_PATH')
    require(sha(ROOT/'WARM_BINDING.disabled.json')==PLAN_SHA,'PLAN_SHA')
    plan=json.loads((ROOT/'WARM_BINDING.disabled.json').read_text())
    require(plan['root']==str(ROOT),'PLAN_ROOT')
    for file,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256'),
                     ('identity_checks.py','helper_sha256'),('EXPERIMENT_PROPOSAL.json','proposal_sha256'),
                     ('S1-pre-source-identity.json','s1_identity_sha256')]:
        require(sha(ROOT/file)==plan[key],'SUPPORT_SHA:'+file)
    guard=load('warm_observation_idle_guard',ROOT/'idle_guard.py')
    checks=load('warm_observation_identity_checks',ROOT/'identity_checks.py')
    lock_fd=guard.require_idle_under_inherited_lock(plan['device_lock'],ROOT/'host_driver.py',record)
    require(len(plan['phases'])==1 and plan['phases'][0]['name']=='W1','ONE_FIXED_OBSERVATION_ONLY')
    phase=plan['phases'][0]
    root=Path(phase['root'])
    require(root.is_dir() and not os.path.lexists(phase['sandbox']),'ABSENT_PHASE_SANDBOX_REQUIRED')
    metrics=checks.no_symlink(phase['environment']['TORCH_NEURONX_METRICS_DIR'])
    require(metrics.is_dir() and not any(metrics.iterdir()),'EMPTY_PRIVATE_METRICS_DIRECTORY_REQUIRED')
    record('W1-resources.json',resources(plan))
    before=checks.inputs(plan,phase)
    anchor=json.loads((ROOT/'S1-pre-source-identity.json').read_text())
    for key in ('executed_train','expected_imported_prepare','stable_directory','protected_prepare','harness'):
        require(before[key]==anchor[key],'ORIGINAL_S1_INPUT_IDENTITY_CHANGED:'+key)
    record('W1-pre-source-identity.json',before)
    for path in plan['cache_environment'].values():
        require(checks.no_symlink(path).is_dir(),'S1_CACHE_DIRECTORY_MISSING')
    fd=guard.require_idle_under_inherited_lock(plan['device_lock'],ROOT/'host_driver.py',
         lambda file,value:record('W1-pre-'+file,value))
    require(fd==lock_fd,'LOCK_FD_CHANGED')
    record('W1-launch.json',dict(utc=utc(),label=phase['label'],command=phase['command'],
           environment=phase['environment'],training_command=phase['training_command'],
           actual_import_origin_observed=False,expected_prepare_path=plan['stable_directory']+'/prepare.py',
           retained_cache_paths=True,native_cache_hit='UNPROVEN'))
    with (root/'training.log').open('x') as out:
        out.write('HARNESS_START_UTC='+utc()+'\n');out.flush()
        rc=subprocess.call(phase['command'],cwd=root,env=phase['environment'],stdin=subprocess.DEVNULL,
                           stdout=out,stderr=subprocess.STDOUT,pass_fds=(lock_fd,))
        out.write('HARNESS_END_UTC='+utc()+'\n');out.flush()
    record('W1-exit.json',dict(utc=utc(),returncode=rc))
    after=checks.inputs(plan,phase)
    record('W1-post-source-identity.json',after)
    require(after==before,'SOURCE_OR_PACKAGE_CHANGED_DURING_PHASE')
    fd=guard.require_idle_under_inherited_lock(plan['device_lock'],ROOT/'host_driver.py',
         lambda file,value:record('W1-post-'+file,value))
    require(fd==lock_fd,'LOCK_FD_CHANGED')
    require(rc==0,'HARNESS_EXIT_NONZERO:W1')
    observed=checks.validate_phase(plan,phase)
    observed['useful_warm_target_seconds']=120
    observed['target_disposition']=classify_target(observed)
    record('W1-mechanical-observation.json',observed)
    # Only this new observational proposal makes forensic availability non-gating.
    # Do not change or resume the original failed pair or assert any cache hit.
    try:
        evidence=checks.capture_inputs(plan,phase)
        evidence['capture_status']='COMPLETE' if evidence['exact_input_bytes_complete'] else 'PRINCIPAL_BYTES_MISSING'
    except Exception as error:
        evidence=dict(capture_status='UNAVAILABLE_OR_INCOMPLETE',error=type(error).__name__+': '+str(error),
                      exact_input_bytes_complete=False,native_neff_reuse='UNPROVEN',cache_hit=False,
                      partial_artifacts_preserved=True,no_broader_search=True)
    record('W1-compiler-evidence.json',evidence)
    record('terminal.json',dict(utc=utc(),status='WARM_OBSERVATION_COMPLETE_PENDING_INDEPENDENT_REVIEW',
           target_disposition=observed['target_disposition'],fifth_boundary_seconds=observed['fifth_boundary_seconds'],
           native_cache_reuse='UNPROVEN',path_only_causality='UNPROVEN',actual_import_origin_observed=False,
           original_pair_status='PAIR_REFUSED_OR_INCOMPLETE',original_S2_launched=False,
           no_eval=True,no_automatic_full=True,no_retry=True,source_adoption=False,submission_authorized=False))
except Exception as error:
    if not (ROOT/'terminal.json').exists():
        record('terminal.json',dict(utc=utc(),status='WARM_OBSERVATION_REFUSED_OR_INCOMPLETE',error=str(error),
               original_pair_status='PAIR_REFUSED_OR_INCOMPLETE',original_S2_launched=False,
               no_retry=True,no_eval=True,no_automatic_full=True))
    raise
