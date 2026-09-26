"""Tiny author binding negatives; no host/native imports or stage operation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import host_driver as driver
from binding import BINDING
from common import HERE, LABEL, GIB, source_pins, digest, require


def rejected(fn, message):
    try:fn()
    except RuntimeError as exc:
        require(str(exc)==message,'WRONG_FAILURE:'+str(exc))
        return
    raise RuntimeError('NEGATIVE_ACCEPTED:'+message)


def main():
    checks=[]
    # Exact real disabled entrypoint: no mocked host/SDK operations.
    child=subprocess.run([sys.executable,str(HERE/'host_driver.py')],capture_output=True,text=True,timeout=2)
    require(child.returncode!=0 and 'DISABLED_NO_SEPARATE_HOST_RELEASE' in child.stderr,'DEFAULT_NOT_DISABLED')
    checks.append('actual_disabled_entrypoint')
    driver.require_original_lock(dict(device=66305,inode=596823))
    for value in (dict(device=66306,inode=596823),dict(device=66305,inode=596824),dict(device=66305),{}):
        rejected(lambda v=value:driver.require_original_lock(v),'ORIGINAL_LOCK_IDENTITY_MISMATCH')
    checks.append('original_lock_exact_and_four_negatives')
    # A flag alone must not turn an incomplete candidate into an executable run.
    changed=copy.deepcopy(BINDING); changed['launch_authorized']=True
    with patch.object(driver,'BINDING',changed):
        rejected(driver.main,'WRONG_BOUND_PACKET_ROOT')
    changed['packet_root']=str(HERE)
    with patch.object(driver,'BINDING',changed):
        rejected(driver.main,'INCOMPLETE_BOUND_PROJECT_OR_STAGE')
    changed.update(project_commit='mock',stage_authorized=True)
    with patch.object(driver,'BINDING',changed),patch.object(driver.socket,'gethostname',lambda:changed['hostname']):
        rejected(driver.main,'MISSING_RELEASE')
    checks.append('enable_flag_root_incomplete_project_missing_release')
    env=driver.environment(Path(BINDING['output_root']),9)
    require(env['NEURON_LOGICAL_NC_CONFIG']=='2' and env['NEURON_COMPETITION_R1_SEED']=='42' and
            env['NEURON_CC_FLAGS']=='--optlevel=1 --auto-cast matmult --auto-cast-type bf16','SCIENCE_ENV')
    require(all(env[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'THREADS')
    require(env['TRN_EXP_LABEL']==LABEL and env['K2_LOCK_FD']=='9','LABEL_LOCK_ENV')
    require(len(str(Path(BINDING['output_root'])/'control.sock').encode())<108,'UNIX_PATH_LENGTH')
    checks.append('exact_env_threads_private_paths_socket_length')
    pins=source_pins()
    baseline=json.loads((HERE/'PROTOTYPE_PINS.json').read_text())
    unchanged=('consumer_nki.py','oracle.py','fixtures.py','generate.py','worker.py','evidence.py',
               'artifacts.py','monitor.py','idle_guard.py','coordinator.py')
    require(all(pins[name]==baseline[name] for name in unchanged),'REPAIRED_CORE_CHANGED')
    checks.append('ten_repaired_science_route_lifecycle_sources_identical')
    proposal=json.loads((HERE.parent/'EXPERIMENT_PROPOSAL.json').read_text())
    require(proposal['limits']['minimum_initial_free_disk_gib']==16 and
            proposal['limits']['total_seconds']==1200 and proposal['launch_authorized'] is False,
            'PROPOSAL_RESOURCE_OR_AUTHORITY')
    require(proposal['last_reported_free_bytes']<16*GIB,'DISK_BLOCKER_LOST')
    checks.append('durable_disabled_proposal_and_disk_blocker')
    print(json.dumps(dict(scope='AUTHOR_CPU_BINDING_ONLY',checks=checks,source_pins=pins,
                         binding_sha256=digest(HERE/'binding.py'),prototype_preserved=True,
                         native=False,host_contact=False),sort_keys=True))


if __name__=='__main__':main()
