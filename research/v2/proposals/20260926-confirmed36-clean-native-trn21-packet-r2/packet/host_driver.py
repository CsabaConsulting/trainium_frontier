"""One disabled clean36 trn21 native diagnostic; no staging or dispatch here."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

ROOT = Path('/home/ubuntu/trainium_confirmed36_clean_native_consistency_20260926_trn21_r2')
PROJECT = Path('/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2')
LABEL = 'fs-20260926-confirmed36-clean-native-consistency-trn21-r2'
HOST = 'ip-172-31-34-85'
LOCK = '/home/ubuntu/.cache/trainium_freshstart/device0.lock'
LOCK_ID = (66305,596823)
GIB = 1024**3
INITIAL_DISK, RUNTIME_DISK, RAM = 27*GIB//2, 21*GIB//2, 64*GIB
ARTIFACT_CAP, RANK_CAP = 256*1024**2, 16*1024**2
CHILD_SECONDS, TERM_SECONDS = 900, 15
NATIVE = '/home/ubuntu/workspace/native_venv/bin/'
PROBE = 'proposals/20260926-confirmed36-clean-native-consistency/probe.py'
FOUNDATION = 'proposals/20260926-confirmed36-clean-checkpoint-cpu/compare_checkpoints.py'
PINS = {
    PROBE:'48962e675ab15cefb1a7ea27b89b7ba26f039a9ba847d679d97bc4544d5e2a77',
    FOUNDATION:'65e7eb1a21b0e880ee2362461ff3664c4dbd52933816e5ec0684645c92cd75e8',
    'INPUTS.json':'0cc0569936b0b1653bcc946edcdedd1bf91d3f671fabbc205a0a7b883bf9a15e',
    'lifetime_guard.py':'e461ec5c76c784fd51471c9e8c268cdde42fe1def68b31bfc9524480f980c312',
    'idle_guard.py':'adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737',
    'dispatch_driver.py':'c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a',
}
# Exact supported executables must receive observed identities in a separately
# reviewed release. No native package/version identity is fabricated locally.
RUNTIME_PATHS = ('/usr/bin/flock','/usr/bin/python3','/usr/bin/timeout','/usr/bin/ps',
                 '/usr/bin/fuser','/opt/aws/neuron/bin/neuron-ls',NATIVE+'python',NATIVE+'torchrun')
CACHE_NAMES = ('neff','local','hlo','inductor','compiler','debug')
CACHE_KEYS = ('TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR',
              'TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR',
              'NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR')
ENV = {
    'HOME':'/home/ubuntu','PATH':NATIVE+':/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin',
    'NKI_VENV_PATH':'/home/ubuntu/workspace/native_venv','PYTHONPATH':str(PROJECT),
    'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','NEURON_LOGICAL_NC_CONFIG':'2',
    'NEURON_CC_FLAGS':'--optlevel=1','NEURON_COMPETITION_R1_SEED':'42',
    'NEURON_COMPETITION_R1_CACHE_DIR':'/home/ubuntu/.cache/neuron_competition_r1',
    'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
    'TRN_EXP_LABEL':LABEL,'TORCH_NEURONX_ENABLED_METRIC_TABLES':'graph_stats',
    'TORCH_NEURONX_METRICS_DIR':str(ROOT/'metrics'),
    **{k:str(ROOT/'caches'/n) for k,n in zip(CACHE_KEYS,CACHE_NAMES)},
    'NEURON_COMPILE_CACHE_URL':str(ROOT/'caches/compiler'),
}
CHILD_ARGV = [NATIVE+'torchrun','--standalone','--nproc_per_node=4',str(ROOT/PROBE),
              '--enable-native','--label',LABEL,'--inputs',str(ROOT/'INPUTS.json'),
              '--inputs-sha256',PINS['INPUTS.json'],'--out-dir',str(ROOT/'receipts')]
COMMAND = ['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=15s','900s',*CHILD_ARGV]
OUTER = ['/usr/bin/flock','-n',LOCK,'/usr/bin/python3','-I','-B',str(ROOT/'host_driver.py')]


def require(ok,message):
    if not ok: raise RuntimeError(message)


def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()


def plain(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts,'NONCANONICAL_PATH')
    require(all(not p.is_symlink() for p in (path,*path.parents)),'SYMLINK:'+str(path))
    return path


def file_record(path):
    path = plain(path)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode),'NONREGULAR:'+str(path))
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
        closed = os.fstat(stream.fileno())
    after = path.stat()
    identity = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(identity(before)==identity(opened)==identity(closed)==identity(after),'FILE_CHANGED:'+str(path))
    return {'sha256':digest,'bytes':after.st_size}


def runtime_record(path):
    # Runtime symlinks (e.g. python3) are observed, never silently substituted.
    lexical = Path(path)
    resolved = lexical.resolve(strict=True)
    return {'resolved_path':str(resolved),'link_target':os.readlink(lexical) if lexical.is_symlink() else None,
            **file_record(resolved)}


def record(name,value):
    with (ROOT/name).open('x') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False); stream.write('\n')


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def resources():
    disk = os.statvfs(ROOT if ROOT.exists() else ROOT.parent)
    mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {'free_disk_bytes':disk.f_bavail*disk.f_frsize,
            'available_ram_bytes':int(mem['MemAvailable'].split()[0])*1024}


def resource_gate(observation,initial):
    require(observation['free_disk_bytes'] >= (INITIAL_DISK if initial else RUNTIME_DISK),'DISK_FLOOR')
    if initial: require(observation['available_ram_bytes'] >= RAM,'RAM_FLOOR')


def stops():
    bases = (Path('/home/ubuntu'),PROJECT,ROOT,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()])
    for base in bases:
        for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):
            require(not os.path.lexists(base/rel),'STOP:'+str(base/rel))


def retained_inputs(raw=None):
    if raw is None: raw = plain(ROOT/'INPUTS.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==PINS['INPUTS.json'],'INPUTS_PIN')
    files = json.loads(raw)['files']
    require(len(files)==17,'EXACT_17_INPUTS')
    observed = {}
    for path,pin in files.items():
        observed[path] = file_record(Path(path))
        require(observed[path]==pin,'RETAINED_INPUT_CHANGED:'+path)
    return observed


def validate_release(release,stage):
    require(release.get('label')==LABEL and release.get('host')=='trn21','RELEASE_BINDING')
    require(release.get('stage_authorized') is True,'STAGING_DISABLED')
    if not stage: require(release.get('launch_authorized') is True,'LAUNCH_DISABLED')
    require(release.get('submission_authorized') is False,'NO_SUBMISSION_AUTHORITY')
    for key in ('mechanics_review_sha256','operator_assignment_sha256','packet_manifest_sha256','runtime_inspection_sha256'):
        require(isinstance(release.get(key),str) and re.fullmatch('[a-f0-9]{64}',release[key]),'MISSING_RELEASE_PIN:'+key)
    require(set(release.get('runtime_pins',{}))==set(RUNTIME_PATHS),'MISSING_RUNTIME_PINS')
    for path,pin in release['runtime_pins'].items():
        require(isinstance(pin,dict) and set(pin)=={'resolved_path','link_target','sha256','bytes'},'UNBOUND_RUNTIME:'+path)
    control = release.get('fresh_controller_receipt')
    require(isinstance(control,dict),'MISSING_CONTROLLER_RECEIPT')
    require(control.get('label')==LABEL and control.get('local_stops_present')==[],'CONTROLLER_STOP')
    require(control.get('host_launch_allowed') is True,'CONTROLLER_HOST_HOLD')
    require(control.get('global_correctness_breaker') is False and control.get('trn21_breaker') is False,'BREAKER')
    observed = dt.datetime.fromisoformat(control['utc']).timestamp()
    require(0 <= time.time()-observed <= 60,'STALE_CONTROLLER_RECEIPT')


def staging_preflight(release,input_bytes,lock_fd,lifetime,idle):
    """Read-only entry for a future separately reviewed Operator staging transport.

    The transport must retain this original lock while checking, copying, and
    recording its result. No such transport/release is supplied by this packet.
    """
    validate_release(release,True)
    require(socket.gethostname()==HOST and os.getuid()==1000,'HOST_IDENTITY')
    lifetime.require_original_lock(lock_fd,LOCK,LOCK_ID)
    require(not os.path.lexists(ROOT),'STAGING_ROOT_ALREADY_EXISTS')
    stops()
    observed = resources(); resource_gate(observed,True)
    evidence = {}
    idle.check_idle(subprocess.run,os.getpid(),os.getppid(),evidence)
    require(Path('/dev/neuron0').is_char_device(),'DEVICE_NODE_MISSING')
    holders = subprocess.run(['/usr/bin/fuser','/dev/neuron0'],capture_output=True,text=True,timeout=15,env=idle.CHECK_ENV)
    evidence['fuser'] = {'returncode':holders.returncode,'stdout':holders.stdout,'stderr':holders.stderr}
    require(holders.returncode==1 and not holders.stdout.strip() and not holders.stderr.strip(),'STAGING_DEVICE_BUSY')
    for path,pin in release['runtime_pins'].items():
        require(runtime_record(path)==pin,'RUNTIME_CHANGED:'+path)
    inputs = retained_inputs(input_bytes)
    lifetime.require_original_lock(lock_fd,LOCK,LOCK_ID)
    return {'utc':utc(),'stage':'before_staging_under_original_lock',**observed,
            'idle':evidence,'retained_inputs':inputs,'lock_identity':list(LOCK_ID)}


def preflight(release,stage=False):
    validate_release(release,stage)
    require(socket.gethostname()==HOST and os.getuid()==1000,'HOST_IDENTITY')
    stops()
    observed = resources(); resource_gate(observed,True)
    for path,pin in release['runtime_pins'].items():
        require(runtime_record(path)==pin,'RUNTIME_CHANGED:'+path)
    for name,pin in PINS.items():
        require(file_record(ROOT/name)['sha256']==pin,'PACKET_PIN:'+name)
    require(file_record(ROOT/'runtime-inspection.json')['sha256']==release['runtime_inspection_sha256'],'RUNTIME_INSPECTION_PIN')
    inputs = retained_inputs()
    require(not any(os.path.lexists(ROOT/name) for name in ('receipts','caches','metrics')),'OUTPUT_ALREADY_EXISTS')
    return {'utc':utc(),'stage':'staging' if stage else 'inside_lifetime_lock',**observed,'retained_inputs':inputs}



def accounted_entries(root, cache_link):
    """Descriptor-relative census. Never open or resolve a symlink target."""
    root = Path(root)
    identity = lambda s: (s.st_dev, s.st_ino, s.st_mode)
    stamp = lambda s: (*identity(s), s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    def walk(fd, relative):
        with os.scandir(fd) as items:
            for item in items:
                name = item.name
                rel = relative / name
                before = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(before.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        require(identity(before) == identity(os.fstat(child)), 'SCAN_DIRECTORY_CHANGED')
                        yield rel, before, None
                        yield from walk(child, rel)
                        require(identity(before) == identity(os.stat(name, dir_fd=fd, follow_symlinks=False)),
                                'SCAN_DIRECTORY_CHANGED')
                    finally:
                        os.close(child)
                else:
                    target = None
                    if stat.S_ISLNK(before.st_mode):
                        require(cache_link(rel), 'NONCACHE_SYMLINK:' + str(rel))
                        target = os.readlink(name, dir_fd=fd)
                    else:
                        require(stat.S_ISREG(before.st_mode), 'SCAN_NONREGULAR:' + str(rel))
                    after = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    require(stamp(before) == stamp(after), 'SCAN_ENTRY_CHANGED:' + str(rel))
                    yield rel, before, target
    before = root.lstat()
    require(stat.S_ISDIR(before.st_mode), 'SCAN_ROOT_NOT_DIRECTORY')
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        require(identity(before) == identity(os.fstat(fd)), 'SCAN_ROOT_CHANGED')
        yield from walk(fd, Path())
        require(identity(before) == identity(root.lstat()), 'SCAN_ROOT_CHANGED')
    finally:
        os.close(fd)


def link_record(st, target):
    return dict(bytes=st.st_size, target=target, device=st.st_dev, inode=st.st_ino,
                mode=st.st_mode, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def compiler_link(relative):
    return len(relative.parts) >= 3 and relative.parts[0] == 'caches' and relative.parts[1] in (
        'neff', 'local', 'hlo', 'inductor', 'compiler', 'debug')

def sizes(root):
    artifacts, caches = 0, 0
    links = {}
    ranks = {str(rank):0 for rank in range(4)}
    for relative, st, target in accounted_entries(root, compiler_link):
        if stat.S_ISDIR(st.st_mode):
            continue
        if relative.parts[0] == 'caches':
            caches += st.st_size
            if target is not None:
                links[str(relative)] = link_record(st, target)
        else:
            artifacts += st.st_size
        if relative.parts[0] == 'receipts':
            for rank in ranks:
                if relative.name.startswith(('rank'+rank+'.', '.rank'+rank+'.')):
                    ranks[rank] += st.st_size
    return {'artifact_bytes':artifacts, 'cache_bytes':caches,
            'cache_symlinks':links, 'rank_receipt_bytes':ranks}


def artifact_gate(observation):
    require(observation['artifact_bytes']<=ARTIFACT_CAP,'ARTIFACT_CAP')
    require(all(n<=RANK_CAP for n in observation['rank_receipt_bytes'].values()),'RANK_RECEIPT_CAP')


class Monitor:
    """Fixed diagnostic watchdog; cancellation is handled by the unchanged guardian."""
    def __init__(self,guardian):
        self.guardian = guardian
        self.stop = threading.Event()
        self.error = None
        self.latest = None
        self.peak_artifact_bytes = 0
        self.peak_cache_bytes = 0
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.watch,name='clean36-resource-watch',daemon=True)

    def check(self):
        observation = {'utc':utc(),'elapsed_seconds':time.monotonic()-self.started,**resources(),**sizes(ROOT)}
        owned_rss = []
        for (_,start),(_,row) in tuple(self.guardian.owned.items()):
            try:
                identity = Path('/proc/'+str(row['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
                if int(identity[19])!=start: continue
                status = Path('/proc/'+str(row['pid'])+'/status').read_text().splitlines()
                rss = next((int(line.split()[1])*1024 for line in status if line.startswith('VmRSS:')),0)
                owned_rss.append({'pid':row['pid'],'start_ticks':start,'rss_bytes':rss})
            except (FileNotFoundError,ProcessLookupError): pass
        observation['owned_process_rss'] = owned_rss
        self.peak_artifact_bytes = max(self.peak_artifact_bytes,observation['artifact_bytes'])
        self.peak_cache_bytes = max(self.peak_cache_bytes,observation['cache_bytes'])
        observation.update(peak_artifact_bytes=self.peak_artifact_bytes,peak_cache_bytes=self.peak_cache_bytes)
        self.latest = observation
        # One bounded latest observation also survives a later logging failure.
        with (ROOT/'resource-progress.tmp').open('x') as stream:
            json.dump(observation,stream,allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(ROOT/'resource-progress.tmp',ROOT/'resource-progress.json')
        resource_gate(observation,False); artifact_gate(observation)
        require(observation['elapsed_seconds']<CHILD_SECONDS,'NATIVE_900_SECOND_DEADLINE')
        return observation

    def watch(self):
        try:
            while not self.stop.is_set():
                self.check()
                self.stop.wait(min(.25,max(0,CHILD_SECONDS-(time.monotonic()-self.started))))
        except BaseException as error:
            self.error = type(error).__name__+': '+str(error)
            # Handler records only a flag. Guardian.run owns every pidfd signal,
            # descendants and cleanup; no process-name or process-group killing.
            self.guardian._signal(signal.SIGTERM,None)

    def finish(self):
        self.stop.set(); self.thread.join(timeout=1)
        if self.thread.is_alive(): self.guardian.park('RESOURCE_MONITOR_UNRESOLVED')


def run_child(guardian):
    monitor = Monitor(guardian)
    record('diagnostic-launch.json',{'utc':utc(),'label':LABEL,'argv':COMMAND,'environment':ENV,
                                    'child_seconds':CHILD_SECONDS,'term_seconds':TERM_SECONDS,'quality_evidence':False})
    with (ROOT/'diagnostic.log').open('x') as output:
        monitor.thread.start()
        try:
            rc = guardian.run(COMMAND,cwd=ROOT,env=ENV,out=output)
        finally:
            monitor.finish()
    observation = {**resources(),**sizes(ROOT)}
    record('resource-observations.json',{'latest':monitor.latest,'terminal':observation,'error':monitor.error,
                                       'polling_can_overshoot':True})
    resource_gate(observation,False); artifact_gate(observation)
    require(monitor.error is None,'MONITOR_FAILED:'+str(monitor.error))
    return rc


def device_idle(idle,stage):
    require(Path('/dev/neuron0').is_char_device(),'DEVICE_NODE_MISSING')
    evidence = {}
    try:
        idle.check_idle(subprocess.run,os.getpid(),os.getppid(),evidence)
        fuser = subprocess.run(['/usr/bin/fuser','/dev/neuron0'],capture_output=True,text=True,timeout=15,env=idle.CHECK_ENV)
        evidence['fuser'] = {'returncode':fuser.returncode,'stdout':fuser.stdout,'stderr':fuser.stderr}
        require(fuser.returncode==1 and not fuser.stdout.strip() and not fuser.stderr.strip(),'DEVICE_HOLDERS_OR_FUSER_FAILED')
    finally:
        record(stage+'-idle.json',evidence)


def main():
    release_path = Path(__file__).parent/'RELEASE.json'
    if not release_path.exists():
        print('{"status":"DISABLED_NO_OPERATOR_RELEASE","stage_authorized":false,"launch_authorized":false}')
        return 2
    release = json.loads(plain(release_path).read_text())
    validate_release(release,False)
    require(Path(__file__).absolute()==ROOT/'host_driver.py','EXACT_DRIVER_ROOT')
    manifest = ROOT/'MANIFEST.json'
    require(file_record(manifest)['sha256']==release.get('packet_manifest_sha256'),'MANIFEST_RELEASE_PIN')
    for name,pin in json.loads(manifest.read_text()).items():
        require(file_record(ROOT/name)==pin,'MANIFEST_FILE:'+name)
    idle = module('clean36_idle',ROOT/'idle_guard.py')
    lifetime = module('clean36_lifetime',ROOT/'lifetime_guard.py')
    held = idle.inherited_lock(LOCK,ROOT/'host_driver.py')
    lifetime.require_original_lock(held['fd'],LOCK,LOCK_ID)
    guardian = lifetime.Guardian(held['fd'],LOCK,LOCK_ID,ROOT/'lifetime-events.jsonl',grace=TERM_SECONDS)
    code = 2
    try:
        record('in-lock-preflight.json',preflight(release))
        device_idle(idle,'preflight')
        lifetime.require_original_lock(held['fd'],LOCK,LOCK_ID)
        for name in ('receipts','metrics','caches'):
            (ROOT/name).mkdir()
        for name in CACHE_NAMES: (ROOT/'caches'/name).mkdir()
        record('fresh-caches.json',{'utc':utc(),'initially_absent':True,
               'paths':[str(ROOT/'caches'/name) for name in CACHE_NAMES],
               'effective_environment':ENV,'cache_hit_or_graph_coverage_claim':False})
        code = run_child(guardian)
        record('child-exit.json',{'utc':utc(),'returncode':code})
        require(code==0,'NATIVE_CHILD_EXIT:'+str(code))
        # Final receipt validation is offline and separately accepted after raw
        # retrieval. Zero child exit is deliberately not a native PASS here.
        record('terminal.json',{'utc':utc(),'status':'CHILD_EXIT_ZERO_PENDING_RECEIPT_AND_INDEPENDENT_REVIEW',
                                'launch_authorized':False,'quality_evidence':False})
    except BaseException as error:
        code = 2
        record('terminal.json',{'utc':utc(),'status':'UNRESOLVED','error':type(error).__name__+': '+str(error)})
    finally:
        def postflight():
            if not guardian.job_started:
                record('postflight-no-job.json',{'utc':utc(),'no_job_started':True,
                       'lineage_quiescence_required_by_guardian':True})
                return
            device_idle(idle,'postflight')
            record('post-inputs.json',{'utc':utc(),'retained_inputs':retained_inputs()})
            record('final-resources.json',{'utc':utc(),**resources(),**sizes(ROOT)})
            resource_gate(resources(),False)
            artifact_gate(sizes(ROOT))
        guardian.finish(postflight)
    return code if code>=0 else 128-code


if __name__=='__main__': raise SystemExit(main())
