"""Disabled, fixed trn21 packet staging/retrieval. No connection or launch code."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import sys
import types

ROOT = Path('/home/ubuntu/trainium_confirmed36_clean_native_consistency_20260926_trn21_r2')
LABEL = 'fs-20260926-confirmed36-clean-native-consistency-trn21-r2'
HOST = 'ip-172-31-34-85'
LOCK = '/home/ubuntu/.cache/trainium_freshstart/device0.lock'
LOCK_ID = (66305,596823)
MANIFEST_SHA = '0833adb1f4d4006ff485fdad45f6ff2488820b0cba0cba5b0045e64cbc84d9c0'
DRIVER_SHA = '044729e6dec4ac1fd87e960c38d892180c0d4bd94403ec044cde1610ae813954'
BINDING_SHA = 'd05b85bf08d921d2cdd59d0f8604d50f76dd68522f7119e4c96655bfa8aa9234'
CHECKPOINT_SHAS = {'623577bb7e6f60d183a1944b28a5624cd55767c16fcf03a9248fd4573f6a6762',
                   '3c7efed1e053cf7607e2099d52643cb368e8a1f14aebe2a36ba3f1444b63c9c9'}
CHUNK = 256*1024
PACKET_LIMIT = 1024*1024
PACKET_NAMES = {'INPUTS.json','NATIVE_BINDING.disabled.json','MANIFEST.json','host_driver.py',
                'idle_guard.py','lifetime_guard.py','dispatch_driver.py',
                'proposals/20260926-confirmed36-clean-native-consistency/probe.py',
                'proposals/20260926-confirmed36-clean-checkpoint-cpu/compare_checkpoints.py'}


def require(ok,message):
    if not ok: raise RuntimeError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def safe_name(name):
    require(isinstance(name,str) and name and '\\' not in name,'RELATIVE_NAME')
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path)==name and all(p not in ('.','..') for p in path.parts),'RELATIVE_NAME')
    return path


def packet_modules(files):
    require(set(files)==PACKET_NAMES and sum(map(len,files.values()))<=PACKET_LIMIT,'EXACT_SMALL_PACKET')
    require(sha(files['MANIFEST.json'])==MANIFEST_SHA,'PACKET_MANIFEST_PIN')
    manifest = json.loads(files['MANIFEST.json'])
    require(set(manifest)==PACKET_NAMES-{'MANIFEST.json'},'MANIFEST_FILE_SET')
    for name,pin in manifest.items():
        require({'sha256':sha(files[name]),'bytes':len(files[name])}==pin,'PACKET_FILE:'+name)
    require(sha(files['host_driver.py'])==DRIVER_SHA and sha(files['NATIVE_BINDING.disabled.json'])==BINDING_SHA,'DRIVER_BINDING_PIN')
    modules = []
    for name in ('host_driver.py','lifetime_guard.py','idle_guard.py'):
        module = types.ModuleType('fixed_'+name[:-3]); module.__file__ = str(ROOT/name)
        exec(compile(files[name],module.__file__,'exec'),module.__dict__)
        modules.append(module)
    return modules


def authorization(release,operation,source_sha):
    require(release.get('label')==LABEL and release.get('host')=='trn21','TRANSPORT_SCOPE')
    require(release.get('submission_authorized') is False,'NO_SUBMISSION_AUTHORITY')
    transport = release.get('transport',{})
    require(transport.get(operation+'_authorized') is True,'TRANSPORT_DISABLED')
    require(transport.get('source_sha256')==source_sha,'TRANSPORT_SOURCE_PIN')
    for key in ('review_sha256','operator_assignment_sha256','connection_receipt_sha256'):
        require(isinstance(transport.get(key),str) and re.fullmatch('[a-f0-9]{64}',transport[key]),'MISSING_TRANSPORT_PIN:'+key)
    require(transport.get('remote_command')==['/usr/bin/python3','-I','-B','-'],'EXACT_REMOTE_COMMAND')


def decode_payload(payload,source_sha):
    require(payload.get('operation') in ('stage','retrieve'),'FIXED_OPERATION')
    require(set(payload['files'])==PACKET_NAMES and sum(map(len,payload['files'].values()))<=2*PACKET_LIMIT,'ENCODED_PACKET_BOUND')
    files = {name:base64.b64decode(value,validate=True) for name,value in payload['files'].items()}
    driver,guardian,idle = packet_modules(files)
    release_raw = base64.b64decode(payload['release'],validate=True)
    require(len(release_raw)<=64*1024,'RELEASE_SIZE')
    release = json.loads(release_raw)
    authorization(release,payload['operation'],source_sha)
    require(socket.gethostname()==HOST and os.getuid()==1000,'HOST_IDENTITY')
    driver.plain(ROOT)
    return files,release_raw,release,driver,guardian,idle


def write_new(path,raw,mode=0o444):
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    path.chmod(mode)


def json_bytes(value): return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()


def stage(payload,source_sha):
    files,release_raw,release,d,g,idle = decode_payload(payload,source_sha)
    runtime = base64.b64decode(payload['runtime_inspection'],validate=True)
    require(len(runtime)<=CHUNK and sha(runtime)==release.get('runtime_inspection_sha256'),'RUNTIME_INSPECTION_PIN')
    require(release.get('packet_manifest_sha256')==MANIFEST_SHA,'RELEASE_MANIFEST_PIN')
    require(not os.path.lexists(ROOT),'STAGING_ROOT_ALREADY_EXISTS')
    fd = os.open(LOCK,os.O_RDWR|os.O_NOFOLLOW)
    created = False
    try:
        require((os.fstat(fd).st_dev,os.fstat(fd).st_ino)==LOCK_ID,'ORIGINAL_LOCK_IDENTITY_CHANGED')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before = d.staging_preflight(release,files['INPUTS.json'],fd,g,idle)
        ROOT.mkdir(mode=0o700); created = True
        d.record('staging-preflight.json',before)
        for name,raw in sorted(files.items()):
            relative = safe_name(name); target = ROOT/relative
            target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            d.plain(target); write_new(target,raw)
            require(d.file_record(target)=={'sha256':sha(raw),'bytes':len(raw)},'STAGED_PIN:'+name)
        write_new(ROOT/'runtime-inspection.json',runtime)
        after_inputs = d.retained_inputs()
        after_resources = d.resources(); d.resource_gate(after_resources,True)
        d.stops(); d.validate_release(release,True)
        for path,pin in release['runtime_pins'].items():
            require(d.runtime_record(path)==pin,'RUNTIME_CHANGED:'+path)
        d.device_idle(idle,'staging-postflight')
        g.require_original_lock(fd,LOCK,LOCK_ID)
        d.artifact_gate(d.sizes(ROOT))
        receipt = {'status':'STAGED_ONLY_NOT_LAUNCHED','label':LABEL,'utc':d.utc(),
                   'transport_source_sha256':source_sha,'lock_identity':list(LOCK_ID),
                   'retained_inputs':after_inputs,'resources':after_resources,
                   'staged_files':{name:{'sha256':sha(raw),'bytes':len(raw)} for name,raw in files.items()},
                   'checkpoint_contents_copied':False,'native_launch_performed':False}
        d.record('staging-complete.json',receipt)
        # Publish the separately authorized release after every staging gate.
        write_new(ROOT/'RELEASE.json',release_raw)
        parent_fd = os.open(ROOT,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(parent_fd)
        finally: os.close(parent_fd)
        return receipt
    except BaseException as error:
        if created:
            try: d.record('staging-failure.json',{'status':'HOLD_PARTIAL_RETAINED','error':repr(error),'utc':d.utc()})
            except BaseException: pass
        raise
    finally:
        os.close(fd)  # No job is spawned; synchronous preflight helpers have returned.



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

def scan(root):
    for relative, st, target in accounted_entries(root, compiler_link):
        if not stat.S_ISDIR(st.st_mode):
            yield root / relative, st, target


def census(d):
    result = {'utc':d.utc()}
    try: result['resources'] = d.resources()
    except Exception as error: result['resource_error'] = repr(error)
    for name,argv,seconds in [('ps',['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],15),
                              ('neuron_ls',['/opt/aws/neuron/bin/neuron-ls','--json-output'],20),
                              ('fuser',['/usr/bin/fuser','/dev/neuron0'],15)]:
        try:
            done = subprocess.run(argv,capture_output=True,text=True,timeout=seconds,env={'PATH':'/usr/bin:/bin:/opt/aws/neuron/bin'})
            result[name] = {'returncode':done.returncode,'stdout':done.stdout,'stderr':done.stderr}
        except Exception as error: result[name] = {'error':repr(error)}
    try:
        fd = os.open(LOCK,os.O_RDWR|os.O_NOFOLLOW)
        try:
            current = os.fstat(fd); result['lock_identity'] = [current.st_dev,current.st_ino]
            require(tuple(result['lock_identity'])==LOCK_ID,'ORIGINAL_LOCK_IDENTITY_CHANGED')
            try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: result['lock_busy'] = True
            else: result['lock_busy'] = False
        finally: os.close(fd)
    except Exception as error: result['lock_observation_error'] = repr(error)
    # This is observation, never a quiescence or native-result acceptance.
    return result


def authenticated_chunks(fd,offset,size,expected):
    """Authenticate immutable chunk bytes against a bounded fanout hash tree.

    Each parent is rehashed before its child hashes are trusted. Only a leaf
    bytes object whose hash matches that chain can leave this generator.
    At most 64 child hashes per level and two CHUNK buffers are resident;
    depth is at most eight for signed 64-bit file sizes. No temporary writes.
    """
    if size<=CHUNK:
        raw = os.pread(fd,size,offset)
        require(len(raw)==size and sha(raw)==expected,'RETRIEVAL_CHUNK_CHANGED')
        if raw: yield raw
        return
    width = (size+63)//64; children = []; whole = hashlib.sha256()
    for start in range(offset,offset+size,width):
        length = min(width,offset+size-start); digest = hashlib.sha256()
        for position in range(start,start+length,CHUNK):
            count = min(CHUNK,start+length-position)
            raw = os.pread(fd,count,position)
            require(len(raw)==count,'RETRIEVAL_CHUNK_CHANGED')
            whole.update(raw); digest.update(raw)
        children.append((start,length,digest.hexdigest()))
    require(whole.hexdigest()==expected,'RETRIEVAL_CHUNK_CHANGED')
    del raw
    for child in children: yield from authenticated_chunks(fd,*child)


def retrieve(payload,source_sha,emit):
    files,release_raw,release,d,g,idle = decode_payload(payload,source_sha)
    require(ROOT.is_dir(),'RETRIEVAL_ROOT_MISSING')
    records = {}; caches = {}; links = {}; artifact_bytes = 0; ranks = {str(i):0 for i in range(4)}
    before = census(d)
    runtime_errors = {}
    for path in d.RUNTIME_PATHS:
        try: require(d.runtime_record(path)==release.get('runtime_pins',{}).get(path),'RUNTIME_PIN')
        except Exception as error: runtime_errors[path] = repr(error)
    emit({'kind':'begin','root':str(ROOT),'label':LABEL,'transport_source_sha256':source_sha,'before':before})
    input_error = None
    try: before_inputs = d.retained_inputs(files['INPUTS.json'])
    except Exception as error: before_inputs = {}; input_error = repr(error)
    for path, st, target in scan(ROOT):
        name = str(path.relative_to(ROOT)); safe_name(name)
        if path.relative_to(ROOT).parts[0]=='caches':
            caches[name] = st.st_size
            if target is not None: links[name] = link_record(st, target)
            continue
        pin = d.file_record(path)
        require(pin['sha256'] not in CHECKPOINT_SHAS,'NO_CHECKPOINT_CONTENT_TRANSFER:'+name)
        artifact_bytes += pin['bytes']
        if path.relative_to(ROOT).parts[0]=='receipts':
            for rank in ranks:
                if path.name.startswith(('rank'+rank+'.','.rank'+rank+'.')): ranks[rank] += pin['bytes']
        fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            require(stat.S_ISREG(os.fstat(fd).st_mode),'RETRIEVAL_NONREGULAR')
            emit({'kind':'file','name':name,**pin})
            digest = hashlib.sha256(); size = 0
            for raw in authenticated_chunks(fd,0,pin['bytes'],pin['sha256']):
                size += len(raw); digest.update(raw)
                emit({'kind':'chunk','b64':base64.b64encode(raw).decode('ascii')})
        finally: os.close(fd)
        require({'bytes':size,'sha256':digest.hexdigest()}==pin and d.file_record(path)==pin,'RETRIEVAL_FILE_CHANGED:'+name)
        records[name] = pin; emit({'kind':'end_file','name':name,**pin})
    # Reinventory and rehash: additions/removals and post-read changes fail completeness.
    after_records = {}; after_caches = {}; after_links = {}
    for p, st, target in scan(ROOT):
        name = str(p.relative_to(ROOT))
        if p.relative_to(ROOT).parts[0] == 'caches':
            after_caches[name] = st.st_size
            if target is not None: after_links[name] = link_record(st, target)
        else:
            after_records[name] = d.file_record(p)
    require(after_records==records,'RETRIEVAL_INVENTORY_CHANGED')
    require(after_caches==caches and after_links==links,'RETRIEVAL_CACHE_CHANGED')
    try:
        after_inputs = d.retained_inputs(files['INPUTS.json'])
        require(before_inputs==after_inputs,'RETAINED_INPUTS_CHANGED')
    except Exception as error: after_inputs = {}; input_error = input_error or repr(error)
    gate = artifact_bytes<=d.ARTIFACT_CAP and all(n<=d.RANK_CAP for n in ranks.values())
    after = census(d)
    observations_ok = all('resource_error' not in row and 'lock_observation_error' not in row
                          and row.get('lock_identity')==list(LOCK_ID)
                          and all('error' not in row.get(key,{}) for key in ('ps','neuron_ls','fuser')) for row in (before,after))
    receipt = {'kind':'complete','status':'COMPLETE_COPY_PENDING_INDEPENDENT_REVIEW' if gate and input_error is None and not runtime_errors and observations_ok else 'COMPLETE_COPY_FAILED_GATE',
               'files':records,'artifact_bytes':artifact_bytes,'rank_receipt_bytes':ranks,
               'artifact_gate_pass':gate,'cache_file_bytes':caches,'cache_bytes':sum(caches.values()),
               'cache_symlinks':links, 'cache_targets_read':False,
               'retained_inputs':after_inputs,'input_error':input_error,'runtime_errors':runtime_errors,'after':after,
               'checkpoint_contents_copied':False,'native_acceptance':False}
    emit(receipt)
    return receipt


def receive(lines,destination):
    require(destination.is_absolute() and all(not p.is_symlink() for p in (destination,*destination.parents)),'LOCAL_DESTINATION')
    destination.mkdir(mode=0o700)
    (destination/'files').mkdir(mode=0o700)
    begin = None; complete = None; records = {}; current = None; stream = None
    try:
        for line in lines:
            require(len(line)<=2*CHUNK,'WIRE_LINE_BOUND')
            item = json.loads(line); kind = item.get('kind')
            require(complete is None,'TRAILING_WIRE_RECORD')
            if kind=='begin':
                require(begin is None and current is None and item.get('root')==str(ROOT) and item.get('label')==LABEL,'WIRE_BEGIN')
                begin = item
            elif kind=='file':
                require(begin is not None and current is None,'WIRE_FILE_ORDER')
                name = safe_name(item['name']); require(name.parts[0]!='caches','NO_CACHE_CONTENTS')
                require(type(item['bytes']) is int and item['bytes']>=0 and re.fullmatch('[a-f0-9]{64}',item['sha256']),'WIRE_FILE_PIN')
                require(item['sha256'] not in CHECKPOINT_SHAS,'NO_CHECKPOINT_CONTENT_TRANSFER')
                require(item['name'] not in records,'DUPLICATE_FILE')
                target = destination/'files'/name; target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
                stream = target.open('xb'); current = {'name':item['name'],'bytes':item['bytes'],'sha256':item['sha256']}
                digest = hashlib.sha256(); count = 0
            elif kind=='chunk':
                require(current is not None,'WIRE_CHUNK_ORDER')
                raw = base64.b64decode(item['b64'],validate=True); require(len(raw)<=CHUNK,'WIRE_CHUNK_BOUND')
                count += len(raw); require(count<=current['bytes'],'WIRE_FILE_SIZE')
                stream.write(raw); digest.update(raw)
            elif kind=='end_file':
                require(current is not None and {k:item[k] for k in ('name','bytes','sha256')}==current,'WIRE_END_FILE')
                require(count==current['bytes'] and digest.hexdigest()==current['sha256'],'WIRE_FILE_HASH')
                stream.flush(); os.fsync(stream.fileno()); stream.close(); stream = None
                records[current['name']] = {'bytes':count,'sha256':digest.hexdigest()}; current = None
            elif kind=='complete':
                require(begin is not None and current is None and item['files']==records,'WIRE_COMPLETE')
                require(item.get('artifact_bytes')==sum(p['bytes'] for p in records.values()),'WIRE_ARTIFACT_BYTES')
                require(item.get('native_acceptance') is False and item.get('checkpoint_contents_copied') is False,'WIRE_NO_NATIVE_ACCEPTANCE')
                complete = item
            else: raise RuntimeError('WIRE_ERROR_OR_UNKNOWN:'+str(item))
        require(complete is not None,'WIRE_INCOMPLETE')
        write_new(destination/'retrieval.json',json_bytes({'begin':begin,'complete':complete}))
        return complete
    except BaseException as error:
        try: write_new(destination/'retrieval-failure.json',json_bytes({'status':'INCOMPLETE_PARTIAL_RETAINED','error':repr(error),'completed_files':records}))
        except BaseException: pass
        raise
    finally:
        if stream is not None: stream.close()


def make_wire(packet,release_path,runtime_path,operation):
    require(all(not p.is_symlink() for p in (packet,*packet.parents)),'LOCAL_PACKET_SYMLINK')
    files = {}
    for name in PACKET_NAMES:
        path = packet/name
        require(all(not p.is_symlink() for p in (path,*path.parents)) and path.is_file(),'LOCAL_PACKET_FILE')
        files[name] = path.read_bytes()
    packet_modules(files)
    source = Path(__file__).read_bytes(); source_sha = sha(source)
    binding = json.loads(Path(__file__).with_name('TRANSPORT_BINDING.disabled.json').read_bytes())
    require(binding['transport_source_sha256']==source_sha,'LOCAL_TRANSPORT_PIN')
    release = release_path.read_bytes(); authorization(json.loads(release),operation,source_sha)
    payload = {'operation':operation,'files':{n:base64.b64encode(raw).decode('ascii') for n,raw in files.items()},
               'release':base64.b64encode(release).decode('ascii')}
    if operation=='stage': payload['runtime_inspection'] = base64.b64encode(runtime_path.read_bytes()).decode('ascii')
    return ('P='+repr(payload)+'\nTRANSPORT_SOURCE_SHA='+repr(source_sha)+'\n').encode()+source


def main():
    if 'P' in globals():
        emit = lambda item: print(json.dumps(item,sort_keys=True,allow_nan=False),flush=True)
        try:
            if P['operation']=='stage': emit(stage(P,TRANSPORT_SOURCE_SHA))
            else: retrieve(P,TRANSPORT_SOURCE_SHA,emit)
        except BaseException as error:
            emit({'kind':'error','status':'HOLD_PARTIAL_RETAINED','error':repr(error)})
            return 2
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path)
    parser.add_argument('--release',type=Path)
    parser.add_argument('--runtime-inspection',type=Path)
    parser.add_argument('--emit-wire',choices=('stage','retrieve'))
    parser.add_argument('--receive',type=Path)
    args = parser.parse_args()
    if args.receive:
        require(not args.emit_wire,'ONE_OPERATION')
        receive(iter(lambda:sys.stdin.buffer.readline(2*CHUNK+1),b''),args.receive)
    elif args.emit_wire:
        require(args.packet is not None and args.release is not None,'SEPARATE_RELEASE_REQUIRED')
        require(args.emit_wire!='stage' or args.runtime_inspection is not None,'RUNTIME_INSPECTION_REQUIRED')
        sys.stdout.buffer.write(make_wire(args.packet,args.release,args.runtime_inspection,args.emit_wire))
    else:
        print('{"status":"DISABLED_NO_OPERATOR_RELEASE","stage_authorized":false,"launch_authorized":false}')
        return 2
    return 0


if __name__=='__main__': raise SystemExit(main())
