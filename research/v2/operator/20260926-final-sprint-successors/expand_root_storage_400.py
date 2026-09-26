"""Human-authorized exact-root online growth, with fresh identity/geometry checks."""
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
CONNECTOR=REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
assert hashlib.sha256(CONNECTOR.read_bytes()).hexdigest()=='7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899'
spec=importlib.util.spec_from_file_location('connector',CONNECTOR);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
CONFIG={
 'trn21':{'hostname':'ip-172-31-34-85','serial':'vol09b9c7946aeaa8c57','lock_inode':596823,'guardian':(3231387,95011264),'workers':[(3231435,95011719),(3231436,95011719),(3231437,95011719),(3231438,95011719)]},
 'trn22':{'hostname':'ip-172-31-32-253','serial':'vol0d64a9cafbb4912e8','lock_inode':832589,'guardian':(3345115,102409331),'workers':[(3345162,102409821),(3345163,102409821),(3345164,102409821),(3345165,102409821)]}}
BODY=r'''
import datetime,fcntl,json,os,re,shutil,socket,stat,subprocess,time,traceback
from pathlib import Path
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
out={'host':HOST,'utc_start':utc(),'human_authority':'Human confirms both root EBS volumes set to400GiB and delegates safe online guest expansion; no AWS action','commands':[],'mutations':[]}
env={'PATH':'/usr/sbin:/usr/bin:/sbin:/bin','LANG':'C','LC_ALL':'C'}
def run(argv,timeout=15):
    start=utc()
    try:
        r=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,env=env)
        row={'argv':argv,'utc_start':start,'utc_end':utc(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    except subprocess.TimeoutExpired as e:
        row={'argv':argv,'utc_start':start,'utc_end':utc(),'returncode':'TIMEOUT','stdout':(e.stdout or b'').decode() if isinstance(e.stdout,bytes) else(e.stdout or ''),'stderr':(e.stderr or b'').decode() if isinstance(e.stderr,bytes) else(e.stderr or '')}
    if len(row['stdout'].encode())+len(row['stderr'].encode())>65536:raise RuntimeError('OUTPUT_BOUND')
    out['commands'].append(row)
    if row['returncode']!=0:raise RuntimeError('COMMAND_FAILED '+repr(argv)+' '+str(row['returncode'])+' '+row['stderr'])
    return row['stdout']
def process(pid,expected):
    p=Path('/proc')/str(pid);raw=(p/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
    row={'pid':pid,'start_ticks':int(v[19]),'state':v[0],'ppid':int(v[1]),'pgid':int(v[2]),'sid':int(v[3])}
    if row['start_ticks']!=expected or row['state']=='Z':raise RuntimeError('PROTECTED_PROCESS_IDENTITY_CHANGED')
    return row
def protected():
    rows=[process(*CFG['guardian'])]+[process(*p) for p in CFG['workers']]
    lock=Path('/home/ubuntu/.cache/trainium_freshstart/device0.lock')
    fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        s=os.fstat(fd)
        if (s.st_dev,s.st_ino)!=(66305,CFG['lock_inode']):raise RuntimeError('LOCK_IDENTITY_CHANGED')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:held=True
        else:
            fcntl.flock(fd,fcntl.LOCK_UN);held=False
        if not held:raise RuntimeError('PROTECTED_LIFETIME_LOCK_NOT_HELD')
    finally:os.close(fd)
    return {'processes':rows,'original_lock':{'device':s.st_dev,'inode':s.st_ino,'held':held}}
def geometry():
    mount=json.loads(run(['/usr/bin/findmnt','-J','-b','-o','SOURCE,FSTYPE,SIZE,USED,AVAIL,TARGET','/']))['filesystems']
    if len(mount)!=1:raise RuntimeError('ROOT_MOUNT_AMBIGUOUS')
    mount=mount[0]
    disks=json.loads(run(['/usr/bin/lsblk','-b','-J','-o','NAME,SIZE,FSTYPE,MOUNTPOINTS,SERIAL,PKNAME','/dev/nvme0n1']))['blockdevices']
    if len(disks)!=1:raise RuntimeError('ROOT_DISK_AMBIGUOUS')
    disk=disks[0]
    if disk['name']!='nvme0n1' or disk['serial']!=CFG['serial'] or disk['size']!=400*1024**3:raise RuntimeError('EXACT_ROOT_EBS_IDENTITY_OR_400GIB_NOT_CONFIRMED')
    partitions=[p for p in disk.get('children',[]) if '/' in p.get('mountpoints',[])]
    if len(partitions)!=1:raise RuntimeError('ROOT_PARTITION_AMBIGUOUS')
    part=partitions[0]
    if part['name']!='nvme0n1p1' or part['pkname']!='nvme0n1' or part['fstype']!='ext4' or mount['source']!='/dev/nvme0n1p1' or mount['fstype']!='ext4':raise RuntimeError('ROOT_LAYOUT_NOT_EXACT_NVME0N1_P1_EXT4')
    for path in ('/dev/nvme0n1','/dev/nvme0n1p1'):
        if not stat.S_ISBLK(os.lstat(path).st_mode):raise RuntimeError('ROOT_DEVICE_NOT_REGULAR_BLOCK_PATH')
    return {'mount':mount,'disk':disk,'root_partition':part,'df_hT':run(['/usr/bin/df','-hT','/'])}
try:
    if socket.gethostname()!=CFG['hostname']:raise RuntimeError('HOSTNAME_MISMATCH')
    out['protected_before']=protected();out['before']=geometry()
    grow=shutil.which('growpart',path=env['PATH']);resize=shutil.which('resize2fs',path=env['PATH']);tune=shutil.which('tune2fs',path=env['PATH'])
    if not all((grow,resize,tune)):raise RuntimeError('REQUIRED_INSTALLED_GUEST_TOOL_MISSING')
    # These boot partitions precede p1 in the already validated root layout.
    # Grow only p1 when more than the expected boot-prefix size is unused.
    disk=out['before']['disk'];part=out['before']['root_partition']
    if disk['size']-part['size']>2*1024**3:
        out['mutations'].append({'action':'grow_verified_root_partition1','utc':utc()})
        run(['/usr/bin/sudo','-n',grow,'/dev/nvme0n1','1'],timeout=30)
    out['after_partition']=geometry()
    extent=run(['/usr/bin/sudo','-n',tune,'-l','/dev/nvme0n1p1'])
    block_count=re.search(r'^Block count:\s+(\d+)\s*$',extent,re.M);block_size=re.search(r'^Block size:\s+(\d+)\s*$',extent,re.M)
    if not block_count or not block_size:raise RuntimeError('EXT4_SIZE_FIELDS_MISSING')
    fs_bytes=int(block_count[1])*int(block_size[1]);part_bytes=out['after_partition']['root_partition']['size']
    out['ext4_extent_before_resize_bytes']=fs_bytes
    if fs_bytes>part_bytes:raise RuntimeError('EXT4_EXTENT_EXCEEDS_PARTITION')
    if part_bytes-fs_bytes>=1024*1024:
        out['mutations'].append({'action':'grow_verified_root_ext4_online','utc':utc()})
        run(['/usr/bin/sudo','-n',resize,'/dev/nvme0n1p1'],timeout=45)
    out['after']=geometry();out['protected_after']=protected()
    if out['after']['disk']['size']-out['after']['root_partition']['size']>2*1024**3:raise RuntimeError('ROOT_PARTITION_STILL_SMALL')
    if out['after']['mount']['size']<0.9*out['after']['root_partition']['size']:raise RuntimeError('ROOT_FILESYSTEM_STILL_SMALL')
    out['status']='VERIFIED_ROOT400GIB_ONLINE_EXPANSION_COMPLETE' if out['mutations'] else 'ALREADY_EXPANDED_VERIFIED_NO_MUTATION'
except BaseException as e:
    out['status']='STORAGE_FOLLOWTHROUGH_FAILED_NO_RETRY';out['error']=str(e);out['traceback']=traceback.format_exc()
out['utc_end']=utc()
print(json.dumps(out))
'''
def one(host):
    out=HERE/(host+'-root400-expansion.json');attempt=HERE/(host+'-root400-expansion-attempt.json')
    assert not out.exists() and not attempt.exists()
    wire=('HOST='+repr(host)+'\nCFG='+repr(CONFIG[host])+'\n'+BODY).encode();assert not m.SENSITIVE.search(wire)
    with attempt.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'payload_sha256':hashlib.sha256(wire).hexdigest(),'target_root_volume_serial':CONFIG[host]['serial']},f,indent=2);f.write('\n')
    argv,*_=m.load_transport(host);argv[-1]='exec /usr/bin/python3 -I -B -S -'
    r=subprocess.run(argv,input=wire,capture_output=True,timeout=120);assert not m.SENSITIVE.search(r.stdout+r.stderr)
    receipt={'returncode':r.returncode,'stderr':r.stderr.decode(),'payload_sha256':hashlib.sha256(wire).hexdigest(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
    with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    z=receipt['remote'] or {}
    print(json.dumps({'host':host,'status':z.get('status'),'mutations':z.get('mutations'),'error':z.get('error'),'utc_end':z.get('utc_end'),'before':z.get('before'),'after':z.get('after'),'protected_after':z.get('protected_after'),'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(one,('trn21','trn22')))
