"""Author stdlib/local-file faults only. No hosts, credentials, native, or tensors."""
import base64
import contextlib
import datetime as dt
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).absolute().parent
PACKET = HERE.parent/'20260926-confirmed36-clean-native-trn21-packet-r1/packet'
spec = importlib.util.spec_from_file_location('fixed_transport',HERE/'transport.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
FILES = {name:(PACKET/name).read_bytes() for name in t.PACKET_NAMES}
SOURCE_SHA = t.sha((HERE/'transport.py').read_bytes())


def release(d):
    return {'label':t.LABEL,'host':'trn21','submission_authorized':False,'stage_authorized':True,'launch_authorized':False,
            'mechanics_review_sha256':'1'*64,'operator_assignment_sha256':'2'*64,'packet_manifest_sha256':t.MANIFEST_SHA,
            'runtime_inspection_sha256':t.sha(b'CPU-only inspection fixture'),
            'runtime_pins':{p:{'resolved_path':p,'link_target':None,'sha256':'3'*64,'bytes':1} for p in d.RUNTIME_PATHS},
            'fresh_controller_receipt':{'label':t.LABEL,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                'local_stops_present':[],'host_launch_allowed':True,'global_correctness_breaker':False,'trn21_breaker':False},
            'transport':{'stage_authorized':True,'retrieve_authorized':True,'source_sha256':SOURCE_SHA,
                'review_sha256':'4'*64,'operator_assignment_sha256':'5'*64,'connection_receipt_sha256':'6'*64,
                'remote_command':['/usr/bin/python3','-I','-B','-']}}


def payload(d,operation='stage'):
    return {'operation':operation,'files':{k:base64.b64encode(v).decode() for k,v in FILES.items()},
            'release':base64.b64encode(t.json_bytes(release(d))).decode(),
            'runtime_inspection':base64.b64encode(b'CPU-only inspection fixture').decode()}


def busy(path):
    fd = os.open(path,os.O_RDWR)
    try:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return True
        return False
    finally: os.close(fd)


@contextlib.contextmanager
def fixture(root,operation='stage'):
    d,g,idle = t.packet_modules(FILES)
    lock = root/'local.lock'; lock.touch(); identity = (lock.stat().st_dev,lock.stat().st_ino)
    output = root/'packet'; good = {'free_disk_bytes':d.INITIAL_DISK,'available_ram_bytes':d.RAM}
    body = payload(d,operation)
    with contextlib.ExitStack() as stack:
        for obj,name,value in [(t,'ROOT',output),(t,'LOCK',str(lock)),(t,'LOCK_ID',identity),
                               (d,'ROOT',output),(d,'LOCK',str(lock)),(d,'LOCK_ID',identity)]:
            stack.enter_context(mock.patch.object(obj,name,value))
        stack.enter_context(mock.patch.object(t,'packet_modules',return_value=(d,g,idle)))
        stack.enter_context(mock.patch.object(t.socket,'gethostname',return_value=t.HOST))
        stack.enter_context(mock.patch.object(d,'resources',return_value=good))
        stack.enter_context(mock.patch.object(d,'stops'))
        retained = stack.enter_context(mock.patch.object(d,'retained_inputs',return_value={'seventeen':'CPU-metadata-only'}))
        stack.enter_context(mock.patch.object(d,'runtime_record',side_effect=lambda p:release(d)['runtime_pins'][p]))
        stack.enter_context(mock.patch.object(idle,'check_idle'))
        stack.enter_context(mock.patch.object(Path,'is_char_device',return_value=True))
        stack.enter_context(mock.patch.object(d.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','')))
        yield output,lock,body,d,g,retained


class TransportTests(unittest.TestCase):
    def test_disabled_and_exact_unchanged_packet(self):
        binding = json.loads((HERE/'TRANSPORT_BINDING.disabled.json').read_text())
        self.assertEqual(binding['transport_source_sha256'],SOURCE_SHA)
        self.assertEqual(t.sha(FILES['MANIFEST.json']),t.MANIFEST_SHA)
        self.assertEqual(t.sha(FILES['host_driver.py']),t.DRIVER_SHA)
        self.assertEqual(t.sha(FILES['NATIVE_BINDING.disabled.json']),t.BINDING_SHA)
        self.assertEqual(t.LOCK_ID,(66305,596823))
        self.assertEqual(t.packet_modules(FILES)[0].CHILD_SECONDS,900)
        self.assertEqual(t.packet_modules(FILES)[0].TERM_SECONDS,15)
        for k in ('stage_authorized','retrieval_authorized','launch_authorized','native_launch_authorized','submission_authorized','host_reserved'):
            self.assertIs(binding[k],False)
        observation_raw = (HERE/'evidence/RUNTIME_INSPECTION.json').read_bytes()
        self.assertEqual(t.sha(observation_raw),binding['runtime_inspection_sha256'])
        self.assertEqual(binding['runtime_pins'],json.loads(observation_raw)['remote']['runtime_pins'])
        self.assertEqual(len(binding['runtime_pins']),8)
        self.assertTrue(all(pin is None for pin in json.loads(FILES['NATIVE_BINDING.disabled.json'])['runtime_pins'].values()))
        result = subprocess.run([sys.executable,'-B',str(HERE/'transport.py')],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('DISABLED_NO_OPERATOR_RELEASE',result.stdout)

    def test_payload_release_and_source_faults(self):
        d,_,_ = t.packet_modules(FILES)
        for key in ('stage_authorized','source_sha256','review_sha256','operator_assignment_sha256','connection_receipt_sha256','remote_command'):
            bad = release(d); bad['transport'][key] = None
            with self.subTest(key=key),self.assertRaises(RuntimeError): t.authorization(bad,'stage',SOURCE_SHA)
        for name in t.PACKET_NAMES:
            bad = dict(FILES); bad[name] += b'x'
            with self.subTest(name=name),self.assertRaises(RuntimeError): t.packet_modules(bad)
        bad = dict(FILES); bad['extra'] = b'x'
        with self.assertRaises(RuntimeError): t.packet_modules(bad)

    def test_stage_exact_small_files_under_real_lock_no_launch(self):
        with tempfile.TemporaryDirectory() as tmp,fixture(Path(tmp)) as (output,lock,body,d,g,retained):
            original = t.write_new; observations = []
            def checked(path,raw,mode=0o444):
                observations.append(busy(lock)); return original(path,raw,mode)
            with mock.patch.object(t,'write_new',side_effect=checked): result = t.stage(body,SOURCE_SHA)
            self.assertTrue(observations and all(observations)); self.assertFalse(busy(lock))
            self.assertEqual(result['status'],'STAGED_ONLY_NOT_LAUNCHED')
            self.assertFalse(result['native_launch_performed']); self.assertFalse(result['checkpoint_contents_copied'])
            for name,raw in FILES.items(): self.assertEqual((output/name).read_bytes(),raw)
            self.assertEqual(json.loads((output/'RELEASE.json').read_bytes())['launch_authorized'],False)
            self.assertEqual(retained.call_count,2)
            self.assertFalse((output/'receipts').exists()); self.assertFalse((output/'caches').exists())
            with self.assertRaisesRegex(RuntimeError,'STAGING_ROOT_ALREADY_EXISTS'): t.stage(body,SOURCE_SHA)

    def test_stage_faults_preserve_unused_root_and_original_lock(self):
        for mode in ('disk','ram','stale','runtime','inputs','wrong_inode','busy','host'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp,fixture(Path(tmp)) as (output,lock,body,d,g,retained):
                with contextlib.ExitStack() as faults:
                    if mode in ('disk','ram'):
                        obs = {'free_disk_bytes':d.INITIAL_DISK-(mode=='disk'),'available_ram_bytes':d.RAM-(mode=='ram')}
                        faults.enter_context(mock.patch.object(d,'resources',return_value=obs))
                    elif mode=='stale':
                        raw = release(d); raw['fresh_controller_receipt']['utc']='2020-01-01T00:00:00+00:00'; body['release']=base64.b64encode(t.json_bytes(raw)).decode()
                    elif mode=='runtime': faults.enter_context(mock.patch.object(d,'runtime_record',return_value={}))
                    elif mode=='inputs': faults.enter_context(mock.patch.object(d,'retained_inputs',side_effect=RuntimeError('INPUT_CHANGED')))
                    elif mode=='wrong_inode': faults.enter_context(mock.patch.object(t,'LOCK_ID',(0,1)))
                    elif mode=='host': faults.enter_context(mock.patch.object(t.socket,'gethostname',return_value='wrong'))
                    elif mode=='busy':
                        fd = os.open(lock,os.O_RDWR); fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); faults.callback(os.close,fd)
                    with self.assertRaises((RuntimeError,BlockingIOError)): t.stage(body,SOURCE_SHA)
                    self.assertFalse(output.exists())
                self.assertFalse(busy(lock))

    def test_stage_write_failure_keeps_partial_without_release(self):
        with tempfile.TemporaryDirectory() as tmp,fixture(Path(tmp)) as (output,lock,body,d,g,retained):
            original = t.write_new; calls = []
            def fail(path,raw,mode=0o444):
                calls.append(path.name); self.assertTrue(busy(lock))
                if len(calls)==3: raise OSError('injected disk write failure')
                return original(path,raw,mode)
            with mock.patch.object(t,'write_new',side_effect=fail),self.assertRaisesRegex(OSError,'injected'): t.stage(body,SOURCE_SHA)
            self.assertTrue(output.exists()); self.assertTrue((output/'staging-failure.json').exists())
            self.assertFalse((output/'RELEASE.json').exists()); self.assertFalse(busy(lock))
            self.assertTrue(any(p.is_file() for p in output.iterdir()))

    def retrieval(self,root,cap=False,input_fail=False):
        with fixture(root,'retrieve') as (output,lock,body,d,g,retained):
            output.mkdir(); (output/'receipts').mkdir(); (output/'caches').mkdir()
            expected = {'receipts/.rank0.interrupted.tmp':b'a'*17,'receipts/rank0.partial.json':b'{"partial":true}',
                        'binary.log':bytes(range(256))*(t.CHUNK//256+1),'resource-progress.tmp':b'progress'}
            for name,raw in expected.items(): (output/name).write_bytes(raw)
            (output/'caches/private').write_bytes(b'cache-content-not-transferred')
            observation = {'lock_identity':list(t.LOCK_ID),'ps':{},'neuron_ls':{},'fuser':{},'resources':d.resources()}
            records = []
            with mock.patch.object(t,'census',return_value=observation):
                if cap: d.ARTIFACT_CAP=1
                if input_fail: retained.side_effect=RuntimeError('INPUT_CHANGED')
                result = t.retrieve(body,SOURCE_SHA,records.append)
            complete = t.receive([t.json_bytes(row) for row in records],root/'received')
            for name,raw in expected.items(): self.assertEqual((root/'received/files'/name).read_bytes(),raw)
            self.assertFalse((root/'received/files/caches').exists())
            self.assertEqual(result,complete)
            self.assertFalse(result['native_acceptance'])
            return records,result

    def test_complete_hidden_partial_binary_and_cache_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            records,result = self.retrieval(Path(tmp))
            self.assertEqual(len(result['files']),4)
            self.assertEqual(result['status'],'COMPLETE_COPY_PENDING_INDEPENDENT_REVIEW')
            self.assertEqual(result['cache_bytes'],29)
            self.assertTrue(any(row['kind']=='chunk' for row in records))

    def test_cap_and_input_failures_retrieve_all_without_truncation(self):
        for cap,inputs in ((True,False),(False,True)):
            with self.subTest(cap=cap),tempfile.TemporaryDirectory() as tmp:
                records,result = self.retrieval(Path(tmp),cap,inputs)
                self.assertEqual(len(result['files']),4)
                self.assertEqual(result['status'],'COMPLETE_COPY_FAILED_GATE')

    def test_retrieval_scan_errors_and_symlinks_reject(self):
        for mode in ('root','nested','symlink'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp,fixture(Path(tmp),'retrieve') as (output,lock,body,d,g,retained):
                output.mkdir(); (output/'receipts').mkdir(); (output/'receipts/.rank0.tmp').write_bytes(b'x')
                if mode=='symlink': (output/'alias').symlink_to(output/'receipts/.rank0.tmp')
                original = os.scandir; target = output if mode=='root' else output/'receipts'
                def denied(path):
                    if mode!='symlink' and Path(path)==target: raise PermissionError('scan denied')
                    return original(path)
                with mock.patch.object(t,'census',return_value={}),mock.patch.object(t.os,'scandir',side_effect=denied),self.assertRaises((PermissionError,RuntimeError)):
                    t.retrieve(body,SOURCE_SHA,lambda row:None)

    def test_receiver_errors_keep_partial_and_refuse_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            records,result = self.retrieval(Path(tmp))
            for mode in ('early','hash','escape','absolute','checkpoint','duplicate','trailing'):
                bad = json.loads(json.dumps(records))
                first = next(i for i,row in enumerate(bad) if row['kind']=='file')
                if mode=='early': bad=bad[:-2]
                elif mode=='hash': bad[first]['sha256']='f'*64
                elif mode=='escape': bad[first]['name']='../escape'
                elif mode=='absolute': bad[first]['name']='/tmp/escape'
                elif mode=='checkpoint': bad[first]['sha256']=next(iter(t.CHECKPOINT_SHAS))
                elif mode=='duplicate': bad.insert(first+1,bad[first])
                elif mode=='trailing': bad.append({'kind':'begin'})
                with self.subTest(mode=mode),self.assertRaises(RuntimeError):
                    t.receive([t.json_bytes(row) for row in bad],Path(tmp)/('failed-'+mode))
                self.assertTrue((Path(tmp)/('failed-'+mode)/'retrieval-failure.json').exists())
            with self.assertRaises(FileExistsError): t.receive([],Path(tmp)/'received')
            self.assertFalse((Path(tmp)/'escape').exists())

    def test_midstream_mutation_and_new_inventory_file_refuse_completeness(self):
        for mode in ('mutation','new_file'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp,fixture(Path(tmp),'retrieve') as (output,lock,body,d,g,retained):
                output.mkdir(); path=output/'one.log'; path.write_bytes(b'original')
                seen=[]
                def emit(row):
                    seen.append(row)
                    if mode=='mutation' and row['kind']=='file': path.write_bytes(b'changed!')
                    if mode=='new_file' and row['kind']=='end_file': (output/'late.log').write_bytes(b'late')
                with mock.patch.object(t,'census',return_value={}),self.assertRaisesRegex(RuntimeError,'RETRIEVAL_'):
                    t.retrieve(body,SOURCE_SHA,emit)
                self.assertFalse(any(row['kind']=='complete' for row in seen))

    def test_sender_checkpoint_identity_refuses_content_transfer(self):
        with tempfile.TemporaryDirectory() as tmp,fixture(Path(tmp),'retrieve') as (output,lock,body,d,g,retained):
            output.mkdir(); (output/'unexpected.pt').write_bytes(b'tiny-only')
            records=[]
            with mock.patch.object(t,'census',return_value={}), \
                 mock.patch.object(d,'file_record',return_value={'bytes':9,'sha256':next(iter(t.CHECKPOINT_SHAS))}), \
                 self.assertRaisesRegex(RuntimeError,'NO_CHECKPOINT_CONTENT_TRANSFER'):
                t.retrieve(body,SOURCE_SHA,records.append)
            self.assertFalse(any(row['kind']=='chunk' for row in records))

    def test_wire_preparation_only_no_connection(self):
        d,_,_=t.packet_modules(FILES)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rel=root/'release.json'; run=root/'runtime.json'
            rel.write_bytes(t.json_bytes(release(d))); run.write_bytes(b'CPU-only inspection fixture')
            for operation in ('stage','retrieve'):
                raw=t.make_wire(PACKET,rel,run,operation)
                compile(raw,'<fixed-disabled-wire>','exec')
                self.assertIn(b'TRANSPORT_SOURCE_SHA=',raw)
            bad=release(d); bad['transport']['stage_authorized']=False; rel.write_bytes(t.json_bytes(bad))
            with self.assertRaisesRegex(RuntimeError,'TRANSPORT_DISABLED'): t.make_wire(PACKET,rel,run,'stage')
            alias=root/'alias'; alias.symlink_to(PACKET,target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError,'LOCAL_PACKET_SYMLINK'): t.make_wire(alias,rel,run,'stage')

    def test_checkpoint_replacement_and_mutation_never_reach_wire_or_receiver(self):
        prohibited = b'tiny prohibited checkpoint bytes'
        allowed = b'x'*len(prohibited)
        for mode in ('before_open','replace_after_header','mutate_after_header','mutate_after_read'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp,fixture(Path(tmp),'retrieve') as (output,lock,body,d,g,retained):
                output.mkdir(); path=output/'artifact.log'; path.write_bytes(allowed)
                records=[]; changed=False; original_record=d.file_record; original_read=os.pread
                def replace():
                    other=output/'replacement'; other.write_bytes(prohibited); os.replace(other,path)
                def record(target):
                    nonlocal changed
                    pin=original_record(target)
                    if mode=='before_open' and not changed:
                        changed=True; replace()
                    return pin
                def pread(fd,count,offset):
                    nonlocal changed
                    raw=original_read(fd,count,offset)
                    if mode=='mutate_after_read' and not changed:
                        changed=True; path.write_bytes(prohibited)
                    return raw
                def emit(row):
                    records.append(row)
                    if row['kind']=='file':
                        if mode=='replace_after_header': replace()
                        if mode=='mutate_after_header': path.write_bytes(prohibited)
                with mock.patch.object(t,'CHECKPOINT_SHAS',{t.sha(prohibited)}), \
                     mock.patch.object(t,'census',return_value={}), \
                     mock.patch.object(d,'file_record',side_effect=record), \
                     mock.patch.object(t.os,'pread',side_effect=pread), \
                     self.assertRaisesRegex(RuntimeError,'RETRIEVAL_'):
                    t.retrieve(body,SOURCE_SHA,emit)
                wire=b''.join(base64.b64decode(row['b64']) for row in records if row['kind']=='chunk')
                self.assertIn(wire,(b'',allowed)); self.assertNotEqual(wire,prohibited)
                self.assertFalse(any(row['kind']=='complete' for row in records))
                destination=Path(tmp)/'received'
                with self.assertRaisesRegex(RuntimeError,'WIRE_INCOMPLETE'):
                    t.receive([t.json_bytes(row) for row in records],destination)
                self.assertEqual((destination/'files/artifact.log').read_bytes(),wire)
                self.assertNotEqual((destination/'files/artifact.log').read_bytes(),prohibited)

    def test_multilevel_authentication_mutation_is_rejected_before_changed_chunk(self):
        with tempfile.TemporaryDirectory() as tmp,fixture(Path(tmp),'retrieve') as (output,lock,body,d,g,retained):
            output.mkdir(); path=output/'artifact.log'; allowed=b'x'*1024; prohibited=b'q'*1024
            path.write_bytes(allowed); records=[]; changed=False
            def emit(row):
                nonlocal changed
                records.append(row)
                if row['kind']=='chunk' and not changed:
                    changed=True; path.write_bytes(prohibited)
            with mock.patch.object(t,'CHUNK',8),mock.patch.object(t,'CHECKPOINT_SHAS',{t.sha(prohibited)}), \
                 mock.patch.object(t,'census',return_value={}),self.assertRaisesRegex(RuntimeError,'RETRIEVAL_CHUNK_CHANGED'):
                t.retrieve(body,SOURCE_SHA,emit)
            wire=b''.join(base64.b64decode(row['b64']) for row in records if row['kind']=='chunk')
            self.assertTrue(wire); self.assertEqual(wire,allowed[:len(wire)]); self.assertNotIn(b'q',wire)
            self.assertFalse(any(row['kind']=='complete' for row in records))

    def test_multilevel_authentication_is_bounded_and_exact_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture'; raw=bytes(range(251))*40; path.write_bytes(raw)
            fd=os.open(path,os.O_RDONLY); original=os.pread; reads=[]
            def bounded(descriptor,count,offset):
                reads.append(count); self.assertLessEqual(count,8)
                return original(descriptor,count,offset)
            try:
                with mock.patch.object(t,'CHUNK',8),mock.patch.object(t.os,'pread',side_effect=bounded):
                    copied=b''.join(t.authenticated_chunks(fd,0,len(raw),t.sha(raw)))
            finally: os.close(fd)
            self.assertEqual(copied,raw); self.assertTrue(reads)
            self.assertEqual(path.read_bytes(),raw)

    def test_readonly_census_distinguishes_busy_original_lock(self):
        with tempfile.TemporaryDirectory() as tmp,fixture(Path(tmp),'retrieve') as (output,lock,body,d,g,retained):
            for held in (False,True):
                fd=os.open(lock,os.O_RDWR)
                try:
                    if held: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    observation=t.census(d)
                    self.assertEqual(observation['lock_identity'],list(t.LOCK_ID))
                    self.assertEqual(observation['lock_busy'],held)
                    self.assertEqual(observation['fuser']['returncode'],1)
                finally: os.close(fd)


if __name__=='__main__': unittest.main(verbosity=2)
