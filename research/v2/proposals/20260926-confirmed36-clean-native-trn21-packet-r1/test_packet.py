"""Author CPU/static mechanical checks. No host/native/scientific execution."""
import copy
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location('fixed_packet_driver',HERE/'packet/host_driver.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)


def release():
    return {'label':d.LABEL,'host':'trn21','stage_authorized':True,'launch_authorized':True,'submission_authorized':False,
            'mechanics_review_sha256':'1'*64,'operator_assignment_sha256':'2'*64,'packet_manifest_sha256':'3'*64,
            'runtime_inspection_sha256':'4'*64,
            'runtime_pins':{p:{'resolved_path':p,'link_target':None,'sha256':'5'*64,'bytes':1} for p in d.RUNTIME_PATHS},
            'fresh_controller_receipt':{'label':d.LABEL,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                'local_stops_present':[],'host_launch_allowed':True,'global_correctness_breaker':False,'trn21_breaker':False}}


def live(row):
    try:
        fields = Path('/proc/'+str(row['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
        return fields[0]!='Z' and int(fields[19])==row['start_ticks']
    except FileNotFoundError: return False


def busy(lock):
    fd = os.open(lock,os.O_RDWR)
    try:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return True
        return False
    finally: os.close(fd)


class PacketTests(unittest.TestCase):
    def test_exact_frozen_inputs_and_unchanged_guardians(self):
        for name,pin in d.PINS.items():
            self.assertEqual(hashlib.sha256((HERE/'packet'/name).read_bytes()).hexdigest(),pin,name)
        inputs = json.loads((HERE/'packet/INPUTS.json').read_text())
        self.assertEqual(len(inputs['files']),17)
        self.assertEqual({a['checkpoint_bytes'] for a in inputs['arms'].values()},{453048594,453044626})
        self.assertEqual(Path(d.PROBE).parent.parent/ '20260926-confirmed36-clean-checkpoint-cpu/compare_checkpoints.py',Path(d.FOUNDATION))

    def test_fixed_argv_environment_resources(self):
        self.assertEqual(d.COMMAND[:5],['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=15s','900s'])
        self.assertEqual(d.CHILD_ARGV[:3],[d.NATIVE+'torchrun','--standalone','--nproc_per_node=4'])
        self.assertEqual(d.LOCK_ID,(66305,596823))
        self.assertEqual((d.CHILD_SECONDS,d.TERM_SECONDS),(900,15))
        self.assertEqual((d.INITIAL_DISK,d.RUNTIME_DISK,d.RAM),(14495514624,11274289152,68719476736))
        self.assertEqual((d.ARTIFACT_CAP,d.RANK_CAP),(268435456,16777216))
        self.assertEqual(d.ENV['NEURON_LOGICAL_NC_CONFIG'],'2')
        self.assertEqual(d.ENV['NEURON_CC_FLAGS'],'--optlevel=1')
        self.assertEqual(d.ENV['NEURON_COMPILE_CACHE_URL'],d.ENV['NEURON_COMPILE_CACHE_DIR'])
        self.assertEqual(len({d.ENV[k] for k in d.CACHE_KEYS}),6)
        self.assertNotIn('prepare.py',d.CHILD_ARGV)

    def test_binding_stays_disabled_and_runtime_pins_unknown(self):
        binding = json.loads((HERE/'packet/NATIVE_BINDING.disabled.json').read_text())
        for key in ('stage_authorized','launch_authorized','native_launch_authorized','submission_authorized','host_reserved'):
            self.assertIs(binding[key],False)
        self.assertTrue(all(value is None for value in binding['runtime_pins'].values()))
        for key,value in [('command',d.COMMAND),('child_argv',d.CHILD_ARGV),('environment',d.ENV),('outer_command',d.OUTER),('support_pins',d.PINS)]:
            self.assertEqual(binding[key],value)
        self.assertFalse((HERE/'packet/RELEASE.json').exists())

    def test_complete_packet_manifest(self):
        manifest = json.loads((HERE/'packet/MANIFEST.json').read_text())
        files = {str(path.relative_to(HERE/'packet')) for path in (HERE/'packet').rglob('*') if path.is_file()}
        self.assertEqual(set(manifest),files-{'MANIFEST.json'})
        for name,pin in manifest.items():
            self.assertEqual(d.file_record(HERE/'packet'/name),pin)

    def test_disabled_default_and_release_faults(self):
        self.assertEqual(d.main(),2)
        d.validate_release(release(),False)
        changes = [('stage_authorized',False),('launch_authorized',False),('submission_authorized',True),
                   ('host','trn22'),('mechanics_review_sha256',None),('runtime_pins',{})]
        for key,value in changes:
            bad = release(); bad[key] = value
            with self.subTest(key=key),self.assertRaises(RuntimeError): d.validate_release(bad,False)
        for key,value in [('global_correctness_breaker',True),('trn21_breaker',True),('host_launch_allowed',False),
                          ('local_stops_present',['STOP']),('utc','2020-01-01T00:00:00+00:00')]:
            bad = release(); bad['fresh_controller_receipt'][key] = value
            with self.subTest(key=key),self.assertRaises(RuntimeError): d.validate_release(bad,False)

    def test_staging_and_launch_exact_resource_boundaries(self):
        d.resource_gate({'free_disk_bytes':d.INITIAL_DISK,'available_ram_bytes':d.RAM},True)
        for disk,ram in [(d.INITIAL_DISK-1,d.RAM),(d.INITIAL_DISK,d.RAM-1)]:
            with self.assertRaises(RuntimeError): d.resource_gate({'free_disk_bytes':disk,'available_ram_bytes':ram},True)
        d.resource_gate({'free_disk_bytes':d.RUNTIME_DISK,'available_ram_bytes':0},False)
        with self.assertRaises(RuntimeError): d.resource_gate({'free_disk_bytes':d.RUNTIME_DISK-1},False)

    def test_staging_checks_original_lock_resources_and_idle_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'unused-root'
            lifetime,idle = mock.Mock(),mock.Mock(CHECK_ENV={})
            released = release()
            good = {'free_disk_bytes':d.INITIAL_DISK,'available_ram_bytes':d.RAM}
            holders = subprocess.CompletedProcess([],1,'','')
            with mock.patch.object(d,'ROOT',root),mock.patch.object(d.socket,'gethostname',return_value=d.HOST), \
                 mock.patch.object(d.os,'getuid',return_value=1000),mock.patch.object(d,'stops'), \
                 mock.patch.object(d,'resources',return_value=good),mock.patch.object(Path,'is_char_device',return_value=True), \
                 mock.patch.object(d.subprocess,'run',return_value=holders) as run, \
                 mock.patch.object(d,'runtime_record',side_effect=lambda p:released['runtime_pins'][p]), \
                 mock.patch.object(d,'retained_inputs',return_value={'test':'metadata_only'}):
                result = d.staging_preflight(released,b'fixture',17,lifetime,idle)
                self.assertEqual(lifetime.require_original_lock.call_args_list,[mock.call(17,d.LOCK,d.LOCK_ID)]*2)
                self.assertEqual(result['free_disk_bytes'],d.INITIAL_DISK)
                self.assertFalse(root.exists())
                idle.check_idle.assert_called_once()
                run.assert_called_once()
                with mock.patch.object(d,'resources',return_value={**good,'free_disk_bytes':d.INITIAL_DISK-1}):
                    with self.assertRaisesRegex(RuntimeError,'DISK_FLOOR'):
                        d.staging_preflight(released,b'fixture',17,lifetime,idle)
                with mock.patch.object(d.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'123','')):
                    with self.assertRaisesRegex(RuntimeError,'STAGING_DEVICE_BUSY'):
                        d.staging_preflight(released,b'fixture',17,lifetime,idle)
                with mock.patch.object(d,'stops',side_effect=RuntimeError('STOP')):
                    with self.assertRaisesRegex(RuntimeError,'STOP'):
                        d.staging_preflight(released,b'fixture',17,lifetime,idle)

    def test_artifact_hidden_partial_final_and_cache_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'receipts').mkdir(); (root/'caches').mkdir()
            for name in ('rank0.json','rank0.partial.json','.rank0.fault.tmp'):
                (root/'receipts'/name).write_bytes(b'x'*7)
            (root/'caches/cache').write_bytes(b'c'*101)
            (root/'stdout.log').write_bytes(b'l'*5)
            result = d.sizes(root)
            self.assertEqual(result,{'artifact_bytes':26,'cache_bytes':101,'rank_receipt_bytes':{'0':21,'1':0,'2':0,'3':0}})
            d.artifact_gate({'artifact_bytes':d.ARTIFACT_CAP,'rank_receipt_bytes':{'0':d.RANK_CAP}})
            for field in ('artifact','rank'):
                bad = {'artifact_bytes':d.ARTIFACT_CAP,'rank_receipt_bytes':{'0':d.RANK_CAP}}
                if field=='artifact': bad['artifact_bytes'] += 1
                else: bad['rank_receipt_bytes']['0'] += 1
                with self.assertRaises(RuntimeError): d.artifact_gate(bad)
            (root/'symlink').symlink_to(root/'stdout.log')
            with self.assertRaises(RuntimeError): d.sizes(root)

    def test_file_and_manifest_pin_faults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); a=root/'a'; a.write_text('a'); b=root/'b'; b.symlink_to(a)
            self.assertEqual(d.file_record(a)['sha256'],hashlib.sha256(b'a').hexdigest())
            with self.assertRaises(RuntimeError): d.file_record(b)
            with self.assertRaises(RuntimeError): d.retained_inputs(b'{}')

    def test_injected_root_and_nested_scan_failures_cancel_monitor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'receipts').mkdir()
            (root/'receipts/.rank0.interrupted.tmp').write_bytes(b'x'*17)
            scan = os.scandir
            for target in (root,root/'receipts'):
                with self.subTest(target=target):
                    error = PermissionError(13,'injected directory scan denied',str(target))
                    def denied_scan(path):
                        if Path(path)==target: raise error
                        return scan(path)
                    fake = mock.Mock(owned={})
                    monitor = d.Monitor(fake)
                    with mock.patch.object(d.os,'scandir',side_effect=denied_scan), \
                         mock.patch.object(d,'ROOT',root),mock.patch.object(d,'resources',return_value={}):
                        with self.assertRaises(PermissionError) as raised:
                            d.artifact_gate(d.sizes(root))
                        self.assertIs(raised.exception,error)
                        monitor.watch()
                    self.assertIn('PermissionError',monitor.error)
                    self.assertIn(str(target),monitor.error)
                    fake._signal.assert_called_once_with(signal.SIGTERM,None)
            self.assertEqual(d.sizes(root)['rank_receipt_bytes']['0'],17)

    def test_real_root_and_nested_permission_and_missing_scan_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'receipts').mkdir()
            (root/'receipts/.rank0.interrupted.tmp').write_bytes(b'x'*17)
            for target in (root,root/'receipts'):
                with self.subTest(target=target):
                    mode = target.stat().st_mode & 0o777
                    target.chmod(0)
                    try:
                        with self.assertRaises(PermissionError) as raised:
                            d.artifact_gate(d.sizes(root))
                        self.assertEqual(raised.exception.filename,str(target))
                    finally:
                        target.chmod(mode)
            with self.assertRaises(FileNotFoundError): d.sizes(root/'missing')
            with self.assertRaises(NotADirectoryError): d.sizes(root/'receipts/.rank0.interrupted.tmp')

    def test_monitor_write_failure_requests_guardian_cancel(self):
        fake = mock.Mock(owned={})
        monitor = d.Monitor(fake)
        with mock.patch.object(monitor,'check',side_effect=OSError('receipt IO failure')):
            monitor.watch()
        self.assertIn('receipt IO failure',monitor.error)
        fake._signal.assert_called_once_with(signal.SIGTERM,None)

    def test_real_guardian_resource_timeout_and_helper_faults(self):
        for mode in ('timeout','disk','rankcap','helperfail','auditfail','scanroot','scannested'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); (root/'device0.lock').touch(); (root/'receipts').mkdir()
                command = [sys.executable]+(['-O'] if sys.flags.optimize else [])+['-B',str(HERE/'cpu_fault_fixture.py'),str(root),mode]
                with (root/'fixture.log').open('w') as out:
                    process = subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT)
                    deadline = time.monotonic()+7
                    observations = 0; parked = False
                    while process.poll() is None and time.monotonic()<deadline:
                        rows = [json.loads(p.read_text()) for p in root.glob('leaf-*.json')]
                        if any(live(row) for row in rows):
                            self.assertTrue(busy(root/'device0.lock')); observations += 1
                        if mode in ('auditfail','scanroot','scannested') and 'FAILED_LOCK_RETAINED' in (root/'fixture.log').read_text():
                            self.assertFalse(any(live(row) for row in rows))
                            self.assertFalse(Path('/proc/'+str(process.pid)+'/task/'+str(process.pid)+'/children').read_text().strip())
                            self.assertTrue(busy(root/'device0.lock'))
                            os.kill(process.pid,signal.SIGKILL)  # Our quiescent CPU fixture only.
                            parked = True; break
                        time.sleep(.01)
                    rc = process.wait(timeout=2)
                self.assertGreater(observations,0,(root/'fixture.log').read_text())
                self.assertEqual(rc,-9 if parked else 0,(root/'fixture.log').read_text())
                self.assertFalse(any(live(json.loads(p.read_text())) for p in root.glob('leaf-*.json')))
                self.assertFalse(busy(root/'device0.lock'))
                if mode in ('auditfail','scanroot','scannested'): self.assertTrue(parked)
                if mode in ('scanroot','scannested'):
                    result = json.loads((root/'result.json').read_text())
                    self.assertIn('injected scan '+mode,result['error'])
                    self.assertFalse((root/'postflight.json').exists())
                    events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
                    self.assertEqual(events[-1]['event'],'FAILED_LOCK_RETAINED')
                    self.assertIn('PermissionError',events[-1]['error'])
                    self.assertTrue(any(e['event']=='guardian_signal' and e['signal']==signal.SIGTERM for e in events))
                    self.assertTrue(any(e['event']=='signal_owned' and e['signal']==signal.SIGKILL for e in events))
                    self.assertTrue(any(e['event']=='lineage_quiescent' and e['proof']=='waitpid_ECHILD' for e in events))
                    self.assertFalse(any(e['event']=='release_ready' for e in events))
                elif mode!='auditfail':
                    result = json.loads((root/'result.json').read_text())
                    self.assertTrue('error' in result or result['rc']==7)
                    events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
                    self.assertEqual(events[-1]['event'],'release_ready')
                    self.assertTrue(any(e['event']=='lineage_quiescent' for e in events))
                print(json.dumps({'case':mode,'lock_busy_observations_with_live_leaves':observations,
                                  'all_owned_exited':True,'failure_park_observed':parked}),flush=True)


if __name__=='__main__': unittest.main(verbosity=2)
