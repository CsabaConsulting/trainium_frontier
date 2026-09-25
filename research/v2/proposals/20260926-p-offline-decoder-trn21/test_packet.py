"""Tiny stdlib fixtures only: no host, SDK, tensors, profiles or data are accessed."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
spec = importlib.util.spec_from_file_location('support_under_test', HERE / 'decoder_support.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
P = json.loads((HERE / 'EXPERIMENT_PROPOSAL.json').read_text())


class Packet(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='trainium-p-decoder-fixture-')
        self.root = Path(self.temp.name)
        self.lock = (self.root / 'fixture-lock').open('w')

    def tearDown(self):
        self.lock.close()
        for path in self.root.rglob('*'):
            if not path.is_symlink():
                path.chmod(0o700 if path.is_dir() else 0o600)
        self.temp.cleanup()

    def small_inputs(self):
        source = self.root / 'source'
        (source / 'neffs/ab/cd').mkdir(parents=True)
        (source / 'session/1').mkdir(parents=True)
        (source / 'neffs/ab/cd/model.neff').write_bytes(b'NEFF')
        (source / 'session/1/trace.pb').write_bytes(b'TRACE')
        (source / 'session/1/empty.pb').write_bytes(b'')
        entries = [dict(path='neffs/ab/cd/model.neff', bytes=4),
                   dict(path='session/1/trace.pb', bytes=5), dict(path='session/1/empty.pb', bytes=0)]
        return source, entries

    def run_tiny(self, script, **overrides):
        limits = dict(address_space=256*1024**2, cpu_seconds=2, file_size=65536,
                      wall_seconds=2, grace=.1, retained_bytes=8192,
                      scratch_bytes=256*1024, interval=.01, entry_cap=128)
        limits.update(overrides)
        remaining = limits.pop('deadline_seconds', 5)
        return h.run_bounded(['/usr/bin/python3', '-I', '-B', '-c', script],
                             {'PATH':'/usr/bin:/bin','GOMAXPROCS':'1'}, self.root, set(),
                             self.lock.fileno(), limits, time.monotonic()+remaining,
                             self.root/'stdout', self.root/'stderr')

    def test_frozen_exact_inventory_and_cli_evidence(self):
        raw = subprocess.check_output(['git','show',P['source_inventory_commit']+':'+P['source_inventory_path']],cwd=REPO)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), P['source_inventory_sha256'])
        archive = json.loads(raw)
        prefix = P['source_root'] + '/'
        rows = sorted([{'path':x['path'][len(prefix):], 'bytes':x['bytes']} for x in archive['profile_companions']
                       if x['path'].startswith(prefix)],key=lambda x:x['path'])
        self.assertEqual(rows, sorted(P['inputs'],key=lambda x:x['path']))
        self.assertEqual(len(rows),86)
        self.assertEqual(len({x['path'] for x in rows}),86)
        self.assertEqual(sum(x['bytes'] for x in rows),4310546731)
        self.assertEqual(sum(x['path'].endswith('.neff') for x in rows),80)
        help_raw = (REPO/P['capability_help_path']).read_bytes()
        self.assertEqual(hashlib.sha256(help_raw).hexdigest(),P['capability_help_sha256'])
        text=json.loads(help_raw)['commands'][0]['stdout']
        self.assertIn('--session-dir=',text)
        self.assertIn('summary-json',text)
        argv=P['core_argv']
        self.assertEqual(argv,[P['binary'],'view','--session-dir',P['root']+'/input/native','--output-format','summary-json','--ignore-nc-buf-usage','--ignore-instruction-hierarchy'])
        for flag in ('--neff-path','--ignore-event-trace','--ignore-instruction-trace','--ignore-dma-trace','--force','--data-path','--ingest-only'):
            self.assertNotIn(flag,argv)
        self.assertFalse(P['launch_authorized'])
        self.assertFalse(P['stage_authorized'])

    def test_exact_copy_hash_readonly_and_post_identity(self):
        source, entries=self.small_inputs()
        target=self.root/'copy'
        receipts=h.copy_inputs(source,target,entries,time.monotonic()+2)
        self.assertEqual(len(receipts),3)
        h.post_inputs(source,target,receipts,time.monotonic()+2)
        for row in receipts:
            self.assertEqual(row['source']['sha256'],row['copy']['sha256'])
            self.assertNotEqual(row['source']['st_ino'],row['copy']['st_ino'])
            self.assertEqual(row['copy']['st_nlink'],1)
            self.assertEqual(row['copy']['st_mode'] & 0o222,0)
        self.assertEqual(target.stat().st_mode & 0o222,0)
        (source/'session/1/trace.pb').write_bytes(b'OTHER')
        with self.assertRaisesRegex(RuntimeError,'INPUT_CHANGED'):
            h.post_inputs(source,target,receipts,time.monotonic()+2)

    def test_refuses_missing_extra_symlink_and_existing_destination(self):
        source, entries=self.small_inputs()
        (source/'extra').write_bytes(b'x')
        with self.assertRaises(RuntimeError):h.exact_layout(source,entries)
        (source/'extra').unlink()
        target=source/'session/1/trace.pb'
        target.unlink()
        with self.assertRaises(FileNotFoundError):h.exact_layout(source,entries)
        target.symlink_to(source/'neffs/ab/cd/model.neff')
        with self.assertRaisesRegex(RuntimeError,'SYMLINK'):h.exact_layout(source,entries)
        target.unlink();target.write_bytes(b'TRACE')
        with self.assertRaisesRegex(RuntimeError,'DESTINATION_EXISTS'):
            h.copy_inputs(source,source,entries,time.monotonic()+2)

    def test_deadline_and_path_escape_fail_closed(self):
        source, entries=self.small_inputs()
        with self.assertRaisesRegex(RuntimeError,'DEADLINE'):
            h.copy_inputs(source,self.root/'copy',entries,time.monotonic()-1)
        with self.assertRaisesRegex(RuntimeError,'UNSAFE_RELATIVE'):
            h.exact_layout(source,[dict(path='../escape',bytes=0)])

    def test_output_cap_retains_at_most_cap_and_terminates(self):
        result=self.run_tiny('import os,time; os.write(1,b"x"*65536); time.sleep(30)',retained_bytes=4096)
        self.assertEqual(result['status'],'REFUSED_OR_INCOMPLETE')
        self.assertIn('COMBINED_OUTPUT_CAP',result['error'])
        self.assertEqual(result['retained_bytes'],4096)
        self.assertEqual((self.root/'stdout').stat().st_size+(self.root/'stderr').stat().st_size,4096)
        self.assertIsNotNone(result['returncode'])
        self.assertLess(result['wall_seconds'],2)

    def test_wall_and_overall_deadlines_terminate(self):
        result=self.run_tiny('import time; time.sleep(30)',wall_seconds=.08)
        self.assertIn('DECODER_WALL_LIMIT',result['error'])
        self.assertIsNotNone(result['returncode'])
        self.assertLess(result['wall_seconds'],2)

    def test_generated_scratch_stop_and_symlink_refusal(self):
        result=self.run_tiny('import pathlib,time; pathlib.Path("generated").write_bytes(b"x"*16384); time.sleep(30)',scratch_bytes=8192)
        self.assertIn('GENERATED_SCRATCH_CAP',result['error'])
        self.assertGreater(result['scratch_peak_overshoot_bytes'],0)
        (self.root/'symlink').symlink_to('/proc')
        with self.assertRaisesRegex(RuntimeError,'SYMLINK'):h.scratch_bytes(self.root,set(),128)

    def test_overall_work_deadline_terminates(self):
        result=self.run_tiny('import time; time.sleep(30)',deadline_seconds=.06)
        self.assertIn('OVERALL_WORK_DEADLINE',result['error'])
        self.assertIsNotNone(result['returncode'])

    def test_cpu_hard_limit_is_a_failure_not_success(self):
        result=self.run_tiny('while True: pass',cpu_seconds=1,wall_seconds=3)
        self.assertNotEqual(result['returncode'],0)
        self.assertEqual(result['status'],'REFUSED_OR_INCOMPLETE')
        self.assertLess(result['child_cpu_seconds'],1.2)

    def test_readonly_exclusion_not_a_growth_bypass(self):
        file=self.root/'input';file.write_bytes(b'x');file.chmod(0o444)
        self.assertEqual(h.scratch_bytes(self.root,{file:1},128),0)
        with self.assertRaisesRegex(RuntimeError,'READONLY_INPUT'):h.scratch_bytes(self.root,{file:0},128)

    def test_standard_resource_limits_and_single_cpu(self):
        script='import json,os,resource; print(json.dumps({"cpus":list(os.sched_getaffinity(0)),"as":resource.getrlimit(resource.RLIMIT_AS),"cpu":resource.getrlimit(resource.RLIMIT_CPU),"fsize":resource.getrlimit(resource.RLIMIT_FSIZE)}))'
        result=self.run_tiny(script)
        self.assertEqual(result['status'],'PROCESS_EXIT_ZERO_COVERAGE_UNKNOWN')
        value=json.loads((self.root/'stdout').read_text())
        self.assertEqual(len(value['cpus']),1)
        self.assertEqual(value['as'],[256*1024**2]*2)
        self.assertEqual(value['cpu'],[2,2])
        self.assertEqual(value['fsize'],[65536,65536])

    def test_nonzero_and_file_size_limit_preserved(self):
        result=self.run_tiny('import pathlib; pathlib.Path("large").write_bytes(b"x"*65536)',file_size=4096)
        self.assertEqual(result['status'],'REFUSED_OR_INCOMPLETE')
        self.assertNotEqual(result['returncode'],0)
        self.assertLessEqual((self.root/'large').stat().st_size,4096)

    def test_descendant_termination_including_new_session(self):
        script='import subprocess,time; subprocess.Popen(["/usr/bin/python3","-I","-c","import time; time.sleep(30)"],start_new_session=True); time.sleep(30)'
        result=self.run_tiny(script)
        self.assertIn('UNEXPECTED_DECODER_DESCENDANT',result['error'])
        self.assertGreaterEqual(len(result['observed_process_identities']),2)
        remaining=h.owned_descendants(h.proc_rows(),-1,result['observed_process_identities'])
        self.assertFalse(remaining)

    def test_reused_pid_is_never_descendant_authority(self):
        rows={10:dict(pid=10,start=200,ppid=1,state='S'),11:dict(pid=11,start=201,ppid=10,state='S')}
        self.assertEqual(h.owned_descendants(rows,10,{10:100}),{})
        self.assertEqual(set(h.owned_descendants(rows,10,{10:200})),{10,11})
        rows[11]['ppid']=1
        self.assertEqual(set(h.owned_descendants(rows,10,{11:201})),{11})

    def test_late_detached_descendant_cannot_retain_success(self):
        script='import os,time; os.close(1); os.close(2); time.sleep(.04); pid=os.fork();\nif pid: os._exit(0)\nos.setsid(); time.sleep(30)'
        result=self.run_tiny(script, interval=.1)
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['status'], 'REFUSED_OR_INCOMPLETE')
        self.assertIn('UNEXPECTED_DECODER_DESCENDANT', result['error'])
        self.assertGreaterEqual(len(result['observed_process_identities']), 2)
        terminated=[row for event in result['termination'] for row in event['identities']]
        self.assertTrue(terminated)
        self.assertFalse(h.owned_descendants(h.proc_rows(),-1,result['observed_process_identities'],os.getpid()))

    def test_fast_double_fork_is_subreaped_and_stopped(self):
        script='import os,time; pid=os.fork();\nif pid: os._exit(0)\npid=os.fork()\nif pid: os._exit(0)\nos.setsid(); time.sleep(30)'
        result=self.run_tiny(script)
        self.assertEqual(result['status'],'REFUSED_OR_INCOMPLETE')
        self.assertIn('UNEXPECTED_DECODER_DESCENDANT',result['error'])
        self.assertTrue(result['child_subreaper'])
        self.assertFalse(h.owned_descendants(h.proc_rows(),-1,result['observed_process_identities'],os.getpid()))

    def test_source_lock_pins_limits_and_no_training_path(self):
        driver=(HERE/'host_driver.py').read_text()
        tree=ast.parse(driver)
        pins=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='PINS')
        for name,expected in pins.items():
            path=HERE/name if (HERE/name).is_file() else REPO/'research/v2/operator/20260926-common-host-idle-guard'/name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),expected)
        self.assertEqual(driver.count('require_idle_under_inherited_lock('),2)
        self.assertIn("post_fd != lock_fd",driver)
        self.assertIn("'ip-172-31-34-85'",driver)
        self.assertIn('started + 580',driver)
        self.assertIn('min(235, math.floor(300 - 60 - 32',driver)
        self.assertIn('cpu_seconds=5',driver)
        self.assertIn('parent_final_cpu_hard',driver)
        self.assertNotIn('torch',driver.lower())
        self.assertNotIn('eval-public',driver)
        self.assertEqual(P['limits']['overall_wall_seconds'],600)
        self.assertEqual(P['limits']['decoder_wall_seconds'],300)
        self.assertEqual(P['limits']['minimum_available_memory_bytes'],24*1024**3)
        self.assertEqual(P['limits']['minimum_free_disk_bytes'],12*1024**3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
