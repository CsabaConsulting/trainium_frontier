"""Local author regressions only; no host, SDK, tensor, or review authority."""
import base64
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
PROPOSALS = HERE.parent
PACKET = PROPOSALS/'20260926-confirmed36-clean-native-trn21-packet-r2/packet'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

d = load('r2_driver', PACKET/'host_driver.py')
t = load('r2_transport', HERE/'r2-transport/transport.py')
r1 = load('r1_recovery', HERE/'r1-recovery/transport.py')
sys.path.insert(0, str(PROPOSALS/'20260926-k2-f0f2-native-evidence/packet'))
k = load('k2_monitor', HERE/'k2/monitor.py')
old = load('transport_regressions', PROPOSALS/'20260926-confirmed36-clean-native-trn21-transport-r1/test_transport.py')
old.t = t
old.PACKET = PACKET
old.HERE = HERE/'r2-transport'
old.FILES = {name:(PACKET/name).read_bytes() for name in t.PACKET_NAMES}
old.SOURCE_SHA = t.sha((HERE/'r2-transport/transport.py').read_bytes())

class CacheChecks(unittest.TestCase):
    def fixtures(self):
        root = Path(tempfile.mkdtemp(prefix='cache-link-author-'))
        (root/'caches/debug').mkdir(parents=True)
        (root/'cache').mkdir()
        (root/'tmp').mkdir()
        (root/'receipts').mkdir()
        outside = Path(tempfile.mkdtemp(prefix='cache-target-author-'))
        (outside/'secret').write_bytes(b'TARGET_BYTES_MUST_NOT_BE_READ'*100)
        for tree in ('caches/debug','cache','tmp'):
            p = root/tree
            (p/'ordinary').write_bytes(b'abc')
            (p/'outside').symlink_to(outside/'secret')
            (p/'broken').symlink_to(outside/'missing')
            (p/'directory').symlink_to(outside, target_is_directory=True)
            (p/'cycle').symlink_to(p)
        return root, outside

    def test_no_target_open_and_no_directory_traversal(self):
        root, outside = self.fixtures()
        original = os.open
        opened = []
        def guarded(path, flags, *args, **kwargs):
            opened.append(str(path))
            self.assertNotIn(str(outside), str(path))
            self.assertNotIn(str(path), ('outside','broken','directory','cycle','secret'))
            return original(path, flags, *args, **kwargs)
        with mock.patch.object(os, 'open', side_effect=guarded):
            # Strict top-level cache/tmp are not the clean diagnostic's namespace.
            for p in ('cache','tmp'):
                os.rename(root/p, root/(p+'-saved'))
            # Only scan the intended clean subtree in a separate fixture root.
            clean = Path(tempfile.mkdtemp(prefix='clean-links-'))
            os.rename(root/'caches',clean/'caches')
            obs = d.sizes(clean)
            links = obs['cache_symlinks']
            self.assertEqual(len(links),4)
            self.assertEqual(obs['cache_bytes'],3+sum(x['bytes'] for x in links.values()))
            self.assertEqual(obs['artifact_bytes'],0)
            for transport in (t,r1):
                entries = list(transport.scan(clean))
                self.assertEqual(len(entries),5)
                self.assertEqual(sum(s.st_size for _,s,_ in entries),obs['cache_bytes'])
            for p in ('cache','tmp'):
                os.rename(root/(p+'-saved'),root/p)
                meta = {}
                count, entries = k.tree_bytes(root/p,cache_metadata=meta)
                self.assertEqual(entries,5)
                self.assertEqual(count,3+sum(x['bytes'] for x in meta.values()))
                self.assertEqual(len(meta),4)

    def test_noncache_links_roots_and_specials_refused(self):
        for module in (d,t,r1):
            for name in ('artifact','receipts/.rank0.tmp','caches/debug'):
                root = Path(tempfile.mkdtemp(prefix='strict-links-'))
                (root/'receipts').mkdir(); (root/'caches').mkdir()
                (root/name).symlink_to('/nonexistent/cache-test-only')
                with self.assertRaises(RuntimeError):
                    list(module.accounted_entries(root,module.compiler_link))
        for name in ('artifacts','logs'):
            root = Path(tempfile.mkdtemp())/name; root.mkdir()
            (root/'alias').symlink_to('/nonexistent/cache-test-only')
            with self.assertRaises(RuntimeError): k.tree_bytes(root)
            with self.assertRaises(RuntimeError): k.tree_bytes(root,cache_metadata={})
        root = Path(tempfile.mkdtemp()); (root/'caches/debug').mkdir(parents=True)
        os.mkfifo(root/'caches/debug/fifo')
        with self.assertRaises(RuntimeError): d.sizes(root)
        alias = root/'root-alias'; alias.symlink_to(root)
        with self.assertRaises(RuntimeError): d.sizes(alias)

    def test_scan_errors_preserved(self):
        for module in (d,t,r1,k):
            root = Path(tempfile.mkdtemp()); (root/'nested').mkdir()
            original = os.scandir
            calls = 0
            def fail(fd):
                nonlocal calls
                calls += 1
                if calls == 2: raise PermissionError('injected nested scan error')
                return original(fd)
            with mock.patch.object(os,'scandir',side_effect=fail),self.assertRaisesRegex(PermissionError,'nested'):
                list(module.accounted_entries(root,lambda p:False))

    def test_link_and_directory_swaps_fail_without_following(self):
        for module in (d,t,r1,k):
            root = Path(tempfile.mkdtemp()); (root/'caches/debug').mkdir(parents=True)
            link = root/'caches/debug/link'; link.symlink_to('/nonexistent/first')
            original = os.readlink
            def swap(name,*args,**kwargs):
                result = original(name,*args,**kwargs)
                os.rename(link,root/'old-link')
                link.symlink_to('/nonexistent/second')
                return result
            with mock.patch.object(os,'readlink',side_effect=swap),self.assertRaisesRegex(RuntimeError,'SCAN_ENTRY_CHANGED'):
                list(module.accounted_entries(root,lambda p:True))
            root = Path(tempfile.mkdtemp()); (root/'nested').mkdir()
            original_open = os.open
            def swap_dir(path,flags,*args,**kwargs):
                if path == 'nested':
                    os.rename(root/'nested',root/'old-nested')
                    (root/'nested').symlink_to('/nonexistent/outside')
                return original_open(path,flags,*args,**kwargs)
            with mock.patch.object(os,'open',side_effect=swap_dir),self.assertRaises(OSError):
                list(module.accounted_entries(root,lambda p:True))

    def test_retrieval_links_metadata_and_wire_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp,old.fixture(Path(tmp),'retrieve') as (output,lock,body,driver,g,retained):
            output.mkdir(); (output/'caches/debug').mkdir(parents=True)
            (output/'caches/debug/link').symlink_to('/nonexistent/sdk-table')
            (output/'artifact.log').write_bytes(b'allowed')
            rows = []
            with mock.patch.object(t,'census',return_value={}):
                result = t.retrieve(body,old.SOURCE_SHA,rows.append)
            self.assertEqual(result['cache_bytes'],len('/nonexistent/sdk-table'))
            self.assertEqual(result['cache_symlinks']['caches/debug/link']['target'],'/nonexistent/sdk-table')
            self.assertEqual(b''.join(base64.b64decode(x['b64']) for x in rows if x['kind']=='chunk'),b'allowed')
            t.receive([t.json_bytes(x) for x in rows],Path(tmp)/'receive')
            self.assertFalse((Path(tmp)/'receive/files/caches').exists())

    def test_retrieval_cache_swap_fails_final_census(self):
        with tempfile.TemporaryDirectory() as tmp,old.fixture(Path(tmp),'retrieve') as (output,lock,body,driver,g,retained):
            output.mkdir(); (output/'caches/debug').mkdir(parents=True)
            link=output/'caches/debug/link'; link.symlink_to('/nonexistent/first')
            (output/'z.log').write_bytes(b'allowed')
            def scan(root):
                yield from original(root)
                if link.readlink()==Path('/nonexistent/first'):
                    link.rename(output/'caches/debug/old'); link.symlink_to('/nonexistent/second')
            original=t.scan
            with mock.patch.object(t,'scan',side_effect=scan),mock.patch.object(t,'census',return_value={}),self.assertRaisesRegex(RuntimeError,'RETRIEVAL_CACHE_CHANGED'):
                t.retrieve(body,old.SOURCE_SHA,lambda x:None)

    def test_r1_readonly_and_binding_identity(self):
        files={name:(PROPOSALS/'20260926-confirmed36-clean-native-trn21-packet-r1/packet'/name).read_bytes() for name in r1.PACKET_NAMES}
        r1.packet_modules(files)
        with self.assertRaisesRegex(RuntimeError,'CONSUMED_R1'): r1.stage({},'unused')
        self.assertTrue(r1.ROOT.name.endswith('_r1')); self.assertTrue(t.ROOT.name.endswith('_r2'))
        probe=PACKET/'proposals/20260926-confirmed36-clean-native-consistency/probe.py'
        inverse=probe.read_bytes().replace(b'clean-native-consistency-trn21-r2',b'clean-native-consistency-trn21-r1')
        self.assertEqual(t.sha(inverse),'937e2f239ad90b5aa61e0b40ccf067b19ac02febe246492735e948a35c6b86af')
        self.assertEqual(d.INITIAL_DISK,14495514624)
        self.assertEqual((d.CHILD_SECONDS,d.TERM_SECONDS,d.LOCK_ID),(900,15,(66305,596823)))
        for sub in ('r1-recovery','r2-transport'):
            binding=json.loads((HERE/sub/'TRANSPORT_BINDING.disabled.json').read_bytes())
            self.assertEqual(binding['transport_source_sha256'],t.sha((HERE/sub/'transport.py').read_bytes()))
            self.assertFalse(binding['stage_authorized']); self.assertFalse(binding['retrieval_authorized'])
        for name in ('INPUTS.json','lifetime_guard.py','idle_guard.py','dispatch_driver.py'):
            self.assertEqual((PACKET/name).read_bytes(),files[name])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CacheChecks)
    # Retained immutable-chunk/checkpoint adversaries and applicable old mechanics.
    for name in unittest.defaultTestLoader.getTestCaseNames(old.TransportTests):
        if name not in ('test_disabled_and_exact_unchanged_packet','test_retrieval_scan_errors_and_symlinks_reject'):
            suite.addTest(old.TransportTests(name))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
