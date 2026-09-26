import ast, hashlib, importlib.util, json, os, stat, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
B=Path('/home/poojith-devan/cache-link-fresh-review-47rwhD')
C=B/'candidate/research/v2/proposals'
A=B/'baseline/research/v2/proposals'
P=C/'20260926-cache-link-repair'
sys.dont_write_bytecode=True
sys.path.insert(0,str(A/'20260926-k2-f0f2-native-evidence/packet'))
def load(name,p):
    m=types.ModuleType(name); m.__file__=str(p); sys.modules[name]=m
    exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__); return m
k=load('k',P/'k2/monitor.py')
t=load('t',P/'r1-recovery/transport.py')
h=load('h',C/'20260926-confirmed36-clean-native-trn21-packet-r2/packet/host_driver.py')
t2=load('t2',P/'r2-transport/transport.py')
class Review(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='independent-cache-')
        self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def test_exact_opaque_counts_and_no_content_reads(self):
        outside=self.root/'outside'; outside.mkdir(); (outside/'secret').write_bytes(b'never read'*100)
        for m in (k,t,h,t2):
            r=self.root/m.__name__; r.mkdir(); cache=r/'cache'; cache.mkdir()
            (cache/'ordinary').write_bytes(b'1234567')
            targets=[str(outside/'secret'),str(outside/'missing'),str(outside),'.']
            for i,target in enumerate(targets): (cache/str(i)).symlink_to(target)
            original_open=os.open; original_stat=os.stat
            def checked_open(path,flags,*a,**kw):
                self.assertTrue(flags & os.O_DIRECTORY); self.assertTrue(flags & os.O_NOFOLLOW)
                return original_open(path,flags,*a,**kw)
            def checked_stat(path,*a,**kw):
                self.assertIs(kw.get('follow_symlinks'),False)
                return original_stat(path,*a,**kw)
            with patch.object(os,'open',checked_open), patch.object(os,'stat',checked_stat), patch.object(os,'read',side_effect=AssertionError('content read')),patch.object(os,'pread',side_effect=AssertionError('content pread')):
                rows=list(m.accounted_entries(cache,lambda p:True))
            self.assertEqual(len(rows),5)
            self.assertEqual(sum(s.st_size for _,s,_ in rows),7+sum(len(x.encode()) for x in targets))
            self.assertEqual(sorted(target for _,_,target in rows if target is not None),sorted(targets))
            meta={}; self.assertEqual(k.tree_bytes(cache,cache_metadata=meta),(7+sum(len(x.encode()) for x in targets),5))
            self.assertEqual(len(meta),4)
    def test_noncache_roots_special_scan_failure(self):
        for m in (k,t,h,t2):
            r=self.root/m.__name__; r.mkdir(); (r/'link').symlink_to('/missing')
            with self.assertRaisesRegex(RuntimeError,'NONCACHE_SYMLINK'): list(m.accounted_entries(r,lambda p:False))
            (r/'link').unlink(); os.mkfifo(r/'fifo')
            with self.assertRaisesRegex(RuntimeError,'SCAN_NONREGULAR'): list(m.accounted_entries(r,lambda p:True))
            with patch.object(os,'scandir',side_effect=PermissionError('independent scan denial')):
                with self.assertRaises(PermissionError): list(m.accounted_entries(r,lambda p:True))
            linked=self.root/(m.__name__+'-rootlink'); linked.symlink_to(r)
            with self.assertRaisesRegex(RuntimeError,'SCAN_ROOT_NOT_DIRECTORY'): list(m.accounted_entries(linked,lambda p:True))
    def test_real_link_and_directory_swaps(self):
        for m in (k,t,h,t2):
            r=self.root/m.__name__; r.mkdir(); link=r/'link'; link.symlink_to('/absent-original')
            orig=os.readlink
            def swap(path,*a,**kw):
                value=orig(path,*a,**kw); link.unlink(); link.symlink_to('/absent-substitute'); return value
            with patch.object(os,'readlink',swap):
                with self.assertRaisesRegex(RuntimeError,'SCAN_ENTRY_CHANGED'): list(m.accounted_entries(r,lambda p:True))
            link.unlink(); d=r/'dir'; d.mkdir(); outside=self.root/(m.__name__+'-outside'); outside.mkdir(); (outside/'secret').write_bytes(b'outside')
            orig_open=os.open
            def swapdir(path,flags,*a,**kw):
                if path=='dir': d.rename(r/'old'); d.symlink_to(outside)
                return orig_open(path,flags,*a,**kw)
            with patch.object(os,'open',swapdir):
                with self.assertRaises(OSError): list(m.accounted_entries(r,lambda p:True))
    def test_clean_domains_counts_and_caps(self):
        r=self.root; (r/'caches/debug').mkdir(parents=True); (r/'receipts').mkdir()
        (r/'caches/debug/a').write_bytes(b'abc'); (r/'caches/debug/link').symlink_to('/no-target')
        (r/'receipts/.rank0.tmp').write_bytes(b'12345'); (r/'.partial').write_bytes(b'xy')
        s=h.sizes(r); self.assertEqual(s['artifact_bytes'],7); self.assertEqual(s['cache_bytes'],13); self.assertEqual(s['rank_receipt_bytes']['0'],5)
        h.artifact_gate(s)
        with self.assertRaisesRegex(RuntimeError,'ARTIFACT_CAP'): h.artifact_gate(dict(s,artifact_bytes=h.ARTIFACT_CAP+1))
        (r/'caches/neff').symlink_to('/no-target')
        with self.assertRaisesRegex(RuntimeError,'NONCACHE_SYMLINK'): h.sizes(r)
        cache=r/'cache'; cache.mkdir(); (cache/'a').write_bytes(b'x'); os.link(cache/'a',cache/'b')
        with self.assertRaisesRegex(RuntimeError,'SPECIAL_OR_HARDLINK'): k.tree_bytes(cache,cache_metadata={})
        with self.assertRaisesRegex(RuntimeError,'ENTRY_CAP'): k.tree_bytes(cache,cap=0,cache_metadata={})
    def test_immutable_chunk_and_checkpoint_exclusion(self):
        f=self.root/'artifact'; raw=b'A'*(t.CHUNK+17); f.write_bytes(raw)
        fd=os.open(f,os.O_RDWR)
        try:
            self.assertEqual(b''.join(t.authenticated_chunks(fd,0,len(raw),t.sha(raw))),raw)
            original=os.pread; count=0
            def mutate(fd,n,off):
                nonlocal count
                count+=1
                if count==65: os.pwrite(fd,b'Z',0)
                return original(fd,n,off)
            with patch.object(os,'pread',mutate):
                with self.assertRaisesRegex(RuntimeError,'RETRIEVAL_CHUNK_CHANGED'): list(t.authenticated_chunks(fd,0,len(raw),t.sha(raw)))
        finally: os.close(fd)
        for m in (t,t2):
            begin={'kind':'begin','root':str(m.ROOT),'label':m.LABEL}
            packet={'kind':'file','name':'checkpoint','bytes':1,'sha256':next(iter(m.CHECKPOINT_SHAS))}
            with self.assertRaisesRegex(RuntimeError,'NO_CHECKPOINT_CONTENT_TRANSFER'): m.receive(map(json.dumps,[begin,packet]),self.root/(m.__name__+'recv'))
    def test_r1_stage_hard_disabled(self):
        with self.assertRaisesRegex(RuntimeError,'CONSUMED_R1_RETRIEVAL_ONLY'): t.stage(None,None)
        with self.assertRaisesRegex(RuntimeError,'CONSUMED_R1_RETRIEVAL_ONLY'): t.make_wire(None,None,None,'stage')
    def test_r1_retrieve_lossless_metadata_recheck_and_sender_checkpoint(self):
        r=self.root/'job'; (r/'caches/debug').mkdir(parents=True); (r/'receipts').mkdir()
        link=r/'caches/debug/link'; link.symlink_to('/missing-sdk-table')
        f=r/'receipts/.rank0.tmp'; f.write_bytes(b'partial evidence')
        d=types.SimpleNamespace(RUNTIME_PATHS=(),retained_inputs=lambda raw:{},file_record=h.file_record,ARTIFACT_CAP=1,RANK_CAP=1)
        observation={'lock_identity':list(t.LOCK_ID),'ps':{},'neuron_ls':{},'fuser':{}}
        with patch.object(t,'ROOT',r),patch.object(t,'decode_payload',return_value=({'INPUTS.json':b''},b'',{},d,None,None)),patch.object(t,'census',return_value=observation):
            rows=[]; result=t.retrieve(None,None,rows.append)
            self.assertEqual(result['status'],'COMPLETE_COPY_FAILED_GATE')
            self.assertFalse(result['artifact_gate_pass']); self.assertEqual(result['cache_bytes'],len('/missing-sdk-table'))
            self.assertEqual(result['cache_symlinks']['caches/debug/link']['target'],'/missing-sdk-table')
            dest=self.root/'retrieved'; t.receive(map(json.dumps,rows),dest)
            self.assertEqual((dest/'files/receipts/.rank0.tmp').read_bytes(),b'partial evidence')
            self.assertFalse((dest/'files/caches').exists())
            original_scan=t.scan; scans=0
            def changed(root):
                nonlocal scans
                yield from original_scan(root)
                scans+=1
                if scans==1: link.unlink(); link.symlink_to('/different-target')
            with patch.object(t,'scan',changed):
                with self.assertRaisesRegex(RuntimeError,'RETRIEVAL_CACHE_CHANGED'): t.retrieve(None,None,lambda row:None)
            d.file_record=lambda path:{'bytes':1,'sha256':next(iter(t.CHECKPOINT_SHAS))}
            rows=[]
            with self.assertRaisesRegex(RuntimeError,'NO_CHECKPOINT_CONTENT_TRANSFER'): t.retrieve(None,None,rows.append)
            self.assertFalse(any(row['kind'] in ('file','chunk') for row in rows))
    def test_frozen_identity_and_unchanged_functions(self):
        old=A/'20260926-confirmed36-clean-native-trn21-packet-r1/packet'; new=C/'20260926-confirmed36-clean-native-trn21-packet-r2/packet'
        for name in ('INPUTS.json','lifetime_guard.py','idle_guard.py','dispatch_driver.py',h.FOUNDATION): self.assertEqual((old/name).read_bytes(),(new/name).read_bytes())
        before=(old/h.PROBE).read_bytes(); after=(new/h.PROBE).read_bytes()
        self.assertEqual(after.replace(b"LABEL = 'fs-20260926-confirmed36-clean-native-consistency-trn21-r2'",b"LABEL = 'fs-20260926-confirmed36-clean-native-consistency-trn21-r1'"),before)
        self.assertEqual(hashlib.sha256(before).hexdigest(),'937e2f239ad90b5aa61e0b40ccf067b19ac02febe246492735e948a35c6b86af')
        def funcs(p): return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        pairs=[(old/'host_driver.py',new/'host_driver.py',{'sizes'}),(A/'20260926-k2-f0f2-native-evidence/packet/monitor.py',P/'k2/monitor.py',{'tree_bytes','sample'}),(A/'20260926-confirmed36-clean-native-trn21-transport-r1/transport.py',P/'r1-recovery/transport.py',{'stage','scan','retrieve','make_wire'})]
        for a,b,changed in pairs:
            aa,bb=funcs(a),funcs(b)
            for name in aa.keys()-changed: self.assertEqual(aa[name],bb[name],name)
        oldpins=json.loads((A/'20260926-k2-f0f2-native-evidence/packet/SOURCE_PINS.json').read_text()); newpins=json.loads((P/'k2/SOURCE_PINS.json').read_text())
        self.assertEqual({name for name in oldpins if oldpins[name]!=newpins[name]},{'monitor.py'})
        self.assertEqual(newpins['monitor.py'],hashlib.sha256((P/'k2/monitor.py').read_bytes()).hexdigest())
        for m,packet in ((t,old),(t2,new)):
            m.packet_modules({name:(packet/name).read_bytes() for name in m.PACKET_NAMES})
        for binding in (P/'r1-recovery/TRANSPORT_BINDING.disabled.json',P/'r2-transport/TRANSPORT_BINDING.disabled.json',new/'NATIVE_BINDING.disabled.json'):
            obj=json.loads(binding.read_text())
            def check(x):
                if isinstance(x,dict):
                    for key,val in x.items():
                        if key.endswith('_authorized'): self.assertIs(val,False,key)
                        check(val)
                elif isinstance(x,list):
                    for val in x: check(val)
            check(obj)
if __name__=='__main__':
    print('FROZEN_MANIFEST_SHA256',hashlib.sha256((B/'SHA256SUMS').read_bytes()).hexdigest(),flush=True)
    print('CPU ONLY; no SDK imports, no native or host evidence; optimize=',sys.flags.optimize,flush=True)
    unittest.main(verbosity=2)
