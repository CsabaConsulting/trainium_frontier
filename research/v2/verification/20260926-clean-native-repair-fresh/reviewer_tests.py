import os, sys, json, time, signal, tempfile, subprocess, hashlib, struct, copy, unittest, importlib.util, ast, re
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/home/poojith-devan/clean36-native-repair-fresh-IrAY3D')
PROBE=ROOT/'research/v2/proposals/20260926-confirmed36-clean-native-consistency/probe.py'
s=importlib.util.spec_from_file_location('probe',PROBE); p=importlib.util.module_from_spec(s); s.loader.exec_module(p)
h=p.foundation(); t=h.torch; t.set_num_threads(1)
def digest(x): return hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()
def fixture(b,n): return t.randint(1,9,(b,n),generator=t.Generator().manual_seed(42))
class Model(t.nn.Module):
    def __init__(self,kind='good'):
        super().__init__(); self.kind=kind; self.count=0
        self.w=t.nn.Parameter(t.tensor([1.]),requires_grad=False)
        self.register_buffer('buf',t.tensor([2.]),persistent=False)
        self.cache=t.tensor([3.]); self.eval()
    def forward(self,x):
        self.count+=1
        out=x.cumsum(1).float().unsqueeze(-1)+t.arange(9).float()
        if self.kind=='leak': out=out+x[:,-1,None,None].float()
        if self.kind=='small': out=out+x[:,-1,None,None].float()/4096
        if self.kind=='param': self.w.add_(1)
        if self.kind=='buffer': self.buf.add_(1)
        if self.kind=='attr': self.cache.add_(1)
        if self.kind=='mode': self.train()
        if self.kind=='grad': self.w.requires_grad_(True)
        if self.kind=='input': x.add_(1)
        if self.kind=='hook': self.register_forward_hook(lambda *a:None)
        if self.kind=='nan': out[0,0,0]=float('nan')
        if self.kind=='inf': out[0,0,0]=float('inf')
        if self.kind=='batch': out+=len(x)
        if self.kind=='repeat': out+=self.count
        if self.kind=='alias':
            if not hasattr(self,'scratch'): self.scratch=out
            else: self.scratch.add_(1)
            return self.scratch
        if self.kind=='term' and self.count==2: os.kill(os.getpid(),signal.SIGTERM)
        return out
def runshape(m,x,case=None,progress=None):
    return p.verify_shape(h,m,x,9,t.device('cpu'),lambda:None,p.snapshot(h,m,lambda:None),case=case,progress=progress)
def report(): return dict(rank=0,label=p.LABEL,status='UNRESOLVED',arms={})
class Tests(unittest.TestCase):
    def test_exact_diff_and_predicates(self):
        old=(ROOT/'baseline/probe.py').read_text().splitlines(keepends=True); result=[]; cursor=0
        lines=(ROOT/'implementation.patch').read_text().splitlines(keepends=True); i=0
        while i<len(lines):
            if not lines[i].startswith('@@'): i+=1; continue
            start=int(re.match(r'@@ -(\d+)',lines[i]).group(1))-1
            result.extend(old[cursor:start]); cursor=start; i+=1
            while i<len(lines) and not lines[i].startswith('@@'):
                line=lines[i]; i+=1
                if line[0] in ' -': self.assertEqual(old[cursor],line[1:]); cursor+=1
                if line[0] in ' +': result.append(line[1:])
        result.extend(old[cursor:]); self.assertEqual(''.join(result),PROBE.read_text())
        for rel in ['science/candidate/workflow/candidates/confirmed-clock36-116188/train.py','science/reference/workflow/proposals/research-v2/train.py']:
            tree=ast.parse((ROOT/rel).read_text()); conv=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='CausalConv')
            forward=next(n for n in conv.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
            test=next(n.test for n in forward.body if isinstance(n,ast.If))
            lane=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='_Conv2NKI')
            choice=next(n.value for n in ast.walk(lane) if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='lanes' for a in n.targets))
            for b,n in p.SHAPES:
                env={'x':SimpleNamespace(device=SimpleNamespace(type='neuron'),dtype=t.bfloat16,shape=(b,n,1024)),'self':SimpleNamespace(kernel_size=2),'torch':t,'USE_NKI_LOCAL_CONV':True}
                self.assertEqual(eval(compile(ast.Expression(test),'<predicate>','eval'),env),n==256)
                self.assertEqual(eval(compile(ast.Expression(choice),'<lane>','eval'),env),2 if b==2 else 1)
    def test_scalar_all_shapes(self):
        count=0
        for b,n in p.SHAPES:
            x=fixture(b,n); out,c=runshape(Model(),x)
            vals=[]
            for row in x.tolist():
                accum=0
                for token in row:
                    accum+=token
                    vals.extend(float(accum+k) for k in range(9))
            self.assertEqual(digest(out),hashlib.sha256(struct.pack('<'+'f'*len(vals),*vals)).hexdigest())
            self.assertEqual(c['input']['sha256'],digest(x)); self.assertEqual(c['status'],'COMPLETE')
            self.assertEqual(len(c['calls']),b+3); count+=len(c['calls'])
            self.assertEqual(c['causality']['max_abs_diff'],0)
        self.assertEqual(count,30)
    def test_causal_durable_details(self):
        with tempfile.TemporaryDirectory() as d:
            r=report(); case={}; r['arms']={'reference':{'shapes':[case]}}
            rr=p.RankReceipt(h,Path(d),r,time.monotonic()); x=fixture(3,257)
            with self.assertRaisesRegex(h.CheckFailure,'CAUSAL_PREFIX') as e: runshape(Model('leak'),x,case,rr.checkpoint)
            saved=json.loads(rr.partial.read_text()); c=saved['arms']['reference']['shapes'][0]
            altered=x.clone(); altered[:,128:]=altered[:,128:].flip(1)
            expected=Model('leak')(x)[:,:127]; actual=Model('leak')(altered)[:,:127]
            z=c['causality']; self.assertEqual(z,e.exception.details)
            self.assertEqual(z['max_abs_diff'],float((expected-actual).abs().max()))
            self.assertEqual(z['expected']['sha256'],digest(expected)); self.assertEqual(z['actual']['sha256'],digest(actual))
            self.assertEqual((z['prefix_start'],z['prefix_stop'],z['cut']),(0,127,128))
            self.assertEqual(z['original_input']['sha256'],digest(x)); self.assertEqual(z['altered_input']['sha256'],digest(altered))
            self.assertEqual(len(c['calls']),6); self.assertTrue(all(v['status']=='COMPLETE' for v in c['calls']))
            self.assertEqual(c['current_call'],'reversed_suffix'); self.assertEqual(c['status'],'REFUSE')
    def test_tolerance_not_exact(self):
        out,c=runshape(Model('small'),fixture(2,256)); self.assertFalse(c['causality']['prefix_exact']); self.assertLessEqual(c['causality']['max_abs_diff'],.01)
    def test_purity_negative(self):
        for kind in ['param','buffer','attr','mode','grad','input','hook','nan','inf','batch','repeat','alias','leak']:
            with self.subTest(kind=kind), self.assertRaises(h.CheckFailure): runshape(Model(kind),fixture(2,256))
        with self.assertRaisesRegex(h.CheckFailure,'DEGENERATE'): runshape(Model(),t.ones((1,256),dtype=t.int64))
    def test_byte_negative(self):
        for a,b in [(t.tensor([-0.]),t.tensor([0.])),(t.nextafter(t.ones(1),t.tensor([2.])),t.ones(1)),(t.ones(1).double(),t.ones(1)),(t.ones(2),t.ones(1))]:
            with self.assertRaises(h.CheckFailure): p.exact(h,a,b,'EXACT')
        self.assertEqual(h.comparison(t.ones(1),t.zeros(1))['relative_rms'],'UNDEFINED_ZERO_REFERENCE')
    def test_masks(self):
        mask=t.tensor([[[[0 if j<=i and j>i-3 else -float('inf') for j in range(7)] for i in range(7)]]],dtype=t.float32)
        h.mask_record(mask,7,3,'tiny')
        for value in [0.,float('nan'),float('inf')]:
            bad=mask.clone(); bad[0,0,0,1]=value
            with self.assertRaises(h.CheckFailure): h.mask_record(bad,7,3,'bad')
    def test_partial_atomic_failures(self):
        for failure in ['write','fsync','replace','directory','cap']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d:
                r=report(); rr=p.RankReceipt(h,Path(d),r,time.monotonic()); before=rr.partial.read_bytes()
                r['next']='new'
                if failure=='write':
                    real=os.fdopen
                    class Bad:
                        def __init__(self,*a): self.f=real(*a)
                        def __enter__(self): return self
                        def __exit__(self,*a): self.f.close()
                        def write(self,v): raise OSError('injected write')
                    ctx=patch.object(p.os,'fdopen',Bad)
                elif failure=='fsync': ctx=patch.object(p.os,'fsync',side_effect=OSError('injected fsync'))
                elif failure=='replace': ctx=patch.object(p.os,'replace',side_effect=OSError('injected rename'))
                elif failure=='directory': ctx=patch.object(rr,'sync_directory',side_effect=OSError('injected directory'))
                else: ctx=patch.object(p,'CAP',300)
                with ctx,self.assertRaises((OSError,RuntimeError)): rr.checkpoint('next')
                saved=json.loads(rr.partial.read_text()); self.assertEqual(saved['receipt_state'],'PARTIAL')
                if failure!='directory': self.assertEqual(rr.partial.read_bytes(),before)
                self.assertFalse(rr.final.exists()); self.assertLessEqual(sum(f.stat().st_size for f in Path(d).iterdir()),p.CAP)
    def test_final_exclusive_and_failure(self):
        for failure in ['link','fsync','directory','unlink']:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as d:
                rr=p.RankReceipt(h,Path(d),report(),time.monotonic()); before=rr.partial.read_bytes()
                if failure=='link': ctx=patch.object(p.os,'link',side_effect=OSError('link'))
                elif failure=='fsync': ctx=patch.object(p.os,'fsync',side_effect=OSError('fsync'))
                elif failure=='directory': ctx=patch.object(rr,'sync_directory',side_effect=OSError('dir'))
                else: ctx=patch.object(Path,'unlink',side_effect=OSError('unlink'))
                with ctx,self.assertRaises(OSError): rr.finish()
                self.assertEqual(rr.partial.read_bytes(),before)
        with tempfile.TemporaryDirectory() as d:
            rr=p.RankReceipt(h,Path(d),report(),time.monotonic()); rr.final.write_text('existing')
            with self.assertRaises(FileExistsError): rr.finish()
            self.assertEqual(rr.final.read_text(),'existing'); self.assertTrue(rr.partial.exists())
        with tempfile.TemporaryDirectory() as d:
            rr=p.RankReceipt(h,Path(d),report(),time.monotonic()); rr.finish(); final=rr.final.read_bytes()
            with self.assertRaises(RuntimeError): rr.finish()
            self.assertEqual(rr.final.read_bytes(),final); self.assertFalse(rr.partial.exists())
    def test_real_sigterm(self):
        for mode in ['term','rename']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                child=subprocess.run([sys.executable,'-B',__file__,'child',d,mode],capture_output=True)
                self.assertEqual(child.returncode,-signal.SIGTERM,child.stderr)
                files=list(Path(d).glob('rank*.json')); self.assertEqual([f.name for f in files],['rank0.partial.json'])
                r=json.loads(files[0].read_text()); c=r['arms']['reference']['shapes'][0]
                self.assertEqual(c['calls'][0]['status'],'COMPLETE'); self.assertEqual(c['calls'][0]['output']['sha256'],digest(Model()(fixture(1,256))))
                self.assertEqual(r['receipt_state'],'PARTIAL'); self.assertEqual(r['status'],'UNRESOLVED')
                self.assertLessEqual(sum(f.stat().st_size for f in Path(d).iterdir()),p.CAP)
    def test_offline_partial_never_final(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(4):
                r=report(); r.update(rank=i,receipt_state='PARTIAL',status='LOCAL_COMPLETE',post_input_failures=[])
                (Path(d)/('rank'+str(i)+'.json')).write_text(json.dumps(r))
            with patch.object(sys,'argv',['probe','--validate-receipts',d]),self.assertRaisesRegex(RuntimeError,'TERMINAL'): p.main()
    def test_structural_cases_ranks(self):
        cases=[]
        for b,n in p.SHAPES:
            _,c=runshape(Model(),fixture(b,n))
            for call in c['calls']: call['output']['shape'][-1]=8192
            cases.append(c)
        arms={a:{'status':'COMPLETE','shapes':copy.deepcopy(cases)} for a in ['reference','candidate']}
        ranks=[dict(rank=i,status='LOCAL_COMPLETE',arms=copy.deepcopy(arms)) for i in range(4)]
        p.validate_ranks(h,ranks)
        for mutate in [lambda r:r.pop(),lambda r:r[0].update(rank=1),lambda r:r[0]['arms']['reference']['shapes'].pop(),lambda r:r[0]['arms']['reference']['shapes'][0]['calls'].pop(),lambda r:r[0]['arms']['reference']['shapes'][0].update(status='IN_PROGRESS'),lambda r:r[0]['arms']['candidate']['shapes'][0]['calls'][0]['output'].update(sha256='0'*64)]:
            bad=copy.deepcopy(ranks); mutate(bad)
            with self.assertRaises(h.CheckFailure): p.validate_ranks(h,bad)
    def test_cleanup_fail_closed(self):
        import torch.distributed as dist
        for failure in ['run','cleanup','success']:
            with tempfile.TemporaryDirectory() as d:
                def body(h,a,r,progress):
                    r['post_input_failures']=[]; progress('tiny')
                    if failure=='run': raise RuntimeError('init failure surrogate')
                def cleanup():
                    if failure=='cleanup': raise RuntimeError('cleanup failure')
                argv=['probe','--enable-native','--label',p.LABEL,'--inputs','/unused','--out-dir',d]
                with patch.object(sys,'argv',argv),patch.dict(os.environ,{'RANK':'0'}),patch.object(p,'run',body),patch.object(t,'set_num_interop_threads'),patch.object(dist,'is_initialized',return_value=True),patch.object(dist,'destroy_process_group',side_effect=cleanup): rc=p.main()
                r=json.loads((Path(d)/'rank0.json').read_text())
                self.assertEqual(rc,0 if failure=='success' else 2)
                self.assertEqual(r['status'],{'success':'LOCAL_COMPLETE','run':'REFUSE','cleanup':'UNRESOLVED'}[failure])
        with self.assertRaises(TypeError): p.snapshot(h,Model(),None)
    def test_files_manifest_payload(self):
        manifest=json.loads((ROOT/'research/v2/proposals/20260926-confirmed36-clean-checkpoint-cpu/INPUTS.json').read_text()); self.assertEqual(len(h.validate_manifest(manifest)),17)
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'f'; f.write_text('initial'); pin=h.file_record(f); f.write_text('changed')
            with self.assertRaises(h.CheckFailure): h.authenticate_files({str(f):pin})
            sy=Path(d)/'link'; sy.symlink_to(f)
            with self.assertRaises(h.CheckFailure): h.file_record(sy)
        for payload in [{},{'model_config':{'n_layer':8,'n_embd':1024,'vocab_size':8192,'sequence_len':257},'model_state':{str(i):t.ones(1) for i in range(65)}}]:
            with self.assertRaises(h.CheckFailure): h.full_payload(payload)
if len(sys.argv)>1 and sys.argv[1]=='child':
    directory,mode=Path(sys.argv[2]),sys.argv[3]
    def tiny_run(h,args,r,progress):
        c={}; r['arms']={'reference':{'shapes':[c]}}
        if mode=='rename':
            real=p.os.replace
            def die(src,dst):
                if c.get('current_call')=='repeat': os.kill(os.getpid(),signal.SIGTERM)
                return real(src,dst)
            p.os.replace=die
        runshape(Model('term' if mode=='term' else 'good'),fixture(1,256),c,progress)
    p.run=tiny_run; os.environ['RANK']='0'
    sys.argv=['probe','--enable-native','--label',p.LABEL,'--inputs','/unused','--out-dir',str(directory)]
    raise SystemExit(p.main())
if __name__=='__main__': unittest.main(verbosity=2)
