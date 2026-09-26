import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace as NS
import os
import sys
import subprocess
import torch

ROOT=Path('/home/poojith-devan/clean36-native-fresh-WIP7Cm')
BASE=ROOT/'research/v2/proposals/20260926-confirmed36-clean-native-consistency'
spec=importlib.util.spec_from_file_location('reviewed_probe', BASE/'probe.py')
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
h=p.foundation()
torch.set_num_threads(1)

def raw(x):
    return hashlib.sha256(bytes(x.detach().contiguous().reshape(-1).view(torch.uint8).tolist())).hexdigest()

def oracle(x):
    # Separate Python scalar implementation, deliberately no batch arithmetic.
    result=[]
    for row in x.tolist():
        total=0; values=[]
        for token in row:
            total += token
            values.append([float((total+j)%19) for j in range(17)])
        result.append(values)
    return torch.tensor(result)

class Tiny(torch.nn.Module):
    def __init__(self, fault=None):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.ones(1),requires_grad=False)
        self.register_buffer('cache',torch.zeros(1),persistent=False)
        self.direct=torch.zeros(1)
        self.fault=fault
        self.calls=0
        self.eval()
    def forward(self,x):
        self.calls += 1
        y=((x.cumsum(1).unsqueeze(-1)+torch.arange(17))%19).float()
        if self.fault=='parameter': self.weight.add_(1)
        if self.fault=='buffer': self.cache.add_(1)
        if self.fault=='direct': self.direct.add_(1)
        if self.fault=='mode': self.train()
        if self.fault=='grad': self.weight.requires_grad_(True)
        if self.fault=='input': x[0,0] += 1
        if self.fault=='nan': y[0,0,0]=float('nan')
        if self.fault=='inf': y[0,0,0]=float('inf')
        if self.fault=='batch': y += x.shape[0]
        if self.fault=='future': y += x[:,-1,None,None]
        if self.fault=='repeat': y += self.calls
        if self.fault=='alias':
            # A shared external scratch is excluded from direct model state;
            # clone-at-return must prevent it concealing changed earlier output.
            global scratch
            scratch.resize_(y.shape).copy_(y+self.calls)
            return scratch
        return y

scratch=torch.empty(0)
def check(model,b=2,t=256):
    x=torch.randint(1,17,(b,t),generator=torch.Generator().manual_seed(42))
    baseline=p.snapshot(h,model,lambda:None)
    return x,p.verify_shape(h,model,x,17,'cpu',lambda:None,baseline)

class Review(unittest.TestCase):
    def test_six_shapes_independent_oracle(self):
        count=0
        for b,t in p.SHAPES:
            x,(out,case)=check(Tiny(),b,t)
            self.assertEqual(raw(out),raw(oracle(x)))
            self.assertEqual(case['input']['sha256'],raw(x))
            self.assertEqual(case['tokens'],x.tolist())
            self.assertTrue(case['repeat']['exact_bytes'])
            self.assertTrue(case['rowwise']['exact_bytes'])
            self.assertEqual(case['causality']['max_abs_diff'],0)
            count += len(case['calls'])
        self.assertEqual(count,30)

    def test_per_call_adversaries(self):
        for fault in ['parameter','buffer','direct','mode','grad','input','nan','inf','batch','future','repeat','alias']:
            with self.subTest(fault=fault),self.assertRaises(h.CheckFailure): check(Tiny(fault))

    def test_byte_exactness(self):
        for a,b in [(torch.tensor([0.]),torch.tensor([-0.])),(torch.tensor([1.]),torch.nextafter(torch.tensor([1.]),torch.tensor([2.]))),(torch.ones(1),torch.ones(1,dtype=torch.float64)),(torch.ones(1),torch.ones(1,1))]:
            with self.assertRaises(h.CheckFailure): p.exact(h,a,b,'negative')
        result=p.exact(h,torch.zeros(2),torch.zeros(2),'positive')
        self.assertEqual(result['relative_rms'],0)
        with self.assertRaises(h.CheckFailure) as caught:p.exact(h,torch.ones(1),torch.zeros(1),'zero')
        self.assertEqual(caught.exception.details['relative_rms'],'UNDEFINED_ZERO_REFERENCE')

    def test_mask_rules(self):
        class Masks(Tiny):
            def __init__(self):
                super().__init__()
                for name,w in [('mask_full',8),('mask_short',4)]:
                    value=torch.tensor([[0. if j<=i and j>i-w else -float('inf') for j in range(8)] for i in range(8)]).reshape(1,1,8,8)
                    self.register_buffer(name,value,persistent=False)
        masks={'full_window':8,'short_window':4,'sequence_len':8}
        m=Masks(); baseline=p.snapshot(h,m,lambda:None,masks)
        self.assertEqual(baseline,p.snapshot(h,m,lambda:None,masks,baseline))
        for value in [0.,float('nan'),float('inf')]:
            m=Masks();m.mask_full[0,0,0,7]=value
            with self.assertRaises(h.CheckFailure):p.snapshot(h,m,lambda:None,masks)
        m=Masks();m.direct[0]=-float('inf')
        with self.assertRaises(h.CheckFailure):p.snapshot(h,m,lambda:None,masks)
        m=Masks();m.mask_short[0,0,7,0]=0
        with self.assertRaises(h.CheckFailure):p.snapshot(h,m,lambda:None,masks,baseline)

    def test_modes_hooks_sync_degenerate(self):
        m=Tiny(); m.register_forward_hook(lambda *a:None)
        with self.assertRaises(h.CheckFailure):p.snapshot(h,m,lambda:None)
        def failure():raise RuntimeError('synchronization failed')
        with self.assertRaises(RuntimeError):p.snapshot(h,Tiny(),failure)
        m=Tiny(); baseline=p.snapshot(h,m,lambda:None)
        with self.assertRaises(h.CheckFailure):p.verify_shape(h,m,torch.ones(1,256,dtype=torch.int64),17,'cpu',lambda:None,baseline)

    def test_receipt_structure(self):
        cases=[]
        for b,t in p.SHAPES:
            _,(_,case)=check(Tiny(),b,t)
            # Structural validator mandates production width. No claim these
            # metadata-only fixtures are full-width numeric evidence.
            for call in case['calls']:call['output']['shape'][-1]=8192
            cases.append(case)
        arms={'reference':{'shapes':cases},'candidate':{'shapes':copy.deepcopy(cases)}}
        ranks=[{'rank':i,'status':'LOCAL_COMPLETE','arms':copy.deepcopy(arms)} for i in range(4)]
        p.validate_ranks(h,ranks)
        mutations=[lambda r:r.pop(),lambda r:r[0].update(rank=1),lambda r:r[0]['arms']['reference']['shapes'].pop(),lambda r:r[0]['arms']['reference']['shapes'][0]['calls'].pop(),lambda r:r[0]['arms']['reference']['shapes'][0]['calls'].append(r[0]['arms']['reference']['shapes'][0]['calls'][0]),lambda r:r[0]['arms']['candidate']['shapes'][0]['calls'][0]['output'].update(sha256='0'*64)]
        for mutate in mutations:
            altered=copy.deepcopy(ranks);mutate(altered)
            with self.assertRaises(h.CheckFailure):p.validate_ranks(h,altered)

    def test_input_identity(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            f=Path(d)/'input';f.write_text('abc');pin=h.file_record(f)
            h.authenticate_files({str(f):pin})
            f.write_text('abd')
            with self.assertRaises(h.CheckFailure):h.authenticate_files({str(f):pin})
            s=Path(d)/'symlink';s.symlink_to(f)
            with self.assertRaises(h.CheckFailure):h.file_record(s)
        manifest=json.loads((ROOT/'research/v2/proposals/20260926-confirmed36-clean-checkpoint-cpu/INPUTS.json').read_text())
        self.assertEqual(len(h.validate_manifest(manifest)),17)
        manifest['files'][str(h.ROOT/'candidate/train.py')]['sha256']='0'*64
        with self.assertRaises(h.CheckFailure):h.validate_manifest(manifest)

    def test_payload_rejects_without_allocation(self):
        for payload in [{},{'model_config':{'n_layer':8,'n_embd':1024,'vocab_size':8192,'sequence_len':257},'model_state':{}},{'model_config':{'n_layer':8,'n_embd':1024,'vocab_size':8192,'sequence_len':257},'model_state':{str(i):torch.zeros(1) for i in range(65)}}]:
            with self.assertRaises(h.CheckFailure):h.full_payload(payload)

    def test_restored_route_boundary_without_native(self):
        class Lane:
            def forward(ctx,x,weight):
                lanes=2 if x.shape[0] % 2 == 0 else 1
                return lanes
        class Model:
            def parameters(self):return [NS(device=NS(type='neuron'))]
        model=Model()
        model.config=NS(sequence_len=8192,n_embd=1024,vocab_size=8192)
        model.transformer=NS(h=[NS(attn=object() if i in [0,2,4,6,7] else None,local_conv=NS(kernel_size=2,n_embd=1024) if i in [0,2,4,6,7] else None) for i in range(8)])
        model.unet_skip_gates=[torch.zeros(1) for _ in range(4)]
        model.value_embeds={'7':object()}
        flags={'USE_UNET_SKIP':True,'USE_PARTIAL_ATTN_8':True,'USE_QK_SHIFT':False,'USE_NKI_LOCAL_CONV':True,'USE_BF16_NORM_OUTPUT':True,'COMPILE_SDPA_DIRECT':True,'SHORT_WINDOW_FRAC':.5}
        module=NS(_HAS_NKI=True,COMPUTE_DTYPE=torch.bfloat16,EVAL_MIN_SEQ_LEN=8192,CausalConv=Lane,_Conv2NKI=Lane,_SubmissionEvalGPT=Model,_EVAL_TOGGLE_DEFAULTS={'SHORT_WINDOW_FRAC':.5},**flags)
        payload={'toggles':copy.deepcopy(flags),'model_config':{'sequence_len':257}}
        masks,route=p.restored_route(h,'reference',model,module,payload,NS(type='neuron'))
        self.assertEqual(masks['short_window'],4096)
        self.assertEqual([x['base_batch'] for x in route['predicate_coverage']],[1]*6)
        for key in flags:
            changed=copy.deepcopy(payload);del changed['toggles'][key]
            with self.assertRaises(h.CheckFailure):p.restored_route(h,'reference',model,module,changed,NS(type='neuron'))
        for field,value in [('COMPUTE_DTYPE',torch.float32),('_HAS_NKI',False),('EVAL_MIN_SEQ_LEN',4096)]:
            changed=copy.copy(module);setattr(changed,field,value)
            with self.assertRaises(h.CheckFailure):p.restored_route(h,'reference',model,changed,payload,NS(type='neuron'))

    def test_main_failure_cleanup(self):
        # CPU mocks exercise the real main finally, not a native import/runtime.
        import torch.distributed as dist
        for cleanup_failure in (False,True):
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
                argv=['probe','--enable-native','--label',p.LABEL,'--inputs',str(Path(d)/'input'),'--out-dir',d]
                with mock.patch.object(sys,'argv',argv),mock.patch.dict(os.environ,{'RANK':'0'}),mock.patch.object(p,'foundation',return_value=h),mock.patch.object(p,'run',side_effect=RuntimeError('native init failed')),mock.patch.object(torch,'set_num_interop_threads'),mock.patch.object(dist,'is_initialized',return_value=True),mock.patch.object(dist,'destroy_process_group',side_effect=RuntimeError('cleanup failed') if cleanup_failure else None) as destroy:
                    self.assertEqual(p.main(),2)
                destroy.assert_called_once()
                receipt=json.loads((Path(d)/'rank0.json').read_text())
                self.assertEqual(receipt['status'],'UNRESOLVED' if cleanup_failure else 'REFUSE')
                self.assertEqual('cleanup_error' in receipt,cleanup_failure)

    def test_demonstrates_term_loses_completed_partial_receipt(self):
        # This passing test reproduces a specification defect, not acceptance.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            code='''import importlib.util,sys,os,signal
from pathlib import Path
s=importlib.util.spec_from_file_location('probe',sys.argv[1]);p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
def interrupted(h,args,report):
    report['arms']['reference']={'shapes':[{'shape':[1,256],'completed':True}]}
    print('completed partial shape in report; delivering SIGTERM',flush=True)
    os.kill(os.getpid(),signal.SIGTERM)
p.run=interrupted
sys.argv=['probe','--enable-native','--label',p.LABEL,'--inputs',str(Path(sys.argv[2])/'input'),'--out-dir',sys.argv[2]]
os.environ['RANK']='0'
p.main()
'''
            result=subprocess.run([sys.executable,'-B','-c',code,str(BASE/'probe.py'),d],capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,-15)
            self.assertIn('completed partial shape',result.stdout)
            self.assertEqual(list(Path(d).glob('rank*.json')),[])

    def test_demonstrates_causality_failure_loses_metric(self):
        with self.assertRaises(h.CheckFailure) as caught:check(Tiny('future'))
        self.assertEqual(str(caught.exception),'CAUSAL_PREFIX')
        self.assertIsNone(caught.exception.details)

    def test_exact_scientific_source_predicates_and_loaders(self):
        paths=[ROOT/'science/reference/workflow/proposals/research-v2/train.py',ROOT/'science/candidate/workflow/candidates/confirmed-clock36-116188/train.py']
        for path,pin in zip(paths,h.SOURCES.values()):
            source=path.read_text();self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),pin)
            tree=ast.parse(source)
            classes={n.name:n for n in ast.walk(tree) if isinstance(n,ast.ClassDef)}
            conv=next(n for n in classes['CausalConv'].body if isinstance(n,ast.FunctionDef) and n.name=='forward')
            condition=next(n for n in conv.body if isinstance(n,ast.If)).test
            # Execute only the branch boolean from AST, never scientific imports.
            expr=compile(ast.Expression(condition),str(path),'eval')
            from types import SimpleNamespace as NS
            for b,t in p.SHAPES:
                x=NS(device=NS(type='neuron'),dtype=torch.bfloat16,shape=(b,t,1024))
                self.assertEqual(eval(expr,{'torch':torch,'USE_NKI_LOCAL_CONV':True,'self':NS(kernel_size=2),'x':x}),t==256)
            lanes=next(n.value for n in ast.walk(classes['_Conv2NKI']) if isinstance(n,ast.Assign) and any(isinstance(q,ast.Name) and q.id=='lanes' for q in n.targets))
            lane_expr=compile(ast.Expression(lanes),str(path),'eval')
            for b in (1,2,3):self.assertEqual(eval(lane_expr,{'x':NS(shape=(b,256,1024))}),2 if b==2 else 1)
            loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_for_eval')
            text=ast.get_source_segment(source,loader)
            for required in ['compute_dtype_for(device)','strict=True','model.to(device)','model.eval()','model.requires_grad_(False)','EVAL_MIN_SEQ_LEN']:
                self.assertIn(required,text)
        self.assertIn('single_forward(idx[i:i + 1])',paths[0].read_text())

if __name__=='__main__':unittest.main(verbosity=2)
