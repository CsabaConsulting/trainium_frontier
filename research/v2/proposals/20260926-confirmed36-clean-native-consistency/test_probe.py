"""Author tests: tiny CPU models only; never import scientific model sources."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('probe',Path(__file__).with_name('probe.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
h = p.foundation()
torch = h.torch
torch.set_num_threads(1)


class Tiny(torch.nn.Module):
    def __init__(self, mutation=None):
        super().__init__()
        self.register_buffer('weight',torch.arange(17).float())
        self.cache = torch.zeros(1)
        self.mutation = mutation
        self.eval()

    def forward(self,x):
        out = x.unsqueeze(-1).float() + self.weight
        if self.mutation=='state': self.weight.add_(1)
        if self.mutation=='attr': self.cache.add_(1)
        if self.mutation=='input': x.add_(1)
        if self.mutation=='mode': self.train()
        if self.mutation=='future': out = out + x[:,-1:].unsqueeze(-1)
        if self.mutation=='nan': out[0,0,0] = float('nan')
        if self.mutation=='batch': out = out + len(x)
        return out


class Tests(unittest.TestCase):
    def check_model(self, mutation=None, shape=(2,256), degenerate=False):
        model = Tiny(mutation)
        baseline = p.snapshot(h,model,lambda:None)
        x = torch.ones(shape,dtype=torch.long) if degenerate else torch.randint(1,17,shape,generator=torch.Generator().manual_seed(42))
        return p.verify_shape(h,model,x,17,torch.device('cpu'),lambda:None,baseline)

    def test_six_positive_shapes(self):
        for shape in p.SHAPES:
            with self.subTest(shape=shape):
                _, rec = self.check_model(shape=shape)
                self.assertEqual(len(rec['calls']),shape[0]+3)

    def test_mutations(self):
        for mutation in ('state','attr','input','mode','future','nan','batch'):
            with self.subTest(mutation=mutation), self.assertRaises(h.CheckFailure):
                self.check_model(mutation)

    def test_degenerate(self):
        with self.assertRaisesRegex(h.CheckFailure,'DEGENERATE'):
            self.check_model(degenerate=True)

    def test_signed_zero_and_byte(self):
        for a,b in ((torch.tensor([0.]),torch.tensor([-0.])),(torch.tensor([1.]),torch.tensor([1.0000001]))):
            with self.assertRaises(h.CheckFailure) as error:
                p.exact(h,a,b,'DIFF')
            self.assertIsNotNone(error.exception.details['first_different_byte'])

    def test_masks(self):
        mask = torch.zeros(1,1,8,8)
        mask[0,0].masked_fill_(torch.ones(8,8,dtype=torch.bool).triu(1),-float('inf'))
        h.mask_record(mask,8,8,'mask_full')
        for val in (0.,float('nan'),float('inf')):
            bad = mask.clone(); bad[0,0,0,1]=val
            with self.assertRaises(h.CheckFailure): h.mask_record(bad,8,8,'mask_full')

    def test_payload_rejection_without_full_allocation(self):
        for payload in ({}, {'model_config':{'n_layer':7,'n_embd':1024,'vocab_size':8192}},
                        {'model_config':{'n_layer':8,'n_embd':1024,'vocab_size':8192,'sequence_len':256}},
                        {'model_config':{'n_layer':8,'n_embd':1024,'vocab_size':8192,'sequence_len':257},'model_state':{}}):
            with self.assertRaises(h.CheckFailure): h.full_payload(payload)

    def test_aliasing_repeat_corruption(self):
        scratch=torch.zeros(2,256,17)
        class Aliasing(Tiny):
            def forward(self,x):
                scratch.add_(1)
                return scratch[:len(x)]
        model=Aliasing()
        before=p.snapshot(h,model,lambda:None)
        x=torch.randint(1,17,(2,256),generator=torch.Generator().manual_seed(42))
        with self.assertRaisesRegex(h.CheckFailure,'REPEAT_BYTES'):
            p.verify_shape(h,model,x,17,torch.device('cpu'),lambda:None,before)

    def test_file_and_symlink(self):
        with tempfile.TemporaryDirectory() as root:
            a=Path(root)/'a'; a.write_text('fixture')
            pin=h.file_record(a)
            h.authenticate_files({str(a):pin})
            a.write_text('changed')
            with self.assertRaises(h.CheckFailure): h.authenticate_files({str(a):pin})
            b=Path(root)/'b'; b.symlink_to(a)
            with self.assertRaises(h.CheckFailure): h.file_record(b)

    def test_hooks_grad_sync(self):
        model=Tiny(); model.register_forward_hook(lambda *a:None)
        with self.assertRaises(h.CheckFailure): p.snapshot(h,model,lambda:None)
        model=Tiny(); model.param=torch.nn.Parameter(torch.ones(1))
        with self.assertRaises(h.CheckFailure): p.snapshot(h,model,lambda:None)
        with self.assertRaisesRegex(RuntimeError,'sync'):
            p.snapshot(h,Tiny(),lambda:(_ for _ in ()).throw(RuntimeError('sync')))

    def test_complete_cases_and_ranks(self):
        cases=[]
        for b,t in p.SHAPES:
            calls=[{'call':c,'output':{'shape':[1 if c.startswith('row') else b,t,8192]},'input':{},'device_input':{}}
                   for c in ['original','repeat']+['row'+str(i) for i in range(b)]+['reversed_suffix']]
            cases.append({'shape':[b,t],'input':{},'calls':calls})
        ranks=[{'rank':r,'status':'LOCAL_COMPLETE','arms':{name:{'shapes':copy.deepcopy(cases)} for name in ('reference','candidate')}} for r in range(4)]
        p.validate_ranks(h,ranks)
        for kind in ('missing_rank','duplicate_rank','missing_shape','duplicate_call','hash'):
            bad=copy.deepcopy(ranks)
            if kind=='missing_rank': bad.pop()
            if kind=='duplicate_rank': bad[1]['rank']=0
            if kind=='missing_shape': bad[0]['arms']['candidate']['shapes'].pop()
            if kind=='duplicate_call': bad[0]['arms']['candidate']['shapes'][0]['calls'].append(copy.deepcopy(cases[0]['calls'][0]))
            if kind=='hash': bad[0]['arms']['candidate']['shapes'][0]['calls'][0]['output']['sha256']='bad'
            with self.subTest(kind=kind), self.assertRaises(h.CheckFailure): p.validate_ranks(h,bad)


if __name__=='__main__': unittest.main(verbosity=2)
