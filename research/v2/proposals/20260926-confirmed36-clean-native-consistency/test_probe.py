"""Author tests: tiny CPU models only; never import scientific model sources."""
import copy
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
            calls=[{'call':c,'status':'COMPLETE','output':{'shape':[1 if c.startswith('row') else b,t,8192]},'input':{},'device_input':{}}
                   for c in ['original','repeat']+['row'+str(i) for i in range(b)]+['reversed_suffix']]
            cases.append({'shape':[b,t],'status':'COMPLETE','input':{},'calls':calls})
        ranks=[{'rank':r,'status':'LOCAL_COMPLETE','arms':{name:{'shapes':copy.deepcopy(cases)} for name in ('reference','candidate')}} for r in range(4)]
        p.validate_ranks(h,ranks)
        for kind in ('missing_rank','duplicate_rank','missing_shape','duplicate_call','hash','partial_case','partial_call'):
            bad=copy.deepcopy(ranks)
            if kind=='missing_rank': bad.pop()
            if kind=='duplicate_rank': bad[1]['rank']=0
            if kind=='missing_shape': bad[0]['arms']['candidate']['shapes'].pop()
            if kind=='duplicate_call': bad[0]['arms']['candidate']['shapes'][0]['calls'].append(copy.deepcopy(cases[0]['calls'][0]))
            if kind=='hash': bad[0]['arms']['candidate']['shapes'][0]['calls'][0]['output']['sha256']='bad'
            if kind=='partial_case': bad[0]['arms']['candidate']['shapes'][0]['status']='IN_PROGRESS'
            if kind=='partial_call': bad[0]['arms']['candidate']['shapes'][0]['calls'][-1]['status']='IN_PROGRESS'
            with self.subTest(kind=kind), self.assertRaises(h.CheckFailure): p.validate_ranks(h,bad)


def tiny_report():
    return {'rank':0,'label':p.LABEL,'status':'UNRESOLVED','arms':{},'quality_evidence':False}


def receipt_child(directory, mode):
    """Run real main/persistence with a tiny CPU body, never native initialization."""
    def body(helper,args,report,progress):
        for name in ('reference','candidate'):
            arm = {'status':'IN_PROGRESS','shapes':[]}
            report['arms'][name] = arm
            progress('arm_ready',arm=name)
            class ChildTiny(Tiny):
                count = 0
                def forward(self,x):
                    self.count += 1
                    if name=='candidate' and mode=='term_call' and self.count==2:
                        os.kill(os.getpid(),signal.SIGTERM)
                    return super().forward(x)
            model = ChildTiny('future' if name=='candidate' and mode=='future' else None)
            x = torch.randint(1,17,(2,256),generator=torch.Generator().manual_seed(42))
            case = {}
            arm['shapes'].append(case)
            baseline = p.snapshot(h,model,lambda:None)
            p.verify_shape(h,model,x,17,torch.device('cpu'),lambda:None,baseline,case=case,
                progress=lambda stage,**loc:progress(stage,arm=name,shape=[2,256],**loc))
            arm['status'] = 'COMPLETE'
            progress('arm_complete',arm=name)
            if name=='reference' and mode=='term_write':
                def stop_before_replace(*args,**kwargs):
                    os.kill(os.getpid(),signal.SIGTERM)
                with mock.patch.object(p.os,'replace',side_effect=stop_before_replace):
                    progress('unpublished_next_boundary',arm=name)
        report['post_input_failures'] = []
    argv = ['probe','--enable-native','--label',p.LABEL,'--inputs',str(directory/'unused.json'),
            '--inputs-sha256',p.INPUTS_SHA,'--out-dir',str(directory)]
    cleanup_error = RuntimeError('injected cleanup failure') if mode=='cleanup_failure' else None
    with mock.patch.object(p,'run',side_effect=body), mock.patch.object(sys,'argv',argv), \
         mock.patch.dict(os.environ,{'RANK':'0'}), \
         mock.patch('torch.distributed.is_initialized',return_value=True), \
         mock.patch('torch.distributed.destroy_process_group',side_effect=cleanup_error):
        return p.main()


class ReceiptTests(unittest.TestCase):
    def child(self,directory,mode):
        command = [sys.executable] + (['-O'] if sys.flags.optimize else []) + [
            '-B',str(Path(__file__).absolute()),'--receipt-child',str(directory),mode]
        return subprocess.run(command,capture_output=True,text=True,timeout=30)

    def test_call_boundary_snapshots(self):
        with tempfile.TemporaryDirectory() as root:
            report = tiny_report()
            receipt = p.RankReceipt(h,Path(root),report,time.monotonic())
            case = {}
            report['arms']['reference'] = {'shapes':[case]}
            observed = []
            def progress(stage,**location):
                receipt.checkpoint(stage,arm='reference',shape=[2,256],**location)
                row = json.loads(receipt.partial.read_text())
                observed.append((stage,row))
                self.assertEqual(row['receipt_state'],'PARTIAL')
                self.assertEqual(row['status'],'UNRESOLVED')
            model = Tiny()
            x = torch.randint(1,17,(2,256),generator=torch.Generator().manual_seed(42))
            p.verify_shape(h,model,x,17,torch.device('cpu'),lambda:None,p.snapshot(h,model,lambda:None),
                           case=case,progress=progress)
            self.assertEqual([stage for stage,_ in observed].count('call_complete'),5)
            for stage,row in observed:
                rec = row['arms']['reference']['shapes'][0]
                if stage=='call_ready':
                    self.assertEqual(rec['calls'][-1]['status'],'IN_PROGRESS')
                    self.assertIn('device_input',rec['calls'][-1])
                    self.assertNotIn('output',rec['calls'][-1])
                    self.assertTrue(all(c['status']=='COMPLETE' for c in rec['calls'][:-1]))
                if stage=='call_complete':
                    self.assertIn('output',rec['calls'][-1])
                    self.assertTrue(all(c['status']=='COMPLETE' for c in rec['calls']))
            self.assertEqual(observed[-1][1]['arms']['reference']['shapes'][0]['status'],'COMPLETE')
            self.assertFalse(receipt.final.exists())

    def test_future_failure_keeps_actual_details(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.child(Path(root),'future')
            self.assertEqual(result.returncode,2,result.stderr)
            row = json.loads((Path(root)/'rank0.json').read_text())
            self.assertEqual(row['receipt_state'],'FINAL')
            self.assertEqual(row['status'],'REFUSE')
            self.assertEqual(row['error'],'CAUSAL_PREFIX')
            case = row['arms']['candidate']['shapes'][0]
            causal = case['causality']
            x = torch.tensor(case['tokens'])
            differences = (x[:,-1]-x[:,128]).abs()
            self.assertEqual(causal['max_abs_diff'],int(differences.max()))
            self.assertGreater(causal['max_abs_diff'],.01)
            self.assertEqual(causal['max_abs_index'],[int(differences.argmax()),0,0])
            self.assertEqual((causal['cut'],causal['prefix_start'],causal['prefix_stop']),(128,0,127))
            self.assertFalse(causal['prefix_exact'])
            self.assertGreater(causal['relative_rms'],0)
            suffix = x.clone(); suffix[:,128:] = suffix[:,128:].flip(1)
            expected = x.unsqueeze(-1).float()+torch.arange(17).float()+x[:,-1:].unsqueeze(-1)
            actual = suffix.unsqueeze(-1).float()+torch.arange(17).float()+suffix[:,-1:].unsqueeze(-1)
            self.assertEqual(causal['expected'],h.raw_record(expected[:,:127]))
            self.assertEqual(causal['actual'],h.raw_record(actual[:,:127]))
            self.assertEqual(causal['original_input'],h.raw_record(x))
            self.assertEqual(causal['altered_input'],h.raw_record(suffix))
            self.assertEqual(row['details'],causal)
            self.assertEqual(case['details'],causal)
            self.assertEqual(case['current_call'],'reversed_suffix')
            self.assertEqual(case['status'],'REFUSE')
            self.assertEqual(len(case['calls']),5)
            self.assertTrue(all(c['status']=='COMPLETE' for c in case['calls']))
            self.assertEqual(row['arms']['reference']['shapes'][0]['status'],'COMPLETE')
            self.assertFalse((Path(root)/'rank0.partial.json').exists())

    def test_causal_tolerance_stays_numerical_with_unequal_prefix_bytes(self):
        class SmallDifference(Tiny):
            def forward(self,x):
                return super().forward(x) + x[:,-1:].unsqueeze(-1)*.00001
        model = SmallDifference()
        x = torch.randint(1,17,(2,256),generator=torch.Generator().manual_seed(42))
        _,case = p.verify_shape(h,model,x,17,torch.device('cpu'),lambda:None,p.snapshot(h,model,lambda:None))
        self.assertEqual(case['status'],'COMPLETE')
        self.assertFalse(case['causality']['prefix_exact'])
        self.assertGreater(case['causality']['max_abs_diff'],0)
        self.assertLessEqual(case['causality']['max_abs_diff'],.01)
        self.assertEqual(case['causality']['tolerance'],.01)

    def test_overflow_causal_error_remains_a_serializable_refusal(self):
        class OverflowDifference(Tiny):
            def forward(self,x):
                sign = torch.where(x[:,-1]>x[:,128],1.,-1.).reshape(-1,1,1)
                return (sign*3e38).expand(len(x),256,17).clone()
        model = OverflowDifference()
        x = torch.randint(1,17,(2,256),generator=torch.Generator().manual_seed(42))
        case = {}
        with self.assertRaisesRegex(h.CheckFailure,'CAUSAL_PREFIX') as error:
            p.verify_shape(h,model,x,17,torch.device('cpu'),lambda:None,p.snapshot(h,model,lambda:None),case=case)
        self.assertEqual(error.exception.details['max_abs_diff'],'inf')
        self.assertFalse(error.exception.details['finite'])
        self.assertEqual(case['status'],'REFUSE')
        self.assertEqual(len(case['calls']),5)
        json.dumps(case,allow_nan=False)

    def test_real_sigterm_preserves_previous_arm_and_calls(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.child(Path(root),'term_call')
            self.assertEqual(result.returncode,-signal.SIGTERM,result.stderr)
            row = json.loads((Path(root)/'rank0.partial.json').read_text())
            self.assertEqual(row['status'],'UNRESOLVED')
            self.assertEqual(row['receipt_state'],'PARTIAL')
            self.assertEqual(row['progress'],{'stage':'call_ready','arm':'candidate','shape':[2,256],'call':'repeat'})
            self.assertEqual(row['arms']['reference']['status'],'COMPLETE')
            reference = row['arms']['reference']['shapes'][0]
            self.assertEqual(len(reference['calls']),5)
            self.assertTrue(all(c['status']=='COMPLETE' for c in reference['calls']))
            current = row['arms']['candidate']['shapes'][0]
            self.assertEqual(current['current_call'],'repeat')
            self.assertEqual([c['status'] for c in current['calls']],['COMPLETE','IN_PROGRESS'])
            self.assertIn('output',current['calls'][0])
            self.assertIn('device_input',current['calls'][1])
            self.assertNotIn('output',current['calls'][1])
            self.assertFalse((Path(root)/'rank0.json').exists())

    def test_real_sigterm_during_atomic_replace_keeps_last_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.child(Path(root),'term_write')
            self.assertEqual(result.returncode,-signal.SIGTERM,result.stderr)
            directory = Path(root)
            row = json.loads((directory/'rank0.partial.json').read_text())
            self.assertEqual(row['progress'],{'stage':'arm_complete','arm':'reference'})
            self.assertEqual(row['arms']['reference']['status'],'COMPLETE')
            temporaries = list(directory.glob('.rank0.*.tmp'))
            self.assertEqual(len(temporaries),1)
            self.assertEqual(json.loads(temporaries[0].read_text())['progress']['stage'],'unpublished_next_boundary')
            self.assertLessEqual(sum(f.stat().st_size for f in directory.iterdir()),p.CAP)
            self.assertFalse((directory/'rank0.json').exists())

    def test_final_success_and_cleanup_failure(self):
        for mode,status in (('success','LOCAL_COMPLETE'),('cleanup_failure','UNRESOLVED')):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                result = self.child(Path(root),mode)
                self.assertEqual(result.returncode,0 if mode=='success' else 2,result.stderr)
                row = json.loads((Path(root)/'rank0.json').read_text())
                self.assertEqual(row['status'],status)
                self.assertEqual(row['receipt_state'],'FINAL')
                self.assertEqual('cleanup_error' in row,mode=='cleanup_failure')
                self.assertFalse((Path(root)/'rank0.partial.json').exists())
                self.assertFalse(list(Path(root).glob('*.tmp')))

    def test_atomic_failure_cap_and_nonfinite_preserve_last_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            report = tiny_report()
            receipt = p.RankReceipt(h,Path(root),report,time.monotonic())
            original = receipt.partial.read_bytes()
            with mock.patch.object(p.os,'replace',side_effect=OSError('injected replace failure')):
                with self.assertRaisesRegex(OSError,'replace failure'):
                    receipt.checkpoint('not_published')
            self.assertEqual(receipt.partial.read_bytes(),original)
            report['oversized'] = 'x'*4096
            with mock.patch.object(p,'CAP',4096), self.assertRaisesRegex(RuntimeError,'RECEIPT_CAP'):
                receipt.checkpoint('too_big')
            self.assertEqual(receipt.partial.read_bytes(),original)
            del report['oversized']
            report['nonfinite'] = float('nan')
            with self.assertRaises(ValueError): receipt.checkpoint('nonfinite')
            self.assertEqual(receipt.partial.read_bytes(),original)
            self.assertEqual(list(Path(root).iterdir()),[receipt.partial])

    def test_fsync_before_replace_and_exclusive_final(self):
        with tempfile.TemporaryDirectory() as root:
            report = tiny_report()
            receipt = p.RankReceipt(h,Path(root),report,time.monotonic())
            old = receipt.partial.read_bytes()
            fsync, replace = p.os.fsync, p.os.replace
            events = []
            def synced(fd):
                events.append('fsync')
                return fsync(fd)
            def replaced(source,destination):
                self.assertEqual(events[0],'fsync')
                self.assertEqual(receipt.partial.read_bytes(),old)
                self.assertEqual(json.loads(Path(source).read_text())['progress']['stage'],'next')
                events.append('replace')
                return replace(source,destination)
            with mock.patch.object(p.os,'fsync',side_effect=synced), mock.patch.object(p.os,'replace',side_effect=replaced):
                receipt.checkpoint('next')
            self.assertEqual(events,['fsync','replace','fsync'])
            report['status'] = 'LOCAL_COMPLETE'
            receipt.finish()
            final_bytes = receipt.final.read_bytes()
            self.assertEqual(json.loads(final_bytes)['receipt_state'],'FINAL')
            with self.assertRaisesRegex(RuntimeError,'RECEIPT_ALREADY_FINAL'): receipt.checkpoint('late')
            with self.assertRaisesRegex(h.CheckFailure,'RECEIPT_ALREADY_EXISTS'):
                p.RankReceipt(h,Path(root),tiny_report(),time.monotonic())
            self.assertEqual(receipt.final.read_bytes(),final_bytes)

    def test_partial_reuse_final_collision_and_directory_symlink_refuse(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)/'receipts'
            receipt = p.RankReceipt(h,directory,tiny_report(),time.monotonic())
            with self.assertRaisesRegex(h.CheckFailure,'RECEIPT_ALREADY_EXISTS'):
                p.RankReceipt(h,directory,tiny_report(),time.monotonic())
            receipt.final.write_text('previous evidence')
            with self.assertRaises(FileExistsError): receipt.finish()
            self.assertEqual(receipt.final.read_text(),'previous evidence')
            self.assertTrue(receipt.partial.is_file())
            linked = Path(root)/'linked'; linked.symlink_to(directory,target_is_directory=True)
            with self.assertRaisesRegex(h.CheckFailure,'RECEIPT_DIRECTORY_SYMLINK'):
                p.RankReceipt(h,linked,tiny_report(),time.monotonic())

    def test_final_directory_sync_failure_keeps_partial_marker(self):
        with tempfile.TemporaryDirectory() as root:
            receipt = p.RankReceipt(h,Path(root),tiny_report(),time.monotonic())
            original = receipt.partial.read_bytes()
            with mock.patch.object(receipt,'sync_directory',side_effect=OSError('injected directory sync failure')):
                with self.assertRaisesRegex(OSError,'directory sync failure'): receipt.finish()
            self.assertEqual(receipt.partial.read_bytes(),original)
            self.assertTrue(receipt.final.exists())
            self.assertFalse(receipt.finished)
            self.assertEqual(sorted(f.name for f in Path(root).glob('rank*.json')),['rank0.json','rank0.partial.json'])

    def test_offline_validator_refuses_partial_markers_even_if_renamed(self):
        # Metadata only. No production-width output tensors are allocated.
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            for rank in range(4):
                row = {'rank':rank,'label':p.LABEL,'status':'LOCAL_COMPLETE',
                       'post_input_failures':[],'receipt_state':'PARTIAL'}
                (directory/('rank'+str(rank)+'.json')).write_text(json.dumps(row))
            with mock.patch.object(sys,'argv',['probe','--validate-receipts',str(directory)]), \
                 self.assertRaisesRegex(h.CheckFailure,'TERMINAL_RECEIPTS'):
                # Preserve exception identity for this isolated CLI-boundary test.
                with mock.patch.object(p,'foundation',return_value=h):
                    p.main()


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--receipt-child':
        raise SystemExit(receipt_child(Path(sys.argv[2]),sys.argv[3]))
    unittest.main(verbosity=2)
