"""Bounded local selfchecks. Tensor helper test uses <1MiB, never loads a model."""
import ast,copy,hashlib,importlib.util,json,math,subprocess,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
def module(name):
 spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
P=module('probe');V=module('validate_receipts')
class Packet(unittest.TestCase):
 def test_disabled_does_not_import_torch(self):
  output=subprocess.check_output([sys.executable,'-B',str(HERE/'probe.py')],text=True)
  self.assertEqual(json.loads(output),{'status':'DISABLED','model_imported':False})
 def test_nonfinite_safe_and_terminal_classification(self):
  for x in (float('nan'),float('inf'),-float('inf')):
   self.assertIsNone(P.finite_metric(x));json.dumps({'x':P.finite_metric(x)},allow_nan=False)
  self.assertEqual(P.verdict([{'passed':True}]),V.EXACT)
  for c in ([],[{'passed':False}],[{'passed':True},{'passed':None}]):self.assertIn('HOLD',P.verdict(c))
 def test_bindings_guard_and_sources(self):
  for backend in ('cpu','neuron'):
   b=json.loads((HERE/backend/'BINDING.disabled.json').read_text());s=(HERE/backend/'host_driver.py').read_text()
   self.assertFalse(b['launch_authorized']);self.assertFalse(b['stage_authorized'])
   self.assertEqual(b['environment']['NEURON_CC_FLAGS'],'--optlevel=1')
   self.assertEqual(b['environment']['NEURON_COMPILE_CACHE_URL'],b['environment']['NEURON_COMPILE_CACHE_DIR'])
   self.assertIn('--kill-after=15s',b['command'])
   self.assertIn('600s' if backend=='cpu' else '900s',b['command'])
   self.assertIn("pass_fds=(lock_fd,)",s);self.assertIn('require_idle_under_inherited_lock',s)
   self.assertLess(s.index('lock_fd = guard.require_idle_under_inherited_lock'),s.index('child=subprocess.Popen'))
   for token in ('RAM_BELOW_64_GIB','PREPARE_SHA','CHECKPOINT_SHA','CHECKPOINT_CHANGED','ARTIFACT_CAP_STOPPED_DIAGNOSTIC','SUPPORT_SHA','STOP:','DISK_HEADROOM'):
    self.assertIn(token,s)
   for name in ('probe.py','validate_receipts.py'):
    self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),b['source_pins'][name])
   if 'driver_sha256' in b:self.assertEqual(hashlib.sha256((HERE/backend/'host_driver.py').read_bytes()).hexdigest(),b['driver_sha256'])
 def test_fixed_nine_calls_and_no_patching(self):
  tree=ast.parse((HERE/'probe.py').read_text())
  run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  assign=next(n for n in ast.walk(run) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='matrix')
  self.assertEqual([ast.literal_eval(e.elts[0]) for e in assign.value.elts],V.ORDER)
  text=ast.unparse(tree)
  for forbidden in ('compute_dtype_for =','register_forward_hook','register_forward_pre_hook','torch.compile','make_eval_dataloader','ensure_tokenizer','model.train(','backward('):self.assertNotIn(forbidden,text)
  self.assertIn('weights_only=True',text);self.assertIn("model = mod.load_for_eval",text)
 def test_validator_refuses_missing_differences_or_rank_mismatch(self):
  # Fixture only, not model evidence. Validator must reject malformed/incomplete receipts.
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'receipts').mkdir()
   with self.assertRaises(RuntimeError):V.validate(root,'x','cpu')
   r={'rank':0,'world_size':1,'status':V.EXACT,'backend':'cpu','label':'x','candidate_sha256':V.CANDIDATE,'reference_sha256':V.REFERENCE,'checkpoint_sha256':V.CHECKPOINT,'call_order':V.ORDER,'cases':[{'name':name,'passed':True} for name in sorted(V.CASE_NAMES)],'outputs':{k:{n:{'sha256':'test'} for n in V.ORDER} for k in ('reference','candidate')},'inputs':{},'source_review':'REFUSE_UNCHANGED','upload':'HOLD'}
   f=root/'receipts/rank0.json';f.write_text(json.dumps(r));self.assertEqual(V.validate(root,'x','cpu')['status'],V.EXACT)
   for key,value in [('rank',1),('world_size',4),('status','ERROR'),('checkpoint_sha256','wrong'),('call_order',[])]:
    bad=copy.deepcopy(r);bad[key]=value;f.write_text(json.dumps(bad))
    with self.assertRaises(RuntimeError):V.validate(root,'x','cpu')
   r['cases'][0]['passed']=False;f.write_text(json.dumps(r))
   with self.assertRaises(RuntimeError):V.validate(root,'x','cpu')
class TinyTensor(unittest.TestCase):
 @unittest.skipUnless('--tiny-tensors' in sys.argv,'explicit tiny slot only')
 def test_actual_comparison_helper(self):
  import torch
  torch.set_num_threads(1)
  tree=ast.parse((HERE/'probe.py').read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  fns=[n for n in run.body if isinstance(n,ast.FunctionDef) and n.name in ('raw','tensor_record','compare')]
  receipt={'cases':[]};ns={'torch':torch,'hashlib':hashlib,'math':math,'finite_metric':P.finite_metric,'receipt':receipt,'T':1024}
  exec(compile(ast.Module(body=fns,type_ignores=[]),'<exact nested helpers>','exec'),ns)
  compare=ns['compare'];a=torch.tensor([[[1.,2.],[3.,4.]]])
  compare('identical',a,a.clone());self.assertTrue(receipt['cases'][-1]['passed'])
  for delta in (1e-5,0.001):
   b=a.clone();b[0,0,0]+=delta;compare('strict',a,b);self.assertFalse(receipt['cases'][-1]['passed'])
  compare('signedzero',torch.tensor([-0.]),torch.tensor([0.]));self.assertFalse(receipt['cases'][-1]['passed'])
  for value in (float('nan'),float('inf'),-float('inf')):
   b=a.clone();b[0,0,0]=value;compare('nonfinite',b,a);self.assertFalse(receipt['cases'][-1]['passed']);json.dumps(receipt,allow_nan=False)
  b=a.clone();b[0,0,0]+=.001;compare('causal',b,a,causal=True);self.assertTrue(receipt['cases'][-1]['passed'])
if __name__=='__main__':
 if '--tiny-tensors' in sys.argv:sys.argv.remove('--tiny-tensors')
 unittest.main(verbosity=2)
