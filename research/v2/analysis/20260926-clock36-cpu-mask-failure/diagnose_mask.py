"""Source-local tiny mask diagnosis. No checkpoint/model/native/data access."""
import ast,hashlib,importlib.util,json,pathlib,subprocess,torch
torch.set_num_threads(1)
repo=pathlib.Path(__file__).resolve().parents[4]
raw=subprocess.check_output(['git','show','44926a6f:workflow/proposals/research-v2/train.py'],cwd=repo)
if hashlib.sha256(raw).hexdigest()!='4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea':raise RuntimeError('SOURCE_PIN')
tree=ast.parse(raw)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_build_window_mask')
ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen-mask-function','exec'),ns)
hp=repo/'research/v2/proposals/20260926-clock36-recursive-exact-cpu-r2-trn22/compare_checkpoints.py'
if hashlib.sha256(hp.read_bytes()).hexdigest()!='a3d9d767b33967311bc8d8fceb852fb771775aeeac0a82433ed2dd9560fa1b6d':raise RuntimeError('HELPER_PIN')
sp=importlib.util.spec_from_file_location('helper',hp);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
out=[]
for window in (8,4):
 mask=ns['_build_window_mask'](8,window,torch.float32)
 expect=torch.tensor([[0.0 if j<=i and j>i-window else -float('inf') for j in range(8)]for i in range(8)]).reshape(1,1,8,8)
 if not torch.equal(mask,expect):raise RuntimeError('MASK_REFERENCE')
 try:h.tensor_record(mask)
 except RuntimeError as exc:error=str(exc)
 else:raise RuntimeError('EXPECTED_EXISTING_FAILURE_NOT_REPRODUCED')
 if error!='NONFINITE_TENSOR':raise RuntimeError('WRONG_ERROR')
 out.append({'window':window,'shape':list(mask.shape),'negative_infinity_count':int(torch.isneginf(mask).sum()),'nan_count':int(torch.isnan(mask).sum()),'positive_infinity_count':int(torch.isposinf(mask).sum()),'expected_attention_mask_exact':True,'existing_helper_error':error})
print(json.dumps({'scope':'tiny exact frozen function only, no trained checkpoint loaded','source_sha256':hashlib.sha256(raw).hexdigest(),'helper_sha256':hashlib.sha256(hp.read_bytes()).hexdigest(),'source_buffers':'GPT.__init__ registers mask_full and mask_short from this function at lines2003-2004','observations':out,'native_failure_tensor_name':'not recorded in original traceback','interpretation':'Current helper rejects intentionally negative-infinite attention masks; this necessarily blocks an ordinary loaded model snapshot. No evidence here of a trained parameter NaN, and no source/model/evaluator fix is justified by this failure.'},indent=2))
