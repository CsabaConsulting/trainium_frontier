"""Small ordinary-backward CPU evidence; no custom embedding kernel executes."""
import ast
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
TABLE=REPO/'research/v2/lanes/tables-compact-k2'
sys.path.insert(0,str(TABLE))
sys.path.insert(0,str(REPO/'workflow/tools/stub_prepare'))
arrow,parquet=types.ModuleType('pyarrow'),types.ModuleType('pyarrow.parquet')
arrow.parquet=parquet
sys.modules['pyarrow'],sys.modules['pyarrow.parquet']=arrow,parquet
import test_tables as suite
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
a=suite.load('lottery_base',REPO/'workflow/candidates/final-batch262144-lr021/train.py')
b=suite.load('lottery_framework',HERE/'train.py')
assert a.TOTAL_BATCH_SIZE==b.TOTAL_BATCH_SIZE==262144
assert a.MATRIX_LR==b.MATRIX_LR==.021 and b.K2_EMBEDDING is False
def forbidden(*args,**kwargs): raise AssertionError('CUSTOM_EMBEDDING_PATH_EXECUTED')
for name in ('_k2_prepare_metadata','_k2_build_kernel','_k2_build_registered_lookup',
             '_k2_table_build_kernel','_k2_table_build_registered_lookup'):
    setattr(b,name,forbidden)
results={'hashes':[], 'collision_reference':[], 'native_executed':False}
for dtype in (torch.int32,torch.int64):
    suite.INDEX_DTYPE=dtype
    results['hashes'].append(suite.hashes(b))
    # Independent ordered FP32 collision sum distinguishes BF16 accumulation.
    ids=torch.tensor([[1,1,1,3]],dtype=dtype)
    p=torch.nn.Parameter(torch.zeros(8,4,dtype=torch.float32))
    g=torch.tensor([1.,2**-8,-1.,.5],dtype=torch.bfloat16).view(1,4,1).expand(1,4,4)
    reference=torch.zeros_like(p)
    for micro in range(2):
        F.embedding(ids,p).bfloat16().backward(g)
        for row,grad in zip(ids.flatten().tolist(),g.reshape(-1,4)):
            reference[row].add_(grad.float())
        suite.exact(p.grad,reference)
        assert p.grad.dtype==torch.float32
    assert p.grad[1,0].item()==2**-7
    results['collision_reference'].append(str(dtype))
results['rms']=suite.rms(b)
results['clipping']=suite.clipping(b)
b._P2_ROWS=32
def make(module,on=False):
    torch.manual_seed(42)
    kw=dict(sequence_len=8,vocab_size=64,n_layer=8,n_head=2,n_embd=128,mlp_ratio=2)
    if module is b: kw['ngram_ve']=on
    model=module.GPT(module.GPTConfig(**kw));model.init_weights()
    return model
base,off,on=make(a),make(b),make(b,True)
for key,value in base.state_dict().items(): suite.exact(value,on.state_dict()[key])
with contextlib.redirect_stdout(sys.stderr): opts=[m.setup_optimizer() for m in (base,off)]
losses=[]
for step in range(3):
    torch.manual_seed(300+step)
    ids=torch.randint(64,(2,8),dtype=torch.int32);target=torch.randint(64,(2,8))
    pair=[]
    for model,opt in zip((base,off),opts):
        opt.zero_grad(set_to_none=True);loss=model(ids,target);loss.backward();pair.append(loss)
    suite.exact(*pair)
    for p,q in zip(base.parameters(),off.parameters()):
        if p.grad is None or q.grad is None: assert p.grad is q.grad
        else: suite.exact(p.grad,q.grad)
    for opt in opts: opt.step()
    for key,value in base.state_dict().items(): suite.exact(value,off.state_dict()[key])
    losses.append(float(pair[0].detach()))
results['off_3step_loss_grad_model']='BIT_IDENTICAL'
results['off_losses']=losses
with torch.no_grad():
    for p in on.ngram_tables.values(): p.fill_(.125)
reference=copy.deepcopy(on)
# Independent autograd gather built from row indexing, not embedding backward.
def gather(index,weight,*args,**kwargs):
    return torch.stack([weight[int(i)] for i in index.reshape(-1)]).reshape(*index.shape,weight.shape[1])
with contextlib.redirect_stdout(sys.stderr): optimizers=[m.setup_optimizer() for m in (on,reference)]
for step in range(3):
    for model,opt in zip((on,reference),optimizers): opt.zero_grad(set_to_none=True)
    loss=on(ids,target);loss.backward()
    with patch.object(F,'embedding',gather): rloss=reference(ids,target);rloss.backward()
    suite.exact(loss,rloss)
    for p,q in zip(on.parameters(),reference.parameters()):
        if p.grad is None or q.grad is None: assert p.grad is q.grad
        else: torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)
    for model,opt in zip((on,reference),optimizers):
        tables=list(model.ngram_tables.values());tids={id(p) for p in tables}
        assert all(p.dtype==torch.float32 and p.grad.dtype==torch.float32 for p in tables)
        b.clip_and_sync_gradients(model,[p for p in model.parameters() if id(p) not in tids],max_norm=1.,clip_table_params=tables)
        opt.step()
        assert all(opt.state[p]['exp_avg_sq'].dtype==torch.float32 for p in tables)
    for key,value in on.state_dict().items(): torch.testing.assert_close(value,reference.state_dict()[key],rtol=1e-5,atol=1e-6)
on.eval();on.requires_grad_(False)
with torch.no_grad():
    output=on(ids);changed=ids.clone();changed[:,4:]=(changed[:,4:]+7)%64
    suite.exact(on(changed)[:,:4],output[:,:4])
    changed=ids.clone();changed[1]=(changed[1]+13)%64
    suite.exact(on(changed)[0],output[0])
with tempfile.TemporaryDirectory(dir='/home/poojith-devan/.cache',prefix='a1-framework-check-') as directory:
    path=Path(directory)/'model.pt';b.save_checkpoint(path,on,{})
    with patch.object(b,'compute_dtype_for',return_value=torch.bfloat16): loaded=b.load_for_eval(str(path),torch.device('cpu'))
    with torch.no_grad(): suite.exact(output,loaded(ids))
results.update(status='PASS_REDUCED_CPU',on_reference='3_STEP_STANDARD_VS_ROW_INDEX_AUTOGRAD',
               fp32_table_master_grad_rms=True,checkpoint='STRICT_ROUNDTRIP',
               custom_embedding_calls=0,causality='SAME_SHAPE_EXACT',
               source_sha256=hashlib.sha256((HERE/'train.py').read_bytes()).hexdigest(),
               test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               torch_version=torch.__version__,scope='Author CPU evidence; native compile/budget/BPB unknown')
(HERE/'CPU_RECEIPT.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
