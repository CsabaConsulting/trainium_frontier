"""Actual Linear cache plus actual micro loop; no retain/detach/refresh workaround."""
import ast
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_batch import SOURCE, code, source_parts

torch.set_num_threads(1)
tree,main,loop=source_parts()
ns={'torch':torch,'nn':nn,'F':F,'USE_WEIGHT_CACHE':True}
exec(code([next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Linear')]),ns)
Linear=ns['Linear']
micro=next(n for n in loop.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='micro_index')
torch.manual_seed(42)
initial=torch.randn(2,3)
batches=[(torch.randn(2,3).bfloat16(),torch.randn(2,2)),(torch.randn(2,3).bfloat16(),torch.randn(2,2))]
def model():
    m=Linear(3,2,bias=False)
    with torch.no_grad():m.weight.copy_(initial)
    m.refresh_weight_cache(torch.bfloat16)
    return m
refs=[]
for x,y in batches:
    m=model();((m(x).float()-y)**2).mean().backward();refs.append(m.weight.grad.clone())
if torch.equal(refs[0],refs[1]) or any(not bool(g.abs().sum()) for g in refs):raise RuntimeError('degenerate micro fixtures')
m=model();events=[];index=0
def fetch():
    global index
    index+=1
    x,y=batches[index%2]
    return x,y,{'epoch':1}
def debug(stage,*args,**kwargs):
    events.append({'stage':stage,'gradient':None if m.weight.grad is None else m.weight.grad.tolist()})
state=dict(grad_accum_now=2,grad_accum_steps=2,BATCH_RAMP=False,LOSS_SCALE=1.,x=batches[0][0],y=batches[0][1],step=0,fetch=fetch,model=lambda x,y:((m(x).float()-y)**2).mean(),debug_numerics=debug)
failure=None
try:exec(code([micro]),state)
except RuntimeError as e:failure=str(e)
if failure is None:
    torch.testing.assert_close(m.weight.grad,(refs[0]/2+refs[1]/2),rtol=0,atol=0)
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
    torch.optim.SGD(m.parameters(),lr=.01).step()
print(json.dumps({'scope':'actual tiny Linear cache and actual loop; no full model/native claim','distinct_nonzero_reference_gradients':[g.tolist() for g in refs],'events':events,'failure':failure,'verdict':'FAIL_CACHE_BACKWARD' if failure else 'PASS','updates_completed':0 if failure else 1},indent=2))
if failure:raise SystemExit(1)
