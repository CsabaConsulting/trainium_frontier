import ast, argparse, contextlib, difflib, hashlib, io, json, pathlib, sys, types
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path('/home/poojith-devan/clock36-batch-fresh-review-QFHzvD')
checks = []
def check(ok, name):
    if not ok: raise RuntimeError(name)
    checks.append(name)
def execute(nodes, env):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-source-ast>', 'exec'), env)
def assign(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
def load(name):
    text = (ROOT / name).read_text()
    tree = ast.parse(text)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    return text, tree, main
parent, pt, pm = load('parent.py')
source, tree, main = load('train.py')
delta = list(difflib.ndiff(parent.splitlines(True), source.splitlines(True)))
check(sum(x.startswith('+ ') for x in delta) == 5 and not any(x.startswith('- ') for x in delta), 'exactly five additions and no deletions')
check(''.join(x[2:] for x in delta if x[:2] in ('  ', '- ')) == parent, 'deleting additions reconstructs exact parent')
check(hashlib.sha256(source.encode()).hexdigest() == 'eeb3c403b86fbc8344864c96e746d9b4f02f8012e68cf7f33812d7549eea5f9c', 'candidate SHA')
check(hashlib.sha256(parent.encode()).hexdigest() == '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb', 'parent SHA')
parser_nodes = [n for n in main.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and any(isinstance(a, ast.Constant) and a.value == '--total-batch-size' for a in n.value.args)]
positive = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'args.total_batch_size <= 0')
batch_set = next(n for n in main.body if assign(n, 'TOTAL_BATCH_SIZE'))
i = next(i for i,n in enumerate(main.body) if assign(n, 'world_tokens_per_micro'))
arithmetic = main.body[i:i+3]
def parse(argv):
    env = dict(parser=argparse.ArgumentParser(), TOTAL_BATCH_SIZE=131072, DEVICE_BATCH_SIZE=1, PACK_FACTOR=32, SEQ_LEN=1024, world_size=4)
    execute(parser_nodes, env)
    env['args'] = env['parser'].parse_args(argv)
    execute([positive, batch_set] + arithmetic, env)
    return env['TOTAL_BATCH_SIZE'], env['grad_accum_steps'], env['world_tokens_per_micro']
for argv,total,gas in [([],131072,1),(['--total-batch-size','131072'],131072,1),(['--total-batch-size','262144'],262144,2)]:
    check(parse(argv) == (total,gas,131072), 'parse/arithmetic '+str(argv))
for value in ['0','-1','1.5','abc','196608']:
    try:
        with contextlib.redirect_stderr(io.StringIO()): parse(['--total-batch-size',value])
    except (SystemExit, ValueError) as e:
        check(True, 'rejected '+value+' '+type(e).__name__)
    else: check(False,'accepted invalid '+value)

def run(t, m, gas, label):
    env = dict(torch=torch, nn=nn, F=F, USE_WEIGHT_CACHE=True)
    names = ['Linear','sync_gradients','clip_and_sync_gradients','unscale_grads_']
    execute([n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names], env)
    loop = next(n for n in m.body if isinstance(n,ast.While))
    start = next(i for i,n in enumerate(loop.body) if isinstance(n,ast.If) and ast.unparse(n.test) == 'grad_flat_buffer is not None')
    end = next(i for i,n in enumerate(loop.body) if isinstance(n,ast.For) and ast.unparse(n.target) == 'micro_index')
    micro_code = loop.body[start:end+1]
    # These exact calls and conditional update are outside the micro loop.
    post = [n for n in loop.body if (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id in ['unscale_grads_','clip_and_sync_gradients']) or (isinstance(n,ast.If) and ast.unparse(n.test) == 'not skip_step') or (isinstance(n,ast.For) and ast.unparse(n.target) == 'm')]
    check(len(post)==4,label+' one unscale, clip/sync, conditional update and refresh AST')
    torch.manual_seed(42)
    linear = env['Linear'](3,2,bias=False)
    class Model(nn.Module):
        def __init__(self): super().__init__(); self.linear=linear
        def forward(self,x,y): return (self.linear(x).float()-y).square().mean()
    model = Model()
    batches = [(torch.randn(2,3).bfloat16(),torch.randn(2,2)) for _ in range(8)]
    event=[]; snapshots=[]; refs=[]; refreshes=[]
    native_refresh=linear.refresh_weight_cache
    def refresh(dtype):
        native_refresh(dtype)
        check(linear._cached_weight.grad_fn is not None,label+' differentiable cache')
        refreshes.append(1)
    linear.refresh_weight_cache=refresh
    class Optim:
        def zero_grad(self,set_to_none): event.append('zero'); model.zero_grad(set_to_none=set_to_none)
        def step(self,update_adamw):
            check(update_adamw,label+' Adam every step')
            event.append('update')
            with torch.no_grad(): linear.weight.add_(linear.weight.grad,alpha=-0.015)
    def reduce(g,op):
        check(op=='AVG',label+' AVG reduction')
        event.append('reduce')
        # Four identical simulated ranks. Real collectives are not executed.
    env['dist']=types.SimpleNamespace(is_available=lambda:True,is_initialized=lambda:True,ReduceOp=types.SimpleNamespace(AVG='AVG'),all_reduce=reduce)
    clip_original=torch.nn.utils.clip_grad_norm_
    def clip(*a,**kw): event.append('clip'); return clip_original(*a,**kw)
    cursor=1
    def fetch():
        nonlocal cursor
        x,y=batches[cursor]; cursor+=1
        return x,y,{'epoch':0}
    def debug(stage,*args,**kw):
        if stage.endswith('_backward'):
            snapshots.append(linear.weight.grad.detach().clone())
    env.update(model=model,optimizer=Optim(),grad_flat_buffer=None,zero_adamw_now=True,adam_step_now=True,args=types.SimpleNamespace(initial_cache_refresh=True,clip_foreach=False),cached_linears=[linear],COMPUTE_DTYPE=torch.bfloat16,grad_accum_now=gas,grad_accum_steps=gas,BATCH_RAMP=False,LOSS_SCALE=1.,fetch=fetch,debug_numerics=debug,x=batches[0][0],y=batches[0][1],unscale_with=1.,clip_main_params=[linear.weight],clip_ngram_params=[],GRAD_CLIP=1.,CLIP_AFTER_REDUCE=True,skip_step=False)
    weights=[]
    torch.nn.utils.clip_grad_norm_=clip
    try:
        for step in range(3):
            before=linear.weight.detach().clone(); batch_offset=step*gas
            expected=torch.zeros_like(before)
            for j in range(gas):
                w=before.clone().requires_grad_(); x,y=batches[batch_offset+j]
                # Independent per-micro reference, separate fresh cast graph.
                loss=(F.linear(x,w.to(torch.bfloat16)).float()-y).square().mean()
                grad,=torch.autograd.grad(loss/gas,w)
                expected+=grad; refs.append(grad.tolist())
            env['step']=step
            execute(micro_code,env)
            check(torch.equal(linear.weight.grad,expected),label+f' step{step} exact mean gradients')
            check(all(torch.count_nonzero(s).item()>0 for s in snapshots),label+f' step{step} nonzero gradients')
            if gas==2:
                check(not torch.equal(snapshots[-1],snapshots[-2]),label+f' step{step} second distinct micro contributes')
            execute(post,env)
            check(event[step*4:]==['zero','reduce','clip','update'],label+f' step{step} once-only order')
            weights.append(linear.weight.detach().clone())
    finally: torch.nn.utils.clip_grad_norm_=clip_original
    check(len(refreshes)==4,label+' initial plus three post-update refreshes')
    return {'weights':[w.tolist() for w in weights], 'gradients':[s.tolist() for s in snapshots], 'reference_contributions':refs,'events':event,'logical_tokens_per_update':gas*131072,'actual_synthetic_rows_per_update':gas*2}

out={'scope':'Independent frozen-source AST CPU tiny Linear/production micro loop; simulated identical rank AVG only; no full model/native/actual gloo execution','torch':torch.__version__,'optimized':not __debug__}
try:
    out['parent']=run(pt,pm,1,'parent')
    out['default']=run(tree,main,parse([])[1],'default')
    out['explicit131072']=run(tree,main,parse(['--total-batch-size','131072'])[1],'explicit131072')
    check(out['parent']==out['default']==out['explicit131072'],'three-step default/explicit bit preservation')
    out['arm262144']=run(tree,main,2,'arm262144')
    out['verdict']='PASS_SCOPED_CPU_REFERENCE'
except Exception as e:
    out['verdict']='FAIL'; out['failure']=repr(e)
finally:
    out['checks']=checks
    print(json.dumps(out,indent=2))
if out['verdict']=='FAIL': sys.exit(1)
