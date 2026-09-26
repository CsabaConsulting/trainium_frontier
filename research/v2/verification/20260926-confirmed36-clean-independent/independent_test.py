from pathlib import Path
import ast,math,json,copy,hashlib
from dataclasses import dataclass,asdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
import weakref
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
root=Path(__file__).parent
class MemoryLimit(TorchDispatchMode):
    def __init__(self): self.refs={};self.peak=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        out=func(*args,**(kwargs or {}))
        def track(x):
            if isinstance(x,torch.Tensor):
                s=x.untyped_storage();self.refs[s._cdata]=weakref.ref(s)
            elif isinstance(x,(list,tuple)): 
                for y in x:track(y)
        track(out)
        self.refs={k:v for k,v in self.refs.items() if v() is not None}
        size=sum(v().nbytes() for v in self.refs.values() if v() is not None)
        self.peak=max(self.peak,size)
        assert size<=128*1024**2, size
        return out
names={'CausalConv','GPTConfig','norm','Linear','apply_rotary_emb','rope_norm','CausalSelfAttention','reference_forward','_fused_mlp_kernel','_linear_weight','_FusedMLP','MLP','Block','GPT','softcap_cross_entropy','startup_adjusted_schedule_elapsed','lr_multiplier','save_checkpoint','load_for_eval','compute_dtype_for'}
tree=ast.parse((root/'candidate.py').read_text())
nodes=[]
for n in tree.body:
    if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names: nodes.append(n)
    elif isinstance(n,(ast.Assign,ast.AnnAssign)):
        try: ast.literal_eval(n.value);nodes.append(n)
        except (ValueError,TypeError): pass
ns=dict(torch=torch,nn=nn,F=F,math=math,dataclass=dataclass,asdict=asdict,Path=Path,
        _HAS_NKI=False,COMPUTE_DTYPE=torch.bfloat16,TRAIN_STARTUP_STEPS_EXCLUDED=5,
        TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=900,_PROCESS_STARTED=100.)
exec(compile(ast.Module(nodes,type_ignores=[]),'candidate-extracted','exec'),ns)
ns['COMPUTE_DTYPE']=torch.bfloat16
ns['EVAL_MIN_SEQ_LEN']=256
def independent(m,idx):
    # Immediate casts are deliberately independent of candidate's cache.
    p=dict(m.named_parameters())
    def lin(x,name): return F.linear(x,p[name+'.weight'].to(x.dtype))
    def rms(x):return F.rms_norm(x.float(),(x.shape[-1],),eps=1.1920928955078125e-7).to(torch.bfloat16)
    x=rms(F.embedding(idx,p['transformer.wte.weight']).to(torch.bfloat16));x0=x;enc=[]
    for i in range(8):
        prefix=f'transformer.h.{i}'
        if i>=4:x=x+p[f'unet_skip_gates.{i-4}'].to(x.dtype)*enc[7-i]
        x=p[prefix+'.resid_lambda']*x+p[prefix+'.x0_lambda']*x0
        if i in (0,2,4,6,7):
            incoming=x
            z=rms(incoming)
            x=x+F.conv1d(F.pad(z.transpose(1,2),(1,0)),p[prefix+'.local_conv.weight'].to(z.dtype),groups=128).transpose(1,2)
            b,t,_=z.shape
            qz,kz,vz=rms(incoming),rms(incoming),rms(incoming)
            q,k,v=[lin(zc,prefix+'.attn.c_'+c).view(b,t,1,128) for zc,c in zip((qz,kz,vz),('q','k','v'))]
            if i==7:
                gate=2*torch.sigmoid(lin(vz[...,:32],prefix+'.attn.ve_gate'))
                v=v+gate.unsqueeze(-1)*F.embedding(idx,p['value_embeds.7.weight']).to(torch.bfloat16).view(b,t,1,128)
            def rotate(a):
                c,s=m.cos[:,:t],m.sin[:,:t]
                return rms(torch.cat((a[...,:64]*c+a[...,64:]*s,-a[...,:64]*s+a[...,64:]*c),-1)).transpose(1,2)
            att=F.scaled_dot_product_attention(rotate(q),rotate(k),v.to(q.dtype).transpose(1,2),is_causal=True)
            x=x+lin(att.transpose(1,2).contiguous().view(b,t,128),prefix+'.attn.c_proj')
        x=x+lin(F.relu(lin(rms(x),prefix+'.mlp.c_fc')).square(),prefix+'.mlp.c_proj')
        if i<4:enc.append(x)
    logits=lin(rms(x),'lm_head').float()
    return 15*torch.tanh(logits/15)
def eq(a,b,label):
    assert torch.equal(a,b),(label,(a-b).abs().max().item())
def cache(m):
    for x in m.modules():
        if isinstance(x,ns['Linear']):x.refresh_weight_cache(torch.bfloat16)
with MemoryLimit() as mem:
    cfg=ns['GPTConfig'](sequence_len=256,vocab_size=256,n_layer=8,n_head=1,n_embd=128)
    torch.manual_seed(42);a=ns['GPT'](cfg);a.init_weights()
    b=copy.deepcopy(a)
    losses=[]
    for step in range(3):
        cache(a)
        idx=torch.randint(256,(2,16));target=torch.randint(256,(2,16))
        la=a(idx,target);lb=F.cross_entropy(independent(b,idx).reshape(-1,256),target.flatten())
        eq(la,lb,'loss');la.backward();lb.backward()
        for (name,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            assert name==nb
            eq(pa.grad,pb.grad,'grad '+name)
            with torch.no_grad():pa.add_(pa.grad,alpha=-.01);pb.add_(pb.grad,alpha=-.01)
            eq(pa,pb,'update '+name)
        losses.append(la.item());a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
    cache(a);a.eval();b.eval()
    with torch.no_grad():
        for batch in [1,2,3]:
            idx=torch.randint(256,(batch,256))
            actual=a(idx);ref=independent(b,idx)
            eq(actual,ref,'B'+str(batch)+' ref')
            eq(actual,torch.cat([a(row[None]) for row in idx]),'B'+str(batch)+' independent rows')
            altered=idx.clone();altered[:,128:]=(altered[:,128:]+1)%256
            eq(actual[:,:128],a(altered)[:,:128],'B'+str(batch)+' causality')
    ns['save_checkpoint'](root/'tiny-checkpoint.pt',a,{'independent_test':True})
    del b
    loaded=ns['load_for_eval'](str(root/'tiny-checkpoint.pt'),torch.device('cpu'))
    # Loader's CPU policy is float32; compare original under same policy and clear BF16 caches.
    for x in a.modules():
        if isinstance(x,ns['Linear']):x._cached_weight=None
    # Loader re-creates rotary tables in its selected CPU compute dtype.
    a.cos,a.sin=a._precompute_rotary_embeddings(256,128)
    for k,v in a.state_dict().items():eq(v,loaded.state_dict()[k],'checkpoint state '+k)
    with torch.no_grad():eq(a(idx),loaded(idx),'ordinary checkpoint loaded')
    print(json.dumps({'losses':losses,'peak_live_tensor_storage_bytes':mem.peak,'model_parameters':sum(p.numel() for p in a.parameters()),'batches':[1,2,3],'context':256,'result':'PASS'}))
clock=ns['startup_adjusted_schedule_elapsed'];count=0
for step in range(8):
    for start in [100.,200.,970.,1000.,1100.]:
        for elapsed in [0.,1.,30.,100.,800.,900.,1800.]:
            expected=elapsed if step>=5 else max(0.,elapsed+start-100.+30.-900.)
            assert clock(step,elapsed,start)==expected
            count+=1
print(json.dumps({'clock_cases':count,'result':'PASS'}))
