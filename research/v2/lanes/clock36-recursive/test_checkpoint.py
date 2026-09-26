"""Sequential tiny CPU loader/toggle/purity probe, no organizer data or scoring."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SUPPORT=Path('/home/poojith-devan/Desktop/Trainium')
sys.path[:0]=[str(SUPPORT/'workflow/tools/stub_prepare'),str(SUPPORT/'research/v2/lanes/architecture-recursive-stack/test_support')]
import torch
torch.set_num_threads(1)
from test_static import BASE, SOURCE_SHA


def require(ok,message):
    if not ok: raise RuntimeError(message)


def digest(t):
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    # Test-size only, not production loader/source edits. Avoid 8192-square masks.
    module.EVAL_MIN_SEQ_LEN=16
    module.USE_UNET_SKIP=module.USE_PARTIAL_ATTN_8=True
    module.ATTENTION_LAYERS='0,2,4,6,7'
    module.USE_NGRAM_VE=False
    module.COMPUTE_DTYPE=torch.float32
    return module


def main():
    tmp=Path(tempfile.mkdtemp(prefix='clock-recursive-checkpoint-'))
    base=tmp/'base.py'; base.write_text(BASE)
    child=ROOT/'workflow/proposals/research-v2/train.py'
    require(hashlib.sha256(child.read_bytes()).hexdigest()==SOURCE_SHA,'source pin')
    old=load_module(base,'clock_ref_loader')
    new=load_module(child,'clock_child_loader')
    torch.manual_seed(42)
    config=old.GPTConfig(sequence_len=16,vocab_size=64,n_layer=8,n_head=2,n_embd=32)
    model=old.GPT(config)
    model.init_weights()
    x=(torch.arange(32).reshape(2,16)%60+1).long()
    target=(x+1)%64
    loss=model(x,target)
    loss.backward()
    # One explicit tiny synthetic SGD update solely creates non-initial weights.
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None: p.add_(p.grad,alpha=-.001)
    path=tmp/'trained.pt'
    old.save_checkpoint(path,model,{'diagnostic_sgd_steps':1})
    payload=torch.load(path,weights_only=True)
    records=[]
    expected=None
    for label,module,toggle in [('base',old,None),('child-old-missing',new,None),('child-off',new,False),('child-on',new,True)]:
        if toggle is not None: payload['toggles']['RECURSIVE_SCHEDULE']=toggle
        else: payload['toggles'].pop('RECURSIVE_SCHEDULE',None)
        torch.save(payload,path)
        ev=module.load_for_eval(str(path),torch.device('cpu'))
        before={k:digest(v) for k,v in ev.state_dict().items()}
        with torch.no_grad():
            output=ev(x); repeat=ev(x)
            rows=torch.cat([ev(x[i:i+1]) for i in range(2)])
            changed=x.clone(); changed[:,8:]=(changed[:,8:]+7)%64
            suffix=ev(changed)
            permuted=ev(x.flip(0)).flip(0)
        require(torch.equal(output,repeat),'repeat '+label)
        require(torch.equal(output,rows),'independent rows '+label)
        require(torch.equal(output,permuted),'permutation '+label)
        require(torch.equal(output[:,:8],suffix[:,:8]),'synthetic prefix causal '+label)
        require(before=={k:digest(v) for k,v in ev.state_dict().items()},'state invariant '+label)
        if expected is None: expected=output
        require(torch.equal(output,expected),'base/new loader exact output '+label)
        if module is new: require(module.RECURSIVE_SCHEDULE is (False if toggle is None else toggle),'restore toggle '+label)
        records.append(dict(label=label,output_sha256=digest(output),state_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),parameter_count=sum(p.numel() for p in ev.parameters()),saved_recursive=payload['toggles'].get('RECURSIVE_SCHEDULE'),restored_recursive=getattr(module,'RECURSIVE_SCHEDULE',None)))
    print(json.dumps(dict(status='IMPLEMENTER_TINY_CPU_ONLY',torch=torch.__version__,source_sha256=SOURCE_SHA,shape=[2,16],config=dict(n_layer=8,n_embd=32,vocab=64),test_only_eval_min_seq_len=16,synthetic_sgd_loss=float(loss.detach()),records=records,organizer_causality_or_native_claim=False),indent=2))


if __name__=='__main__': main()
