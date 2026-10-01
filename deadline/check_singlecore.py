#!/usr/bin/env python3
"""Native LNC1 inference smoke tests; random weights are not quality evidence."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    os.environ['NEURON_LOGICAL_NC_CONFIG']='1'
    os.environ.pop('NEURON_CC_FLAGS',None)
    label=sys.argv[1]
    for key in ('NEURON_COMPILE_CACHE_DIR','NEURON_COMPILE_CACHE_URL',
                'TORCH_NEURONX_HLO_CACHE_DIR','TORCH_NEURONX_NEFF_CACHE_DIR',
                'TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR',
                'TORCHINDUCTOR_CACHE_DIR'):
        path=ROOT/'out'/label/'caches'/key
        path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    import torch,prepare
    device=prepare._detect_eval_device('neuron')
    prepare._init_eval_distributed(device)
    manifest=json.loads((ROOT/'deadline/candidate-manifest.json').read_text())
    receipts=[]
    try:
        for variant in ['headbias','dualhead','mlpnki','mlp6nki']:
            started=time.monotonic()
            path=ROOT/'candidates'/variant/'train.py'
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            assert sha==manifest['candidates'][variant]['sha256']
            m=prepare._import_train_module(str(path))
            m.COMPUTE_DTYPE=m.compute_dtype_for(device)
            torch.manual_seed(42)
            model=m.GPT(m.build_config(tbl_freeze_tokens=0));model.init_weights()
            with torch.no_grad():
                for block in model.transformer.h:
                    block.mlp.c_proj.weight.normal_(std=.0001)
                model.value_bigram_table.fill_(.01)
                if hasattr(model,'logit_bias'):
                    model.logit_bias.copy_(torch.linspace(-3,3,model.logit_bias.numel()))
            model.requires_grad_(False);model.to(device);model.eval()
            x=torch.randint(0,8192,(2,1024),dtype=torch.int32).to(device)
            with torch.no_grad():
                a=model(x[:1]);assert torch.isfinite(a).all()
                future=x[:1].clone();future[:,512:]=torch.flip(future[:,512:],dims=[1])
                b=model(future);assert torch.isfinite(b).all()
                causal=float((a[:,:511].float()-b[:,:511].float()).abs().max().item())
                assert causal<=1e-2,(variant,causal)
                original=model(x);perturbed=x.clone();perturbed[1]=torch.flip(perturbed[1],dims=[0])
                other=model(perturbed)
                assert torch.isfinite(original).all() and torch.isfinite(other).all()
                batch=float((original[0].float()-other[0].float()).abs().max().item())
                assert batch<=1e-2,(variant,batch)
            receipt=dict(variant=variant,source_sha256=sha,passed=True,lnc='1',
                declared_nki_lanes=m._NKI_LANES,shapes=[[1,1024],[2,1024]],
                width=1024,vocab=8192,causality_max_abs=causal,batch_max_abs=batch,
                nonzero_mlp_outputs=True,nonzero_table=True,trained_weights=False,
                quality_claim=False,seconds=time.monotonic()-started)
            receipts.append(receipt);print('SINGLECORE_CASE '+json.dumps(receipt),flush=True)
            del model,a,b,original,other
        result=dict(label=label,passed=True,receipts=receipts,official_portal_run=False)
        (ROOT/'research/run-logs'/(label+'.singlecore.json')).write_text(json.dumps(result,indent=2)+'\n')
        print('SINGLECORE_CHECK '+json.dumps(result),flush=True)
    finally:prepare._cleanup_eval_distributed()


if __name__=='__main__':main()
