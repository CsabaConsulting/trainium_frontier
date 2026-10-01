#!/usr/bin/env python3
"""Accelerate unchanged public evaluation; prove batch-size numerical agreement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant',required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--label',required=True);args=ap.parse_args()
    source=ROOT/'candidates'/args.variant/'train.py'
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest=json.loads((ROOT/'deadline/candidate-manifest.json').read_text())
    assert source_sha==manifest['candidates'][args.variant]['sha256']
    cp_sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    os.environ['NEURON_LOGICAL_NC_CONFIG']='2';os.environ.pop('NEURON_CC_FLAGS',None)
    cache=ROOT/'out'/args.label/'fast-eval-caches'
    for key,sub in [('NEURON_COMPILE_CACHE_DIR','compiler'),('NEURON_COMPILE_CACHE_URL','compiler'),
        ('TORCH_NEURONX_HLO_CACHE_DIR','hlo'),('TORCH_NEURONX_NEFF_CACHE_DIR','neff'),
        ('TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','local'),('TORCH_NEURONX_DEBUG_DIR','debug'),
        ('TORCHINDUCTOR_CACHE_DIR','inductor')]:
        p=cache/sub;p.mkdir(parents=True,exist_ok=sub=='compiler');os.environ[key]=str(p)
    import torch,prepare
    started=time.monotonic();m=prepare._import_train_module(str(source))
    device=prepare._detect_eval_device('neuron');prepare._init_eval_distributed(device)
    try:
        tok=prepare.ensure_tokenizer(build_if_missing=False)
        model=m.load_for_eval(str(args.checkpoint),device)
        small=prepare.evaluate_bpb(model,tok,1,1024,device,max_tokens=2097152,timeout_seconds=450)
        assert not small['eval_over_timeout'] and 0 < small['val_bpb'] < 100
        print('DEADLINE_PREFIX_B1 '+json.dumps(small),flush=True)
        causal=prepare.causality_check(model,tok,1,1024,device);assert causal['passed'],causal
        print('DEADLINE_CAUSAL '+json.dumps(causal),flush=True)
        batch=prepare.evaluate_bpb(model,tok,32,1024,device,max_tokens=2097152,timeout_seconds=500)
        difference=abs(small['val_bpb']-batch['val_bpb'])
        assert small['num_tokens']==batch['num_tokens']==2097152
        assert small['num_bytes']==batch['num_bytes'] and difference <= 5e-5,(small,batch)
        print('DEADLINE_BATCH_EQUIVALENCE '+json.dumps(dict(prefix_b1=small,prefix_b32=batch,bpb_abs_diff=difference)),flush=True)
        full=prepare.evaluate_bpb(model,tok,32,1024,device,max_tokens=20971520,timeout_seconds=600)
        assert full['num_tokens']==20971520 and not full['eval_over_timeout']
        assert 0 < full['val_bpb'] < 100
        print('DEADLINE_FAST_SCORE '+json.dumps(full),flush=True)
        x,_,_=next(prepare.make_eval_dataloader(tok,32,1024,'public_val',device))
        with torch.no_grad():
            a=model(x);changed=x.clone();changed[31]=torch.flip(changed[31],dims=[0]);b=model(changed)
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            err=float((a[0].float()-b[0].float()).abs().max().item());assert err <= 1e-2,err
        assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==cp_sha
        receipt=dict(label=args.label,variant=args.variant,source_sha256=source_sha,
            checkpoint_sha256=cp_sha,checkpoint_bytes=args.checkpoint.stat().st_size,
            evaluation=full,prefix_b1=small,prefix_b32=batch,prefix_bpb_abs_diff=difference,
            causality=causal,batch_independence_max_abs=err,eval_batch_size=32,
            lnc='2',compiler_flags=os.environ.get('NEURON_CC_FLAGS'),
            elapsed_seconds=time.monotonic()-started,official_portal_score=False)
        (ROOT/'research/run-logs'/(args.label+'.fast-eval.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        print('DEADLINE_FAST_EVAL '+json.dumps(receipt),flush=True)
    finally:prepare._cleanup_eval_distributed()


if __name__=='__main__':main()
