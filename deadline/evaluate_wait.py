#!/usr/bin/env python3
"""Warm inference compilation on an idle host; score only the verified trained checkpoint.

Random-weight warmup results are explicitly not quality evidence. The trusted
prepare.py and the submitted/training model are unchanged.
"""
import argparse
import dataclasses
import gc
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
        cfg=m.build_config(tbl_freeze_tokens=0)
        cfg.sequence_len=max(cfg.sequence_len,m.EVAL_MIN_SEQ_LEN)
        warm_config=dataclasses.asdict(cfg)
        m.COMPUTE_DTYPE=m.compute_dtype_for(device)
        torch.manual_seed(42)
        warm=m.GPT(cfg);warm.init_weights();warm.to(device);warm.eval();warm.requires_grad_(False)
        warm_small=prepare.evaluate_bpb(warm,tok,1,1024,device,max_tokens=1024,timeout_seconds=400)
        warm_causal=prepare.causality_check(warm,tok,1,1024,device)
        assert warm_causal['passed'],warm_causal
        warm_batch=prepare.evaluate_bpb(warm,tok,32,1024,device,max_tokens=32768,timeout_seconds=400)
        wx,_,_=next(prepare.make_eval_dataloader(tok,32,1024,'public_val',device))
        with torch.no_grad():
            wa=warm(wx);wc=wx.clone();wc[31]=torch.flip(wc[31],dims=[0]);wb=warm(wc)
            assert torch.isfinite(wa).all() and torch.isfinite(wb).all()
            warm_batch_error=float((wa[0].float()-wb[0].float()).abs().max().item())
            assert warm_batch_error <= 1e-2,warm_batch_error
        warm_receipt=dict(random_weights=True,quality_claim=False,model_config=warm_config,
                          batch1_seconds=warm_small['eval_seconds'],batch32_seconds=warm_batch['eval_seconds'],
                          causality=warm_causal,batch_independence_max_abs=warm_batch_error)
        print('DEADLINE_EVAL_READY '+json.dumps(warm_receipt),flush=True)
        del warm,wa,wb,wx,wc;gc.collect()
        ready=args.checkpoint.with_suffix('.ready.json')
        wait_started=time.monotonic()
        while not ready.exists():
            assert time.monotonic()-wait_started < 1100,'Checkpoint did not arrive before bounded wait limit'
            time.sleep(2)
        incoming=json.loads(ready.read_text())
        cp_sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        assert cp_sha==incoming['checkpoint_sha256']
        assert args.checkpoint.stat().st_size==incoming['checkpoint_bytes']
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        actual_config=dict(payload['model_config'])
        actual_config['sequence_len']=max(actual_config['sequence_len'],m.EVAL_MIN_SEQ_LEN)
        assert actual_config==warm_config,(actual_config,warm_config)
        del payload;gc.collect()
        score_started=time.monotonic()
        model=m.load_for_eval(str(args.checkpoint),device)
        small=prepare.evaluate_bpb(model,tok,1,1024,device,max_tokens=262144,timeout_seconds=450)
        assert not small['eval_over_timeout'] and 0 < small['val_bpb'] < 100
        print('DEADLINE_PREFIX_B1 '+json.dumps(small),flush=True)
        causal=prepare.causality_check(model,tok,1,1024,device);assert causal['passed'],causal
        print('DEADLINE_CAUSAL '+json.dumps(causal),flush=True)
        batch=prepare.evaluate_bpb(model,tok,32,1024,device,max_tokens=262144,timeout_seconds=500)
        difference=abs(small['val_bpb']-batch['val_bpb'])
        assert small['num_tokens']==batch['num_tokens']==262144
        assert small['num_bytes']==batch['num_bytes'] and difference <= 5e-5,(small,batch)
        print('DEADLINE_BATCH_EQUIVALENCE '+json.dumps(dict(prefix_b1=small,prefix_b32=batch,bpb_abs_diff=difference)),flush=True)
        full=prepare.evaluate_bpb(model,tok,32,1024,device,max_tokens=20971520,timeout_seconds=600)
        assert full['num_tokens']==20971520 and not full['eval_over_timeout']
        assert 0 < full['val_bpb'] < 100
        print('DEADLINE_SPARE_SCORE '+json.dumps(full),flush=True)
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
            elapsed_seconds=time.monotonic()-started,scoring_seconds=time.monotonic()-score_started,
            warmup=warm_receipt,incoming=incoming,prefix_tokens=262144,official_portal_score=False)
        (ROOT/'research/run-logs'/(args.label+'.spare-eval.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        print('DEADLINE_SPARE_EVAL '+json.dumps(receipt),flush=True)
    finally:prepare._cleanup_eval_distributed()


if __name__=='__main__':main()
