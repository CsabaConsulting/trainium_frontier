#!/usr/bin/env python3
"""Score a saved candidate on the unchanged historical 20M public protocol."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--label', required=True)
    args = ap.parse_args()
    source = ROOT / 'candidates' / args.variant / 'train.py'
    manifest = json.loads((ROOT / 'deadline/candidate-manifest.json').read_text())
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha == manifest['candidates'][args.variant]['sha256']
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    checkpoint_bytes = args.checkpoint.stat().st_size
    # Historical fresh public20M receipts use one process, LNC2, import-owned
    # O3 flags with no caller compiler override. Scoring is not training.
    os.environ['NEURON_LOGICAL_NC_CONFIG'] = '2'
    os.environ.pop('NEURON_CC_FLAGS', None)
    import torch
    import prepare
    started = time.monotonic()
    module = prepare._import_train_module(str(source))
    device = prepare._detect_eval_device('neuron')
    prepare._init_eval_distributed(device)
    try:
        tokenizer = prepare.ensure_tokenizer(build_if_missing=False)
        model = module.load_for_eval(str(args.checkpoint), device)
        result = prepare.evaluate_bpb(model, tokenizer, batch_size=1, seq_len=1024,
                                      device=device, split='public_val',
                                      max_tokens=20971520, timeout_seconds=1200)
        assert result['num_tokens'] == 20971520, result
        assert not result['eval_over_timeout'], result
        assert 0 < result['val_bpb'] < 100, result
        # Unchanged official causality test. All validation access happens after
        # training has ended and cannot influence its weights or schedules.
        causal = prepare.causality_check(model, tokenizer, batch_size=1,
                                         seq_len=1024, device=device)
        assert causal['passed'], causal
        loader = prepare.make_eval_dataloader(tokenizer, 2, 1024, 'public_val', device)
        x, _, _ = next(loader)
        with torch.no_grad():
            original = model(x)
            changed = x.clone()
            changed[1] = torch.flip(changed[1], dims=[0])
            perturbation = model(changed)
            batch_error = float((original[0].float()-perturbation[0].float()).abs().max().item())
            assert torch.isfinite(original).all() and torch.isfinite(perturbation).all()
            assert batch_error <= 1e-2, batch_error
        assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == checkpoint_sha
        receipt = dict(label=args.label, variant=args.variant, source_sha256=source_sha,
                       checkpoint_sha256=checkpoint_sha, checkpoint_bytes=checkpoint_bytes,
                       evaluation=result, causality=causal, batch_independence_max_abs=batch_error,
                       lnc='2', compiler_flags=os.environ.get('NEURON_CC_FLAGS'),
                       elapsed_seconds=time.monotonic()-started, matched_historical_public20m=True,
                       official_portal_score=False)
        out = ROOT / 'research/run-logs' / (args.label+'.matched-eval.json')
        out.write_text(json.dumps(receipt, indent=2)+'\n')
        print('DEADLINE_MATCHED_EVAL '+json.dumps(receipt), flush=True)
    finally:
        prepare._cleanup_eval_distributed()


if __name__ == '__main__':
    main()
