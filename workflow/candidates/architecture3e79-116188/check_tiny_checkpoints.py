"""Reduced CPU cross-check of retained G2 checkpoints and the eligible MLP path.

Run with the packet's import-only PyArrow and existing stub-prepare on PYTHONPATH.
Never use these test shims for a native run. No actual training data is read.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_source(name, receipt):
    path = Path(receipt['train_py'])
    require(hashlib.sha256(path.read_bytes()).hexdigest() == receipt['train_py_sha256'], 'source hash')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod.compute_dtype_for = lambda device: torch.bfloat16
    mod.COMPUTE_DTYPE = torch.bfloat16
    return mod


def run():
    reports = [json.loads((HERE / name).read_text()) for name in ('tiny-g2-research.json', 'tiny-g2-clean.json')]
    modules = [load_source(name, r) for name, r in zip(('research_checkpoint_probe', 'clean_checkpoint_probe'), reports)]
    paths = [Path(r['appended_argv'][-1]) / 'final.pt' for r in reports]
    payloads = [torch.load(p, weights_only=True, map_location='cpu') for p in paths]
    require(payloads[0]['model_config'] == payloads[1]['model_config'], 'config mismatch')
    states = [p['model_state'] for p in payloads]
    require(states[0].keys() == states[1].keys(), 'state keys mismatch')
    require(all(torch.equal(states[0][k], states[1][k]) for k in states[0]), 'state values mismatch')
    evals = [m.load_for_eval(str(p), torch.device('cpu')) for m, p in zip(modules, paths)]
    require(type(evals[1]) is modules[1].GPT, 'clean loader must return ordinary GPT')
    rows = []
    with torch.no_grad():
        for length in (16, 64, 256):
            x = torch.randint(1, 8192, (2, length), generator=torch.Generator().manual_seed(17 + length))
            single = [m(x[:1]) for m in evals]
            require(torch.equal(*single), f'B1 logits differ at {length}')
            research_batch, clean_batch = (m(x) for m in evals)
            batch_delta = float((research_batch - clean_batch).abs().max())
            mutated = x[:1].clone()
            mutated[:, length // 2:] = 11
            changed = evals[1](mutated)
            causal_delta = float((single[1][:, :length // 2] - changed[:, :length // 2]).abs().max())
            require(causal_delta <= 1e-2, 'official causality tolerance exceeded')
            require(torch.isfinite(clean_batch).all(), 'nonfinite logits')
            rows.append(dict(length=length, b1_byte_exact=True, b2_vs_independent_rows_max_abs=batch_delta, causal_prefix_max_abs=causal_delta, official_causality_tolerance=1e-2))
    del evals
    # 256 tokens crosses the actual custom-MLP selection boundary; G2's 1x16 does not.
    trained = []
    for mod, payload in zip(modules, payloads):
        model = mod.GPT(mod.GPTConfig(**payload['model_config']))
        model.load_state_dict(payload['model_state'], strict=True)
        if hasattr(mod, '_submission_mlp_install'):
            mod._submission_mlp_install(model, 'cpu', mod, forward_backend='native')
        model.train()
        for layer in model.modules():
            if isinstance(layer, mod.Linear):
                layer.refresh_weight_cache(torch.bfloat16)
        trained.append(model)
    x = torch.randint(1, 8192, (1, 256), generator=torch.Generator().manual_seed(88))
    y = torch.randint(1, 8192, (1, 256), generator=torch.Generator().manual_seed(89))
    losses = [m(x, y) for m in trained]
    require(torch.equal(*losses), 'eligible custom-MLP loss mismatch')
    for loss in losses:
        loss.backward()
    gradients = [dict(m.named_parameters()) for m in trained]
    missing = []
    for name in gradients[0]:
        a, b = (d[name].grad for d in gradients)
        require((a is None) == (b is None), f'missing gradient mismatch {name}')
        if a is None:
            missing.append(name)
        else:
            require(torch.equal(a, b), f'gradient mismatch {name}')
            require(torch.isfinite(a).all(), f'nonfinite gradient {name}')
    return dict(status='PASS_REDUCED_CPU_ONLY', source_hashes=[r['train_py_sha256'] for r in reports], torch=torch.__version__, checkpoint_state_tensors=len(states[0]), checkpoint_states_exact=True, eval_rows=rows, eligible_mlp_input=[1, 256, 128], loss=float(losses[0].detach()), all_parameter_gradients_exact=True, parameter_count=len(gradients[0]), missing_gradients=missing, limits=['width128 CPU BF16 only', 'synthetic tokens', 'no Neuron/NKI/compile/timing/quality proof', 'batch comparison is reported separately, not exactness assumed'])


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
