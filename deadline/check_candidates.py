#!/usr/bin/env python3
"""Check disabled-path parity, causality, batch independence and gradients on CPU."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def config(m, **extra):
    return m.GPTConfig(sequence_len=32, vocab_size=256, n_layer=8, n_head=1,
                       n_embd=128, lane_b=False, **extra)


def main():
    modules = {name: module('deadline_' + name, ROOT / path) for name, path in [
        ('base', 'submission/train.py'), ('control', 'candidates/control/train.py'),
        ('middle', 'candidates/middle/train.py'), ('mlp', 'candidates/mlp/train.py'),
        ('mlp6', 'candidates/mlp6/train.py'), ('rmsfp32', 'candidates/rmsfp32/train.py'),
        ('mimetic', 'candidates/mimetic/train.py')]}
    torch.manual_seed(42)
    base = modules['base'].GPT(config(modules['base']))
    base.init_weights()
    state = base.state_dict()
    x = torch.randint(0, 256, (2, 32))
    targets = torch.randint(0, 256, (2, 32))
    weights = torch.tensor([1., .2])
    control = modules['control'].GPT(config(modules['control']))
    control.load_state_dict(state, strict=True)
    original_logits, control_logits = base(x), control(x)
    assert torch.equal(original_logits, control_logits), 'Disabled path logits differ'
    original_loss = base(x, targets, mtp_weights=weights)
    control_loss = control(x, targets, mtp_weights=weights)
    assert torch.equal(original_loss, control_loss), 'Disabled path loss differs'
    original_loss.backward()
    control_loss.backward()
    compared = 0
    for (n, p), (cn, cp) in zip(base.named_parameters(), control.named_parameters()):
        assert n == cn and ((p.grad is None and cp.grad is None) or (p.grad is not None and cp.grad is not None and torch.equal(p.grad, cp.grad))), ('Disabled path gradient differs', n)
        compared += 1
    outputs = {}
    for name in ['middle', 'mlp', 'mlp6']:
        m = modules[name]
        cfg = config(m, coarse_middle=name == 'middle', coarse_mlp=name in ('mlp', 'mlp6'))
        model = m.GPT(cfg)
        model.load_state_dict(state, strict=True)
        assert [n for n, _ in model.named_parameters()] == [n for n, _ in base.named_parameters()]
        logits = model(x)
        assert logits.shape == (2, 32, 256) and torch.isfinite(logits).all()
        max_future = 0.
        for boundary in [1, 2, 3, 7, 16, 31]:
            changed = x.clone()
            changed[:, boundary:] = (changed[:, boundary:] + 37) % 256
            shifted = model(changed)
            error = float((logits[:, :boundary] - shifted[:, :boundary]).abs().max())
            assert error == 0., (name, 'future leakage', boundary, error)
            max_future = max(max_future, error)
        other = x.clone()
        other[1] = (other[1] + 23) % 256
        assert torch.equal(logits[0], model(other)[0]), (name, 'batch dependence')
        assert torch.equal(logits, model(x)), (name, 'repeat failure')
        loss = model(x, targets, mtp_weights=weights)
        assert torch.isfinite(loss)
        loss.backward()
        gradient_count = 0
        for n, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (name, n)
            gradient_count += 1
        payload = {'model_config': dataclasses.asdict(cfg), 'model_state': model.state_dict()}
        reload = m.GPT(m.GPTConfig(**payload['model_config']))
        reload.load_state_dict(payload['model_state'], strict=True)
        assert torch.equal(logits, reload(x)), (name, 'checkpoint mismatch')
        outputs[name] = {'passed': True, 'future_max_abs': max_future,
                         'finite_parameter_gradients': gradient_count,
                         'loss': float(loss), 'parameters': sum(p.numel() for p in model.parameters()),
                         'state_keys_unchanged': True, 'checkpoint_exact': True,
                         'test_device': 'cpu', 'test_shape': [2, 32, 128]}
    # Exercise the numerical gap directly: with zero gradient, BF16 .999
    # decay rounds to the original moment; FP32 must still decay it.
    rms = modules['rmsfp32']
    table = torch.zeros((4, 8), dtype=torch.bfloat16)
    grad = torch.zeros_like(table)
    second = torch.ones_like(table, dtype=torch.float32)
    lr, correction = torch.tensor(.3), torch.tensor(1.)
    updated, got, cleared = rms.d1_rms_step(table, grad, second, lr, correction)
    assert torch.equal(updated, table) and torch.equal(got, second * .999)
    assert got.dtype == torch.float32 and cleared.dtype == torch.float32
    old = modules['base'].d1_rms_step(table, grad, second.bfloat16(), lr, correction)[1]
    assert torch.equal(old, second.bfloat16())
    for step in range(1, 101):
        grad = (torch.sin(torch.arange(32).reshape(4, 8).float() + step) * .003).bfloat16()
        reference_second = torch.lerp(second, grad.float().square(), .001)
        correction = torch.tensor(1 - .999 ** step)
        reference_table = (table.float() - lr * grad.float() / ((reference_second / correction).sqrt() + 1e-10)).bfloat16()
        table, second, cleared = rms.d1_rms_step(table, grad, second, lr, correction)
        assert torch.equal(second, reference_second) and torch.equal(table, reference_table)
        assert torch.isfinite(table).all() and torch.isfinite(second).all()
    outputs['rmsfp32'] = {'passed': True, 'zero_gradient_decay_preserved': True,
                          'fp32_reference_steps_exact': 100}
    # Seed is a reproducibility control, never a seed search. This arm preserves
    # the global RNG and every unrelated initialized parameter.
    torch.manual_seed(42)
    reference = modules['base'].GPT(config(modules['base']))
    reference.init_weights()
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    mimetic = modules['mimetic'].GPT(config(modules['mimetic']))
    mimetic.init_weights()
    assert torch.equal(reference_rng, torch.get_rng_state()), 'Mimetic changed global RNG'
    changed_names = []
    for (n, rp), (mn, mp) in zip(reference.named_parameters(), mimetic.named_parameters()):
        assert n == mn
        if '.attn.c_v.weight' in n or '.attn.c_proj.weight' in n:
            changed_names.append(n)
        else:
            assert torch.equal(rp, mp), ('Mimetic changed unrelated initialization', n)
    for block in mimetic.transformer.h:
        if block.attn is not None:
            product = block.attn.c_proj.weight.float() @ block.attn.c_v.weight.float()
            expected = torch.eye(128) * modules['mimetic'].INIT_SCALE ** 2 * modules['mimetic'].ATTN_CPROJ_INIT_SCALE
            assert torch.allclose(product, expected, atol=2e-4, rtol=.01)
    logits = mimetic(x)
    assert torch.isfinite(logits).all()
    changed = x.clone(); changed[:, 7:] = (changed[:, 7:] + 37) % 256
    assert torch.equal(logits[:, :7], mimetic(changed)[:, :7])
    loss = mimetic(x, targets, mtp_weights=weights); loss.backward()
    for n, param in mimetic.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), ('mimetic', n)
    outputs['mimetic'] = {'passed': True, 'global_rng_unchanged': True,
                          'unrelated_parameters_unchanged': True, 'changed_parameters': changed_names}
    result = {'passed': True, 'disabled_parameter_gradients_bitwise_equal': compared,
              'variants': outputs, 'cpu_proof_is_not_native_performance_or_quality': True}
    print('DEADLINE_CPU_CHECK ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
