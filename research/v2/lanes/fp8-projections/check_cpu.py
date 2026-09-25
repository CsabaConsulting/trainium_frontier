#!/usr/bin/env python3
"""Tiny numerical self-checks; no native or independent-verifier claim."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-py', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    path = Path(args.train_py).resolve()
    spec = importlib.util.spec_from_file_location('fp8_test_train', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.compute_dtype_for = lambda device: torch.bfloat16
    torch.manual_seed(42)
    checks = {}
    errors = []
    for name, multiplier in [('ordinary', 1.0), ('tiny', 1e-8), ('outlier', 100.0), ('zero', 0.0)]:
        x = (torch.randn(2, 3, 16) * multiplier).to(torch.bfloat16).requires_grad_()
        w = torch.randn(24, 16, requires_grad=True)
        gy = torch.randn(2, 3, 24).to(torch.bfloat16)
        qx, sx, xe = mod._fp8_row_reference(x.reshape(-1, 16))
        qw, sw, we = mod._fp8_row_reference(w)
        expected = ((qx @ qw.T) * sw.T * sx).to(torch.bfloat16).reshape(2, 3, 24)
        y = mod._FP8Projection.apply(x, w)
        y.backward(gy)
        dx = (gy.reshape(-1, 24) @ we).reshape_as(x)
        dw = (gy.reshape(-1, 24).T @ xe).float()
        assert torch.equal(y, expected) and torch.equal(x.grad, dx) and torch.equal(w.grad, dw)
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
        assert x.grad.dtype == torch.bfloat16 and w.grad.dtype == torch.float32
        # Untouched token rows cannot acquire different scaling from companions.
        separate = torch.cat([mod._FP8Projection.apply(x[i:i+1], w) for i in range(2)])
        assert torch.equal(y, separate)
        errors.append(dict(case=name, max_bf16_forward_difference=float(
            (y.float() - torch.nn.functional.linear(x, w.to(x.dtype)).float()).abs().max().detach())))
    checks['projection_forward_vjp_zero_tiny_outlier_batch_independence'] = True
    # A row-dependent scaling witness: pulling scales outside dW's reduction
    # gives the wrong answer; all effective rows must participate in BF16.
    x = torch.randn(5, 16).to(torch.bfloat16)
    x[0] *= 32
    x.requires_grad_()
    w = torch.randn(24, 16, requires_grad=True)
    y = mod._FP8Projection.apply(x, w)
    y.sum().backward()
    _, _, xe = mod._fp8_row_reference(x)
    assert torch.equal(w.grad, (torch.ones_like(y).T @ xe).float())
    checks['token_scale_inside_weight_gradient_reduction'] = True
    mod.USE_FP8_PROJ = True
    mod.USE_NGRAM_VE = False
    mod.USE_BF16_NORM_OUTPUT = True
    mod.USE_QK_SHIFT = False
    config = mod.GPTConfig(sequence_len=16, vocab_size=256, n_layer=2, n_head=2, n_embd=256)
    model = mod.GPT(config)
    model.init_weights()
    marked = [name for name, item in model.named_modules() if isinstance(item, mod.Linear) and item.fp8_projection]
    assert len(marked) == 12 and not model.lm_head.fp8_projection
    assert all(p.dtype == torch.float32 for p in model.parameters())
    checks['six_projections_per_block_only_fp32_masters'] = marked
    token = torch.randint(0, 256, (2, 12))
    target = torch.randint(0, 256, (2, 12))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(token, target)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        before = model(token)
        changed = token.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % 256
        assert torch.equal(before[:, :8], model(changed)[:, :8])
        assert torch.equal(before, model(token))
        changed_input = (token + 3) % 256
        assert not torch.equal(before, model(changed_input))
    checks['tiny_three_step_finite_and_causal'] = losses
    # Bound the test-only eval mask allocation; model numerics are unchanged.
    mod.EVAL_MIN_SEQ_LEN = 16
    with tempfile.TemporaryDirectory(prefix='l1-fp8-cpu-') as tmp:
        checkpoint = Path(tmp) / 'final.pt'
        mod.save_checkpoint(checkpoint, model, {})
        mod.USE_FP8_PROJ = False
        restored = mod.load_for_eval(str(checkpoint), torch.device('cpu'))
        with torch.no_grad():
            assert mod.USE_FP8_PROJ and torch.equal(before, restored(token))
            assert torch.equal(restored(token), torch.cat([restored(token[i:i+1]) for i in range(2)]))
        # Old checkpoints without the new toggle must still select OFF.
        payload = torch.load(checkpoint, weights_only=True)
        payload['toggles'].pop('USE_FP8_PROJ')
        mod.restore_eval_toggles(payload)
        assert mod.USE_FP8_PROJ is False
    checks['checkpoint_restore_and_legacy_default'] = True
    result = dict(verdict='PASS', scope='implementer CPU self-check only', seed=42,
                  torch=torch.__version__, train_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  checks=checks, forward_error_examples=errors,
                  native_kernel_executed=False, performance_claim=False)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
