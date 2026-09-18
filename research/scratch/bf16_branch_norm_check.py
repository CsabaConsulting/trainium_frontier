#!/usr/bin/env python3
"""CPU gates for the proposed precision policy and previously bypassed cache.

Check RMS arithmetic against an FP64 analytic reference, then exercise the real
model with nonzero attention/MLP/conv branches. Cache-on/off must preserve master
gradients and two updates across three independent backward calls per update.
Native compilation, throughput, and trained BPB are separate gates.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

from bf16_branch_norm import FP32_EPS, bf16_branch_norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T  # noqa: E402


def normalization_checks():
    reports = []
    for width in (128, 896, 1024):
        for scale in (0.0, 1e-6, 1e-3, 1.0, 1e4):
            x = (torch.randn(2, 17, width) * scale).requires_grad_()
            y = bf16_branch_norm(x)
            g = torch.randn_like(y)
            (dx,) = torch.autograd.grad(y, (x,), g)
            xd, gd = x.detach().double(), g.double()
            inv = (xd.square().mean(-1, keepdim=True) + FP32_EPS).rsqrt()
            yd = xd * inv
            dxd = inv * (gd - xd * inv.square() * (gd * xd).mean(-1, keepdim=True))
            rounded = yd.to(torch.bfloat16)
            next_up = torch.nextafter(rounded, torch.full_like(rounded, float("inf")))
            next_down = torch.nextafter(rounded, torch.full_like(rounded, -float("inf")))
            ulp = torch.maximum((next_up.float() - rounded.float()).abs(),
                                (next_down.float() - rounded.float()).abs())
            assert y.dtype == torch.bfloat16 and dx.dtype == torch.float32
            assert torch.isfinite(y).all() and torch.isfinite(dx).all()
            assert ((y.float() - rounded.float()).abs() <= ulp).all()
            rel = float((dx.double() - dxd).norm() / dxd.norm().clamp_min(1e-30))
            assert rel < 3e-6, (width, scale, rel)
            # Bound reduction error by SUM(abs(products)), not abs(SUM(products)).
            # The first checker used the latter and underestimated dot-product
            # cancellation error (10 elements at C1024/scale1e-3, RMS error 4.88e-8).
            # gamma covers a conservative sequential FP32 reduction; the separate
            # unchanged RMS gate above prevents this bound hiding a broad error.
            u = 2.0 ** -24
            gamma = (width + 8) * u / (1 - (width + 8) * u)
            envelope = ((gd * inv).abs() + xd.abs() * inv.pow(3) *
                        (gd * xd).abs().mean(-1, keepdim=True))
            assert ((dx.double() - dxd).abs() <= 4 * gamma * envelope + 1e-30).all()
            changed = x.detach().clone()
            changed[:, 8:] += 10
            torch.testing.assert_close(bf16_branch_norm(changed)[:, :8], y[:, :8], rtol=0, atol=0)
            reports.append(dict(width=width, scale=scale, dx_relative_rms=rel,
                                forward_within_one_bf16_ulp=True))
    return reports


def model_checks(policy="norm"):
    T.USE_NGRAM_VE = T.USE_ENGRAM = T.USE_OUT_BIGRAM = False
    T.USE_QK_SHIFT = T.USE_NKI_ROPE_NORM = T.USE_NKI_SOFTCAP_CE = False
    T.USE_NKI_RELU2 = False
    T.USE_LOCAL_CONV = T.TRAIN_LOCAL_CONV = True
    T.LOCAL_CONV_KERNEL = 2
    T.COMPUTE_DTYPE = torch.bfloat16
    if policy == "norm":
        T.norm = bf16_branch_norm
        expected_input_dtype = torch.bfloat16
    else:
        from bf16_sdpa_boundary import bf16_sdpa
        T.F.scaled_dot_product_attention = bf16_sdpa
        expected_input_dtype = torch.float32
    cfg = T.GPTConfig(sequence_len=128, vocab_size=256, n_layer=2, n_head=1, n_embd=128)
    torch.manual_seed(923)
    initial = T.GPT(cfg)
    initial.init_weights()
    with torch.no_grad():
        for block in initial.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.02)
            block.mlp.c_proj.weight.normal_(std=0.02)
            block.local_conv.weight.normal_(std=0.1)
    state = {k: v.clone() for k, v in initial.state_dict().items()}
    batches = [(torch.randint(cfg.vocab_size, (2, 128)),
                torch.randint(cfg.vocab_size, (2, 128))) for _ in range(6)]
    outcomes = []
    for use_cache in (False, True):
        T.USE_WEIGHT_CACHE = use_cache
        model = T.GPT(cfg)
        model.load_state_dict(state)
        linears = [m for m in model.modules() if isinstance(m, T.Linear)]
        branch_dtypes, residual_dtypes = [], []
        hooks = [m.register_forward_pre_hook(lambda m, args: branch_dtypes.append(args[0].dtype))
                 for m in linears]
        hooks += [m.register_forward_hook(lambda m, args, out: residual_dtypes.append(out.dtype))
                  for m in model.transformer.h]
        losses = []
        for step in range(2):
            model.zero_grad(set_to_none=True)
            if use_cache:
                for m in linears:
                    m.refresh_weight_cache(torch.bfloat16)
                    assert m._cached_weight.requires_grad
            for x, y in batches[step * 3:(step + 1) * 3]:
                loss = model(x, y)
                assert torch.isfinite(loss)
                losses.append(float(loss.detach()))
                (loss / 3).backward()
            gradients = {n: p.grad.clone() for n, p in model.named_parameters()
                         if p.grad is not None}
            assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
            assert all(g.dtype == torch.float32 for g in gradients.values())
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-0.01)
        assert set(branch_dtypes) == {expected_input_dtype}, set(branch_dtypes)
        assert set(residual_dtypes) == {torch.float32}, set(residual_dtypes)
        for hook in hooks:
            hook.remove()
        if use_cache:
            for m in linears:
                m.refresh_weight_cache(torch.bfloat16)
        with torch.no_grad():
            x = batches[0][0]
            logits = model(x)
            changed = x.clone()
            changed[:, 64:] = (changed[:, 64:] + 19) % cfg.vocab_size
            torch.testing.assert_close(model(changed)[:, :64], logits[:, :64], rtol=0, atol=0)
            changed = x.clone()
            changed[0] = (changed[0] + 27) % cfg.vocab_size
            torch.testing.assert_close(model(changed)[1:], logits[1:], rtol=0, atol=0)
        outcomes.append((losses, gradients, {n: p.detach().clone() for n, p in model.named_parameters()}))
    assert outcomes[0][0] == outcomes[1][0]
    for before, after in zip(outcomes[0][1:], outcomes[1][1:]):
        assert before.keys() == after.keys()
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0, msg=name)
    return dict(policy=policy, steps=2, microbatches_per_step=3, cache_gradients_and_updates_exact=True,
                linear_input_dtype=str(expected_input_dtype), block_residual_dtype="float32",
                future_and_cross_batch_exact=True, losses=outcomes[0][0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", choices=["norm", "attention"], default="norm")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(571)
    if args.policy == "norm":
        print(json.dumps(dict(check="normalization", cases=normalization_checks())), flush=True)
    print(json.dumps(dict(check="model_and_cache", **model_checks(args.policy))), flush=True)
    print("PASS: CPU precision-policy and cache gates; no native speed or BPB claim", flush=True)


if __name__ == "__main__":
    main()
