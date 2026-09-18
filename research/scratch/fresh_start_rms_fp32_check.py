"""CPU-only gates for --ngram-rms-fp32 state AND arithmetic precision.

Run with repository dependencies installed. No Neuron device is selected.
Checks the actual MuonAdamW RMSProp path against an independent high-precision
recurrence, and the default path against the historical scalar update semantics.
Includes sparse/zero/outlier gradients, zero-vs-None distinction, varying learning
rates, foreach/scalar paths, tensor/float LR, and restored-state dtype guards.
"""

import copy
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T


BETA = 0.999
EPS = 1e-10


def make_optimizer(params):
    return T.MuonAdamW([dict(kind="rmsprop", params=params, lr=0.3,
                            betas=(0.0, BETA), eps=EPS, weight_decay=0.0)])


def gradient(shape, step, index, dtype):
    if index == 1 and step % 5 == 0:
        return None
    g = torch.zeros(shape, dtype=dtype)
    if step % 4 != 1:
        flat = g.view(-1)
        flat[(step + index) % flat.numel()] = (-1 if step % 2 else 1) * (0.03125 + step / 32)
        flat[(step * 3 + 2) % flat.numel()] = 1e-6
        if step in (0, 7, 19):
            flat[index] = 64.0 if step == 7 else 1.03125
    return g


def historical_step(p, v, grad, step, lr, tensor_lr):
    """Legacy low-precision semantics, independent of optimizer state/plumbing."""
    if grad is None:
        return p, v, step
    step += 1
    v.lerp_(grad.square(), 1 - BETA)
    u = grad / ((v / (1 - BETA ** step)).sqrt() + EPS)
    if tensor_lr:
        p.sub_(u * torch.tensor(lr, dtype=torch.float32))
    else:
        p.add_(u, alpha=-lr)
    return p, v, step


def precise_step(p, v, grad, step, lr, tensor_lr):
    """FP64 mathematical recurrence, rounded to FP32 state after each step.

    Parameter rounding occurs only after the FP32-scale update. This reference
    neither invokes lerp/foreach nor reads the production optimizer's state.
    """
    if grad is None:
        return p, v, step
    step += 1
    g = grad.float()
    # FP32 representable gradients are squared in FP32, then the moment is
    # evaluated in FP64 to avoid duplicating the production lerp arithmetic.
    v = (BETA * v.double() + (1 - BETA) * g.square().double()).float()
    den = (v / (1 - BETA ** step)).sqrt() + EPS
    update = g / den
    if tensor_lr:
        p = (p.float() - update * torch.tensor(lr, dtype=torch.float32)).to(p.dtype)
    else:
        p = torch.add(p.float(), update, alpha=-lr).to(p.dtype)
    return p, v, step


def run_cases():
    records = []
    for fp32 in (False, True):
        T.NGRAM_RMS_FP32 = fp32
        for foreach in (False, True):
            T.USE_FOREACH_OPTIM = foreach
            for tensor_lr in (False, True):
                T.USE_TENSOR_LR_SCALARS = tensor_lr
                for dtype in (torch.bfloat16, torch.float32):
                    torch.manual_seed(42)
                    params = [torch.nn.Parameter(torch.randn(shape).to(dtype))
                              for shape in ((4, 3), (2, 5))]
                    optimizer = make_optimizer(params)
                    reference = [(p.detach().clone(), torch.zeros_like(p, dtype=(torch.float32
                                 if fp32 else dtype)), 0) for p in params]
                    worst_state = 0.0
                    for i in range(32):
                        lr = (0.3, 0.071, 0.01, 0.125)[i % 4]
                        optimizer.param_groups[0]["lr"] = lr
                        saved_grads = []
                        for k, p in enumerate(params):
                            p.grad = gradient(p.shape, i, k, dtype)
                            saved_grads.append(None if p.grad is None else p.grad.clone())
                        optimizer.step()
                        for k, p in enumerate(params):
                            g = saved_grads[k]
                            ref_fn = precise_step if fp32 else historical_step
                            reference[k] = ref_fn(*reference[k][:2], g, reference[k][2], lr, tensor_lr)
                            rp, rv, count = reference[k]
                            assert p.dtype == dtype
                            if g is not None:
                                assert p.grad.dtype == dtype and torch.equal(p.grad, g)
                            if count == 0:
                                assert not optimizer.state[p]
                                continue
                            state = optimizer.state[p]
                            assert state["step"] == count
                            assert state["exp_avg_sq"].dtype == (torch.float32 if fp32 else dtype)
                            if fp32:
                                torch.testing.assert_close(state["exp_avg_sq"], rv, rtol=3e-6, atol=1e-14)
                                torch.testing.assert_close(p, rp, rtol=3e-6, atol=2e-7)
                            else:
                                assert torch.equal(state["exp_avg_sq"], rv), (foreach, tensor_lr, dtype, i, "v")
                                assert torch.equal(p, rp), (foreach, tensor_lr, dtype, i, "p")
                            assert torch.isfinite(p).all() and torch.isfinite(state["exp_avg_sq"]).all()
                            worst_state = max(worst_state, float((state["exp_avg_sq"] - rv).abs().max()))
                    records.append({"fp32_state_math": fp32, "foreach": foreach,
                                    "tensor_lr": tensor_lr, "parameter_dtype": str(dtype),
                                    "steps": 32, "max_state_abs_error": worst_state})
    return records


def decay_probe(fp32):
    T.NGRAM_RMS_FP32 = fp32
    T.USE_FOREACH_OPTIM = True
    T.USE_TENSOR_LR_SCALARS = True
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    optimizer = make_optimizer([p])
    p.grad = torch.tensor([1.03125, 0], dtype=p.dtype)
    optimizer.step()
    state = optimizer.state[p]
    first = float(state["exp_avg_sq"][0])
    if fp32:
        expected = float(p.grad[0].float().square()) * (1 - BETA)
        assert abs(first - expected) < 1e-9
        bf16_square = float(p.grad[0].square()) * (1 - BETA)
        assert abs(first - bf16_square) > 5e-7, "gradient was squared before FP32 cast"
    saved_p = p.detach().clone()
    for _ in range(256):
        p.grad.zero_()
        optimizer.step()
    ratio = float(state["exp_avg_sq"][0]) / first
    assert torch.equal(p, saved_p), "zero gradients moved a parameter"
    if fp32:
        assert math.isclose(ratio, BETA ** 256, rel_tol=3e-6), ratio
    else:
        assert ratio == 1.0, ratio
    prior = state["exp_avg_sq"].clone()
    prior_step = state["step"]
    p.grad = None
    optimizer.step()
    assert state["step"] == prior_step and torch.equal(state["exp_avg_sq"], prior)
    return {"fp32_state_math": fp32, "zero_steps": 256, "moment_ratio": ratio,
            "mathematical_ratio": BETA ** 256}


def restore_guard():
    T.NGRAM_RMS_FP32 = True
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    optimizer = make_optimizer([p])
    p.grad = torch.ones_like(p)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    restored = make_optimizer([q])
    restored.load_state_dict(saved)
    q.grad = torch.zeros_like(q)
    if restored.state[q]["exp_avg_sq"].dtype == torch.float32:
        restored.step()
        result = "preserved FP32 state accepted"
    else:
        before = q.detach().clone()
        step = restored.state[q]["step"]
        try:
            restored.step()
        except ValueError as error:
            assert "second-moment dtype" in str(error)
        else:
            raise AssertionError("lossy restored state was silently accepted")
        assert torch.equal(q, before) and restored.state[q]["step"] == step
        result = "parameter-dtype-cast state rejected before update"
    return result


def adamw_unchanged():
    outputs = []
    for enabled in (False, True):
        T.NGRAM_RMS_FP32 = enabled
        p = torch.nn.Parameter(torch.tensor([1.0, -0.5]))
        optimizer = T.MuonAdamW([dict(kind="adamw", params=[p], lr=0.03,
                                      betas=(0.8, 0.95), eps=EPS, weight_decay=0.0)])
        for i in range(4):
            p.grad = torch.tensor([1.0 + i, -0.25])
            optimizer.step()
        outputs.append((p.detach().clone(), copy.deepcopy(optimizer.state[p])))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert all(torch.equal(outputs[0][1][name], outputs[1][1][name])
               for name in ("exp_avg", "exp_avg_sq"))


def main():
    torch.set_num_threads(1)
    names = ("NGRAM_RMS_FP32", "USE_FOREACH_OPTIM", "USE_TENSOR_LR_SCALARS")
    original = {name: getattr(T, name) for name in names}
    assert T.NGRAM_RMS_FP32 is False and T.current_toggles()["NGRAM_RMS_FP32"] is False
    try:
        cases = run_cases()
        decay = [decay_probe(enabled) for enabled in (False, True)]
        restored = restore_guard()
        adamw_unchanged()
        print(json.dumps({"status": "PASS", "device": "cpu", "cases": cases,
                          "default_off_bit_exact": True, "gradients_preserved": True,
                          "zero_gradient_decay": decay, "restored_state": restored,
                          "adamw_unchanged": True, "hardware_quality_or_speed_claim": False}), flush=True)
    finally:
        for name, value in original.items():
            setattr(T, name, value)


if __name__ == "__main__":
    main()
