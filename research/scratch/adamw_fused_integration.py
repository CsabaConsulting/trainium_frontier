"""Opt-in integration of the gated functional FP32 AdamW kernel.

Only the dense1024 model's five large FP32 tables are eligible. Other groups,
including scalar/conv parameters, retain the actual native foreach path. The
original per-parameter state prologue handles missing gradients and counters.
Parameter objects and storage are retained; copy_ also advances their version.
"""
import types

import torch

from adamw_fused_fp32 import build_kernel, scalar_values


def make_step(train_module, native, math_mode=0):
    from torch_neuronx import wrap_nki

    assert math_mode in (0, 2), "only original or component-gated reciprocal refinement"
    kernel = wrap_nki(build_kernel())

    @torch.no_grad()
    def fused_step(self, group):
        active = [p for p in group["params"] if p.grad is not None]
        if not active:
            return
        supported = (train_module.USE_TENSOR_LR_SCALARS
                     and not train_module.USE_CAUTIOUS_WD
                     and group["eps"] == 1e-10
                     and all(p.device.type == "neuron" and p.dtype == torch.float32
                             and p.shape == (8192, 1024) and p.is_contiguous()
                             and p.grad.dtype == torch.float32 and p.grad.is_contiguous()
                             for p in active))
        if not supported:
            return native(self, group)
        params, grads, moments, variances, _, _ = self._adamw_state(group)
        if not hasattr(self, "_fresh_start_adamw_buffers"):
            self._fresh_start_adamw_buffers = {}
            self._fresh_start_adamw_calls = 0
        beta1, beta2 = group["betas"]
        for p, g, m, v in zip(params, grads, moments, variances):
            assert m.dtype == v.dtype == torch.float32
            assert m.is_contiguous() and v.is_contiguous()
            if p not in self._fresh_start_adamw_buffers:
                host = torch.empty(128, 7, dtype=torch.float32)
                device = torch.empty_like(host, device=p.device)
                self._fresh_start_adamw_buffers[p] = (host, device)
            host, device = self._fresh_start_adamw_buffers[p]
            state = self.state[p]
            values = scalar_values(group["lr"], state["step"], beta1, beta2,
                                   group["weight_decay"], group["eps"])
            host.copy_(values.expand_as(host))
            device.copy_(host)
            new_p, new_m, new_v = kernel[2](
                p.view(4096, 2048), g.view(4096, 2048),
                m.view(4096, 2048), v.view(4096, 2048), device, 2, False, math_mode)
            p.copy_(new_p.view_as(p))
            state["exp_avg"] = new_m.view_as(p)
            state["exp_avg_sq"] = new_v.view_as(p)
            self._fresh_start_adamw_calls += 1

    return fused_step


def install_optimizer(optimizer, train_module, math_mode=0):
    """Patch only this instance for isolated native/candidate comparisons."""
    assert "_adamw_step_foreach" not in optimizer.__dict__, "already installed"
    native = type(optimizer)._adamw_step_foreach
    optimizer._adamw_step_foreach = types.MethodType(make_step(train_module, native, math_mode), optimizer)
    optimizer._fresh_start_adamw_math_mode = math_mode


def install(train_module, math_mode=0):
    """Patch future and existing instances only in an explicit scratch process."""
    assert not getattr(train_module, "_fresh_start_fused_adamw", False), "already installed"
    native = train_module.MuonAdamW._adamw_step_foreach
    train_module.MuonAdamW._adamw_step_foreach = make_step(train_module, native, math_mode)
    train_module._fresh_start_fused_adamw = True
