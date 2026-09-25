def _device_scalar(self, group, key: str, value: float, ref: torch.Tensor) -> torch.Tensor:
    """Carry a per-step-changing hyperparameter in a persistent DEVICE tensor
        instead of a Python float. Measured on Trn2: a Python-float lr is baked into
        the Neuron graph as a constant, so a new value every step forces a fresh
        neuronx-cc compile every step -- dt 0.90s -> 2.2-4.3s for the whole LR
        cooldown, i.e. the last WARMDOWN_RATIO of every run. Writing the value into a
        pre-allocated tensor keeps the graph shape-and-constant identical forever."""
    cache = self._scalar_cache.setdefault(id(group), {})
    entry = cache.get(key)
    if entry is None:
        entry = (torch.empty((), dtype=torch.float32), torch.zeros((), dtype=torch.float32, device=ref.device))
        cache[key] = entry
    host, dev = entry
    host.fill_(float(value))
    dev.copy_(host)
    return dev

def _adamw_state(self, group):
    """Shared prologue for both AdamW paths: filter to params that actually have a
        grad this step, lazily create their state, advance the per-parameter step
        counter, and return the parallel lists the update needs.

        The step counter is per-parameter and NOT per-group because a parameter whose
        grad was None on some earlier step legitimately lags the rest of its group; the
        bias corrections are therefore returned as per-parameter scalar lists.
        """
    beta1, beta2 = group['betas']
    params, grads, exp_avgs, exp_avg_sqs, bias1, bias2 = ([], [], [], [], [], [])
    for p in group['params']:
        if p.grad is None:
            continue
        state = self.state[p]
        if not state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)
        state['step'] += 1
        params.append(p)
        grads.append(p.grad)
        exp_avgs.append(state['exp_avg'])
        exp_avg_sqs.append(state['exp_avg_sq'])
        bias1.append(1 - beta1 ** state['step'])
        bias2.append(1 - beta2 ** state['step'])
    return (params, grads, exp_avgs, exp_avg_sqs, bias1, bias2)

@torch.no_grad()
def _adamw_step_foreach(self, group):
    """Batched-dispatch equivalent of _adamw_step. Same update math, restructured so
        each stage is ONE torch._foreach_* call over the whole group instead of one aten
        op per parameter.

        Motivation is pure launch overhead, not arithmetic: the native TorchNeuron eager
        backend compiles one NEFF per aten op, so a K-parameter group costs ~8*K tiny
        launches in the loop form, each one paying a full host dispatch gap with every
        engine idle. A profile of the locked config measured MuonAdamW.step() at 10% of
        step time (89.6ms) across 647 launches / 171 un-fused NEFFs, 73.2% of which was
        dispatch gap.

        This is required to be BIT-IDENTICAL to _adamw_step, not merely close:
          * grad.square() is emitted as _foreach_mul(g, g) -- verified bitwise identical
            to .square() for both fp32 and bf16 (research/scratch/foreach_api_probe.py).
          * bias1/bias2 go in as ScalarLists, one entry per parameter, preserving the
            per-parameter step counter exactly (see _adamw_state).
          * the weight-decay multiply is skipped when weight_decay == 0. That is exactly,
            not approximately, a no-op: the original computes p.mul_(1 - lr*0) =
            p.mul_(1.0), and multiplying a finite IEEE754 value by exactly 1.0 (fp32,
            then rounded back to the parameter dtype) returns it unchanged. Four of the
            locked config's AdamW groups have weight_decay=0.0, so this is also where a
            good chunk of the saved launches come from.
        foreach ops require same dtype/device but NOT the same shape, so the ragged
        scalar group batches fine -- confirmed against this installed torch (2.11.0) in
        research/scratch/foreach_api_probe.py rather than assumed from a newer API.
        """
    params, grads, exp_avgs, exp_avg_sqs, bias1, bias2 = self._adamw_state(group)
    if not params:
        return
    lr = group['lr']
    if USE_TENSOR_LR_SCALARS:
        lr = self._device_scalar(group, 'lr', lr, params[0])
    beta1, beta2 = group['betas']
    wd = group['weight_decay']
    torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
    torch._foreach_lerp_(exp_avg_sqs, torch._foreach_mul(grads, grads), 1 - beta2)
    denom = torch._foreach_div(exp_avg_sqs, bias2)
    torch._foreach_sqrt_(denom)
    torch._foreach_add_(denom, group['eps'])
    update = torch._foreach_div(exp_avgs, bias1)
    torch._foreach_div_(update, denom)
    if wd != 0:
        if USE_CAUTIOUS_WD:
            _cautious_weight_decay_(params, update, lr, wd)
        else:
            torch._foreach_mul_(params, 1 - lr * wd)
    if USE_TENSOR_LR_SCALARS:
        torch._foreach_mul_(update, lr)
        torch._foreach_sub_(params, update)
    else:
        torch._foreach_add_(params, update, alpha=-lr)
