"""Exactness harness for the T2.3 (USE_CAUTIOUS_WD) restructuring of MuonAdamW's AdamW paths.

Two separate claims, both checked bitwise on the real model at the locked config, CPU fp32:

 (1) FLAG OFF == PRE-EDIT CODE. Adding cautious weight decay required moving the decoupled decay
     multiply from ABOVE the two moment lerps to BELOW the `update` computation (the mask needs
     `update`). The claim is that this is exactly, not approximately, equivalence-preserving:
     `update` reads only exp_avg / exp_avg_sq / grad and never `p`, and `p` is still at its
     pre-decay value when the multiply reads it. This file re-implements the pre-edit ordering
     verbatim (_preedit_foreach / _preedit_loop below, both copied from git HEAD~ and diffed by
     eye against the current bodies) and demands 0.0 max abs diff on every parameter and every
     optimizer-state tensor after several steps.

 (2) FLAG ON == THE SPEC, and only where it is supposed to bite. With USE_CAUTIOUS_WD=True the
     result must equal an independent, explicitly-written reference
     `p * (1 - lr*wd*((update*p) > 0)) - lr*update` for the two groups with wd != 0 (wte, lm_head),
     and must be bitwise UNCHANGED for every parameter in a wd == 0 group. Also reports the
     fraction of elements the mask actually spares, which is the only interesting sanity number:
     a mask that spares ~0% or ~100% of elements means the flag is doing nothing / everything.

Run: python research/scratch/cautious_wd_equiv.py
"""
import copy
import os
import sys

os.environ.setdefault("NEURON_COMPILE_CACHE_DIR", "/tmp/neuron_cache")
sys.argv = ["equiv"]  # keep train.py's argparse out of this

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import train as T

STEPS = 5
SEED = 1234


# --- pre-edit reference implementations (git HEAD before the T2.3 commit) --------------------
# Only the position of the weight-decay multiply differs from the current bodies; everything else
# is a verbatim copy, deliberately duplicated rather than parameterised so the two orderings can
# be compared as independent code.

def _preedit_foreach(self, group):
    params, grads, exp_avgs, exp_avg_sqs, bias1, bias2 = self._adamw_state(group)
    if not params:
        return
    lr = group["lr"]
    if T.USE_TENSOR_LR_SCALARS:
        lr = self._device_scalar(group, "lr", lr, params[0])
    beta1, beta2 = group["betas"]
    wd = group["weight_decay"]
    if wd != 0:
        torch._foreach_mul_(params, 1 - lr * wd)          # <-- ABOVE the lerps: the pre-edit spot
    torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
    torch._foreach_lerp_(exp_avg_sqs, torch._foreach_mul(grads, grads), 1 - beta2)
    denom = torch._foreach_div(exp_avg_sqs, bias2)
    torch._foreach_sqrt_(denom)
    torch._foreach_add_(denom, group["eps"])
    update = torch._foreach_div(exp_avgs, bias1)
    torch._foreach_div_(update, denom)
    if T.USE_TENSOR_LR_SCALARS:
        torch._foreach_mul_(update, lr)
        torch._foreach_sub_(params, update)
    else:
        torch._foreach_add_(params, update, alpha=-lr)


@torch.no_grad()
def _preedit_loop(self, group):
    beta1, beta2 = group["betas"]
    lr = group["lr"]
    if T.USE_TENSOR_LR_SCALARS and group["params"]:
        lr = self._device_scalar(group, "lr", lr, group["params"][0])
    for p in group["params"]:
        if p.grad is None:
            continue
        grad = p.grad
        state = self.state[p]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        state["step"] += 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        p.mul_(1 - lr * group["weight_decay"])            # <-- ABOVE the lerps: the pre-edit spot
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2)
        bias1 = 1 - beta1 ** state["step"]
        bias2 = 1 - beta2 ** state["step"]
        update = exp_avg / bias1 / ((exp_avg_sq / bias2).sqrt() + group["eps"])
        if T.USE_TENSOR_LR_SCALARS:
            p.sub_(update * lr)
        else:
            p.add_(update, alpha=-lr)


# --- harness --------------------------------------------------------------------------------

def build():
    torch.manual_seed(SEED)
    return T.GPT(T.build_config())


def make_grads(model, step):
    g = torch.Generator().manual_seed(9000 + step)
    return {n: (torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.01).to(p.dtype)
            for n, p in model.named_parameters()}


def run(model, grads_per_step, *, preedit):
    """Run STEPS optimizer steps. `preedit` swaps in the pre-edit method bodies."""
    saved = (T.MuonAdamW._adamw_step_foreach, T.MuonAdamW._adamw_step)
    if preedit:
        T.MuonAdamW._adamw_step_foreach = _preedit_foreach
        T.MuonAdamW._adamw_step = _preedit_loop
    try:
        opt = model.setup_optimizer()
        named = dict(model.named_parameters())
        for s in range(STEPS):
            for name, p in named.items():
                p.grad = grads_per_step[s][name].clone().to(p.dtype)
            for grp in opt.param_groups:      # exercise the real per-step lr write
                grp["lr"] = grp["initial_lr"] * (1.0 - 0.3 * s / STEPS)
            opt.step()
        return opt
    finally:
        T.MuonAdamW._adamw_step_foreach, T.MuonAdamW._adamw_step = saved


def compare(label, m_ref, m_new, opt_ref, opt_new, *, expect_differ=()):
    """expect_differ: parameter names allowed (and required) to differ."""
    print(f"\n--- {label} ---")
    worst, worst_where, nparam, ndiff = 0.0, "", 0, 0
    differed = set()
    for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_new.named_parameters()):
        assert n1 == n2
        nparam += 1
        d = (p1.float() - p2.float()).abs().max().item()
        if not torch.equal(p1, p2):
            differed.add(n1)
            ndiff += 1
            if n1 not in expect_differ:
                print(f"   UNEXPECTED MISMATCH param {n1}: max abs {d:.3e}")
        if n1 not in expect_differ and d > worst:
            worst, worst_where = d, f"param {n1}"
    keys = ("exp_avg", "exp_avg_sq", "momentum_buffer", "second_momentum_buffer")
    nstate = 0
    for (pr, sr), (pn, sn) in zip(opt_ref.state.items(), opt_new.state.items()):
        for k in keys:
            if k in sr and k in sn:
                nstate += 1
                d = (sr[k].float() - sn[k].float()).abs().max().item()
                if not torch.equal(sr[k], sn[k]):
                    print(f"   MISMATCH state[{k}] shape {tuple(sr[k].shape)}: max abs {d:.3e}")
                if d > worst:
                    worst, worst_where = d, f"state[{k}]"
    print(f"   compared {nparam} params + {nstate} state tensors over {STEPS} steps; "
          f"{ndiff} params differed")
    missing = set(expect_differ) - differed
    if missing:
        print(f"   WARNING: expected these to differ and they did NOT: {sorted(missing)}")
    print(f"   max abs diff outside the expected set = {worst:.3e}  ({worst_where or 'exact'})")
    ok = worst == 0.0 and not missing
    print(f"   VERDICT: {'PASS' if ok else 'FAIL'}")
    return ok


print(f"torch {torch.__version__}  |  locked config DEPTH={T.DEPTH} ASPECT_RATIO={T.ASPECT_RATIO} "
      f"USE_FOREACH_OPTIM={T.USE_FOREACH_OPTIM}")
all_ok = True

# ---- claim (1): flag off == pre-edit, on both paths and both lr representations ----
for foreach in (True, False):
    for tensor_lr in (True, False):
        T.USE_FOREACH_OPTIM = foreach
        T.USE_TENSOR_LR_SCALARS = tensor_lr
        T.USE_CAUTIOUS_WD = False
        m_ref = build()
        m_new = copy.deepcopy(m_ref)
        grads = [make_grads(m_ref, s) for s in range(STEPS)]
        opt_ref = run(m_ref, grads, preedit=True)
        opt_new = run(m_new, grads, preedit=False)
        all_ok &= compare(f"(1) flag OFF vs pre-edit  foreach={foreach} tensor_lr={tensor_lr}",
                          m_ref, m_new, opt_ref, opt_new)

# ---- claim (2): flag on changes exactly the wd != 0 params, and matches the written spec ----
T.USE_FOREACH_OPTIM = True
T.USE_TENSOR_LR_SCALARS = True
m_off = build()
m_on = copy.deepcopy(m_off)
grads = [make_grads(m_off, s) for s in range(STEPS)]
T.USE_CAUTIOUS_WD = False
opt_off = run(m_off, grads, preedit=False)
T.USE_CAUTIOUS_WD = True
opt_on = run(m_on, grads, preedit=False)

# which params live in a wd != 0 group?
T.USE_CAUTIOUS_WD = False
probe = build()
opt_probe = probe.setup_optimizer()
ids = {id(p): n for n, p in probe.named_parameters()}
wd_names, wd_by_name = [], {}
for g in opt_probe.param_groups:
    if g["kind"] == "adamw" and g["weight_decay"] != 0:
        for p in g["params"]:
            wd_names.append(ids[id(p)])
            wd_by_name[ids[id(p)]] = g["weight_decay"]
print(f"\nAdamW groups with wd != 0: {[(n, wd_by_name[n]) for n in wd_names]}")
all_ok &= compare("(2) cautious ON vs OFF -- only the wd != 0 params may move",
                  m_off, m_on, opt_off, opt_on, expect_differ=tuple(wd_names))

# independent single-tensor reference for the masked update, plus the mask occupancy
print("\n--- (2b) masked update vs an independently written reference ---")
T.USE_CAUTIOUS_WD = True
for name in wd_names:
    p0 = dict(build().named_parameters())[name].detach().clone()
    wd = wd_by_name[name]
    # one step, by hand, at the same lr the group used on step 0
    grp = [g for g in opt_probe.param_groups if any(ids[id(q)] == name for q in g["params"])][0]
    lr = grp["initial_lr"]
    g0 = grads[0][name].float()
    beta1, beta2 = grp["betas"]
    exp_avg = g0 * (1 - beta1)
    exp_avg_sq = g0.square() * (1 - beta2)
    update = exp_avg / (1 - beta1) / ((exp_avg_sq / (1 - beta2)).sqrt() + grp["eps"])
    mask = (update * p0.float()) > 0
    ref = p0.float() * (1 - lr * wd * mask.float()) - lr * update
    got = p0.clone()
    T._cautious_weight_decay_([got], [update.to(got.dtype)], lr, wd)
    got = got - lr * update.to(got.dtype)
    d = (ref.to(got.dtype) - got).abs().max().item()
    print(f"   {name:28s} wd={wd:<6} spared {100.0 * (~mask).float().mean().item():5.1f}% of "
          f"elements from decay | max abs diff vs reference {d:.3e}")
    all_ok &= d == 0.0

print(f"\n================ OVERALL: {'ALL CHECKS PASS' if all_ok else 'FAILURE'} ================")
sys.exit(0 if all_ok else 1)
