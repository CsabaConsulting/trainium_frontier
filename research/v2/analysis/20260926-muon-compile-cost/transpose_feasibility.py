#!/usr/bin/env python3
"""Bounded prospective update-mapping experiment; no candidate implementation.

Extracts only two arithmetic functions from pinned e1e2. Small matrices,
single CPU thread, no model, compiler, native backend, data or host access.
"""
import ast
import hashlib
import json
from pathlib import Path
import resource
import time

resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
started = time.monotonic()
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(42)
ROOT = Path(__file__).resolve().parents[4]
source = (ROOT / "workflow/candidates/confirmed-clock36-116188/train.py").read_bytes()
source_sha = hashlib.sha256(source).hexdigest()
if source_sha != "e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43":
    raise RuntimeError("source hash changed")
tree = ast.parse(source)
nodes = [n for n in tree.body if
         (isinstance(n, ast.FunctionDef) and n.name in ("_muon_math", "_muon_lr_g")) or
         (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "POLAR_EXPRESS_COEFFS" for t in n.targets))]
namespace = {"torch": torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<pinned e1e2 arithmetic>", "exec"), namespace)
math_fn = namespace["_muon_math"]


def metrics(reference, candidate):
    difference = (reference.double() - candidate.double())
    return {"bit_equal": torch.equal(reference, candidate),
            "max_abs": difference.abs().max().item(),
            "relative_l2": (difference.norm() / reference.double().norm().clamp_min(1e-30)).item(),
            "finite": bool(torch.isfinite(candidate).all())}


def scalars(step, ratio):
    # Same FP32 product staging as _muon_compile_scalars; schedule changes.
    lr = torch.tensor(0.015 * (1 - 0.075 * step), dtype=torch.float32)
    lr.mul_(ratio)
    lrwd = lr.clone().mul_(0.2 * (1 - 0.1 * step))
    mom = 0.85 + 0.0125 * step
    return torch.tensor(mom), torch.tensor(1 - mom), lr, lrwd


def update(state, grad, step, ratio, tall, red_dim, bf16):
    mom, one_mom, lr, lrwd = scalars(step, ratio)
    return math_fn(state[0], grad, state[1], state[2], mom, one_mom,
                   lr, lrwd, tall, 5, red_dim, grad.size(red_dim), 0.1,
                   bf16, False)


def transpose_case(rows, cols, bf16):
    logical_shape = (2, rows, cols)
    w = torch.randn(logical_shape) * 0.05
    w[1].zero_()  # c_proj-like zero-initialized weight, preserved independently.
    red = -1 if rows >= cols else -2
    second_shape = (2, rows, 1) if red == -1 else (2, 1, cols)
    baseline = (w, torch.zeros_like(w), torch.zeros(second_shape))
    mapped = tuple(x.mT.contiguous() for x in baseline)
    results = []
    for step in range(8):
        grad = torch.randn(logical_shape) * (0.05 + 0.01 * step)
        if step == 2:
            grad[0].zero_()
        if step == 3:
            grad.zero_()  # current _muon_step maps None gradients to zero.
        ratio = max(1.0, rows / cols) ** 0.5
        baseline = update(baseline, grad, step, ratio, rows > cols, red, bf16)
        mapped = update(mapped, grad.mT.contiguous(), step, ratio,
                        cols > rows, -2 if red == -1 else -1, bf16)
        results.append({"step": step, **{name: metrics(a, b.mT) for name, a, b in
                       zip(("weights", "momentum", "second_moment"), baseline, mapped)}})
    return {"logical_shape": list(logical_shape), "bf16_iterate": bf16,
            "logical_ratio_preserved": ratio, "results": results}


def merged_case(bf16):
    # fc: native32x8, ratio2. proj: native8x32 transposed only in optimizer
    # scratch, ratio1. No Parameter/storage/caller implementation is changed.
    fc = torch.randn(2, 32, 8) * 0.05
    proj = torch.zeros(2, 8, 32)
    fc_state = (fc, torch.zeros_like(fc), torch.zeros(2, 32, 1))
    proj_state = (proj, torch.zeros_like(proj), torch.zeros(2, 1, 32))
    merged = tuple(torch.cat((a, b.mT.contiguous())) for a, b in zip(fc_state, proj_state))
    results = []
    for step in range(8):
        fc_grad = torch.randn_like(fc) * 0.05
        proj_grad = torch.randn_like(proj) * 0.05
        if step == 2:
            fc_grad[0].zero_()
        if step == 3:
            proj_grad.zero_()
        fc_state = update(fc_state, fc_grad, step, 2.0, True, -1, bf16)
        proj_state = update(proj_state, proj_grad, step, 1.0, False, -2, bf16)
        mom, one_mom, fc_lr, fc_lrwd = scalars(step, 2.0)
        _, _, proj_lr, proj_lrwd = scalars(step, 1.0)
        lr = torch.stack((fc_lr, fc_lr, proj_lr, proj_lr)).view(4, 1, 1)
        lrwd = torch.stack((fc_lrwd, fc_lrwd, proj_lrwd, proj_lrwd)).view(4, 1, 1)
        merged = math_fn(merged[0], torch.cat((fc_grad, proj_grad.mT.contiguous())),
                         merged[1], merged[2], mom, one_mom, lr, lrwd,
                         True, 5, -1, 8, 0.1, bf16, False)
        row = {"step": step}
        for name, a, b, candidate in zip(("weights", "momentum", "second_moment"), fc_state, proj_state, merged):
            row[name] = metrics(torch.cat((a, b.mT.contiguous())), candidate)
        results.append(row)
    return {"bf16_iterate": bf16, "physical_shape": [4, 32, 8],
            "per_matrix_lr_ratios": [2, 2, 1, 1], "results": results}


cases = [transpose_case(rows, cols, bf16) for rows, cols in ((8, 32), (32, 8)) for bf16 in (False, True)]
merged = [merged_case(bf16) for bf16 in (False, True)]
comparisons = [row[name] for case in cases + merged for row in case["results"]
               for name in ("weights", "momentum", "second_moment")]
result = {"status": "PROSPECTIVE_SMALL_CPU_MAPPING_ONLY", "source_sha256": source_sha,
          "torch": torch.__version__, "threads": torch.get_num_threads(),
          "caps": {"address_space_gib": 6, "cpu_seconds": 60, "external_wall_seconds": 90},
          "seed": 42, "transpose_cases": cases, "mixed_stack_cases": merged,
          "comparison_count": len(comparisons),
          "all_finite": all(c["finite"] for c in comparisons),
          "all_bit_equal": all(c["bit_equal"] for c in comparisons),
          "max_abs_across_comparisons": max(c["max_abs"] for c in comparisons),
          "max_relative_l2_across_comparisons": max(c["relative_l2"] for c in comparisons),
          "elapsed_seconds": time.monotonic() - started,
          "limits": ["small rectangles only, not production1024/4096 dimensions",
                     "prospective mixed scratch arithmetic, not an optimizer/caller patch",
                     "no model-level OFF identity, native compiler, timing or quality validation",
                     "no AdamW gate equivalence claimed"]}
print(json.dumps(result, indent=2, sort_keys=True))
