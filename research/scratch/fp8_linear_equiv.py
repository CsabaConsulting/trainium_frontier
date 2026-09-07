"""
FP8 cast-and-scale Linear: offline numerical-equivalence harness (Phase 1 of the FP8
design, see research/trn2-experiment-results.md). CPU-only, does NOT touch train.py or
prepare.py -- torch.float8_e4m3fn is CPU-constructible even without a Neuron device, so
the core cast/scale math can be verified here before ever touching hardware.

Design (from the FP8 research pass): current-scaling (not delayed-scaling), forward-only
fp8 (both operands cast to float8_e4m3fn for the matmul), backward stays in bf16 -- the
de-risked variant recommended given this competition's short (~850-step) horizon, where
delayed-scaling's real, documented 50-200-step amax-history warmup instability would eat
a much larger fraction of the budget than in a normal long pretrain.

Verification metric: relative L2 error of the fp8-cast-and-scale matmul's forward output
against a plain bf16 reference, and relative L2 error of the backward gradients (weight
and input) using the de-risked bf16-backward scheme -- NOT an exact-match tolerance (fp8
E4M3 has ~2-3 effective mantissa bits, an inherent, expected precision loss unlike the
rope/softcap kernels' lossless fusions). Checked across a few activation-magnitude
regimes representative of this model's real scale (post-RMSNorm activations are
order-1 in magnitude; pre-activation matmul inputs/weights follow the init scheme in
train.py: uniform(-s, s), s = INIT_SCALE * sqrt(3) * n_embd**-0.5).
"""
import torch
import torch.nn.functional as F

FP8_E4M3_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


def current_scale(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Current-scaling (NVIDIA Transformer Engine's Float8CurrentScaling): use this
    tensor's own amax directly, no history/delay -- avoids delayed-scaling's warmup
    instability, at the cost of one extra reduction pass over an already-resident tensor
    (cheap relative to the matmul it's scaling)."""
    amax = t.abs().amax().clamp_min(eps)
    return FP8_E4M3_MAX / amax


def cast_fp8(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (t.float() * scale).to(torch.float8_e4m3fn)


def fp8_linear_forward(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reference (unfused) implementation of the forward-only-fp8 matmul: cast both
    operands to fp8 using their own current-scale, matmul in fp8 (accumulate in fp32 --
    matches how a real Trainium2 FP8 GEMM accumulates), rescale back."""
    x_scale = current_scale(x)
    w_scale = current_scale(weight)
    x_fp8 = cast_fp8(x, x_scale)
    w_fp8 = cast_fp8(weight, w_scale)
    # fp8 x fp8 -> fp32 accumulation, then undo both scales
    y = torch.matmul(x_fp8.float(), w_fp8.float().t()) / (x_scale * w_scale)
    return y


class Fp8LinearFn(torch.autograd.Function):
    """torch.autograd.Function wrapper matching the design doc: forward in fp8 (both
    operands), backward in bf16/fp32 (de-risked -- gradients keep full precision, only
    the forward matmul actually gets the fp8 speed path)."""

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return fp8_linear_forward(x, weight)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_x = grad_out @ weight
        grad_w = grad_out.transpose(-2, -1) @ x if x.dim() == 2 else torch.einsum("...o,...i->oi", grad_out, x)
        return grad_x, grad_w


def relative_l2_error(ref: torch.Tensor, approx: torch.Tensor) -> float:
    return ((ref - approx).norm() / ref.norm().clamp_min(1e-12)).item()


def run():
    torch.manual_seed(0)

    # Realistic shapes/scales for this model: MLP c_fc is [4*n_embd, n_embd], real
    # n_embd=640 at the locked DEPTH=6/ASPECT_RATIO=96 config.
    n_embd = 640
    hidden = 4 * n_embd
    init_scale = 1.0  # matches train.py's INIT_SCALE default
    s = init_scale * (3.0 ** 0.5) * n_embd ** -0.5

    print("=== Forward equivalence, real init-scale weights, varied activation magnitude ===")
    weight = torch.empty(hidden, n_embd).uniform_(-s, s)
    for act_scale in [0.1, 1.0, 3.0, 10.0]:
        x = torch.randn(32, n_embd) * act_scale  # order-1-ish post-RMSNorm activation regime

        y_ref = F.linear(x, weight)
        y_fp8 = fp8_linear_forward(x, weight)
        err = relative_l2_error(y_ref, y_fp8)
        print(f"  act_scale={act_scale:5.1f}: forward rel_l2_err={err:.4f}")

    print()
    print("=== Backward gradient equivalence (bf16-backward scheme) ===")
    x = torch.randn(32, n_embd, requires_grad=True)
    weight_req = weight.clone().requires_grad_(True)

    # Reference: plain fp32 matmul, real autograd
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weight_req.detach().clone().requires_grad_(True)
    y_ref = F.linear(x_ref, w_ref)
    y_ref.sum().backward()

    # fp8-forward, bf16-backward via the custom autograd Function
    x_fp8 = x.detach().clone().requires_grad_(True)
    w_fp8 = weight_req.detach().clone().requires_grad_(True)
    y_fp8 = Fp8LinearFn.apply(x_fp8, w_fp8)
    y_fp8.sum().backward()

    grad_x_err = relative_l2_error(x_ref.grad, x_fp8.grad)
    grad_w_err = relative_l2_error(w_ref.grad, w_fp8.grad)
    print(f"  grad_x rel_l2_err={grad_x_err:.4f}")
    print(f"  grad_w rel_l2_err={grad_w_err:.4f}")
    print(f"  grad has NaN: x={torch.isnan(x_fp8.grad).any().item()} w={torch.isnan(w_fp8.grad).any().item()}")

    print()
    print("=== Numerical-stability sweep: overflow/NaN check at extreme activation scales ===")
    for act_scale in [1e-3, 1e2, 1e4, 1e6]:
        x = torch.randn(8, n_embd) * act_scale
        y = fp8_linear_forward(x, weight)
        bad = torch.isnan(y).any() or torch.isinf(y).any()
        print(f"  act_scale={act_scale:8.0e}: nan_or_inf={bool(bad)}")


if __name__ == "__main__":
    run()
