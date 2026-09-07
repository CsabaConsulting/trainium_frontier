"""
YAT/SLAY Phase 2: small, clearly-separated real-hardware smoke test
(research/yat-slay-attention-research.md).

Purpose ONLY: is SLAY's op sequence even throughput-competitive with the existing
fused SDPA kernel, at this competition's REAL shape? Not a training run. Does not
import train.py, does not touch the locked default config, does not run in the same
wall-clock slot as an in-flight Bucket A-E run. Forward+backward timing only.

Real shape (from train.py: DEPTH=6, ASPECT_RATIO=96 -> n_embd=640, HEAD_DIM=128 ->
n_head=5, SEQ_LEN=2048, DEVICE_BATCH_SIZE=1):
    B=1, H=5, L=2048, D=128

Back-of-envelope FLOP estimate done BEFORE this script was written (see chat/doc):
causal SDPA ~ L^2*d ~ 5.37e8 flops/head. SLAY chunked-linear cost is O(L*C*m) where
C=chunk size, m=R*P*prf_dim (P = low-rank poly-feature proj dim, replacing the exact
d^2 poly feature used ONLY in Phase 1's toy-scale correctness oracle -- d^2=16384 at
real head_dim=128 is not usable at this scale). Beating SDPA needs C*m < L*d = 262144.
At a modest R=4,P=32,prf_dim=32 (m=4096), that forces C<64 -- a tiny chunk. This script
measures the real wall-clock at a couple of (R,P,prf_dim,chunk) points to check that
prediction against real hardware, not just arithmetic.

Poly feature map: Random Maclaurin features (Kar & Karnick 2012) -- phi(u)_j =
(1/sqrt(P)) * (a_j^T u)(b_j^T u) for independent Gaussian a_j,b_j. Unbiased estimator
of (u^Tv)^2 via Sigma_j phi(u)_j*phi(v)_j, pure matmul + elementwise (Tensor + Vector
engine friendly), no FFT/hashing -- unlike Phase 1's exact vec(uu^T) which is O(d^2)
and unusable at real head_dim. This is a DIFFERENT, established random-feature
technique (not re-verified for numerical accuracy here -- that is a separate
question from this script's only job, throughput).
"""
import time

import torch
import torch.nn.functional as F


def detect_device() -> torch.device:
    if hasattr(torch, "neuron"):
        try:
            return torch.device("neuron")
        except Exception:
            pass
    return torch.device("cpu")


def sdpa_baseline(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def random_maclaurin_poly_features(u, a, b, P):
    """phi(u)_j = (1/sqrt(P)) * (a_j^T u)(b_j^T u). u: [...,d], a,b: [P,d]."""
    pa = torch.matmul(u, a.t())  # [...,P]
    pb = torch.matmul(u, b.t())  # [...,P]
    return (pa * pb) / (P ** 0.5)


def prf_feature(u_hat, s, omega):
    D = omega.shape[0]
    proj = torch.matmul(u_hat, omega.t()) * torch.sqrt(2 * s)
    return torch.exp(proj - s) / torch.sqrt(torch.tensor(float(D), device=u_hat.device))


def build_slay_features(q, k, R, P, prf_dim, device, gen):
    """Returns psi_q, psi_k: [B,H,L,m], m = R*P*prf_dim. Same Bernstein/Laplace
    Gauss-Laguerre construction as Phase 1's prototype, poly term swapped for the
    low-rank Random Maclaurin map instead of exact vec(uu^T)."""
    import numpy as np

    d = q.shape[-1]
    q_hat = F.normalize(q, dim=-1)
    k_hat = F.normalize(k, dim=-1)

    eps = 1e-5
    C = 2.0 + eps
    t_nodes, alpha_weights = np.polynomial.laguerre.laggauss(R)
    t_nodes = torch.tensor(t_nodes, dtype=torch.float32, device=device)
    alpha_weights = torch.tensor(alpha_weights, dtype=torch.float32, device=device)
    s_nodes = t_nodes / C
    w_nodes = (alpha_weights / C).clamp_min(0.0)

    a = torch.randn(P, d, generator=gen).to(device)
    b = torch.randn(P, d, generator=gen).to(device)
    phi_poly_q = random_maclaurin_poly_features(q_hat, a, b, P)  # [B,H,L,P]
    phi_poly_k = random_maclaurin_poly_features(k_hat, a, b, P)

    psi_q_parts, psi_k_parts = [], []
    for r in range(R):
        omega = torch.randn(prf_dim, d, generator=gen).to(device)
        s_r = s_nodes[r]
        scale = torch.sqrt(w_nodes[r])
        phi_prf_q = prf_feature(q_hat, s_r, omega)  # [B,H,L,prf_dim]
        phi_prf_k = prf_feature(k_hat, s_r, omega)
        kron_q = (phi_poly_q.unsqueeze(-1) * phi_prf_q.unsqueeze(-2)).flatten(-2)
        kron_k = (phi_poly_k.unsqueeze(-1) * phi_prf_k.unsqueeze(-2)).flatten(-2)
        psi_q_parts.append(scale * kron_q)
        psi_k_parts.append(scale * kron_k)
    return torch.cat(psi_q_parts, dim=-1), torch.cat(psi_k_parts, dim=-1)


def slay_chunked_causal(psi_q, psi_k, v, chunk_size, eps=1e-5):
    """Chunked/blocked causal linear attention -- the real hardware-friendly design
    flagged in the research doc, NOT Phase 1's literal torch.cumsum. Within a chunk:
    masked matmul (parallel, Tensor-Engine-friendly). Across chunks: small running
    KV-state carried sequentially, L/chunk_size steps instead of L."""
    B, H, L, m = psi_q.shape
    d_v = v.shape[-1]
    assert L % chunk_size == 0
    num_chunks = L // chunk_size

    state_kv = torch.zeros(B, H, m, d_v, device=psi_q.device, dtype=psi_q.dtype)
    state_z = torch.zeros(B, H, m, device=psi_q.device, dtype=psi_q.dtype)

    tri = torch.tril(torch.ones(chunk_size, chunk_size, device=psi_q.device, dtype=torch.bool))

    outputs = []
    for c in range(num_chunks):
        sl = slice(c * chunk_size, (c + 1) * chunk_size)
        qc, kc, vc = psi_q[:, :, sl, :], psi_k[:, :, sl, :], v[:, :, sl, :]

        # inter-chunk contribution (all prior chunks, via carried state)
        inter_numer = torch.matmul(qc, state_kv)  # [B,H,C,d_v]
        inter_denom = torch.matmul(qc, state_z.unsqueeze(-1))  # [B,H,C,1]

        # intra-chunk causal contribution via masked matmul
        scores = torch.matmul(qc, kc.transpose(-2, -1))  # [B,H,C,C]
        scores = scores.masked_fill(~tri, 0.0)
        intra_numer = torch.matmul(scores, vc)
        intra_denom = scores.sum(-1, keepdim=True)

        numer = inter_numer + intra_numer
        denom = inter_denom + intra_denom + eps
        outputs.append(numer / denom.clamp_min(eps))

        # update carried state for next chunk (sequential, but only L/chunk_size times)
        state_kv = state_kv + torch.matmul(kc.transpose(-2, -1), vc)
        state_z = state_z + kc.sum(-2)

    return torch.cat(outputs, dim=2)


def time_fwd_bwd(fn, *inputs, warmup=3, iters=10):
    for t in inputs:
        t.requires_grad_(True)
    for _ in range(warmup):
        out = fn(*inputs)
        out.sum().backward()
        for t in inputs:
            t.grad = None
    if inputs[0].device.type in ("cuda",):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*inputs)
        out.sum().backward()
        for t in inputs:
            t.grad = None
    if inputs[0].device.type in ("cuda",):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def run():
    device = detect_device()
    print(f"device: {device}")
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    B, H, L, D = 1, 5, 2048, 128  # real locked-config shape

    q = torch.randn(B, H, L, D, device=device)
    k = torch.randn(B, H, L, D, device=device)
    v = torch.randn(B, H, L, D, device=device)

    dt_sdpa = time_fwd_bwd(sdpa_baseline, q.clone(), k.clone(), v.clone())
    print(f"SDPA (causal, fused kernel):            {dt_sdpa*1000:.3f} ms/iter (fwd+bwd)")

    configs = [
        dict(R=4, P=32, prf_dim=32, chunk_size=64),
        dict(R=4, P=32, prf_dim=32, chunk_size=256),
        dict(R=3, P=16, prf_dim=16, chunk_size=64),
    ]

    for cfg in configs:
        R, P, prf_dim, chunk_size = cfg["R"], cfg["P"], cfg["prf_dim"], cfg["chunk_size"]
        m = R * P * prf_dim

        def slay_fn(qq, kk, vv):
            psi_q, psi_k = build_slay_features(qq, kk, R, P, prf_dim, device, gen)
            return slay_chunked_causal(psi_q, psi_k, vv, chunk_size)

        dt = time_fwd_bwd(slay_fn, q.clone(), k.clone(), v.clone())
        ratio = dt / dt_sdpa
        print(f"SLAY R={R} P={P} prf_dim={prf_dim} chunk={chunk_size:4d} m={m:5d}: "
              f"{dt*1000:.3f} ms/iter (fwd+bwd)  -- {ratio:.2f}x SDPA "
              f"({'SLOWER' if ratio > 1 else 'faster'})")


if __name__ == "__main__":
    run()
