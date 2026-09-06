"""
Phase 1 of the YAT-kernel / SLAY attention research track (research/yat-slay-attention-research.md).

Standalone, CPU-only, does NOT touch train.py or prepare.py. Pure numerical-correctness
prototype, matching this session's established discipline: build the exact reference
first, verify the fast/approximate version against it, before ever touching real
hardware.

Implements, from the papers directly (arXiv:2603.12276, arXiv:2602.04915):

1. `exact_yat_attention_softmax` -- paper 1's plain E-product + real softmax, no
   spherical constraint. The best-performing quadratic variant in the SLAY paper's own
   Table 5 ("Yat (Exact)": val loss 4.5747, beats standard softmax's 4.6417).
2. `exact_spherical_yat_attention` -- L2-normalize Q/K, compute the spherical E-product
   exactly over the full LxL matrix, causal-masked, then a normalized-kernel-weighted
   average (sum-and-divide, NOT softmax -- this is the form SLAY's Eq. 11 approximates,
   and what the paper itself uses as SLAY's ground-truth reference in Table 2/5, "Yat
   (Spherical)": val loss 4.7180, worse than plain exact Yat).
3. `slay_attention` -- the O(L) linear-time approximation: spherical normalization,
   degree-2 polynomial feature map for (q^Tk)^2, positive random features (Performer/
   FAVOR+-style) for the exponential term at each of R Gauss-Laguerre quadrature nodes,
   fused via literal Kronecker product (no sketching -- this prototype targets
   correctness, not speed), causal via cumulative sum over the sequence dimension.

Verification metric matches the paper's own (Table 2): relative L2 error and cosine
similarity of SLAY's *output* against exact_spherical_yat_attention's output, at fixed
feature budgets.
"""
import numpy as np
import torch
import torch.nn.functional as F


def exact_yat_attention_softmax(q, k, v, eps=1e-5, causal=True):
    """Paper 1's plain E-product + real softmax. No spherical constraint.
    q, k, v: [B, H, L, D]
    """
    alignment = torch.matmul(q, k.transpose(-2, -1))  # [B,H,L,L], q^T k
    numerator = alignment.square()
    q_norm = q.square().sum(-1, keepdim=True)  # [B,H,L,1]
    k_norm = k.square().sum(-1, keepdim=True).transpose(-2, -1)  # [B,H,1,L]
    distance_sq = (q_norm - 2 * alignment + k_norm).clamp_min(0.0)
    denominator = distance_sq + eps
    scores = numerator / denominator
    if causal:
        L = q.shape[-2]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


def exact_spherical_yat_attention(q, k, v, eps=1e-5, causal=True):
    """SLAY's own ground-truth reference: spherically-constrained E-product, exact
    O(L^2) computation, normalized-kernel-weighted average (Eq. 11's numerator/
    denominator form, applied directly to the exact E_sph matrix instead of the
    approximate Psi features)."""
    q_hat = F.normalize(q, dim=-1)
    k_hat = F.normalize(k, dim=-1)
    x = torch.matmul(q_hat, k_hat.transpose(-2, -1))  # [B,H,L,L], cos similarity
    C = 2.0 + eps
    scores = x.square() / (C - 2 * x)  # E_sph, Eq. 5
    if causal:
        L = q.shape[-2]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, 0.0)  # zero, not -inf: this is a direct weighted average, not softmax
    numer = scores @ v
    denom = scores.sum(-1, keepdim=True) + eps
    return numer / denom


def gauss_laguerre_nodes(R):
    """R-point Gauss-Laguerre quadrature nodes/weights for integral_0^inf e^{-t} f(t) dt."""
    t, alpha = np.polynomial.laguerre.laggauss(R)
    return torch.tensor(t, dtype=torch.float32), torch.tensor(alpha, dtype=torch.float32)


def poly_feature_exact(u):
    """Exact degree-2 polynomial feature map phi_poly(u) = vec(u u^T), giving
    <phi_poly(a), phi_poly(b)> = (a^T b)^2 exactly. O(d^2) -- fine for a correctness
    prototype at small d; the paper's real implementation uses low-rank anchor features
    instead (Sec 2.4.1) for the O(dP) cost, not needed here."""
    outer = u.unsqueeze(-1) * u.unsqueeze(-2)  # [..., d, d]
    return outer.reshape(*u.shape[:-1], -1)  # [..., d*d]


def prf_feature(u, s, omega):
    """Positive random features (Choromanski et al. 2021 / Performer) for e^{2s u^Tv} on
    unit-norm u. omega: [D, d] ~ N(0, I_d). Returns [..., D].
    phi_PRF(u; s) = (1/sqrt(D)) exp(sqrt(2s) omega^T u - s)
    """
    D = omega.shape[0]
    proj = torch.matmul(u, omega.t()) * torch.sqrt(2 * s)  # [..., D]
    return torch.exp(proj - s) / torch.sqrt(torch.tensor(float(D)))


def slay_attention(q, k, v, eps=1e-5, R=6, prf_dim=64, causal=True, seed=0):
    """SLAY's O(L) linear-time approximation. No sketching (literal Kronecker product) --
    this prototype targets correctness verification, not speed; feature dim grows as
    R * d^2 * prf_dim, fine for tiny toy shapes.
    """
    gen = torch.Generator().manual_seed(seed)
    q_hat = F.normalize(q, dim=-1)
    k_hat = F.normalize(k, dim=-1)
    d = q_hat.shape[-1]

    C = 2.0 + eps
    # Bernstein/Laplace representation: 1/(C-2x) = int_0^inf e^{-s(C-2x)} ds
    #                                             = int_0^inf e^{-sC} e^{2sx} ds
    # change of vars t = Cs -> Gauss-Laguerre quadrature over int_0^inf e^{-t} h(t/C) dt/C
    t_nodes, alpha_weights = gauss_laguerre_nodes(R)
    s_nodes = t_nodes / C           # s_r
    w_nodes = alpha_weights / C     # w_r (already include the 1/C factor)

    phi_poly_q = poly_feature_exact(q_hat)  # [B,H,L,d^2]
    phi_poly_k = poly_feature_exact(k_hat)

    psi_q_parts = []
    psi_k_parts = []
    for r in range(R):
        omega = torch.randn(prf_dim, d, generator=gen)
        s_r = s_nodes[r]
        w_r = w_nodes[r]
        phi_prf_q = prf_feature(q_hat, s_r, omega)  # [B,H,L,prf_dim]
        phi_prf_k = prf_feature(k_hat, s_r, omega)
        # kronecker product per position: outer(phi_poly, phi_prf), flattened
        kron_q = (phi_poly_q.unsqueeze(-1) * phi_prf_q.unsqueeze(-2)).flatten(-2)  # [...,d^2*prf_dim]
        kron_k = (phi_poly_k.unsqueeze(-1) * phi_prf_k.unsqueeze(-2)).flatten(-2)
        scale = torch.sqrt(w_r.clamp_min(0.0))
        psi_q_parts.append(scale * kron_q)
        psi_k_parts.append(scale * kron_k)

    psi_q = torch.cat(psi_q_parts, dim=-1)  # [B,H,L,m], m = R * d^2 * prf_dim
    psi_k = torch.cat(psi_k_parts, dim=-1)
    # x^2 factor from Eq. 8 (the polynomial part is NOT inside the Laplace integral --
    # it multiplies the whole thing): already baked into phi_poly via poly_feature_exact,
    # since <phi_poly(q),phi_poly(k)> = x^2 exactly. Combined with the PRF's
    # <phi_prf_q,phi_prf_k>=e^{2s x}, the kron product's inner product gives
    # w_r * x^2 * e^{2 s_r x} per node -- exactly the quadrature term. Sum over r via
    # concatenation + a single inner product = the full quadrature sum. Correct.

    if not causal:
        numer = psi_q @ (psi_k.transpose(-2, -1) @ v)
        denom = psi_q @ psi_k.sum(-2, keepdim=True).transpose(-2, -1)
        denom = denom + eps
        return numer / denom.clamp_min(eps)

    # Causal: cumulative sum over the key/value dimension. O(L*m*d_v) memory -- fine for
    # a tiny toy-shape correctness check; NOT how this would be done efficiently on real
    # hardware (this is exactly the "sequential recurrence" throughput concern flagged
    # in the research doc -- a real NKI kernel would need a smarter blocked/chunked cumsum,
    # out of scope for this correctness-only prototype).
    B, H, L, m = psi_k.shape
    d_v = v.shape[-1]
    kv_outer = psi_k.unsqueeze(-1) * v.unsqueeze(-2)  # [B,H,L,m,d_v]
    kv_cumsum = torch.cumsum(kv_outer, dim=2)  # running sum over positions <= i
    k_cumsum = torch.cumsum(psi_k, dim=2)  # [B,H,L,m]

    numer = torch.einsum("bhlm,bhlmd->bhld", psi_q, kv_cumsum)  # [B,H,L,d_v]
    denom = torch.einsum("bhlm,bhlm->bhl", psi_q, k_cumsum).unsqueeze(-1) + eps
    return numer / denom.clamp_min(eps)


def relative_l2_error(exact, approx):
    return (exact - approx).norm() / exact.norm().clamp_min(1e-12)


def cosine_sim(exact, approx):
    e = exact.flatten()
    a = approx.flatten()
    return (e @ a) / (e.norm() * a.norm()).clamp_min(1e-12)


def run():
    torch.manual_seed(0)
    B, H, L, D = 2, 2, 16, 8
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    y_softmax_exact = exact_yat_attention_softmax(q, k, v)
    y_spherical_exact = exact_spherical_yat_attention(q, k, v)

    print(f"exact softmax-YAT output norm: {y_softmax_exact.norm().item():.4f}")
    print(f"exact spherical-YAT output norm: {y_spherical_exact.norm().item():.4f}")
    print(f"gradient check (softmax-YAT): ", end="")
    q_req = q.clone().requires_grad_(True)
    loss = exact_yat_attention_softmax(q_req, k, v).sum()
    loss.backward()
    print(f"grad has NaN: {torch.isnan(q_req.grad).any().item()}, "
          f"grad norm: {q_req.grad.norm().item():.4f}")

    print()
    for R, prf_dim in [(3, 32), (6, 64), (10, 128)]:
        y_slay = slay_attention(q, k, v, R=R, prf_dim=prf_dim, seed=0)
        err = relative_l2_error(y_spherical_exact, y_slay).item()
        cos = cosine_sim(y_spherical_exact, y_slay).item()
        print(f"SLAY(R={R:2d}, prf_dim={prf_dim:3d}): "
              f"rel_l2_err={err:.4f}  cos_sim={cos:.4f}  "
              f"(paper's own best config: rel_l2=0.527, cos=0.850)")

    print()
    print("Gradient sanity on SLAY:")
    q_req2 = q.clone().requires_grad_(True)
    loss2 = slay_attention(q_req2, k, v, R=6, prf_dim=64, seed=0).sum()
    loss2.backward()
    print(f"grad has NaN: {torch.isnan(q_req2.grad).any().item()}, "
          f"grad norm: {q_req2.grad.norm().item():.4f}")

    print()
    print("Numerical-stability sweep (varied input scale, checking for NaN/inf):")
    for scale in [1e-3, 1e-1, 1.0, 10.0, 40.0]:
        qs, ks, vs = q * scale, k * scale, v * scale
        y_exact = exact_spherical_yat_attention(qs, ks, vs)
        y_slay = slay_attention(qs, ks, vs, R=6, prf_dim=64, seed=0)
        bad_exact = torch.isnan(y_exact).any() or torch.isinf(y_exact).any()
        bad_slay = torch.isnan(y_slay).any() or torch.isinf(y_slay).any()
        err = relative_l2_error(y_exact, y_slay).item() if not (bad_exact or bad_slay) else float("nan")
        print(f"  scale={scale:6.3f}: exact_bad={bool(bad_exact)} slay_bad={bool(bad_slay)} rel_l2_err={err:.4f}")


if __name__ == "__main__":
    run()
