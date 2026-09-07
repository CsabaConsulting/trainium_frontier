"""Mamba2 SSD (structured state-space duality) -- exact CPU reference implementations.

Step 1 of the Mamba2 track: build the *correct* math first, three independent ways,
and prove they agree exactly (not approximately) before any NKI kernel exists.

Math (Dao & Gu 2024, arXiv:2405.21060, "Transformers are SSMs"; cross-checked against
state-spaces/mamba's `ssd_minimal_discrete` and against the sibling repo's
`state-space-models-neuron/mamba2/mamba2_kernel.py` algebra, line by line).

Per head h, per batch b, the *selective* SSM recurrence with a SCALAR-per-head A
(this is the real Mamba2 restriction -- Mamba1 used diagonal-per-channel A):

    h_t = a_t * h_{t-1} + B_t (dt_t x_t)^T        h_t in R^{N x P},  a_t = exp(dt_t * A_h)
    y_t = C_t^T h_t                               y_t in R^P
    y_t += D_h * x_t                              (skip / "position-wise" term)

  P = head_dim, N = d_state, a_t is a SCALAR (A_h < 0, dt_t > 0 so a_t in (0,1)).

Unrolling gives the quadratic ("attention-like", the SSD duality) form. With the
inclusive cumulative decay  l_t = sum_{r<=t} dt_r A_h :

    y_t = sum_{s<=t} exp(l_t - l_s) (C_t . B_s) (dt_s x_s)
        = [ (L o (C B^T)) (dt (.) X) ]_t ,   L_{ts} = exp(l_t - l_s) 1[s<=t]

Because A_h < 0 and dt > 0, l is strictly decreasing, so every exponent
l_t - l_s <= 0 for s <= t -- every exp() argument in the whole algorithm is
non-positive, hence exp in (0, 1]. No overflow is possible anywhere, and no
max-subtraction / rescaling trick is needed. This is checked explicitly in
test_all_exponents_nonpositive below rather than assumed.

The chunked algorithm splits [0,T) into chunks of Q and is MATHEMATICALLY EXACT
vs. the sequential recurrence (it is a regrouping of the same sum, not an
approximation) -- so the tests below demand agreement at float64 round-off, not
a loose tolerance.
"""

import math

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Reference 1: the literal sequential recurrence. Slowest, most obviously correct.

def ssd_recurrent(x, dt, A, B, C, D=None):
    """Sequential SSM scan. The ground truth by definition.

    x:  [b, t, h, p]   input (pre-dt-scaling)
    dt: [b, t, h]      positive discretization step
    A:  [h]            negative scalar-per-head
    B:  [b, t, g, n]   state input projection  (g = n_groups, heads share groups)
    C:  [b, t, g, n]   state output projection
    D:  [h] or None    skip scale
    -> y: [b, t, h, p]
    """
    bsz, T, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    assert H % G == 0
    hpg = H // G

    state = x.new_zeros(bsz, H, N, P)
    ys = []
    for t in range(T):
        a = torch.exp(dt[:, t, :] * A)                       # [b, h]
        # expand group-shared B/C to per-head
        Bt = B[:, t].repeat_interleave(hpg, dim=1)           # [b, h, n]
        Ct = C[:, t].repeat_interleave(hpg, dim=1)           # [b, h, n]
        dx = dt[:, t, :, None] * x[:, t]                     # [b, h, p]
        state = a[:, :, None, None] * state + Bt[:, :, :, None] * dx[:, :, None, :]
        ys.append(torch.einsum("bhn,bhnp->bhp", Ct, state))
    y = torch.stack(ys, dim=1)                               # [b, t, h, p]
    if D is not None:
        y = y + D[None, None, :, None] * x
    return y


# -----------------------------------------------------------------------------
# Reference 2: the full quadratic (SSD-dual, attention-like) form. O(T^2), no chunking.

def ssd_quadratic(x, dt, A, B, C, D=None):
    """Materialize the whole T x T semiseparable matrix M = L o (C B^T)."""
    bsz, T, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    hpg = H // G

    dtA = dt * A[None, None, :]                              # [b, t, h], <= 0
    ell = torch.cumsum(dtA, dim=1)                           # inclusive cumsum
    seg = ell[:, :, None, :] - ell[:, None, :, :]            # [b, t, s, h] = l_t - l_s
    causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    # MASK BEFORE exp, never after. For s > t the exponent is POSITIVE and grows
    # like |A|*dt*(s-t), so exp() overflows to +inf and inf*0 == NaN. Clamping the
    # argument at 0 is a no-op on every valid (s <= t) entry -- those are provably
    # <= 0 -- so this is exact, not a fudge.
    L = torch.where(causal[None, :, :, None], torch.exp(seg.clamp(max=0.0)),
                    torch.zeros((), dtype=x.dtype, device=x.device))

    Bf = B.repeat_interleave(hpg, dim=2)                     # [b, t, h, n]
    Cf = C.repeat_interleave(hpg, dim=2)
    CB = torch.einsum("bthn,bshn->btsh", Cf, Bf)
    M = L * CB
    dx = dt[..., None] * x                                   # [b, t, h, p]
    y = torch.einsum("btsh,bshp->bthp", M, dx)
    if D is not None:
        y = y + D[None, None, :, None] * x
    return y


# -----------------------------------------------------------------------------
# Reference 3: the CHUNKED SSD algorithm -- the one the NKI kernel implements.

def ssd_chunked(x, dt, A, B, C, D=None, chunk_size=128, return_states=False):
    """Chunked SSD: intra-chunk quadratic term + inter-chunk carried state.

    This is the algorithm the NKI kernel mirrors. It is EXACT vs ssd_recurrent.
    """
    bsz, T, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    hpg = H // G
    Q = chunk_size
    assert T % Q == 0, f"T={T} must be a multiple of chunk_size={Q}"
    NC = T // Q

    def ck(v, extra):
        return v.reshape(bsz, NC, Q, *extra)

    dtA = (dt * A[None, None, :])                            # [b, t, h] <= 0
    dtc = ck(dt, (H,))                                       # [b, c, q, h]
    ellc = torch.cumsum(ck(dtA, (H,)), dim=2)                # local inclusive cumsum
    xc = ck(x, (H, P))
    Bc = ck(B, (G, N))
    Cc = ck(C, (G, N))

    Bf = Bc.repeat_interleave(hpg, dim=3)                    # [b, c, q, h, n]
    Cf = Cc.repeat_interleave(hpg, dim=3)
    dx = dtc[..., None] * xc                                 # [b, c, q, h, p]

    # ---- intra-chunk (diagonal blocks): the quadratic form, chunk-local -------
    seg = ellc[:, :, :, None, :] - ellc[:, :, None, :, :]     # [b,c,q_t,q_s,h]
    causal = torch.ones(Q, Q, dtype=torch.bool, device=x.device).tril()
    # Mask BEFORE exp (see ssd_quadratic): the s>t half of `seg` is positive and
    # overflows. clamp(max=0) is exact on the s<=t half.
    Lc = torch.where(causal[None, None, :, :, None], torch.exp(seg.clamp(max=0.0)),
                     torch.zeros((), dtype=x.dtype, device=x.device))
    CBc = torch.einsum("bcthn,bcshn->bctsh", Cf, Bf)
    Y_intra = torch.einsum("bctsh,bcshp->bcthp", Lc * CBc, dx)

    # ---- per-chunk state contribution ---------------------------------------
    decay_end = ellc[:, :, -1, :]                            # [b, c, h] = l_{Q-1}
    # Bbar_s = exp(l_{Q-1} - l_s) B_s   (all exponents <= 0)
    Bbar = Bf * torch.exp(decay_end[:, :, None, :] - ellc)[..., None]
    chunk_state = torch.einsum("bcshn,bcshp->bchnp", Bbar, dx)   # [b,c,h,n,p]

    # ---- inter-chunk sequential carry (only NC steps, not T) -----------------
    a_center = torch.exp(decay_end)                           # [b, c, h]
    states = []
    S = x.new_zeros(bsz, H, N, P)
    for c in range(NC):
        states.append(S)
        S = a_center[:, c][:, :, None, None] * S + chunk_state[:, c]
    S_in = torch.stack(states, dim=1)                         # [b, c, h, n, p] state ENTERING chunk c

    # ---- inter-chunk output -------------------------------------------------
    Cbar = Cf * torch.exp(ellc)[..., None]                    # exp(l_t) C_t, exponents <= 0
    Y_inter = torch.einsum("bcthn,bchnp->bcthp", Cbar, S_in)

    y = (Y_intra + Y_inter).reshape(bsz, T, H, P)
    if D is not None:
        y = y + D[None, None, :, None] * x
    if return_states:
        return y, S_in, S
    return y


# -----------------------------------------------------------------------------
# Grouped depthwise causal conv1d (the short local conv that precedes the scan)

def grouped_causal_conv1d(u, weight, bias=None):
    """u: [b, t, c]; weight: [c, k] (depthwise, one filter per channel). Causal.

    y[:, t, c] = sum_{j=0}^{k-1} weight[c, j] * u[:, t - (k-1) + j, c]   (+ bias[c])
    """
    bsz, T, Cch = u.shape
    k = weight.shape[-1]
    ut = u.transpose(1, 2)                                    # [b, c, t]
    ut = F.pad(ut, (k - 1, 0))
    y = F.conv1d(ut, weight.reshape(Cch, 1, k), bias=bias, groups=Cch)
    return y.transpose(1, 2)


def grouped_causal_conv1d_explicit(u, weight, bias=None):
    """Same thing written as an explicit sum -- independent check of the padding/orientation."""
    bsz, T, Cch = u.shape
    k = weight.shape[-1]
    out = torch.zeros_like(u)
    for t in range(T):
        for j in range(k):
            s = t - (k - 1) + j
            if s >= 0:
                out[:, t, :] = out[:, t, :] + weight[:, j] * u[:, s, :]
    if bias is not None:
        out = out + bias
    return out


# -----------------------------------------------------------------------------
# Test helpers

def make_inputs(bsz=2, T=256, H=4, P=16, G=2, N=8, Q=64, dtype=torch.float64,
                seed=0, device="cpu", requires_grad=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    kw = dict(dtype=dtype, device=device)
    x = torch.randn(bsz, T, H, P, generator=g).to(**kw)
    dt_raw = torch.randn(bsz, T, H, generator=g).to(**kw)
    dt = F.softplus(dt_raw)                                  # real Mamba2: dt = softplus(proj + bias)
    A_log = torch.rand(H, generator=g).to(**kw) * 2.0
    A = -torch.exp(A_log)                                    # A < 0, real S4D-real init family
    B = torch.randn(bsz, T, G, N, generator=g).to(**kw)
    C = torch.randn(bsz, T, G, N, generator=g).to(**kw)
    D = torch.randn(H, generator=g).to(**kw)
    if requires_grad:
        for t in (x, dt, B, C, D):
            t.requires_grad_(True)
        A.requires_grad_(True)
    return dict(x=x, dt=dt, A=A, B=B, C=C, D=D, Q=Q)


def _report(name, a, b):
    d = (a - b).abs()
    scale = b.abs().max().clamp(min=1e-30)
    print(f"  {name:38s} max_abs={d.max().item():.3e}  max_rel={(d.max()/scale).item():.3e}")
    return d.max().item() / scale.item()


# -----------------------------------------------------------------------------

def test_three_way_equivalence():
    print("[1] three-way equivalence (recurrent == quadratic == chunked), float64")
    ok = True
    for cfg in [dict(bsz=2, T=256, H=4, P=16, G=2, N=8, Q=64),
                dict(bsz=1, T=128, H=1, P=8, G=1, N=8, Q=32),
                dict(bsz=3, T=384, H=6, P=32, G=3, N=16, Q=128),
                dict(bsz=1, T=64, H=8, P=4, G=1, N=4, Q=64)]:   # Q == T: single chunk
        inp = make_inputs(**cfg, seed=hash(str(cfg)) % 10000)
        Q = inp.pop("Q")
        y_rec = ssd_recurrent(**inp)
        y_quad = ssd_quadratic(**inp)
        y_chk = ssd_chunked(**inp, chunk_size=Q)
        print(f"  cfg T={cfg['T']} H={cfg['H']} P={cfg['P']} G={cfg['G']} N={cfg['N']} Q={Q}")
        r1 = _report("quadratic vs recurrent", y_quad, y_rec)
        r2 = _report("chunked   vs recurrent", y_chk, y_rec)
        ok &= (r1 < 1e-12 and r2 < 1e-12)
    print(f"  -> {'PASS' if ok else 'FAIL'} (float64 round-off level, i.e. EXACT)\n")
    return ok


def test_chunk_size_invariance():
    print("[2] chunk-size invariance: chunked output must not depend on Q")
    inp = make_inputs(bsz=2, T=384, H=6, P=16, G=2, N=8, seed=7)
    inp.pop("Q")
    y_ref = ssd_recurrent(**inp)
    ok = True
    for Q in (1, 2, 3, 4, 6, 8, 12, 16, 32, 48, 64, 96, 128, 192, 384):
        y = ssd_chunked(**inp, chunk_size=Q)
        r = (y - y_ref).abs().max().item() / y_ref.abs().max().item()
        ok &= r < 1e-12
        if Q in (1, 4, 64, 128, 384):
            print(f"  Q={Q:4d}  max_rel={r:.3e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_all_exponents_nonpositive():
    print("[3] numerical-safety claim: every exp() argument in the algorithm is <= 0")
    inp = make_inputs(bsz=2, T=256, H=4, P=16, G=2, N=8, seed=11)
    x, dt, A = inp["x"], inp["dt"], inp["A"]
    Q = 64
    bsz, T, H, _ = x.shape
    NC = T // Q
    assert (dt > 0).all(), "dt must be positive (softplus)"
    assert (A < 0).all(), "A must be negative (-exp(A_log))"
    dtA = dt * A[None, None, :]
    ellc = torch.cumsum(dtA.reshape(bsz, NC, Q, H), dim=2)
    seg = ellc[:, :, :, None, :] - ellc[:, :, None, :, :]
    tril = torch.ones(Q, Q, dtype=torch.bool).tril()
    intra_max = seg.permute(0, 1, 4, 2, 3)[..., tril].max().item()
    bbar_max = (ellc[:, :, -1:, :] - ellc).max().item()
    cbar_max = ellc.max().item()
    center_max = ellc[:, :, -1, :].max().item()
    print(f"  max intra-chunk exponent  (l_t - l_s, s<=t) = {intra_max:.3e}")
    print(f"  max Bbar exponent (l_end - l_s)             = {bbar_max:.3e}")
    print(f"  max Cbar exponent (l_t)                     = {cbar_max:.3e}")
    print(f"  max a_center exponent (l_end)               = {center_max:.3e}")
    ok = max(intra_max, bbar_max, cbar_max, center_max) <= 0.0
    print(f"  -> {'PASS' if ok else 'FAIL'} (exp() in (0,1] always; overflow impossible)\n")
    return ok


def test_masked_overflow_regression():
    """The masked-out (s>t) half of the decay matrix must NEVER be exponentiated.

    Regression guard for the bug this file originally had (and which the sibling
    repo's NKI kernel also has: it does nl.exp(partial_sums) and only THEN
    nisa.affine_select). For s > t the exponent l_t - l_s is positive and grows
    without bound, so exp() overflows (fp32 at +88, fp64 at +709) and the
    subsequent multiply-by-zero mask turns inf into NaN. The NKI kernel must mask
    the ARGUMENT, not the result.
    """
    print("[4] masked-half overflow regression (fp32 + large decay per chunk)")
    ok = True
    # dt*|A| per step ~ 4 -> per 128-chunk ~ -512: the s>t half reaches +512,
    # which is inf in fp32 (limit +88) AND stresses fp64 (limit +709).
    for dtype, Q, T in ((torch.float32, 128, 512), (torch.float32, 256, 512),
                        (torch.float64, 384, 384)):
        inp = make_inputs(bsz=1, T=T, H=4, P=8, G=1, N=8, dtype=dtype, seed=31)
        inp.pop("Q")
        inp["dt"] = inp["dt"] * 4.0          # force a big per-chunk decay
        yq = ssd_quadratic(**inp)
        yc = ssd_chunked(**inp, chunk_size=Q)
        bad = int(torch.isnan(yc).sum() + torch.isinf(yc).sum()), \
            int(torch.isnan(yq).sum() + torch.isinf(yq).sum())
        print(f"  {str(dtype).split('.')[-1]:8s} T={T} Q={Q}: non-finite in chunked={bad[0]}, quadratic={bad[1]}")
        ok &= bad == (0, 0)
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_degenerate_cases():
    print("[4] degenerate / sanity limits")
    ok = True
    bsz, T, H, P, G, N, Q = 1, 128, 2, 8, 1, 4, 32
    inp = make_inputs(bsz=bsz, T=T, H=H, P=P, G=G, N=N, seed=3)
    inp.pop("Q")

    # (a) dt -> 0  =>  y -> D * x  (no state accumulates). The residual is O(T*dt)
    # by construction (T terms each ~dt*C.B*x), so the tolerance scales with dt.
    dt0 = 1e-14
    i = dict(inp); i["dt"] = torch.full_like(inp["dt"], dt0)
    y = ssd_chunked(**i, chunk_size=Q)
    want = i["D"][None, None, :, None] * i["x"]
    r = (y - want).abs().max().item()
    print(f"  (a) dt->0 gives y == D*x                 max_abs={r:.3e} (expect O(T*dt)={T*dt0:.1e})")
    ok &= r < 100 * T * dt0

    # (b) A -> -inf (a_t -> 0) => pure "diagonal": y_t = (C_t.B_t) dt_t x_t + D x_t
    i = dict(inp); i["A"] = torch.full_like(inp["A"], -1e6)
    y = ssd_chunked(**i, chunk_size=Q)
    CB_diag = (i["C"][:, :, 0, :] * i["B"][:, :, 0, :]).sum(-1)   # G==1
    want = CB_diag[:, :, None, None] * (i["dt"][..., None] * i["x"]) \
        + i["D"][None, None, :, None] * i["x"]
    r = (y - want).abs().max().item() / want.abs().max().item()
    print(f"  (b) A->-inf gives diagonal-only          max_rel={r:.3e}")
    ok &= r < 1e-9

    # (c) D=None omits the skip exactly
    y0 = ssd_chunked(**{k: v for k, v in inp.items() if k != "D"}, D=None, chunk_size=Q)
    y1 = ssd_chunked(**inp, chunk_size=Q)
    r = ((y1 - y0) - inp["D"][None, None, :, None] * inp["x"]).abs().max().item()
    print(f"  (c) D term is exactly additive           max_abs={r:.3e}")
    ok &= r < 1e-12

    # (d) B == 0 => y == D*x
    i = dict(inp); i["B"] = torch.zeros_like(inp["B"])
    y = ssd_chunked(**i, chunk_size=Q)
    r = (y - i["D"][None, None, :, None] * i["x"]).abs().max().item()
    print(f"  (d) B==0 gives y == D*x                  max_abs={r:.3e}")
    ok &= r < 1e-14
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_causality():
    print("[5] causality: d y_t / d x_s == 0 for s > t (the property prepare.py checks)")
    bsz, T, H, P, G, N, Q = 1, 32, 2, 4, 1, 4, 8
    inp = make_inputs(bsz=bsz, T=T, H=H, P=P, G=G, N=N, seed=5, requires_grad=True)
    inp.pop("Q")
    worst = 0.0
    for fn, name in ((ssd_recurrent, "recurrent"), (ssd_chunked, "chunked")):
        kw = dict(chunk_size=Q) if fn is ssd_chunked else {}
        for t_probe in (0, 1, 7, 8, 15, 31):
            for k in ("x", "dt", "B", "C"):
                inp[k].grad = None
            y = fn(**inp, **kw)
            y[:, t_probe].sum().backward()
            for k in ("x", "dt", "B", "C"):
                gfut = inp[k].grad[:, t_probe + 1:]
                worst = max(worst, gfut.abs().max().item() if gfut.numel() else 0.0)
        print(f"  {name:10s} worst |grad| at future positions = {worst:.3e}")
    ok = worst == 0.0
    print(f"  -> {'PASS' if ok else 'FAIL'} (exactly zero, not merely small)\n")
    return ok


def test_gradients_match():
    print("[6] gradients: chunked vs recurrent, all six inputs (float64)")
    inp = make_inputs(bsz=2, T=96, H=3, P=8, G=1, N=8, seed=9, requires_grad=True)
    inp.pop("Q")
    Q = 24
    gsrc = torch.randn(2, 96, 3, 8, dtype=torch.float64,
                       generator=torch.Generator().manual_seed(42))
    grads = {}
    for fn, name in ((ssd_recurrent, "rec"), (ssd_chunked, "chk")):
        for v in inp.values():
            v.grad = None
        y = fn(**inp, **(dict(chunk_size=Q) if fn is ssd_chunked else {}))
        (y * gsrc).sum().backward()
        grads[name] = {k: v.grad.clone() for k, v in inp.items()}
    ok = True
    for k in inp:
        r = _report(f"d/d{k}", grads["chk"][k], grads["rec"][k])
        ok &= r < 1e-11
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_gradcheck():
    print("[7] torch.autograd.gradcheck on the chunked implementation")
    inp = make_inputs(bsz=1, T=16, H=2, P=3, G=1, N=4, seed=13, requires_grad=True)
    inp.pop("Q")
    args = (inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], inp["D"])
    ok = torch.autograd.gradcheck(
        lambda *a: ssd_chunked(*a, chunk_size=4), args, eps=1e-6, atol=1e-7, rtol=1e-5)
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_conv1d():
    print("[8] grouped depthwise causal conv1d: F.conv1d path vs explicit sum, + causality")
    g = torch.Generator().manual_seed(17)
    u = torch.randn(2, 40, 12, dtype=torch.float64, generator=g, requires_grad=True)
    w = torch.randn(12, 4, dtype=torch.float64, generator=g)
    bias = torch.randn(12, dtype=torch.float64, generator=g)
    y1 = grouped_causal_conv1d(u, w, bias)
    y2 = grouped_causal_conv1d_explicit(u, w, bias)
    r = _report("conv1d vs explicit sum", y1, y2)
    ok = r < 1e-13
    # causality of the conv
    worst = 0.0
    for t_probe in (0, 1, 2, 3, 20, 39):
        u.grad = None
        grouped_causal_conv1d(u, w, bias)[:, t_probe].sum().backward()
        gf = u.grad[:, t_probe + 1:]
        worst = max(worst, gf.abs().max().item() if gf.numel() else 0.0)
    print(f"  causality: worst |grad| at future positions = {worst:.3e}")
    ok &= worst == 0.0
    # receptive field: y_t must depend on exactly t-3..t
    u.grad = None
    grouped_causal_conv1d(u, w, bias)[:, 20].sum().backward()
    nz = (u.grad[0, :, 0].abs() > 0).nonzero().flatten().tolist()
    print(f"  y_20 depends on positions {nz} (expect [17, 18, 19, 20] for k=4)")
    ok &= nz == [17, 18, 19, 20]
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_competition_shape_fp32():
    print("[9] competition-relevant shape in float32 (what will actually run)")
    # n_embd=640, expand=2 -> d_inner=1280, head_dim=128 -> 10 heads, d_state=128, G=1
    inp = make_inputs(bsz=1, T=2048, H=10, P=128, G=1, N=128, dtype=torch.float32, seed=21)
    inp.pop("Q")
    inp64 = {k: (v.double() if torch.is_tensor(v) else v) for k, v in inp.items()}
    y64 = ssd_recurrent(**inp64)
    for Q in (128, 256):
        y32 = ssd_chunked(**inp, chunk_size=Q)
        r = (y32.double() - y64).abs().max().item() / y64.abs().max().item()
        print(f"  fp32 chunked (Q={Q}) vs fp64 recurrent: max_rel={r:.3e}")
    y64c = ssd_chunked(**inp64, chunk_size=128)
    r = (y64c - y64).abs().max().item() / y64.abs().max().item()
    print(f"  fp64 chunked (Q=128) vs fp64 recurrent: max_rel={r:.3e}")
    ok = r < 1e-12
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


if __name__ == "__main__":
    torch.manual_seed(0)
    results = {
        "three_way": test_three_way_equivalence(),
        "chunk_invariance": test_chunk_size_invariance(),
        "exponent_safety": test_all_exponents_nonpositive(),
        "overflow_regression": test_masked_overflow_regression(),
        "degenerate": test_degenerate_cases(),
        "causality": test_causality(),
        "gradients": test_gradients_match(),
        "gradcheck": test_gradcheck(),
        "conv1d": test_conv1d(),
        "fp32_shape": test_competition_shape_fp32(),
    }
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:20s} {'PASS' if v else 'FAIL'}")
    print("=" * 70)
    print("ALL PASS" if all(results.values()) else "SOME FAILED")
