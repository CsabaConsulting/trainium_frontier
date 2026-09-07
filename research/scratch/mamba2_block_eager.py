"""Mamba2 mixer block -- eager PyTorch, native-Neuron-friendly, loop-free.

Step 2 of the Mamba2 track. Takes the verified chunked SSD math from
`mamba2_ssd_ref.py` and rewrites it so that:

  * there is NO python loop over chunks. The inter-chunk state recurrence
    S_{c+1} = a_c S_c + G_c has scalar a_c, so it collapses to the SAME masked-decay
    matmul form as the intra-chunk term, just at the chunk level (NC x NC instead
    of Q x Q). Two levels of the identical trick, zero sequential steps.
  * there is NO torch.cumsum. The within-chunk inclusive cumsum is a matmul with a
    constant lower-triangular ones matrix (which is what the NKI kernel does too).
  * every remaining op is matmul / mul / add / exp / softplus / silu -- the ops the
    native-eager Neuron backend handles well.

Why loop-free matters here: this repo's native-eager backend lowers each individual
op to its own NEFF (see train.py's rope-kernel comment: ~30 NEFFs for one
rope+norm), so a 16-step python loop over chunks would cost 16x the dispatch.

Everything in this file is checked against `mamba2_ssd_ref.ssd_recurrent` (the
literal sequential scan) before it is allowed anywhere near hardware.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# Masked entries of a decay matrix get this added to their exponent before exp().
# exp(-30000) is exactly 0.0 in fp32/bf16/fp64, and adding it to an argument that
# has already been clamped to <= 0 can never produce a NaN (unlike masking after
# exp, which multiplies a possible +inf by 0). See mamba2_ssd_ref's
# test_masked_overflow_regression.
_NEG = -30000.0


def _tri_masks(Q, NC, device, dtype):
    """Constant masks. cumsum_op is [Q,Q] lower-tri ones (inclusive cumsum as matmul)."""
    qi = torch.arange(Q, device=device)
    cumsum_op = (qi[:, None] >= qi[None, :]).to(dtype)              # [Q,Q] tril ones
    intra_add = torch.where(qi[:, None] >= qi[None, :],
                            torch.zeros((), device=device, dtype=dtype),
                            torch.full((), _NEG, device=device, dtype=dtype))
    ci = torch.arange(NC, device=device)
    # strictly lower triangular: chunk c only sees chunks c' < c
    inter_add = torch.where(ci[:, None] > ci[None, :],
                            torch.zeros((), device=device, dtype=dtype),
                            torch.full((), _NEG, device=device, dtype=dtype))
    # exclusive-cumsum operator over chunks: Pexcl[c] = sum_{r<c} de[r]
    excl_op = (ci[:, None] > ci[None, :]).to(dtype)                 # [NC,NC]
    incl_op = (ci[:, None] >= ci[None, :]).to(dtype)
    return cumsum_op, intra_add, inter_add, excl_op, incl_op


def ssd_chunked_matmul(x, dt, A, B, C, D, Q, masks=None, acc_dtype=torch.float32):
    """Loop-free chunked SSD.

    x:  [b, T, H, P]   dt: [b, T, H]   A: [H] (<0)
    B, C: [b, T, G, N]  D: [H] or None
    -> y: [b, T, H, P]
    """
    b, T, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    assert T % Q == 0
    NC = T // Q
    hpg = H // G
    dev = x.device
    ad = acc_dtype

    if masks is None:
        masks = _tri_masks(Q, NC, dev, ad)
    cumsum_op, intra_add, inter_add, excl_op, incl_op = masks

    # ---- reshape to [b, NC, H, Q, ...] (chunk & head as batch dims) -----------
    xc = x.reshape(b, NC, Q, H, P).permute(0, 1, 3, 2, 4).to(ad)          # [b,NC,H,Q,P]
    dtc = dt.reshape(b, NC, Q, H).permute(0, 1, 3, 2).to(ad)              # [b,NC,H,Q]
    Bc = B.reshape(b, NC, Q, G, N).permute(0, 1, 3, 2, 4).to(ad)          # [b,NC,G,Q,N]
    Cc = C.reshape(b, NC, Q, G, N).permute(0, 1, 3, 2, 4).to(ad)
    if hpg > 1:                                                          # expand groups -> heads
        Bf = Bc.repeat_interleave(hpg, dim=2)
        Cf = Cc.repeat_interleave(hpg, dim=2)
    else:
        Bf, Cf = Bc, Cc                                                  # G==1 -> broadcast over H

    dx = dtc.unsqueeze(-1) * xc                                          # [b,NC,H,Q,P]

    # ---- within-chunk inclusive cumsum of dt*A, as a matmul -------------------
    dtA = dtc * A.to(ad).reshape(1, 1, H, 1)                              # [b,NC,H,Q], <= 0
    ell = torch.matmul(dtA, cumsum_op.transpose(0, 1))                    # [b,NC,H,Q]
    # ell[..., t] = sum_{s<=t} dtA[..., s]   (cumsum_op[t,s] = 1 iff s<=t)

    # ---- intra-chunk (diagonal) term ----------------------------------------
    seg = ell.unsqueeze(-1) - ell.unsqueeze(-2)                           # [b,NC,H,Qt,Qs]
    Lc = torch.exp(seg.clamp(max=0.0) + intra_add)                        # masked BEFORE exp
    CB = torch.matmul(Cf, Bf.transpose(-1, -2))                           # [b,NC,H|G,Q,Q]
    Y = torch.matmul(Lc * CB, dx)                                         # [b,NC,H,Q,P]

    # ---- per-chunk state contribution G_c -----------------------------------
    de = ell[..., -1]                                                     # [b,NC,H] = l_{Q-1} <= 0
    Bbar = Bf * torch.exp((de.unsqueeze(-1) - ell).clamp(max=0.0)).unsqueeze(-1)
    Gst = torch.matmul(Bbar.transpose(-1, -2), dx)                        # [b,NC,H,N,P]

    # ---- inter-chunk carry, closed form (second level of the same trick) -----
    # S_in[c] = sum_{c'<c} exp(Pexcl[c] - Pincl[c']) G_{c'},  exponent <= 0.
    deh = de.permute(0, 2, 1)                                             # [b,H,NC]
    Pex = torch.matmul(deh, excl_op.transpose(0, 1))                       # [b,H,NC] sum_{r<c}
    Pin = torch.matmul(deh, incl_op.transpose(0, 1))                       # [b,H,NC] sum_{r<=c}
    L2 = torch.exp((Pex.unsqueeze(-1) - Pin.unsqueeze(-2)).clamp(max=0.0) + inter_add)
    Gflat = Gst.permute(0, 2, 1, 3, 4).reshape(b, H, NC, N * P)            # [b,H,NC,N*P]
    S_in = torch.matmul(L2, Gflat).reshape(b, H, NC, N, P).permute(0, 2, 1, 3, 4)

    # ---- inter-chunk output --------------------------------------------------
    Cbar = Cf * torch.exp(ell.clamp(max=0.0)).unsqueeze(-1)
    if hpg == 1 and G == 1:
        Cbar = Cbar.expand(b, NC, H, Q, N)
    Y = Y + torch.matmul(Cbar, S_in)                                       # [b,NC,H,Q,P]

    y = Y.permute(0, 1, 3, 2, 4).reshape(b, T, H, P)
    if D is not None:
        y = y + D.to(ad).reshape(1, 1, H, 1) * x.to(ad)
    return y


# -----------------------------------------------------------------------------

def rms_norm_gated(y, z, weight, eps=1e-5):
    """Mamba2's gated RMSNorm: rmsnorm(y * silu(z)) * weight, over the d_inner axis."""
    h = y.float() * F.silu(z.float())
    var = h.pow(2).mean(-1, keepdim=True)
    return (h * torch.rsqrt(var + eps)) * weight.float()


class Mamba2Mixer(nn.Module):
    """A real Mamba2 mixer: in_proj -> grouped causal conv1d -> SSD scan -> gated RMSNorm -> out_proj.

    Follows the reference Mamba2 block (state-spaces/mamba `Mamba2`, and the sibling
    repo's Mamba2Mixer) but with plain nn.Linear instead of tensor-parallel layers,
    and the loop-free chunked SSD above instead of an XLA/NxD kernel.
    """

    def __init__(self, n_embd, d_state=128, head_dim=128, expand=2, n_groups=1,
                 conv_kernel=4, chunk_size=128, linear_cls=None,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        super().__init__()
        Lin = linear_cls if linear_cls is not None else (lambda i, o: nn.Linear(i, o, bias=False))
        self.n_embd = n_embd
        self.d_state = d_state
        self.head_dim = head_dim
        self.d_inner = expand * n_embd
        assert self.d_inner % head_dim == 0, (self.d_inner, head_dim)
        self.n_heads = self.d_inner // head_dim
        self.n_groups = n_groups
        assert self.n_heads % n_groups == 0
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size
        self.d_conv = self.d_inner + 2 * n_groups * d_state
        self.dt_min, self.dt_max, self.dt_init_floor = dt_min, dt_max, dt_init_floor

        # z (gate) | xBC (conv'd) | dt  -- one projection, as in real Mamba2
        self.in_proj = Lin(n_embd, self.d_inner + self.d_conv + self.n_heads)
        self.conv_weight = nn.Parameter(torch.empty(self.d_conv, conv_kernel))
        self.dt_bias = nn.Parameter(torch.empty(self.n_heads))
        self.A_log = nn.Parameter(torch.empty(self.n_heads))
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.norm_weight = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = Lin(self.d_inner, n_embd)
        self._masks = None
        self.reset_ssm_parameters()

    @torch.no_grad()
    def reset_ssm_parameters(self):
        H = self.n_heads
        # dt_bias: inverse-softplus of a log-uniform dt in [dt_min, dt_max] -- the real
        # Mamba/Mamba2 init (state-spaces/mamba). Guarantees dt starts in a sane range.
        dt = torch.exp(torch.rand(H) * (math.log(self.dt_max) - math.log(self.dt_min))
                       + math.log(self.dt_min)).clamp(min=self.dt_init_floor)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))     # softplus^-1(dt)
        # A = -exp(A_log), A_log = log(1..H): the real "S4D-real" scalar-per-head init.
        self.A_log.copy_(torch.log(torch.arange(1, H + 1, dtype=torch.float32)))
        self.D.fill_(1.0)
        self.norm_weight.fill_(1.0)
        # depthwise causal conv: uniform, fan_in = kernel_size (matches nn.Conv1d default
        # for groups==channels)
        bound = 1.0 / math.sqrt(self.conv_kernel)
        self.conv_weight.uniform_(-bound, bound)

    def _get_masks(self, T, device, dtype):
        NC = T // self.chunk_size
        key = (T, device, dtype)
        if self._masks is None or self._masks[0] != key:
            self._masks = (key, _tri_masks(self.chunk_size, NC, device, dtype))
        return self._masks[1]

    def forward(self, u):
        """u: [b, T, n_embd] -> [b, T, n_embd]"""
        b, T, _ = u.shape
        H, P, N, G = self.n_heads, self.head_dim, self.d_state, self.n_groups
        zxbcdt = self.in_proj(u)
        z, xBC, dt_raw = torch.split(
            zxbcdt, [self.d_inner, self.d_conv, self.n_heads], dim=-1)

        # grouped (depthwise) causal conv1d + silu -- the short local-context op
        xBC = F.silu(causal_conv1d(xBC, self.conv_weight.to(xBC.dtype)))
        x, Bm, Cm = torch.split(xBC, [self.d_inner, G * N, G * N], dim=-1)

        dt = F.softplus(dt_raw.float() + self.dt_bias.float())          # [b,T,H] > 0
        A = -torch.exp(self.A_log.float())                             # [H] < 0

        y = ssd_chunked_matmul(
            x.reshape(b, T, H, P), dt, A,
            Bm.reshape(b, T, G, N), Cm.reshape(b, T, G, N), self.D,
            self.chunk_size, masks=self._get_masks(T, u.device, torch.float32))

        y = rms_norm_gated(y.reshape(b, T, self.d_inner), z, self.norm_weight)
        return self.out_proj(y.to(u.dtype))


def causal_conv1d(u, weight):
    """Depthwise causal conv1d via matmul over the k shifted copies.

    Written as k shifts + k muls + adds rather than F.conv1d: on the native-eager
    Neuron backend a depthwise grouped conv1d is not a well-supported primitive,
    while pad/slice/mul/add are. k == 4 so this is 4 terms.
    Exactly equal to F.conv1d(pad(u, (k-1,0)), w[:,None,:], groups=C).
    """
    k = weight.shape[-1]
    out = None
    for j in range(k):
        shift = k - 1 - j                                     # u[t - shift]
        if shift == 0:
            us = u
        else:
            us = F.pad(u[:, :-shift, :], (0, 0, shift, 0))
        term = us * weight[:, j]
        out = term if out is None else out + term
    return out
