def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """relu(x - RELU2_TAU)**2, via the example NKI kernel when enabled, else eager.

    T3.9: RELU2_TAU is Recursive's "shifted ReLU^2" (tau=0.5 in their recipe, 0.0 = the
    historical activation). The shift is a compile-time Python float folded into one subtraction,
    NOT a per-step-changing scalar, so it is baked into the graph exactly once -- the rule about
    new scalars needing to be device tensors applies to values that CHANGE per step.
    """
    if USE_NKI_RELU2 and _HAS_NKI and (x.device.type == 'neuron') and (x.numel() // x.shape[-1] % _NKI_PARTITION == 0):
        return _ReluSquaredNKI.apply(x)
    if RELU2_TAU != 0.0:
        return F.relu(x - RELU2_TAU).square()
    return F.relu(x).square()

class Linear(nn.Linear):
    """Linear without bias that casts weight to input dtype (allows bf16 input with f32 weight).

    When USE_WEIGHT_CACHE is on, the cast copy is refreshed once per optimizer step
    (see refresh_weight_cache, called from the training loop) instead of being
    recomputed on every forward call across the grad-accum inner loop.

    T1.6: when `weight_transposed` is set (by transpose_linear_weights(), driven by
    USE_TRANSPOSED_LINEAR), `self.weight` holds [in_features, out_features] instead of
    nn.Linear's [out_features, in_features] and forward does a plain `x @ weight`. The flag
    is per-instance rather than read off the global in forward() so that a module which was
    never converted still works while the global is on -- forward must describe the tensor it
    actually holds, never what the global wishes it held.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_weight: torch.Tensor | None = None
        self.weight_transposed = False

    def refresh_weight_cache(self, dtype: torch.dtype) -> None:
        self._cached_weight = self.weight.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if USE_WEIGHT_CACHE and self._cached_weight is not None and (self._cached_weight.dtype == x.dtype):
            w = self._cached_weight
        else:
            w = self.weight.to(dtype=x.dtype)
        if self.weight_transposed:
            return x @ w
        return F.linear(x, w, None)

def transpose_linear_weights(root: nn.Module) -> int:
    """T1.6: replace every Linear's [out, in] weight with its [in, out] transpose, in place.

    Returns the number of modules converted (0 when the flag is off, which callers print so a
    silent no-op can never masquerade as a measured arm).

    Called at the END of GPT.init_weights(), which is the single point both main() and
    load_for_eval() go through, so the training model and the eval model always agree on the
    layout. `state_dict` keys are UNCHANGED (still ".weight"); only the shape flips. That makes
    USE_TRANSPOSED_LINEAR shape-critical, hence its entry in SHAPE_CRITICAL_TOGGLES: a
    checkpoint written with it on will not strict-load with it off, and vice versa. That is
    load_for_eval() restores the saved, explicitly whitelisted architecture
    toggles before construction, so a CLI-selected layout also survives scoring
    in a fresh process without repeating the training command line.

    `.t().contiguous()` and not just `.t()`: a non-contiguous parameter would leave the compiler
    to materialize the transpose anyway, which is the entire cost this change exists to remove.
    """
    n = 0
    for m in root.modules():
        if isinstance(m, Linear) and (not m.weight_transposed):
            with torch.no_grad():
                wt = m.weight.detach().t().contiguous()
            m.weight = nn.Parameter(wt, requires_grad=m.weight.requires_grad)
            m.weight_transposed = True
            m._cached_weight = None
            n += 1
    return n

class MLP(nn.Module):

    def __init__(self, config: GPTConfig, coalesced: str='none'):
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.swiglu_hidden = coalesced_swiglu_hidden(config.n_embd, config.mlp_ratio) if USE_COALESCED_SWIGLU else 0
        if self.swiglu_hidden:
            hidden = self.swiglu_hidden
        fc_out = 2 * hidden if self.swiglu_hidden else hidden
        self.c_fc = None if coalesced in ('all', 'vfc') else Linear(config.n_embd, fc_out, bias=False)
        self.c_gate = Linear(config.n_embd, hidden, bias=False) if USE_GATED_MLP else None
        self.c_proj = Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, fc: torch.Tensor | None=None) -> torch.Tensor:
        h = self.c_fc(x) if fc is None else fc
        if self.swiglu_hidden:
            gate = h[..., :self.swiglu_hidden]
            up = h[..., self.swiglu_hidden:]
            return self.c_proj(F.silu(gate) * up)
        if self.c_gate is not None:
            return self.c_proj(h * F.silu(self.c_gate(x)))
        return self.c_proj(relu_squared(h))

def _muon_math(w, g, momentum_buffer, second, mom, one_mom, lr, lrwd, tall: bool, ns_steps: int, red_dim: int, red_dim_size: int, one_beta2, to_bf16: bool, lr_fp32: bool=False):
    """Muon's per-group math body, as a pure tensors-in/tensors-out function.

    This is the ONLY form of the Muon step that `torch.compile(backend="neuron")` can
    take, and every deviation from `_muon_step`'s eager body is forced by that:

    * **No `torch._foreach_*` anywhere.** `torch._foreach_copy_` is what blocked T1.7:
      the neuron backend fails Torch-MLIR legalization on it
      ("failed to legalize operation 'torch.operator'"). The stacked read-in and the
      write-back to the live parameters keep using `_foreach_copy_`, but they stay
      OUTSIDE this function, in eager, where foreach is legal and cheap (1 launch a side).
    * **No in-place ops on graph inputs.** `lerp` not `lerp_`, and `w` is returned
      rather than mutated. That costs nothing: `stacked_params` is already scratch that
      `_muon_step` refills from the live params every step, and `momentum_buffer` /
      `second_momentum_buffer` are rebound in the state dict, which is a Python-level
      assignment with no device work (the same trick the NKI path already uses for
      `stacked_params`).
    * **No per-step Python float reaches a device op.** `lr`, `lr*wd`, `momentum` and
      `1-momentum` come in as 0-dim fp32 DEVICE tensors, because under
      `dynamic=False` a Python float argument is a Dynamo guard: the LR schedule alone
      would force a fresh neuronx-cc compile on every single step. This is the
      USE_TENSOR_LR_SCALARS lesson applied to a compiled region, where it is much worse
      than in eager -- eager only bakes `alpha=`-style scalars, Dynamo bakes all of them.
      `tall` / `ns_steps` / `red_dim` / `red_dim_size` / `to_bf16` / `lr_fp32` ARE
      Python constants on purpose: they are fixed per group for the whole run, so
      guarding on them is free and it keeps the traced graph straight-line.
      `one_beta2` is in that class only while the plan-§11.1 NorMuon ramp is OFF, which is
      the default; when MUON_BETA2_FINAL > 0 it changes every step and arrives as a 0-dim
      fp32 device tensor instead, for exactly the reason `lr` does. `torch.lerp` takes
      either overload, so the body is written once. Under plan-§11.1 per-depth scaling
      `mom` / `one_mom` / `lr` / `lrwd` arrive as [K, 1, 1] instead of 0-dim and broadcast
      over the stacked [K, rows, cols] buffers; nothing in this body needs to know.

    Bit-exactness against the eager body is the standard, and these are the three places
    it could have been lost:

    1. `1-momentum` is formed on the HOST in float64 and uploaded, not computed on
       device from an fp32 `momentum`. `fp32(1.0 - 0.95)` in float64-then-round is
       0.05f, while `1.0f - 0.95f` on device is 0.0500000119 -- a real difference.
       `_muon_compile_scalars` does the float64 subtraction, exactly like the Python
       float the eager path hands to `lerp_.Scalar`.
    2. `lr*ratio` and `lr*ratio*wd` are formed on the host in float32 (NOT float64), so
       they equal the device's own fp32 products bit for bit -- the identical argument
       `_muon_lrwd` already makes for the NKI path.
    3. `lerp`'s weight arrives as a 0-dim tensor here and as a Python scalar in eager,
       i.e. the `lerp.Tensor` overload versus `lerp.Scalar`. Verified equal, not
       assumed: research/scratch/compile_optim_equiv.py compares both overloads
       directly and then compares 6 full optimizer steps element by element.
    """
    momentum_buffer = torch.lerp(momentum_buffer, g, one_mom)
    g = torch.lerp(g, momentum_buffer, mom)
    x = g.bfloat16() if to_bf16 else g
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-06)
    if tall:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            xtx = x.mT @ x
            x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            xxt = x @ x.mT
            x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    g = x
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second = torch.lerp(second, v_mean, one_beta2)
    step_size = second.clamp_min(1e-10).rsqrt()
    scaled = v_mean * red_dim_size * step_size.square()
    v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
    mask = g * w >= 0
    w = w - (_muon_lr_g(lr, g, lr_fp32) + lrwd * w * mask)
    return (w, momentum_buffer, second)

def _muon_lr_g(lr, g, lr_fp32: bool=False):
    """`lr * g`, computed the way a 0-dim `lr` computes it whatever `lr`'s shape.

    Plan §11.1 item 3 makes `lr` a [K, 1, 1] operand instead of a 0-dim one, and that flips
    PyTorch's type promotion. `g` is bf16 here (the Newton-Schulz iterate) while `lr` is fp32:

      * 0-dim `lr`  -> same-category, so the 0-dim operand's dtype loses. Common dtype is bf16, and
        TensorIterator CASTS lr to bf16 before multiplying. The product is bf16 of a bf16 lr.
      * [K,1,1] `lr` -> both dimensioned, so common dtype is fp32: lr keeps its full precision and
        the product comes out fp32.

    Left alone, the per-depth arm would therefore also carry a silent precision UPGRADE of the
    weight update -- roughly 0.4% of relative lr accuracy recovered -- riding along with the change
    being measured, on the arm least able to afford a confound. Measured, not theorised:
    research/scratch/sched_tricks_probe.py intercepts `_muon_math` and shows every scalar operand
    bit-identical and only `w_out` differing, by ~3e-5 on fp32 weights, with neutral multipliers.

    So the cast happens BEFORE the multiply, not after: rounding the fp32 product back to bf16 is
    NOT the same number, because the 0-dim path rounds lr first and the product second. This is what
    makes the neutral-multiplier case in research/scratch/sched_tricks_equiv.py bitwise.

    Recorded while here, as a separate matter: the locked path really does round lr to bf16 before
    scaling the update, i.e. every Muon group's effective lr carries up to ~0.4% quantisation error.
    It is uniform per group per step, so it acts as tiny lr jitter rather than as noise on the
    direction -- but keeping lr in fp32 is a candidate free win in its own right, and it must be its
    own flag and its own A/B, not something smuggled in under per-depth scaling.

    `lr_fp32` (USE_MUON_LR_FP32 / --muon-lr-fp32) is that flag, and it is that A/B. Default False
    reproduces the paragraph above byte for byte -- this branch is only reached when the dimensioned
    (`lr.dim() > 0`) branch above was NOT taken, which is exactly the 0-dim locked-default case, so
    it can never touch the per-depth path's intentional bf16-matching behaviour. When on, instead of
    letting the multiply's own type promotion cast the fp32 `lr` down to `g`'s dtype first, `g` is
    upcast to fp32 and the product is formed and returned in fp32. Both call sites add that product
    to another fp32 term (`lrwd * w * mask`) and subtract the fp32 sum from an fp32 `w`, so the
    lower-precision cast that used to happen to `lr` before the multiply now simply does not happen
    anywhere before the final fp32 subtract -- there is no dtype for `g.float() * lr` to be cast
    down to here, unlike the `lr.dim() > 0` branch above where the cast is the point.
    """
    if isinstance(lr, torch.Tensor) and lr.dim() > 0 and (lr.dtype != g.dtype):
        lr = lr.to(g.dtype)
        return lr * g
    if lr_fp32 and isinstance(lr, torch.Tensor) and (lr.dim() == 0) and (lr.dtype != g.dtype):
        return lr * g.float()
    return lr * g

def reference_forward(x, up, down):
    relu = F.relu(F.linear(x, up.to(x.dtype)))
    return (F.linear(relu.square(), down.to(x.dtype)), relu)

def make_autograd(kernel_call, transposed_relu=False, normal_input_gradient=False, packed_weight_inputs=False):

    class FusedMLP(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, up, down, up_packed=None, down_packed=None):
            if packed_weight_inputs:
                assert up_packed is not None and down_packed is not None
                result, relu = kernel_call(x, up_packed, down_packed)
            else:
                result, relu = kernel_call(x, up, down)
            ctx.save_for_backward(x, up, down, relu)
            return result

        @staticmethod
        def backward(ctx, grad):
            x, up, down, relu = ctx.saved_tensors
            if transposed_relu:
                activated_t = relu.square()
                dhidden_t = down.to(x.dtype).T @ grad.T
                dpre_t = dhidden_t * (relu * 2.0)
                if normal_input_gradient:
                    dx = dpre_t.T @ up.to(x.dtype)
                else:
                    dx = (up.to(x.dtype).T @ dpre_t).T
                dup = (dpre_t @ x).to(up.dtype)
                ddown = (grad.T @ activated_t.T).to(down.dtype)
            else:
                activated = relu.square()
                dhidden = grad @ down.to(x.dtype)
                dpre = dhidden * (relu * 2.0)
                dx = dpre @ up.to(x.dtype)
                dup = (dpre.T @ x).to(up.dtype)
                ddown = (grad.T @ activated).to(down.dtype)
            if packed_weight_inputs:
                return (dx, dup, ddown, None, None)
            return (dx, dup, ddown)
    return FusedMLP.apply

def make_forward(device, native_forward, use_weight_cache, normal_input_gradient=False, token_tile=512, grouped_io=False, weight_mode=0, forward_backend='nki', separate_region=False):
    assert forward_backend in ('nki', 'native')
    if forward_backend == 'native':
        assert weight_mode == 0

        def kernel_call(x, up, down):
            result, relu = reference_forward(x, up, down)
            return (result, relu.T.contiguous())
    elif str(device).startswith('neuron'):
        from torch_neuronx import wrap_nki
        if weight_mode == 4:
            raise ValueError('Unsubmitted MLP variant; use native forward')
            kernel = wrap_nki(build_fixed_kernel())
        else:
            kernel = wrap_nki(build_kernel())

        def kernel_call(x, up, down):
            tile = min(token_tile, x.shape[0] // 2)
            return kernel[2](x, up, down, 2, tile, True, True, True, grouped_io, weight_mode)
    else:

        def kernel_call(x, up, down):
            if weight_mode >= 3:
                up, down = (unpack_weight(up), unpack_weight(down))
            result, relu = reference_forward(x, up, down)
            return (result, relu.T)
    fused = make_autograd(kernel_call, transposed_relu=True, normal_input_gradient=normal_input_gradient, packed_weight_inputs=weight_mode >= 3)
    if separate_region:

        def training_call(x, up, down, up_packed=None, down_packed=None):
            if weight_mode >= 3:
                return fused(x, up, down, up_packed, down_packed)
            return fused(x, up, down)
        compiled_training = torch.compile(training_call, backend='neuron' if str(device).startswith('neuron') else 'inductor', fullgraph=True, dynamic=False)

        def apply_fused(*inputs):
            return call_separate_region(compiled_training, *inputs)
    else:
        apply_fused = fused

    def weight(linear, dtype):
        cached = linear._cached_weight
        if use_weight_cache and cached is not None and (cached.dtype == dtype):
            return cached
        return linear.weight.to(dtype=dtype)

    def forward(self, x, fc=None):
        tokens = x.numel() // x.shape[-1]
        tile = min(token_tile, tokens // 2)
        if not self.training or fc is not None or x.dtype != torch.bfloat16 or (tile < 128) or tile % 128 or tokens % (2 * tile):
            return native_forward(self, x, fc)
        up, down = (weight(self.c_fc, x.dtype), weight(self.c_proj, x.dtype))
        if weight_mode >= 3:
            if use_weight_cache and self.c_fc._cached_weight is not None and (self.c_proj._cached_weight is not None) and (self.c_fc._cached_weight.dtype == x.dtype) and (self.c_proj._cached_weight.dtype == x.dtype):
                up_packed = self.c_fc._nki_packed_weight
                down_packed = self.c_proj._nki_packed_weight
            else:
                up_packed, down_packed = (prepack_weight(up), prepack_weight(down))
            result = apply_fused(x.reshape(tokens, x.shape[-1]), up, down, up_packed, down_packed)
        else:
            result = apply_fused(x.reshape(tokens, x.shape[-1]), up, down)
        return result.reshape(x.shape)
    return forward

POLAR_EXPRESS_COEFFS = [(8.156554524902461, -22.48329292557795, 15.878769915207462), (4.042929935166739, -2.808917465908714, 0.5000178451051316), (3.8916678022926607, -2.772484153217685, 0.5060648178503393), (3.285753657755655, -2.3681294933425376, 0.46449024233003106), (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)]
