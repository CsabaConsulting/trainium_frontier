"""Training-only packed E4M3 forward with the gated BF16 surrogate backward.

Masters and checkpoint format stay FP32/native. Evaluation deliberately uses
the original BF16 MLP: this is a quantized-training experiment, not an exact
replacement. The caller must supply gain-free RMS-normalized BF16 inputs and
refresh Linear caches after every parameter update, as the ordinary loop does.
All online calibration and packing are performed inside that timed refresh.
"""
import types

import torch
import torch.nn.functional as F

from fp8_mlp_island_autograd import make_autograd
from fp8_rms_scale_contract import policy


def make_ops():
    """Build the already independently gated kernels; do not change their math."""
    from torch_neuronx import wrap_nki
    from fp8_weight_cache_nki import build_kernels
    from fp8_packed_weight_cache_nki import build_kernel as build_packer
    from fp8_mlp_token_rows_nki import build_kernel as build_forward
    calibrate = wrap_nki(build_kernels()[0])
    pack = wrap_nki(build_packer())
    kernel = wrap_nki(build_forward())

    def prepare(weight):
        value = weight.detach().bfloat16()
        rows = value.shape[0]
        groups = rows // min(512, rows)
        lanes = 2 if groups % 2 == 0 else 1
        meta = calibrate[1](value)
        qdq, packed = pack[lanes](value, meta, lanes)
        return qdq, meta, packed

    def forward(x, up, down, up_meta, down_meta, packed_up, packed_down):
        scale = policy(x.shape[1])[0]
        return kernel[2](x, up, down, up_meta, down_meta, 2, 128, False,
                         scale, packed_up, packed_down)

    apply = make_autograd(forward, relu_token_major=True, token_row_backward=True,
                          packed_weights=True)
    return prepare, apply


def install(model, device, train_module):
    """Attach per-instance methods without adding parameters, buffers or RNG draws.

    A missing initial cache is prepared from current masters inside the forward
    call. Subsequent ordinary refreshes prepare one FP8 cache per update, reused
    by every accumulated microbatch. Short calls pad only this token-independent
    MLP, never the attention sequence; no shape-dependent BF16 training fallback.
    """
    if not str(device).startswith('neuron'):
        raise ValueError('Packed FP8 integration requires Neuron')
    if not train_module.USE_WEIGHT_CACHE or not train_module.USE_BF16_NORM_OUTPUT:
        raise ValueError('Require normal weight refresh and gain-free BF16 norm outputs')
    if (train_module.RELU2_TAU != 0.0 or train_module.USE_GATED_MLP
            or train_module.USE_COALESCED_SWIGLU
            or train_module.USE_PARALLEL_BLOCK):
        raise ValueError('Only the gated ordinary ReLU-squared MLP is supported')
    before_keys = tuple(model.state_dict())
    before_ids = tuple(id(p) for p in model.parameters())
    before_rng = torch.get_rng_state().clone()
    prepare, apply = make_ops()
    selected = []
    for index, block in enumerate(model.transformer.h):
        module = block.mlp
        if 'forward' in module.__dict__:
            raise ValueError('An MLP forward override is already installed')
        if module.c_fc is None or module.c_gate is not None or module.swiglu_hidden:
            raise ValueError('Unsupported MLP projections')
        hidden, channels = module.c_fc.weight.shape
        policy(channels)
        if hidden != 4 * channels or module.c_proj.weight.shape != (channels, hidden):
            raise ValueError('Only the gated four-times expansion is supported')
        for linear in (module.c_fc, module.c_proj):
            if linear.bias is not None or linear.weight_transposed:
                raise ValueError('Require ordinary biasless Linear weights')
            if 'refresh_weight_cache' in linear.__dict__:
                raise ValueError('A Linear cache override is already installed')
            native_refresh = type(linear).refresh_weight_cache
            linear._fp8_training_cache = None
            linear._fp8_refresh_count = 0

            def refresh(self, dtype, _native_refresh=native_refresh):
                _native_refresh(self, dtype)
                self._fp8_training_cache = (prepare(self._cached_weight)
                                            if dtype == torch.bfloat16 else None)
                self._fp8_refresh_count += 1

            linear.refresh_weight_cache = types.MethodType(refresh, linear)
        native_forward = type(module).forward

        def forward(self, x, fc=None, _native_forward=native_forward):
            if not self.training:
                return _native_forward(self, x, fc)
            if fc is not None or x.dtype != torch.bfloat16:
                raise ValueError('FP8 training requires an ordinary normalized BF16 MLP input')
            tokens, channels = x.numel() // x.shape[-1], x.shape[-1]
            # A minimum of512 also preserves the gated two-core outer-loop
            # geometry. Padding is per-token zeros and is included in timing.
            padded_tokens = max(512, ((tokens + 511) // 512) * 512)
            flat = x.reshape(tokens, channels)
            if padded_tokens != tokens:
                flat = F.pad(flat, (0, 0, 0, padded_tokens - tokens))
            up = self.c_fc._fp8_training_cache
            down = self.c_proj._fp8_training_cache
            if up is None:
                up = prepare(self.c_fc.weight)
            if down is None:
                down = prepare(self.c_proj.weight)
            uq, um, pu = up
            dq, dm, pd = down
            value = apply(flat, self.c_fc.weight, self.c_proj.weight,
                          uq, dq, um, dm, pu, pd)
            return value[:tokens].reshape(x.shape)

        module.forward = types.MethodType(forward, module)
        selected.append(index)
    assert selected
    assert tuple(model.state_dict()) == before_keys
    assert tuple(id(p) for p in model.parameters()) == before_ids
    assert torch.equal(before_rng, torch.get_rng_state())
    return dict(layers=selected, training_forward='legacy E4M3 packed up/down GEMMs',
                backward='gated BF16 identity-surrogate VJP into FP32 masters',
                evaluation='original BF16 MLP', cache_preparation='ordinary timed Linear refresh',
                input_contract='gain-free RMS norm, BF16, widths256/1024',
                short_input_policy='zero-pad inside MLP to multiple512; discard padded rows')
