"""Scratch MLP integration retaining native modules, parameters and BF16 caches.

Only supported training calls use the gated fusion. Evaluation calls the
original implementation, and saved checkpoints contain no extra state.
"""
import types

import torch

from mlp_fused_bf16 import build_kernel, make_autograd, prepack_weight, reference_forward, unpack_weight


def make_forward(device, native_forward, use_weight_cache, normal_input_gradient=False,
                 token_tile=512, grouped_io=False, weight_mode=0):
    if str(device).startswith("neuron"):
        from torch_neuronx import wrap_nki
        if weight_mode == 4:
            from mlp_prepacked_fixed_sbuf import build_kernel as build_fixed_kernel
            kernel = wrap_nki(build_fixed_kernel())
        else:
            kernel = wrap_nki(build_kernel())

        def kernel_call(x, up, down):
            tile = min(token_tile, x.shape[0] // 2)
            return kernel[2](x, up, down, 2, tile, True, True, True, grouped_io, weight_mode)
    else:
        def kernel_call(x, up, down):
            if weight_mode >= 3:
                up, down = unpack_weight(up), unpack_weight(down)
            result, relu = reference_forward(x, up, down)
            return result, relu.T

    fused = make_autograd(kernel_call, transposed_relu=True,
                          normal_input_gradient=normal_input_gradient,
                          packed_weight_inputs=(weight_mode >= 3))

    def weight(linear, dtype):
        cached = linear._cached_weight
        if use_weight_cache and cached is not None and cached.dtype == dtype:
            return cached
        return linear.weight.to(dtype=dtype)

    def forward(self, x, fc=None):
        tokens = x.numel() // x.shape[-1]
        tile = min(token_tile, tokens // 2)
        if (not self.training or fc is not None or x.dtype != torch.bfloat16
                or tile < 128 or tile % 128 or tokens % (2 * tile)):
            return native_forward(self, x, fc)
        up, down = weight(self.c_fc, x.dtype), weight(self.c_proj, x.dtype)
        if weight_mode >= 3:
            if (use_weight_cache and self.c_fc._cached_weight is not None
                    and self.c_proj._cached_weight is not None
                    and self.c_fc._cached_weight.dtype == x.dtype
                    and self.c_proj._cached_weight.dtype == x.dtype):
                up_packed = self.c_fc._nki_packed_weight
                down_packed = self.c_proj._nki_packed_weight
            else:
                up_packed, down_packed = prepack_weight(up), prepack_weight(down)
            result = fused(x.reshape(tokens, x.shape[-1]), up, down, up_packed, down_packed)
        else:
            result = fused(x.reshape(tokens, x.shape[-1]), up, down)
        return result.reshape(x.shape)

    return forward


def install(model, device, train_module, normal_input_gradient=False,
            token_tile=512, grouped_io=False, weight_mode=0):
    assert train_module.RELU2_TAU == 0.0
    assert token_tile in (128, 256, 512)
    assert weight_mode in (0, 1, 2, 3, 4) and (weight_mode == 0 or grouped_io)
    keys = tuple(model.state_dict())
    parameter_ids = tuple(id(p) for p in model.parameters())
    for block in model.transformer.h:
        module = block.mlp
        assert module.c_fc is not None and module.c_gate is None and not module.swiglu_hidden
        assert module.c_fc.bias is None and module.c_proj.bias is None
        assert not module.c_fc.weight_transposed and not module.c_proj.weight_transposed
        hidden, channels = module.c_fc.weight.shape
        assert module.c_proj.weight.shape == (channels, hidden)
        assert channels % 128 == 0 and hidden % 128 == 0
        live_bytes = 4 * channels * hidden + 2 * (channels + hidden) * token_tile
        if weight_mode:
            live_bytes = 2 * (channels + hidden) * token_tile + 4 * 128 * (channels + hidden)
            if weight_mode == 1:
                live_bytes += 2 * channels * hidden
        if grouped_io:
            live_bytes += 2 * channels * token_tile
        assert live_bytes + 2 * 1024**2 <= 24 * 1024**2
        if weight_mode >= 3:
            for linear in (module.c_fc, module.c_proj):
                native_refresh = type(linear).refresh_weight_cache

                def refresh(self, dtype, _native_refresh=native_refresh):
                    _native_refresh(self, dtype)
                    self._nki_packed_weight = prepack_weight(self._cached_weight)

                linear.refresh_weight_cache = types.MethodType(refresh, linear)
                # Installation is also valid after a native cache exists.
                linear._nki_packed_weight = (prepack_weight(linear._cached_weight)
                                             if linear._cached_weight is not None else None)
        forward = make_forward(device, type(module).forward, train_module.USE_WEIGHT_CACHE,
                               normal_input_gradient, token_tile, grouped_io, weight_mode)
        module.forward = types.MethodType(forward, module)
    assert tuple(model.state_dict()) == keys
    assert tuple(id(p) for p in model.parameters()) == parameter_ids
