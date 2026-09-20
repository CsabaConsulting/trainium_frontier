"""Attach the gated library training MLP to ordinary Linear caches and masters.

No parameters, persistent buffers, initialization draws or evaluation semantics
change. BF16 weight casts remain in the ordinary differentiable cache refresh;
the library's contiguous weight transposes remain inside every timed forward.
"""
import types

import torch

from mlp_fused_bf16 import make_autograd
from mlp_fused_integration import call_separate_region
from nki_library_probe_common import verify_sources
from nki_library_training_mlp_hxt import install as install_kernel, saved_relu_mlp_hxt


def install(model, device, train_module, separate_region=False):
    if not str(device).startswith('neuron'):
        raise ValueError('The library training adapter requires Neuron')
    assert train_module.RELU2_TAU == 0.0
    assert not train_module.USE_GATED_MLP and not train_module.USE_COALESCED_SWIGLU
    assert not train_module.USE_PARALLEL_BLOCK
    assert train_module.USE_BF16_NORM_OUTPUT
    keys = tuple(model.state_dict())
    ids = tuple(id(p) for p in model.parameters())
    rng = torch.get_rng_state().clone()
    verify_sources()
    install_kernel()
    from torch_neuronx import wrap_nki
    kernel = wrap_nki(saved_relu_mlp_hxt)

    def kernel_call(x, up, down):
        return kernel[2](x, up.T.contiguous(), down.T.contiguous())

    apply = make_autograd(kernel_call, transposed_relu=True)
    if separate_region:
        compiled = torch.compile(apply, backend='neuron', fullgraph=True, dynamic=False)

        def separated(*values):
            return call_separate_region(compiled, *values)

        apply_training = separated
    else:
        apply_training = apply

    use_cache = train_module.USE_WEIGHT_CACHE

    def weight(linear, dtype):
        cached = linear._cached_weight
        if use_cache and cached is not None and cached.dtype == dtype:
            return cached
        return linear.weight.to(dtype=dtype)

    forwards = {}
    for block in model.transformer.h:
        module = block.mlp
        assert module.c_fc is not None and module.c_gate is None and not module.swiglu_hidden
        assert module.c_fc.bias is None and module.c_proj.bias is None
        assert not module.c_fc.weight_transposed and not module.c_proj.weight_transposed
        hidden, channels = module.c_fc.weight.shape
        assert module.c_proj.weight.shape == (channels, hidden)
        assert channels % 128 == 0 and hidden % 128 == 0
        cls = type(module)
        if cls not in forwards:
            native_forward = cls.forward

            def forward(self, x, fc=None, _native=native_forward):
                if not self.training:
                    return _native(self, x, fc)
                # Unsupported training calls fail explicitly; there is no
                # shape-dependent fallback concealing a candidate bypass.
                assert fc is None and x.dtype == torch.bfloat16
                rows = x.numel() // x.shape[-1]
                assert rows >= 256 and rows % 256 == 0
                up, down = weight(self.c_fc, x.dtype), weight(self.c_proj, x.dtype)
                result = apply_training(x.reshape(rows, x.shape[-1]), up, down)
                return result.reshape(x.shape)

            forwards[cls] = forward
        module.forward = types.MethodType(forwards[cls], module)
    assert tuple(model.state_dict()) == keys
    assert tuple(id(p) for p in model.parameters()) == ids
    assert torch.equal(torch.get_rng_state(), rng)
    return dict(library_commit='92d11f63a9a8ec1ade34e6e1a3b8db66ef31307e',
        modules=len(model.transformer.h), token_row_sharding=True, saved_relu_layout='HxT',
        backward='original_bf16_four_products', separate_compiled_region=separate_region,
        ordinary_differentiable_weight_cache=bool(use_cache), weight_transposes_inside_forward=True,
        native_evaluation=True, checkpoint_keys_parameters_and_rng_unchanged=True)
