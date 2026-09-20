"""Select library MLP layers while retaining the same native HxT backward.

Install both existing adapters unchanged, then restore the exact bound native
methods on unselected layers. No new arithmetic, fallback, compiler boundary,
parameter, buffer or weight cache is introduced by this routing diagnostic.
"""
import torch

from mlp_fused_integration import install as install_native
from nki_library_mlp_integration import install as install_library


def install(model, device, train_module, layer_indices):
    total = len(model.transformer.h)
    selected = tuple(layer_indices)
    assert len(set(selected)) == len(selected)
    assert all(isinstance(i, int) and 0 <= i < total for i in selected)
    keys = tuple(model.state_dict())
    parameter_ids = tuple(id(p) for p in model.parameters())
    rng = torch.get_rng_state().clone()
    install_native(model, device, train_module, forward_backend='native')
    native_forwards = [block.mlp.forward for block in model.transformer.h]
    contract = install_library(model, device, train_module)
    library_forwards = [block.mlp.forward for block in model.transformer.h]
    for index, block in enumerate(model.transformer.h):
        if index not in selected:
            block.mlp.forward = native_forwards[index]
        expected = library_forwards[index] if index in selected else native_forwards[index]
        assert block.mlp.forward is expected
    assert tuple(model.state_dict()) == keys
    assert tuple(id(p) for p in model.parameters()) == parameter_ids
    assert torch.equal(torch.get_rng_state(), rng)
    contract.update(modules=len(selected), total_mlp_modules=total,
        library_layer_indices=list(selected),
        native_hxt_layer_indices=[i for i in range(total) if i not in selected],
        exact_unselected_native_forward_objects_restored=True,
        all_layers_keep_original_hxt_custom_backward=True,
        existing_native_and_library_adapter_arithmetic_unchanged=True,
        layer_selection_before_compilation=True)
    return contract
