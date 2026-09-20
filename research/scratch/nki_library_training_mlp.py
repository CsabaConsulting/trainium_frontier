"""Adapt the pinned row-sharded CTE MLP to return training's saved ReLU.

All weights enter in ordinary Linear orientation through the caller, which
charges BF16 casts and contiguous transposes. The NKI entry point receives the
library's [C,I]/[I,C] layouts. Save the BF16 ReLU before the existing square,
inside the forward's timed DMA stream; no up-projection recomputation.
Only the isolated library's eviction/sharding helpers are replaced.
"""
import importlib

import nki
import nki.isa as nisa
import nki.language as nl

from nki_library_probe_common import NAMESPACE, emit
from nki_library_relu2_adapter import token_row_sharding
from research.vendor.nki_library_92d11f63.core.mlp.mlp_parameters import (
    MLPParameters, validate_mlp_arguments,
)
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.basic.mlp_cte_basic import mlp_cte_basic
from research.vendor.nki_library_92d11f63.core.utils.common_types import ActFnType, NormType, QuantizationType
from research.vendor.nki_library_92d11f63.core.utils.kernel_assert import kernel_assert


def save_relu_evict(mlp_params, constants, psum_bank, proj_results_sbuf_list,
                    bxs_subtile, int_tile, bias_sbuf, static_scales_sbuf,
                    weight_row_scales_sbuf, hidden_scales_sbuf_list):
    kernel_assert(mlp_params.skip_gate_proj, "Training adapter is ungated")
    kernel_assert(mlp_params.activation_fn == ActFnType.SquaredReLU, "ReLU2 only")
    kernel_assert(not mlp_params.quant_params.is_quant(), "BF16, no quantization")
    kernel_assert(bias_sbuf == None and static_scales_sbuf == None, "No bias or static scale")
    kernel_assert(weight_row_scales_sbuf == None and hidden_scales_sbuf_list == None,
                  "No weight or input row scales")
    kernel_assert(constants.get_intermediate_offset() == 0,
                  "This training screen uses complete intermediate channels per core")
    dst = proj_results_sbuf_list[bxs_subtile.index][
        :bxs_subtile.size, nl.ds(int_tile.start_offset, int_tile.size)]
    nisa.activation(dst=dst, data=psum_bank[:bxs_subtile.size, :int_tile.size], op=nl.relu)
    row = constants.get_bxs_offset() + bxs_subtile.start_offset
    nisa.dma_copy(dst=mlp_params.training_saved_relu[
        nl.ds(row, bxs_subtile.size), nl.ds(int_tile.start_offset, int_tile.size)], src=dst)
    nisa.activation(dst=dst, data=dst, op=nl.square)


@nki.jit
def saved_relu_mlp(x, up, down):
    tokens, channels = x.shape
    kernel_assert(x.dtype == nl.bfloat16 and up.dtype == nl.bfloat16 and down.dtype == nl.bfloat16,
                  "Explicit BF16 input and weight contract")
    hidden = up.shape[1]
    kernel_assert(up.shape[0] == channels and down.shape == (hidden, channels), "Weight layout")
    kernel_assert(channels % 128 == 0 and hidden % 128 == 0 and tokens % 256 == 0, "Tile dimensions")
    output = nl.ndarray((1, tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    saved = nl.ndarray((tokens, hidden), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    params = MLPParameters(x.reshape((1, tokens, channels)), up, up, down,
        activation_fn=ActFnType.SquaredReLU, normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.NONE, skip_gate_proj=True, force_cte_mode=True)
    # An explicit per-invocation tensor field, never a global device tensor.
    # It is consumed only by the isolated save_relu_evict hook above.
    params.training_saved_relu = saved
    validate_mlp_arguments(params)
    mlp_cte_basic(params, output, None)
    return output.reshape((tokens, channels)), saved


def install():
    projection = importlib.import_module(NAMESPACE + ".core.mlp.mlp_cte.basic.mlp_cte_basic_projection")
    core = importlib.import_module(NAMESPACE + ".core.mlp.mlp_cte.basic.mlp_cte_basic")
    projection._evict_gate_res_tile = save_relu_evict
    core.calculate_sharding = token_row_sharding
    emit("NKILIB_TRAINING_ADAPTER", dict(installed=True, isolated_namespace_only=True,
        token_sharding=True, saved_relu_inside_kernel=True, up_projection_recomputation=False,
        master_and_weight_layout_costs_charged_by_caller=True))
