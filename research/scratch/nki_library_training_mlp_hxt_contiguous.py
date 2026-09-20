"""Save the library ReLU as HxT using tiled transpose and contiguous DMA.

All weights enter in ordinary Linear orientation through the caller, which
charges BF16 casts and contiguous transposes. The NKI entry point receives the
library's [C,I]/[I,C] layouts. Save the BF16 ReLU before the existing square,
inside the forward's timed DMA stream; no up-projection recomputation.
A 32KiB SBUF tile is reserved below the library allocator's lower bound.
The current projection PSUM bank is reused only after its activation is
evicted. All transpose/store costs are charged, with no upstream edits.
"""
import importlib

import nki
import nki.isa as nisa
import nki.language as nl

from nki_library_probe_common import NAMESPACE, emit
from nki_library_relu2_adapter import token_row_sharding
from research.vendor.nki_library_92d11f63.core.utils.allocator import SbufManager
from research.vendor.nki_library_92d11f63.core.utils.kernel_helpers import PSUM_BANK_SIZE
from research.vendor.nki_library_92d11f63.core.mlp.mlp_parameters import (
    MLPParameters, validate_mlp_arguments,
)
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.basic.mlp_cte_basic import mlp_cte_basic
from research.vendor.nki_library_92d11f63.core.utils.common_types import ActFnType, NormType, QuantizationType
from research.vendor.nki_library_92d11f63.core.utils.kernel_assert import kernel_assert


def save_relu_evict_hxt_contiguous(mlp_params, constants, psum_bank, proj_results_sbuf_list,
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
    kernel_assert(int_tile.size % 128 == 0 and bxs_subtile.size <= 128,
                  "Tiled transpose requires 128-wide intermediate blocks")
    # Reuse this bank only after its FP32 accumulation has been consumed by
    # the ReLU eviction. The next projection iteration allocates a fresh
    # logical accumulator at its original bank address.
    transposed_psum = nl.ndarray((128, 128), dtype=nl.bfloat16,
        buffer=nl.psum, address=(0, bxs_subtile.index * PSUM_BANK_SIZE))
    scratch = mlp_params.training_transpose_tile
    for chunk in range(int_tile.size // 128):
        nisa.nc_transpose(dst=transposed_psum[:, :bxs_subtile.size],
                          data=dst[:, nl.ds(chunk * 128, 128)])
        nisa.tensor_copy(dst=scratch[:, :bxs_subtile.size],
                         src=transposed_psum[:, :bxs_subtile.size])
        nisa.dma_copy(dst=mlp_params.training_saved_relu[
            nl.ds(int_tile.start_offset + chunk * 128, 128),
            nl.ds(row, bxs_subtile.size)], src=scratch[:, :bxs_subtile.size])
    nisa.activation(dst=dst, data=dst, op=nl.square)


@nki.jit
def saved_relu_mlp_hxt_contiguous(x, up, down):
    tokens, channels = x.shape
    kernel_assert(x.dtype == nl.bfloat16 and up.dtype == nl.bfloat16 and down.dtype == nl.bfloat16,
                  "Explicit BF16 input and weight contract")
    hidden = up.shape[1]
    kernel_assert(up.shape[0] == channels and down.shape == (hidden, channels), "Weight layout")
    kernel_assert(channels % 128 == 0 and hidden % 128 == 0 and tokens % 256 == 0, "Tile dimensions")
    output = nl.ndarray((1, tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    saved = nl.ndarray((hidden, tokens), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    params = MLPParameters(x.reshape((1, tokens, channels)), up, up, down,
        activation_fn=ActFnType.SquaredReLU, normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.NONE, skip_gate_proj=True, force_cte_mode=True)
    # An explicit per-invocation tensor field, never a global device tensor.
    # It is consumed only by the isolated save_relu_evict_hxt_contiguous hook above.
    params.training_saved_relu = saved
    # 128 lanes * 128 BF16 values. The isolated SbufManager factory below
    # reserves [0,256) bytes in every lane, including during tile selection.
    params.training_transpose_tile = nl.ndarray((128, 128), dtype=nl.bfloat16,
        buffer=nl.sbuf, address=(0, 0))
    validate_mlp_arguments(params)
    mlp_cte_basic(params, output, None)
    return output.reshape((tokens, channels)), saved


def reserved_sbuf_manager(sb_lower_bound, sb_upper_bound, logger):
    kernel_assert(sb_lower_bound == 0, "Expected original library allocation range")
    return SbufManager(sb_lower_bound=256, sb_upper_bound=sb_upper_bound, logger=logger)


def install():
    projection = importlib.import_module(NAMESPACE + ".core.mlp.mlp_cte.basic.mlp_cte_basic_projection")
    core = importlib.import_module(NAMESPACE + ".core.mlp.mlp_cte.basic.mlp_cte_basic")
    projection._evict_gate_res_tile = save_relu_evict_hxt_contiguous
    core.calculate_sharding = token_row_sharding
    core.SbufManager = reserved_sbuf_manager
    emit("NKILIB_TRAINING_ADAPTER", dict(installed=True, isolated_namespace_only=True,
        token_sharding=True, saved_relu_inside_kernel=True, saved_relu_layout="HxT", explicit_activation_transpose=True, contiguous_activation_store=True, reserved_sbuf_bytes=32768, up_projection_recomputation=False,
        master_and_weight_layout_costs_charged_by_caller=True))
