"""Save ReLU through the library's existing intermediate transpose.

The up projection still rounds its ReLU to BF16. Its existing transpose
then supplies both the HxT training save and the down projection. Square
the BF16 values only after saving them. Transpose commutes with this
elementwise square and is bit preserving on Trainium2. This avoids an
additional transpose, HBM read, projection, or SBUF allocation.

Only two helpers in the isolated, pinned library namespace are replaced.
Weights and master casts retain the caller's ordinary Linear contract.
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
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.basic.mlp_cte_basic_transpose import (
    transpose_intermediate_tensor_tile as upstream_transpose,
)
from research.vendor.nki_library_92d11f63.core.utils.common_types import ActFnType, NormType, QuantizationType
from research.vendor.nki_library_92d11f63.core.utils.kernel_assert import kernel_assert


def relu_only_evict(mlp_params, constants, psum_bank, proj_results_sbuf_list,
                    bxs_subtile, int_tile, bias_sbuf, static_scales_sbuf,
                    weight_row_scales_sbuf, hidden_scales_sbuf_list):
    kernel_assert(mlp_params.skip_gate_proj, "Training adapter is ungated")
    kernel_assert(mlp_params.activation_fn == ActFnType.SquaredReLU, "ReLU2 only")
    kernel_assert(not mlp_params.quant_params.is_quant(), "BF16, no quantization")
    kernel_assert(bias_sbuf == None and static_scales_sbuf == None, "No bias or static scale")
    kernel_assert(weight_row_scales_sbuf == None and hidden_scales_sbuf_list == None,
                  "No weight or input row scales")
    kernel_assert(constants.get_intermediate_offset() == 0, "Complete intermediate channels per core")
    dst = proj_results_sbuf_list[bxs_subtile.index][
        :bxs_subtile.size, nl.ds(int_tile.start_offset, int_tile.size)]
    nisa.activation(dst=dst, data=psum_bank[:bxs_subtile.size, :int_tile.size], op=nl.relu)


def transpose_save_then_square(mlp_params, tile_info, constants, indices,
                               int_tile_sbuf_list, output_tile_sbuf_list, sbm):
    kernel_assert(not mlp_params.quant_params.is_quant(), "No quantized intermediate")
    kernel_assert(constants.get_intermediate_offset() == 0, "Row sharding only")
    kernel_assert(tile_info.bxs_dim_tile.subtile_dim_info.tile_size == 128,
                  "The upstream transpose packs token subtiles of 128")
    kernel_assert(tile_info.xpose_intermediate_dim_tile.subtile_dim_info.tile_size == 128,
                  "The upstream transpose packs intermediate subtiles of 128")
    kernel_assert(mlp_params.intermediate_size % 512 == 0, "Complete intermediate transpose tiles")
    upstream_transpose(mlp_params, tile_info, constants, indices,
                       int_tile_sbuf_list, output_tile_sbuf_list, sbm)
    bxs_dim = tile_info.bxs_dim_tile
    shard_rows = constants.get_bxs_size(mlp_params)
    for bxs_subtile_idx in range(bxs_dim.subtile_dim_info.tile_count):
        local_row = bxs_dim.get_subtile_start(indices.bxs_tile_idx, bxs_subtile_idx)
        if local_row < shard_rows:
            rows = min(128, shard_rows - local_row)
            global_row = constants.get_bxs_offset() + local_row
            for hidden_block in range(mlp_params.intermediate_size // 128):
                # Upstream's post-transpose storage packs [I-block,T] in
                # the free dimension, with I-within-block as partition.
                tile = output_tile_sbuf_list[bxs_subtile_idx][
                    :128, nl.ds(hidden_block * 128, rows)]
                nisa.dma_copy(dst=mlp_params.training_saved_relu[
                    nl.ds(hidden_block * 128, 128), nl.ds(global_row, rows)], src=tile)
                # Same BF16 ReLU -> BF16 square rounding as the control.
                # The dependency preserves the original ReLU HBM save.
                nisa.activation(dst=tile, data=tile, op=nl.square)


@nki.jit
def saved_relu_mlp_reuse_transpose(x, up, down):
    tokens, channels = x.shape
    kernel_assert(x.dtype == nl.bfloat16 and up.dtype == nl.bfloat16 and down.dtype == nl.bfloat16,
                  "Explicit BF16 input and weight contract")
    hidden = up.shape[1]
    kernel_assert(up.shape[0] == channels and down.shape == (hidden, channels), "Weight layout")
    kernel_assert(channels % 128 == 0 and hidden % 512 == 0 and tokens % 256 == 0, "Tile dimensions")
    output = nl.ndarray((1, tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    saved = nl.ndarray((hidden, tokens), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    params = MLPParameters(x.reshape((1, tokens, channels)), up, up, down,
        activation_fn=ActFnType.SquaredReLU, normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.NONE, skip_gate_proj=True, force_cte_mode=True)
    params.training_saved_relu = saved
    validate_mlp_arguments(params)
    mlp_cte_basic(params, output, None)
    return output.reshape((tokens, channels)), saved


def install():
    projection = importlib.import_module(NAMESPACE + ".core.mlp.mlp_cte.basic.mlp_cte_basic_projection")
    core = importlib.import_module(NAMESPACE + ".core.mlp.mlp_cte.basic.mlp_cte_basic")
    projection._evict_gate_res_tile = relu_only_evict
    core.transpose_intermediate_tensor_tile = transpose_save_then_square
    core.calculate_sharding = token_row_sharding
    emit("NKILIB_TRAINING_ADAPTER", dict(installed=True, isolated_namespace_only=True,
        token_sharding=True, saved_relu_inside_kernel=True, saved_relu_layout="HxT",
        reuses_existing_transpose=True, extra_activation_transpose=False,
        additional_sbuf_bytes=0, square_after_transpose=True, up_projection_recomputation=False,
        master_and_weight_layout_costs_charged_by_caller=True))
