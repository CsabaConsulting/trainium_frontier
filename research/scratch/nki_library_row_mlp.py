"""Pinned library ROW FP8 forward with an explicit BF16 ReLU-squared hook.

All FP8 crosses PyTorch boundaries as raw uint8. Token quantization, weight
casts/layouts and per-output-channel scale broadcasts belong to the caller.
Optional diagnostic stores expose real up-projection sums, rounded ReLU and
intermediate quantized bytes/scales. They are absent from the timed entry path.
The diagnostic and plain output must pass an exact equality gate.
"""
import importlib

import nki
import nki.isa as nisa
import nki.language as nl

from nki_library_probe_common import NAMESPACE, emit
from nki_library_relu2_adapter import token_row_sharding
from research.vendor.nki_library_92d11f63.core.mlp.mlp_parameters import MLPParameters, validate_mlp_arguments
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.basic.mlp_cte_basic import mlp_cte_basic
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.basic.mlp_cte_basic_projection import (
    _evict_gate_res_tile as upstream_evict,
)
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.basic.mlp_cte_basic_quantization import (
    _perform_intermediate_row_quantization as upstream_quantize,
)
from research.vendor.nki_library_92d11f63.core.utils.common_types import ActFnType, NormType, QuantizationType
from research.vendor.nki_library_92d11f63.core.utils.kernel_assert import kernel_assert


def row_relu2_evict(mlp_params, constants, psum_bank, proj_results_sbuf_list,
                    bxs_subtile, int_tile, bias_sbuf, static_scales_sbuf,
                    weight_row_scales_sbuf, hidden_scales_sbuf_list):
    kernel_assert(mlp_params.skip_gate_proj and mlp_params.quant_params.is_quant_row(),
                  'Ungated ROW quantization only')
    kernel_assert(mlp_params.activation_fn == ActFnType.ReLU, 'Original ReLU then explicit square')
    kernel_assert(constants.get_intermediate_offset() == 0, 'Token-row sharding only')
    row = constants.get_bxs_offset() + bxs_subtile.start_offset
    columns = nl.ds(int_tile.start_offset, int_tile.size)
    if mlp_params.capture_intermediates:
        nisa.dma_copy(dst=mlp_params.raw_up[nl.ds(row, bxs_subtile.size), columns],
                      src=psum_bank[:bxs_subtile.size, :int_tile.size])
    upstream_evict(mlp_params, constants, psum_bank, proj_results_sbuf_list,
                   bxs_subtile, int_tile, bias_sbuf, static_scales_sbuf,
                   weight_row_scales_sbuf, hidden_scales_sbuf_list)
    tile = proj_results_sbuf_list[bxs_subtile.index][:bxs_subtile.size, columns]
    if mlp_params.capture_intermediates:
        nisa.dma_copy(dst=mlp_params.relu[nl.ds(row, bxs_subtile.size), columns], src=tile)
    nisa.activation(dst=tile, data=tile, op=nl.square)


def inspect_row_quantization(mlp_params, tile_info, constants, indices, bxs_tile_idx,
                             src_proj_res_sbuf_list, quantized_output_sbuf_list,
                             row_dequant_scales_sbuf_list, sbm):
    upstream_quantize(mlp_params, tile_info, constants, indices, bxs_tile_idx,
                      src_proj_res_sbuf_list, quantized_output_sbuf_list,
                      row_dequant_scales_sbuf_list, sbm)
    if mlp_params.capture_intermediates:
        bxs_dim = tile_info.bxs_dim_tile
        for subtile in range(bxs_dim.subtile_dim_info.tile_count):
            count = bxs_dim.get_subtile_bound(bxs_tile_idx, subtile)
            if count > 0:
                row = constants.get_bxs_offset() + bxs_dim.get_subtile_start(bxs_tile_idx, subtile)
                nisa.dma_copy(dst=mlp_params.intermediate_bytes[
                    nl.ds(row, count), :mlp_params.intermediate_size],
                    src=quantized_output_sbuf_list[subtile][
                        :count, :mlp_params.intermediate_size].view(nl.uint8))
                nisa.dma_copy(dst=mlp_params.intermediate_scale[nl.ds(row, count), :],
                              src=row_dequant_scales_sbuf_list[subtile][:count, :1])


@nki.jit
def row_mlp(packed_input, up_bytes, down_bytes, up_scales, down_scales,
             capture_intermediates: bool = False):
    tokens, packed_width = packed_input.shape
    channels, hidden = up_bytes.shape
    kernel_assert(packed_width == channels + 4 and down_bytes.shape == (hidden, channels),
                  'Packed C+4 input and [C,I]/[I,C] weights')
    kernel_assert(packed_input.dtype == nl.uint8 and up_bytes.dtype == nl.uint8 and down_bytes.dtype == nl.uint8,
                  'Raw legacy E4M3 bytes only')
    kernel_assert(tokens % 256 == 0 and channels % 256 == 0 and hidden % 512 == 0, 'Complete tiles')
    output = nl.ndarray((1, tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    up, down = up_bytes.view(nl.float8_e4m3), down_bytes.view(nl.float8_e4m3)
    params = MLPParameters(packed_input.view(nl.float8_e4m3).reshape((1, tokens, packed_width)),
        up, up, down, activation_fn=ActFnType.ReLU, normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.ROW, gate_w_scale=up_scales, up_w_scale=up_scales,
        down_w_scale=down_scales, output_dtype=nl.bfloat16, skip_gate_proj=True, force_cte_mode=True)
    params.capture_intermediates = capture_intermediates
    if capture_intermediates:
        raw_up = nl.ndarray((tokens, hidden), dtype=nl.float32, buffer=nl.shared_hbm)
        relu = nl.ndarray((tokens, hidden), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        intermediate_bytes = nl.ndarray((tokens, hidden), dtype=nl.uint8, buffer=nl.shared_hbm)
        intermediate_scale = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        params.raw_up, params.relu = raw_up, relu
        params.intermediate_bytes, params.intermediate_scale = intermediate_bytes, intermediate_scale
    validate_mlp_arguments(params)
    mlp_cte_basic(params, output, None)
    if capture_intermediates:
        return output.reshape((tokens, channels)), raw_up, relu, intermediate_bytes, intermediate_scale
    return output.reshape((tokens, channels))


def install():
    projection = importlib.import_module(NAMESPACE + '.core.mlp.mlp_cte.basic.mlp_cte_basic_projection')
    quantization = importlib.import_module(NAMESPACE + '.core.mlp.mlp_cte.basic.mlp_cte_basic_quantization')
    core = importlib.import_module(NAMESPACE + '.core.mlp.mlp_cte.basic.mlp_cte_basic')
    projection._evict_gate_res_tile = row_relu2_evict
    quantization._perform_intermediate_row_quantization = inspect_row_quantization
    core.calculate_sharding = token_row_sharding
    emit('NKILIB_ROW_MLP_ADAPTER', dict(installed=True, isolated_namespace_only=True,
         legacy_e4m3_double_row=True, upstream_row_quantization_and_scaling=True,
         token_row_sharding=True, relu_bf16_then_square_bf16=True,
         diagnostics_excluded_from_timed_path=True, training_backward_implemented=False))
