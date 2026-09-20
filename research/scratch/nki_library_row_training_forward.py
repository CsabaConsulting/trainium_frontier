# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified: ROW ReLU-squared training saves and isolated source191 control dispatch.

"""Save the actual ROW FP8 forward operands needed by a future training VJP.

The five saved tensors are BF16 pre-square ReLU, input FP8 bytes and row
scales, and squared-activation FP8 bytes and row scales. All stores occur in
the same forward kernel. This entry implements no backward approximation.
The original source191 forward remains a separate, unchanged control.

ROW dequantization below derives from the Apache-2.0-licensed pinned library
at92d11f63, core/mlp/mlp_cte/basic/mlp_cte_basic_projection.py:832-845.
"""
import importlib

import nki
import nki.isa as nisa
import nki.language as nl

from nki_library_probe_common import NAMESPACE, emit
import nki_library_row_heap_compat as heap_compat
from nki_library_row_fused_input import install as install_inline, load_hidden as inline_load_hidden
from nki_library_row_mlp import row_relu2_evict, inspect_row_quantization
from research.vendor.nki_library_92d11f63.core.mlp.mlp_parameters import (
    MLPParameters, validate_mlp_arguments,
)
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.basic.mlp_cte_basic import mlp_cte_basic
from research.vendor.nki_library_92d11f63.core.utils.common_types import (
    ActFnType, NormType, QuantizationType,
)
from research.vendor.nki_library_92d11f63.core.utils.kernel_assert import kernel_assert


def save_relu_evict(params, constants, psum_bank, proj_results_sbuf_list,
                    bxs_subtile, int_tile, bias_sbuf, static_scales_sbuf,
                    weight_row_scales_sbuf, hidden_scales_sbuf_list):
    if params.activation_fn != ActFnType.SquaredReLU:
        row_relu2_evict(params, constants, psum_bank, proj_results_sbuf_list,
            bxs_subtile, int_tile, bias_sbuf, static_scales_sbuf,
            weight_row_scales_sbuf, hidden_scales_sbuf_list)
    else:
        kernel_assert(params.skip_gate_proj and params.quant_params.is_quant_row(),
                      'Ungated ROW training forward only')
        kernel_assert(bias_sbuf == None and static_scales_sbuf == None
                      and not params.capture_intermediates, 'No bias or raw PSUM diagnostics')
        kernel_assert(constants.get_intermediate_offset() == 0, 'Token-row sharding only')
        columns = nl.ds(int_tile.start_offset, int_tile.size)
        tile = proj_results_sbuf_list[bxs_subtile.index][:bxs_subtile.size, columns]
        # Keep both original BF16 rounding boundaries, then save before square.
        nisa.tensor_tensor(dst=tile, op=nl.multiply,
            data1=psum_bank[:bxs_subtile.size, :int_tile.size],
            data2=weight_row_scales_sbuf[:bxs_subtile.size, columns])
        nisa.activation(dst=tile, op=nl.relu, data=tile,
            scale=hidden_scales_sbuf_list[bxs_subtile.index][:bxs_subtile.size, 0:1],
            bias=constants.bxs_dim_subtile_zero_bias_vector_sbuf[:bxs_subtile.size, 0:1])
        row = constants.get_bxs_offset() + bxs_subtile.start_offset
        nisa.dma_copy(dst=params.saved_relu[nl.ds(row, bxs_subtile.size), columns], src=tile)
        nisa.activation(dst=tile, data=tile, op=nl.square)


def save_input(params, tile_info, constants, indices, hidden_tiles, scales, stored_add):
    inline_load_hidden(params, tile_info, constants, indices, hidden_tiles, scales, stored_add)
    if params.activation_fn == ActFnType.SquaredReLU:
        dim = tile_info.bxs_dim_tile
        for index in range(dim.subtile_dim_info.tile_count):
            rows = dim.get_subtile_bound(indices.bxs_tile_idx, index)
            if rows > 0:
                row = constants.get_bxs_offset() + dim.get_subtile_start(indices.bxs_tile_idx, index)
                nisa.dma_copy(dst=params.saved_input_bytes[nl.ds(row, rows), :],
                              src=hidden_tiles[index][:rows, :params.hidden_size].view(nl.uint8))
                nisa.dma_copy(dst=params.saved_input_scale[nl.ds(row, rows), :],
                              src=scales[index][:rows, :1])


def save_intermediate(params, tile_info, constants, indices, bxs_tile_idx,
                      src_proj_res_sbuf_list, quantized_output_sbuf_list,
                      row_dequant_scales_sbuf_list, sbm):
    inspect_row_quantization(params, tile_info, constants, indices, bxs_tile_idx,
        src_proj_res_sbuf_list, quantized_output_sbuf_list, row_dequant_scales_sbuf_list, sbm)
    if params.activation_fn == ActFnType.SquaredReLU:
        dim = tile_info.bxs_dim_tile
        for index in range(dim.subtile_dim_info.tile_count):
            rows = dim.get_subtile_bound(bxs_tile_idx, index)
            if rows > 0:
                row = constants.get_bxs_offset() + dim.get_subtile_start(bxs_tile_idx, index)
                nisa.dma_copy(dst=params.saved_hidden_bytes[nl.ds(row, rows), :],
                    src=quantized_output_sbuf_list[index][:rows, :params.intermediate_size].view(nl.uint8))
                nisa.dma_copy(dst=params.saved_hidden_scale[nl.ds(row, rows), :],
                              src=row_dequant_scales_sbuf_list[index][:rows, :1])


@nki.jit
def row_mlp_training_forward(input_values, up_bytes, down_bytes, up_scales, down_scales):
    tokens, channels = input_values.shape
    hidden = up_bytes.shape[1]
    kernel_assert(up_bytes.shape == (channels, hidden) and down_bytes.shape == (hidden, channels),
                  'BF16 [T,C] input and raw FP8 [C,I]/[I,C] weights')
    kernel_assert(input_values.dtype == nl.bfloat16 and up_bytes.dtype == nl.uint8
                  and down_bytes.dtype == nl.uint8, 'Truthful input and weight dtypes')
    kernel_assert(tokens % 256 == 0 and channels % 256 == 0 and hidden % 512 == 0, 'Complete tiles')
    output = nl.ndarray((1, tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    relu = nl.ndarray((tokens, hidden), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    x_bytes = nl.ndarray((tokens, channels), dtype=nl.uint8, buffer=nl.shared_hbm)
    x_scale = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    h_bytes = nl.ndarray((tokens, hidden), dtype=nl.uint8, buffer=nl.shared_hbm)
    h_scale = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    up = up_bytes.view(nl.float8_e4m3)
    down = down_bytes.view(nl.float8_e4m3)
    params = MLPParameters(input_values.reshape((1, tokens, channels)), up, up, down,
        activation_fn=ActFnType.SquaredReLU, normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.ROW, gate_w_scale=up_scales, up_w_scale=up_scales,
        down_w_scale=down_scales, output_dtype=nl.bfloat16, skip_gate_proj=True, force_cte_mode=True)
    params.capture_intermediates = False
    params.saved_relu = relu
    params.saved_input_bytes = x_bytes
    params.saved_input_scale = x_scale
    params.saved_hidden_bytes = h_bytes
    params.saved_hidden_scale = h_scale
    validate_mlp_arguments(params)
    mlp_cte_basic(params, output, None)
    return output.reshape((tokens, channels)), relu, x_bytes, x_scale, h_bytes, h_scale


def install():
    install_inline()
    projection = importlib.import_module(NAMESPACE + '.core.mlp.mlp_cte.basic.mlp_cte_basic_projection')
    quantization = importlib.import_module(NAMESPACE + '.core.mlp.mlp_cte.basic.mlp_cte_basic_quantization')
    projection._evict_gate_res_tile = save_relu_evict
    quantization._perform_intermediate_row_quantization = save_intermediate
    heap_compat.load_hidden_tensor_tile_opt_fused_add = save_input
    emit('NKILIB_ROW_TRAINING_SAVE_ADAPTER', dict(installed=True,
        unchanged_relu_forward_control=True, squared_relu_entry_selects_saves=True,
        actual_consumed_input_and_hidden_bytes_saved=True, both_row_scales_saved=True,
        bf16_pre_square_relu_saved=True, all_stores_inside_forward=True,
        additional_sbuf_allocations=0, upstream_rounding_boundaries_retained=True,
        backward_implemented=False, training_or_bpb_claim=False))
