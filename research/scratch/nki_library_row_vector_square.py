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

# Modified: choose explicit Vector multiplication or the original Scalar square.

"""Change only the engine used to square the BF16 ReLU tile; preserve roundings."""
import importlib

import nki
import nki.isa as nisa
import nki.language as nl
from nki_library_probe_common import NAMESPACE, emit
from nki_library_row_training_forward import (
    MLPParameters, validate_mlp_arguments, mlp_cte_basic, ActFnType, NormType,
    QuantizationType, kernel_assert, row_relu2_evict, install as install_training,
)

def vector_square_evict(params, constants, psum_bank, proj_results_sbuf_list, bxs_subtile, int_tile, bias_sbuf, static_scales_sbuf, weight_row_scales_sbuf, hidden_scales_sbuf_list):
    if params.activation_fn != ActFnType.SquaredReLU:
        row_relu2_evict(params, constants, psum_bank, proj_results_sbuf_list, bxs_subtile, int_tile, bias_sbuf, static_scales_sbuf, weight_row_scales_sbuf, hidden_scales_sbuf_list)
    else:
        kernel_assert(params.skip_gate_proj and params.quant_params.is_quant_row(), 'Ungated ROW training forward only')
        kernel_assert(bias_sbuf == None and static_scales_sbuf == None and (not params.capture_intermediates), 'No bias or raw PSUM diagnostics')
        kernel_assert(constants.get_intermediate_offset() == 0, 'Token-row sharding only')
        columns = nl.ds(int_tile.start_offset, int_tile.size)
        tile = proj_results_sbuf_list[bxs_subtile.index][:bxs_subtile.size, columns]
        nisa.tensor_tensor(dst=tile, op=nl.multiply, data1=psum_bank[:bxs_subtile.size, :int_tile.size], data2=weight_row_scales_sbuf[:bxs_subtile.size, columns])
        nisa.activation(dst=tile, op=nl.relu, data=tile, scale=hidden_scales_sbuf_list[bxs_subtile.index][:bxs_subtile.size, 0:1], bias=constants.bxs_dim_subtile_zero_bias_vector_sbuf[:bxs_subtile.size, 0:1])
        row = constants.get_bxs_offset() + bxs_subtile.start_offset
        nisa.dma_copy(dst=params.saved_relu[nl.ds(row, bxs_subtile.size), columns], src=tile)
        if params.vector_square:
            nisa.tensor_tensor(dst=tile, data1=tile, data2=tile, op=nl.multiply, engine=nisa.vector_engine)
        else:
            nisa.activation(dst=tile, data=tile, op=nl.square)


@nki.jit
def row_mlp_vector_square_forward(input_values, up_bytes, down_bytes, up_scales, down_scales, vector_square: bool=True):
    tokens, channels = input_values.shape
    hidden = up_bytes.shape[1]
    kernel_assert(up_bytes.shape == (channels, hidden) and down_bytes.shape == (hidden, channels), 'BF16 [T,C] input and raw FP8 [C,I]/[I,C] weights')
    kernel_assert(input_values.dtype == nl.bfloat16 and up_bytes.dtype == nl.uint8 and (down_bytes.dtype == nl.uint8), 'Truthful input and weight dtypes')
    kernel_assert(tokens % 256 == 0 and channels % 256 == 0 and (hidden % 512 == 0), 'Complete tiles')
    output = nl.ndarray((1, tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    relu = nl.ndarray((tokens, hidden), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    x_bytes = nl.ndarray((tokens, channels), dtype=nl.uint8, buffer=nl.shared_hbm)
    x_scale = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    h_bytes = nl.ndarray((tokens, hidden), dtype=nl.uint8, buffer=nl.shared_hbm)
    h_scale = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    up = up_bytes.view(nl.float8_e4m3)
    down = down_bytes.view(nl.float8_e4m3)
    params = MLPParameters(input_values.reshape((1, tokens, channels)), up, up, down, activation_fn=ActFnType.SquaredReLU, normalization_type=NormType.NO_NORM, quantization_type=QuantizationType.ROW, gate_w_scale=up_scales, up_w_scale=up_scales, down_w_scale=down_scales, output_dtype=nl.bfloat16, skip_gate_proj=True, force_cte_mode=True)
    params.capture_intermediates = False
    params.saved_relu = relu
    params.saved_input_bytes = x_bytes
    params.saved_input_scale = x_scale
    params.saved_hidden_bytes = h_bytes
    params.saved_hidden_scale = h_scale
    params.vector_square = vector_square
    validate_mlp_arguments(params)
    mlp_cte_basic(params, output, None)
    return (output.reshape((tokens, channels)), relu, x_bytes, x_scale, h_bytes, h_scale)


def install():
    install_training()
    projection = importlib.import_module(NAMESPACE + '.core.mlp.mlp_cte.basic.mlp_cte_basic_projection')
    projection._evict_gate_res_tile = vector_square_evict
    emit('NKILIB_VECTOR_SQUARE_ADAPTER', dict(installed=True,
        only_training_square_engine_changed=True, explicit_vector_multiply=True,
        matched_original_scalar_square_arm=True, original_scalar_diagnostic_control=True,
        both_bf16_rounding_boundaries_preserved=True, all_saves_unchanged=True,
        all_allocator_and_shape_checks_retained=True, backward_or_quality_claim=False))
