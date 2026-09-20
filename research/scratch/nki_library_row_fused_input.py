"""Quantize BF16 token rows directly into the library's FP8 input tiles.

Input shape/dtype remains truthful. The original allocator owns every new
buffer; its tile planner charges the source/absolute-value/reduction scratch
and all per-row scale vectors. Packed-input controls keep their original path.
The same ROW double-row projections, rounding and intermediate quantization
follow. Diagnostic HBM stores are compiled out of the plain forward.
"""
import importlib

import nki
import nki.isa as nisa
import nki.language as nl

from nki_library_probe_common import NAMESPACE, emit
import nki_library_row_heap_compat as heap_compat
from nki_library_row_mlp import install as install_row
from research.vendor.nki_library_92d11f63.core.mlp.mlp_parameters import (
    MLPParameters, validate_mlp_arguments, mlpp_has_dma_xpose as original_dma_xpose,
)
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.basic.mlp_cte_basic import (
    mlp_cte_basic,
)
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.mlp_cte_constants import (
    build_mlp_cte_constants as original_constants,
)
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.basic.mlp_cte_basic_allocation import (
    allocate_hidden_tensor_tile as original_allocate_hidden,
)
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.basic.mlp_cte_basic_tensor_io import (
    load_hidden_tensor_tile_opt_fused_add as original_load_hidden,
)
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.basic.mlp_cte_basic_tile_info import (
    calc_up_proj_footprint as original_up_footprint,
)
from research.vendor.nki_library_92d11f63.core.utils.common_types import (
    ActFnType, NormType, QuantizationType,
)
from research.vendor.nki_library_92d11f63.core.utils.kernel_assert import kernel_assert


def inline_input(params):
    return params.quant_params.is_quant_row() and params.hidden_tensor.dtype == nl.bfloat16


def build_constants(params, sharded_dim, total_programs, sbm, shard_idx, program_id, dim_shard=None):
    constants = original_constants(params, sharded_dim, total_programs, sbm,
                                   shard_idx, program_id, dim_shard)
    if inline_input(params):
        constants.hidden_tile_data_type = constants.src_proj_quant_data_type
    return constants


def up_footprint(params, constants, bxs_dim_tile, hidden_dim_tile, intermediate_dim_tile):
    size = original_up_footprint(params, constants, bxs_dim_tile, hidden_dim_tile,
                                 intermediate_dim_tile)
    if inline_input(params):
        # Two BF16 C-wide buffers, two FP32 scalars, and one FP32 scale
        # per128-row subtile. Their allocations use natural alignment.
        size += 4 * params.hidden_size + 8 + 4 * bxs_dim_tile.subtile_dim_info.tile_count
    return size


def allocate_hidden(params, tile_info, constants, indices, hidden_tiles, scales,
                    alloc_tile, alloc_scale):
    original_allocate_hidden(params, tile_info, constants, indices, hidden_tiles,
                             scales, alloc_tile, alloc_scale)
    if inline_input(params):
        kernel_assert(len(scales) == 0, 'BF16 input has no externally packed scales')
        for index in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
            scales.append(alloc_scale((128, 1), dtype=nl.float32, buffer=nl.sbuf,
                name=indices.get_tensor_name('inline_input_scale', f'sub{index}')))
        params.input_source_sbuf = alloc_tile((128, params.hidden_size),
            dtype=nl.bfloat16, buffer=nl.sbuf, name=indices.get_tensor_name('inline_input_source'))
        params.input_absolute_sbuf = alloc_tile((128, params.hidden_size),
            dtype=nl.bfloat16, buffer=nl.sbuf, name=indices.get_tensor_name('inline_input_absolute'))
        params.input_maximum_sbuf = alloc_tile((128, 1), dtype=nl.float32,
            buffer=nl.sbuf, name=indices.get_tensor_name('inline_input_maximum'))
        params.input_reciprocal_sbuf = alloc_tile((128, 1), dtype=nl.float32,
            buffer=nl.sbuf, name=indices.get_tensor_name('inline_input_reciprocal'))


def load_hidden(params, tile_info, constants, indices, hidden_tiles, scales, stored_add):
    if not inline_input(params):
        original_load_hidden(params, tile_info, constants, indices, hidden_tiles, scales, stored_add)
    else:
        kernel_assert(stored_add == None and params.hidden_tensor.shape[0] == 1,
                      'Flat independent rows without fused add')
        dim = tile_info.bxs_dim_tile
        source = params.input_source_sbuf
        absolute = params.input_absolute_sbuf
        maximum = params.input_maximum_sbuf
        reciprocal = params.input_reciprocal_sbuf
        for index in range(dim.subtile_dim_info.tile_count):
            rows = dim.get_subtile_bound(indices.bxs_tile_idx, index)
            if rows > 0:
                kernel_assert(rows == 128, 'Complete row tiles only')
                row = constants.get_bxs_offset() + dim.get_subtile_start(indices.bxs_tile_idx, index)
                nisa.dma_copy(dst=source,
                    src=params.hidden_tensor[0, nl.ds(row, rows), :params.hidden_size])
                nisa.tensor_scalar_reduce(dst=absolute, data=source, op0=nl.abs,
                    operand0=0.0, reduce_op=nl.maximum, reduce_res=maximum)
                nisa.tensor_scalar(dst=scales[index], data=maximum, op0=nl.multiply,
                    operand0=1.0 / 240.0, op1=nl.maximum, operand1=1e-6)
                nisa.reciprocal(dst=reciprocal, data=scales[index])
                nisa.activation(dst=hidden_tiles[index], data=source,
                    op=nl.copy, scale=reciprocal)
                if params.capture_intermediates:
                    nisa.dma_copy(dst=params.input_bytes[nl.ds(row, rows), :],
                        src=hidden_tiles[index].view(nl.uint8))
                    nisa.dma_copy(dst=params.input_metadata[nl.ds(row, rows), 0:1], src=reciprocal)
                    nisa.dma_copy(dst=params.input_metadata[nl.ds(row, rows), 1:2], src=scales[index])
                    nisa.dma_copy(dst=params.input_metadata[nl.ds(row, rows), 2:3], src=maximum)


def has_dma_xpose(params):
    # Quantization needs channels on free before the existing FP8 PE transpose.
    if inline_input(params):
        return False
    return original_dma_xpose(params)


@nki.jit
def row_mlp_bf16_input(input_values, up_bytes, down_bytes, up_scales, down_scales,
                       capture_intermediates: bool = False):
    tokens, channels = input_values.shape
    hidden = up_bytes.shape[1]
    kernel_assert(up_bytes.shape == (channels, hidden) and down_bytes.shape == (hidden, channels),
                  'BF16 [T,C] input and raw FP8 [C,I]/[I,C] weights')
    kernel_assert(input_values.dtype == nl.bfloat16 and up_bytes.dtype == nl.uint8
                  and down_bytes.dtype == nl.uint8, 'Truthful input and weight dtypes')
    kernel_assert(tokens % 256 == 0 and channels % 256 == 0 and hidden % 512 == 0, 'Complete tiles')
    output = nl.ndarray((1, tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    up = up_bytes.view(nl.float8_e4m3)
    down = down_bytes.view(nl.float8_e4m3)
    params = MLPParameters(input_values.reshape((1, tokens, channels)), up, up, down,
        activation_fn=ActFnType.ReLU, normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.ROW, gate_w_scale=up_scales, up_w_scale=up_scales,
        down_w_scale=down_scales, output_dtype=nl.bfloat16, skip_gate_proj=True, force_cte_mode=True)
    params.capture_intermediates = capture_intermediates
    if capture_intermediates:
        raw_up = nl.ndarray((tokens, hidden), dtype=nl.float32, buffer=nl.shared_hbm)
        relu = nl.ndarray((tokens, hidden), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        intermediate_bytes = nl.ndarray((tokens, hidden), dtype=nl.uint8, buffer=nl.shared_hbm)
        intermediate_scale = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        input_bytes = nl.ndarray((tokens, channels), dtype=nl.uint8, buffer=nl.shared_hbm)
        input_metadata = nl.ndarray((tokens, 3), dtype=nl.float32, buffer=nl.shared_hbm)
        params.raw_up = raw_up
        params.relu = relu
        params.intermediate_bytes = intermediate_bytes
        params.intermediate_scale = intermediate_scale
        params.input_bytes = input_bytes
        params.input_metadata = input_metadata
    validate_mlp_arguments(params)
    mlp_cte_basic(params, output, None)
    if capture_intermediates:
        return (output.reshape((tokens, channels)), raw_up, relu, intermediate_bytes,
                intermediate_scale, input_bytes, input_metadata)
    return output.reshape((tokens, channels))


def install():
    install_row()
    core = importlib.import_module(NAMESPACE + '.core.mlp.mlp_cte.basic.mlp_cte_basic')
    tile_info = importlib.import_module(NAMESPACE + '.core.mlp.mlp_cte.basic.mlp_cte_basic_tile_info')
    core.build_mlp_cte_constants = build_constants
    tile_info.calc_up_proj_footprint = up_footprint
    heap_compat.allocate_hidden_tensor_tile = allocate_hidden
    heap_compat.load_hidden_tensor_tile_opt_fused_add = load_hidden
    heap_compat.mlpp_has_dma_xpose = has_dma_xpose
    emit('NKILIB_INLINE_INPUT_ADAPTER', dict(installed=True, isolated_namespace_only=True,
        input_dtype_bf16=True, online_quantization_inside_loader=True,
        scratch_allocated_by_original_manager=True, scratch_charged_to_tile_planner=True,
        input_hbm_materialization_removed=True, packed_input_control_retained=True,
        upstream_projections_and_rounding_unchanged=True,
        backward_or_training_quality_claim=False))
