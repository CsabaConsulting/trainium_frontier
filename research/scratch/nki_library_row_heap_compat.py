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

"""Pinned ROW CTE shard with allocation-matched scale cleanup.

Copied from upstream92d11f63 mlp_cte_basic._mlp_cte_single_shard. The only AST
change makes the second ROW scale pop conditional on an allocated gate branch.
Up scale is allocated last; release it first. Keep strict allocator underflow
checks and all arithmetic unchanged. STATIC mode is outside this ROW adapter.
"""
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.basic.mlp_cte_basic import (
    GateUpDim,
    MLPCTEBasicTileInfo,
    MLPCTEConstants,
    MLPParameters,
    MlpBxsIndices,
    SbufManager,
    ShardedDim,
    allocate_down_projection_weights,
    allocate_hidden_tensor_tile,
    allocate_intermediate_tensor_tile,
    allocate_output_tensor_tile,
    allocate_src_projection_weights,
    apply_normalization_if_necessary,
    cleanup_heap_for_normalization,
    load_bias_vector,
    load_hidden_tensor_tile_opt_fused_add,
    load_source_projection_row_scales,
    load_vector_across_partitions,
    mlpp_has_dma_xpose,
    mlpp_has_down_projection_bias,
    mlpp_has_gate_projection_bias,
    mlpp_has_normalization,
    mlpp_has_normalization_bias,
    mlpp_has_normalization_weights,
    mlpp_has_up_projection_bias,
    nl,
    perform_down_projection,
    perform_gate_up_projection,
    perform_intermediate_quantization,
    prepare_static_scales,
    store_half_hidden_tensor_tile,
    store_hidden_tensor_tile,
    sync_down_proj_results_across_int_dim,
    transpose_intermediate_tensor_tile,
    transpose_source_tensor_tile,
)


def _mlp_cte_single_shard(program_id: int, shard_idx: int, mlp_params: MLPParameters, tile_info: MLPCTEBasicTileInfo, constants: MLPCTEConstants, output_tensor_hbm: nl.NkiTensor, output_stored_add_tensor_hbm: nl.NkiTensor, sbm: SbufManager):
    if constants.sharded_dim == ShardedDim.BATCH_X_SEQUENCE_LENGTH:
        batch_range = 1
    else:
        batch_range = mlp_params.batch_size
    gate_proj_bias_tensor_sbuf = load_bias_vector(mlp_params.bias_params.gate_proj_bias_tensor, constants.compute_data_type, sbm.alloc_heap) if mlpp_has_gate_projection_bias(mlp_params) else None
    up_proj_bias_tensor_sbuf = load_bias_vector(mlp_params.bias_params.up_proj_bias_tensor, constants.compute_data_type, sbm.alloc_heap) if mlpp_has_up_projection_bias(mlp_params) else None
    down_proj_bias_tensor_sbuf = load_bias_vector(mlp_params.bias_params.down_proj_bias_tensor, constants.compute_data_type, sbm.alloc_heap) if mlpp_has_down_projection_bias(mlp_params) else None
    gate_proj_row_weight_scales_sbuf = load_source_projection_row_scales(mlp_params, tile_info, constants, mlp_params.quant_params.gate_w_scale, sbm, tensor_name=f'gate_proj_row_scales__shard{shard_idx}__prog{program_id}') if mlp_params.quant_params.is_logical_quant_row() and (not mlp_params.skip_gate_proj) else None
    up_proj_row_weight_scales_sbuf = load_source_projection_row_scales(mlp_params, tile_info, constants, mlp_params.quant_params.up_w_scale, sbm, tensor_name=f'up_proj_row_scales__shard{shard_idx}__prog{program_id}') if mlp_params.quant_params.is_logical_quant_row() else None
    gate_up_proj_static_input_scales_sbuf = None
    down_proj_static_input_scales_sbuf = None
    gate_proj_static_weight_scales_sbuf = None
    up_proj_static_weight_scales_sbuf = None
    down_proj_static_weight_scales_sbuf = None
    if mlp_params.quant_params.is_logical_quant_static():
        gate_up_proj_static_input_scales_sbuf = sbm.alloc_heap((nl.tile_size.pmax, 1), dtype=nl.float32, name=f'gate_up_proj_static_input_scales__shard{shard_idx}__prog{program_id}')
        down_proj_static_input_scales_sbuf = sbm.alloc_heap((nl.tile_size.pmax, 1), dtype=nl.float32, name=f'down_proj_static_input_scales__shard{shard_idx}__prog{program_id}')
        if not mlp_params.skip_gate_proj:
            gate_proj_static_weight_scales_sbuf = sbm.alloc_heap((nl.tile_size.pmax, 1), dtype=nl.float32, name=f'gate_proj_static_weight_scales__shard{shard_idx}__prog{program_id}')
        up_proj_static_weight_scales_sbuf = sbm.alloc_heap((nl.tile_size.pmax, 1), dtype=nl.float32, name=f'up_proj_static_weight_scales__shard{shard_idx}__prog{program_id}')
        down_proj_static_weight_scales_sbuf = sbm.alloc_heap((nl.tile_size.pmax, 1), dtype=nl.float32, name=f'down_proj_static_weight_scales__shard{shard_idx}__prog{program_id}')
        prepare_static_scales(mlp_params, constants, gate_up_proj_static_input_scales_sbuf, down_proj_static_input_scales_sbuf, gate_proj_static_weight_scales_sbuf, up_proj_static_weight_scales_sbuf, down_proj_static_weight_scales_sbuf)
    norm_weights_tensor_sbuf = load_vector_across_partitions(mlp_params.norm_params.normalization_weights_tensor, constants.norm_weights_bias_data_type, sbm.alloc_heap, tensor_name=f'norm_weights_tensor__shard{shard_idx}__prog{program_id}') if mlpp_has_normalization_weights(mlp_params) else None
    norm_bias_tensor_sbuf = load_vector_across_partitions(mlp_params.norm_params.normalization_bias_tensor, constants.norm_weights_bias_data_type, sbm.alloc_heap, tensor_name=f'norm_bias_tensor__shard{shard_idx}__prog{program_id}') if mlpp_has_normalization_bias(mlp_params) else None
    for batch_idx in range(batch_range):
        for bxs_tile_idx in range(tile_info.bxs_dim_tile.tile_count):
            indices = MlpBxsIndices(program_id, shard_idx, batch_idx, bxs_tile_idx)
            sbm.open_scope(name='intermediate_alive')
            src_proj_res_sbuf_list = []
            allocate_intermediate_tensor_tile(mlp_params, tile_info, constants, indices, 'src_proj_res_sbuf', constants.compute_data_type, src_proj_res_sbuf_list, sbm.alloc_stack)
            sbm.open_scope(name='gate_up_projection')
            hidden_tile_sbuf_list = []
            hidden_tile_scales_sbuf_list = []
            allocate_hidden_tensor_tile(mlp_params, tile_info, constants, indices, hidden_tile_sbuf_list, hidden_tile_scales_sbuf_list, sbm.alloc_stack, sbm.alloc_stack)
            load_hidden_tensor_tile_opt_fused_add(mlp_params, tile_info, constants, indices, hidden_tile_sbuf_list, hidden_tile_scales_sbuf_list, output_stored_add_tensor_hbm)
            apply_normalization_if_necessary(mlp_params, tile_info, constants, indices, hidden_tile_sbuf_list, hidden_tile_sbuf_list, sbm.alloc_heap)
            if mlpp_has_normalization(mlp_params):
                cleanup_heap_for_normalization(mlp_params, tile_info, sbm)
            src_proj_weights_sbuf_list = []
            allocate_src_projection_weights(mlp_params, tile_info, constants, indices, src_proj_weights_sbuf_list, sbm.alloc_stack, sbm)
            if not mlpp_has_dma_xpose(mlp_params):
                transpose_source_tensor_tile(mlp_params, tile_info, constants, indices, hidden_tile_sbuf_list, norm_weights_tensor_sbuf, norm_bias_tensor_sbuf, hidden_tile_sbuf_list, sbm)
            if not mlp_params.skip_gate_proj:
                perform_gate_up_projection(mlp_params, tile_info, constants, indices, GateUpDim.GATE, hidden_tile_sbuf_list, src_proj_weights_sbuf_list, gate_proj_bias_tensor_sbuf, gate_proj_static_weight_scales_sbuf, gate_proj_row_weight_scales_sbuf, hidden_tile_scales_sbuf_list, src_proj_res_sbuf_list, sbm)
            perform_gate_up_projection(mlp_params, tile_info, constants, indices, GateUpDim.UP, hidden_tile_sbuf_list, src_proj_weights_sbuf_list, up_proj_bias_tensor_sbuf, up_proj_static_weight_scales_sbuf, up_proj_row_weight_scales_sbuf, hidden_tile_scales_sbuf_list, src_proj_res_sbuf_list, sbm)
            sbm.close_scope()
            output_tile_sbuf_list = []
            allocate_output_tensor_tile(mlp_params, tile_info, constants, indices, output_tile_sbuf_list, sbm.alloc_heap)
            if mlp_params.quant_params.is_quant():
                intermediate_dequant_scales_sbuf_list = []
                for bxs_subtile_idx in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
                    if mlp_params.quant_params.is_quant_row():
                        intermediate_dequant_scales_sbuf = sbm.alloc_heap((tile_info.bxs_dim_tile.subtile_dim_info.tile_size, 1), dtype=nl.float32, name=indices.get_tensor_name('intermediate_scale_tensor', f'subbxs{bxs_subtile_idx}'))
                        intermediate_dequant_scales_sbuf_list.append(intermediate_dequant_scales_sbuf)
                intermediate_tensor_sbuf_list = []
                allocate_intermediate_tensor_tile(mlp_params, tile_info, constants, indices, 'intermediate_tensor', constants.down_proj_quant_data_type, intermediate_tensor_sbuf_list, sbm.alloc_heap)
                perform_intermediate_quantization(mlp_params, tile_info, constants, indices, bxs_tile_idx, src_proj_res_sbuf_list, intermediate_tensor_sbuf_list, intermediate_dequant_scales_sbuf_list, down_proj_static_input_scales_sbuf, sbm)
                sbm.close_scope()
            else:
                intermediate_dequant_scales_sbuf_list = None
                intermediate_tensor_sbuf_list = src_proj_res_sbuf_list
            down_proj_weights_sbuf_list = []
            allocate_down_projection_weights(mlp_params, tile_info, constants, indices, down_proj_weights_sbuf_list, sbm.alloc_heap)
            transpose_intermediate_tensor_tile(mlp_params, tile_info, constants, indices, intermediate_tensor_sbuf_list, intermediate_tensor_sbuf_list, sbm)
            perform_down_projection(mlp_params, tile_info, constants, indices, intermediate_tensor_sbuf_list, mlp_params.down_proj_weights_tensor, down_proj_weights_sbuf_list, down_proj_bias_tensor_sbuf, down_proj_static_weight_scales_sbuf, intermediate_dequant_scales_sbuf_list, output_tile_sbuf_list, sbm)
            for weight_buffer_idx in range(len(down_proj_weights_sbuf_list)):
                sbm.pop_heap()
            if mlp_params.quant_params.is_quant():
                for bxs_subtile_idx in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
                    sbm.pop_heap()
            if mlp_params.quant_params.is_quant_row():
                for bxs_subtile_idx in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
                    sbm.pop_heap()
            if not mlp_params.quant_params.is_quant():
                sbm.close_scope()
            if constants.sharded_dim == ShardedDim.INTERMEDIATE:
                sync_down_proj_results_across_int_dim(mlp_params, tile_info, constants, indices, output_tile_sbuf_list, sbm)
                store_half_hidden_tensor_tile(mlp_params, tile_info, constants, indices, output_tile_sbuf_list, output_tensor_hbm)
            else:
                store_hidden_tensor_tile(mlp_params, tile_info, constants, indices, output_tile_sbuf_list, output_tensor_hbm)
            for bxs_subtile_idx in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
                sbm.pop_heap()
            sbm.increment_section()
    if mlpp_has_normalization_bias(mlp_params):
        sbm.pop_heap()
    if mlpp_has_normalization_weights(mlp_params):
        sbm.pop_heap()
    if mlp_params.quant_params.is_logical_quant_row():
        sbm.pop_heap()
        if not mlp_params.skip_gate_proj:
            sbm.pop_heap()
    elif mlp_params.quant_params.is_logical_quant_static():
        sbm.pop_heap()
        sbm.pop_heap()
        sbm.pop_heap()
        sbm.pop_heap()
        sbm.pop_heap()
    if mlpp_has_down_projection_bias(mlp_params):
        sbm.pop_heap()
    if mlpp_has_up_projection_bias(mlp_params):
        sbm.pop_heap()
    if mlpp_has_gate_projection_bias(mlp_params):
        sbm.pop_heap()
