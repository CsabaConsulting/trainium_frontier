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

"""Pinned attention row-max helper with mandatory intermediate readouts.

Derived from AWS NKI Library92d11f63, attention_cte.py lines3669-3949.
Only row_max_reduce_res assignment changes. The installed NKI0.5 rejects None
although newer upstream code intentionally omits intermediate readouts. All
original masks, matmuls, reset/accumulate commands and output columns remain.
The extra readouts use already allocated columns; max(all prefixes) = max(all
tiles), so the later row maximum is unchanged for our finite causal fixtures.
This compatibility version pays the extra readout instructions in any timing.
The bundled control and pinned source remain untouched.
"""
import importlib

from nki_library_probe_common import NAMESPACE, emit
from research.vendor.nki_library_92d11f63.core.attention.attention_cte import (
    AttnConfig,
    AttnInternalBuffers,
    AttnTileParams,
    SectionParams,
    _FLOAT32_MIN,
    _K_TILE_SZ,
    _NUM_K_TILES_IN_LARGE_TILE,
    _Q_GRP_SZ,
    _k_tile_selection,
    _load_position_bias_tile,
    _maximum,
    _row_max_chain_endpoints,
    interleave_copy,
    kernel_assert,
    nisa,
    nl,
    reduce_cmd,
)


def _qk_and_max_large_tile_impl(
    qkmax_grp,
    large_tile_idx,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
    batch_id: int = 0,
):
    """Compute QK^T matmul (MM1) and find row-wise maximum for this Q group and large (2048) K tile.
    Also apply masking and position bias if relevant.

    When position_bias is provided, bias is pre-loaded into PSUM before the matmul, and the matmul
    accumulates QK on top of it (accumulate=True). This gives PSUM = QK + bias with zero extra
    instructions beyond a single tensor_copy (SBUF→PSUM). The masking paths then operate on the
    already-biased scores unchanged, giving: final = mask(scale * (QK + bias)).
    """

    q_seqlen_offset = qkmax_grp * atp.sb_p
    # Valid query rows in this Q group (partition dim); loop-invariant across k-tiles.
    num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)

    # Whether position-bias preload is active for this invocation. The actual
    # bias preload happens per-K-tile in the matmul loop below.
    has_bias = ac.use_position_bias and bufs.position_bias_hbm is not None

    # perform matmul and masking in 512 (_K_TILE_SZ) tile increment on the seqlen dimension
    # Compute each k-tile's selection once, shared by _row_max_chain_endpoints and the loop below.
    k_tile_selections = []
    for k_tile_idx in range(_NUM_K_TILES_IN_LARGE_TILE):
        k_tile_selections.append(_k_tile_selection(qkmax_grp, large_tile_idx, k_tile_idx, ac, atp, sp))
    # Row-max chaining: chain this large tile's executed reduces on the Vector accumulator, reading out
    # once (on the last chained tile) instead of per tile. None = fall back to per-tile reduces.
    row_max_first_tile, row_max_last_tile = _row_max_chain_endpoints(k_tile_selections, ac) or (None, None)
    for k_tile_idx in range(_NUM_K_TILES_IN_LARGE_TILE):
        tile, should_process = k_tile_selections[k_tile_idx]

        if not should_process:
            continue
        # Extract relevant tensor tiles for convenience
        mm1_psum_tile = bufs.mm1_psum[qkmax_grp][large_tile_idx][k_tile_idx]
        if not atp.dynamic_sel_mask:
            mm1_copy_sb_tile = bufs.mm1_copy_sb[qkmax_grp][large_tile_idx][k_tile_idx]
            mm1_affine_select_output_tile = bufs.mm1_affine_select_output[qkmax_grp][large_tile_idx][k_tile_idx]
        mm1_masked_tile = bufs.mm1_masked[qkmax_grp][large_tile_idx]
        mm1_partial_max_tile = bufs.mm1_partial_max[qkmax_grp]

        k_tile_idx_in_section = large_tile_idx * _NUM_K_TILES_IN_LARGE_TILE + k_tile_idx
        num_f = min(tile.num_valid_k, _K_TILE_SZ)

        # Step 1: Load position bias into PSUM before matmul (if present)
        # This pre-seeds PSUM with bias so that after matmul accumulation,
        # PSUM contains QK + bias. The DMA for bias can overlap with Q/K loads.
        use_accumulate = False
        if has_bias:
            abs_k_pos = sp.section_offset + k_tile_idx_in_section * _K_TILE_SZ
            bias_sb_tile = bufs.position_bias_sb[qkmax_grp]
            preload_ok = _load_position_bias_tile(
                ac=ac,
                atp=atp,
                bufs=bufs,
                bias_sb_tile=bias_sb_tile,
                batch_id=batch_id,
                q_seqlen_offset=q_seqlen_offset,
                num_p_bias=num_p,
                abs_k_pos=abs_k_pos,
                num_f=num_f,
            )
            if preload_ok:
                # Preload bias into PSUM; matmul below will accumulate on top.
                interleave_copy(mm1_psum_tile[:num_p, :num_f], bias_sb_tile[:num_p, :num_f], index=k_tile_idx)
                use_accumulate = True

        # Step 2: MM1 matmul (accumulates onto bias if pre-loaded, otherwise overwrites)
        if tile.is_prior and bufs.k_sb_prior is not None:
            k_tile_to_use = bufs.k_sb_prior[tile.k_start_pos // _K_TILE_SZ]
        elif bufs.k_sb_prior is not None:
            k_tile_to_use = bufs.k_sb[tile.k_start_pos // _K_TILE_SZ]
        else:
            k_tile_to_use = bufs.k_sb[k_tile_idx_in_section]

        for d_tile in range(atp.num_d_tiles):
            d_chunk = min(atp.d_tile_size_par_dim, ac.d - d_tile * atp.d_tile_size_par_dim)
            nisa.nc_matmul(
                mm1_psum_tile[:num_p, :num_f],
                bufs.q_sb[qkmax_grp // atp.num_q_grps_per_load][d_tile][
                    :d_chunk,
                    nl.ds((qkmax_grp % atp.num_q_grps_per_load) * _Q_GRP_SZ, num_p),
                ],
                k_tile_to_use[d_tile][:d_chunk, :num_f],
                accumulate=True if use_accumulate else None,
            )

        # Step 3: Masking + scale + max reduce (unchanged from no-bias path)
        # PSUM now contains QK + bias (or just QK if no bias)

        # Row-max chain: reset on the first chained tile, accumulate (reduce) on the rest, and read
        # out only on the last; when not chaining, every tile resets and reads out (per-tile reduce).
        chaining = row_max_first_tile is not None
        row_max_reduce_cmd = (
            reduce_cmd.reduce if (chaining and k_tile_idx != row_max_first_tile) else reduce_cmd.reset_reduce
        )
        # Compatibility: NKI0.5 requires a concrete reduction output. Write
        # each running maximum into the already allocated per-tile column.
        # Every prefix maximum <= the last maximum, so the later max over
        # these columns is identical. Reset/accumulate commands stay unchanged.
        row_max_reduce_res = mm1_partial_max_tile[:num_p, k_tile_idx_in_section]

        # For interleaved KV: skip masking entirely if tile is fully visible
        kvp_tile_fully_visible = False
        if ac.kvp_group_size > 0 and not tile.is_prior:
            if ac.kvp_all_tiles_visible:
                kvp_tile_fully_visible = True
            elif ac.kvp_cp_offset_int > 0:
                stride = ac.kvp_group_size * ac.block_size
                last_block_in_tile = (tile.k_start_pos + num_f - 1) // ac.block_size
                max_k_threshold_in_tile = last_block_in_tile * stride + ac.block_size - 1
                # Use adjusted coordinates: min_adjusted_ubs = min_q - seg_block_offset*stride (rank=0 best case)
                min_adjusted_ubs = qkmax_grp * _Q_GRP_SZ + ac.kvp_cp_offset_int - ac.kvp_seg_block_offset_int * stride
                kvp_tile_fully_visible = min_adjusted_ubs > max_k_threshold_in_tile

        diagonal_sel_mask = (
            should_process and ((qkmax_grp * _Q_GRP_SZ) < (tile.k_start_pos + _K_TILE_SZ))
            if (atp.is_causal and not tile.is_prior and not atp.dynamic_sel_mask and not kvp_tile_fully_visible)
            else False
        )
        if ac.use_swa and atp.is_causal and not tile.is_prior:
            # When using SWA, above condition for diagonal_sel_mask might miss some
            # conditions where masking needs to be applied since it only checks for
            # causal condition. Therefore we either need dynamic mask or affine select mask.
            diagonal_sel_mask = not atp.dynamic_sel_mask

        if diagonal_sel_mask:  # static diagonal mask
            nisa.tensor_copy(
                mm1_copy_sb_tile[:num_p, :num_f],
                mm1_psum_tile[:num_p, :num_f],
            )
            nisa.affine_select(
                mm1_affine_select_output_tile[:num_p, :num_f],
                pattern=[[-1, num_f]],
                offset=qkmax_grp * atp.sb_p - tile.k_start_pos,
                channel_multiplier=1,
                cmp_op=nl.greater_equal,
                on_true_tile=mm1_copy_sb_tile[:num_p, :num_f],
                on_false_value=_FLOAT32_MIN,
            )

            if ac.use_swa:
                nisa.affine_select(
                    mm1_affine_select_output_tile[:num_p, :num_f],
                    pattern=[[1, num_f]],
                    offset=(tile.k_start_pos + ac.sliding_window - 1 - qkmax_grp * atp.sb_p),
                    channel_multiplier=-1,
                    cmp_op=nl.greater_equal,
                    on_true_tile=mm1_affine_select_output_tile[:num_p, :num_f],
                    on_false_value=_FLOAT32_MIN,
                )

            nisa.tensor_scalar_reduce(
                mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                data=mm1_affine_select_output_tile[:num_p, :num_f],
                op0=nl.multiply,
                operand0=ac.scale,
                reduce_op=nl.maximum,
                reduce_res=row_max_reduce_res,
                reduce_cmd=row_max_reduce_cmd,
            )

        elif (
            atp.dynamic_sel_mask or tile.is_prior
        ) and not kvp_tile_fully_visible:  # dynamic (compile-time unknown) mask
            if ac.kvp_group_size > 0 and not tile.is_prior and bufs.k_threshold_sb is not None:
                # Static mask path for interleaved KV (3 instructions)
                k_threshold_tile = bufs.k_threshold_sb[:num_p, nl.ds(tile.k_start_pos, num_f)]
                adjusted_ub = bufs.range_sel_ubs[:num_p, qkmax_grp : qkmax_grp + 1]

                # Instruction 1: mask[q,j] = k_threshold[j] > adjusted_ubs[q]
                nisa.tensor_scalar(
                    dst=mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    data=k_threshold_tile,
                    op0=nl.greater,
                    operand0=adjusted_ub,
                )

                # SWA: also mask positions below lower bound
                if ac.use_swa and bufs.range_sel_lbs is not None:
                    adjusted_lb = bufs.range_sel_lbs[:num_p, qkmax_grp : qkmax_grp + 1]
                    nisa.scalar_tensor_tensor(
                        dst=mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                        data=k_threshold_tile,
                        op0=nl.less,
                        operand0=adjusted_lb,
                        op1=nl.maximum,
                        operand1=mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    )

                # Instruction 2: masked_scores = mask * FLOAT32_MIN + scores
                nisa.scalar_tensor_tensor(
                    dst=mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    data=mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    op0=nl.multiply,
                    operand0=_FLOAT32_MIN,
                    op1=nl.add,
                    operand1=mm1_psum_tile[:num_p, :num_f],
                )

                # Instruction 3: scale + max reduction
                nisa.tensor_scalar_reduce(
                    dst=mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    data=mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    op0=nl.multiply,
                    operand0=ac.scale,
                    reduce_op=nl.maximum,
                    reduce_res=row_max_reduce_res,
                    reduce_cmd=row_max_reduce_cmd,
                )
            elif tile.is_prior:
                bound0 = bufs.range_sel_lbs_prior[:num_p, qkmax_grp] if ac.use_swa else bufs.zero_bias_tensor
                bound1 = bufs.range_sel_ubs_prior[:num_p, qkmax_grp]
                comp_op1 = nl.less  # k < prior_used_len
                kernel_assert(ac.scale == 1.0, "range_select path doesn't support scale != 1.0")
                nisa.range_select(
                    mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    on_true_tile=mm1_psum_tile[:num_p, :num_f],
                    on_false_value=_FLOAT32_MIN,
                    comp_op0=nl.greater_equal,
                    comp_op1=comp_op1,
                    bound0=bound0[:num_p, :1],
                    bound1=bound1[:num_p, :1],
                    reduce_op=_maximum,
                    reduce_res=row_max_reduce_res,
                    reduce_cmd=row_max_reduce_cmd,
                    range_start=tile.k_start_pos,
                )
            elif ac.is_sequence_packed:
                bound0 = bufs.range_sel_lbs[:num_p, nl.ds(qkmax_grp, 1)]
                bound1 = bufs.range_sel_ubs[:num_p, nl.ds(qkmax_grp, 1)]
                comp_op1 = nl.less_equal if atp.is_causal else nl.less
                kernel_assert(ac.scale == 1.0, "range_select path doesn't support scale != 1.0")
                nisa.range_select(
                    mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    on_true_tile=mm1_psum_tile[:num_p, :num_f],
                    on_false_value=_FLOAT32_MIN,
                    comp_op0=nl.greater_equal,
                    comp_op1=comp_op1,
                    bound0=bound0[:num_p, :1],
                    bound1=bound1[:num_p, :1],
                    reduce_op=_maximum,
                    reduce_res=row_max_reduce_res,
                    reduce_cmd=row_max_reduce_cmd,
                    range_start=tile.k_start_pos,
                )
            else:
                bound0 = bufs.range_sel_lbs[:num_p, qkmax_grp] if ac.use_swa else bufs.zero_bias_tensor
                bound1 = bufs.range_sel_ubs[:num_p, qkmax_grp]
                comp_op1 = nl.less_equal  # k <= q + cp_offset
                kernel_assert(ac.scale == 1.0, "range_select path doesn't support scale != 1.0")
                nisa.range_select(
                    mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    on_true_tile=mm1_psum_tile[:num_p, :num_f],
                    on_false_value=_FLOAT32_MIN,
                    comp_op0=nl.greater_equal,
                    comp_op1=comp_op1,
                    bound0=bound0[:num_p, :1],
                    bound1=bound1[:num_p, :1],
                    reduce_op=_maximum,
                    reduce_res=row_max_reduce_res,
                    reduce_cmd=row_max_reduce_cmd,
                    range_start=tile.k_start_pos,
                )

        else:  # no masking
            nisa.tensor_scalar_reduce(
                mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                data=mm1_psum_tile[:num_p, :num_f],
                op0=nl.multiply,
                operand0=ac.scale,
                reduce_op=nl.maximum,
                reduce_res=row_max_reduce_res,
                reduce_cmd=row_max_reduce_cmd,
            )


def install():
    module = importlib.import_module(NAMESPACE + ".core.attention.attention_cte")
    assert module._qk_and_max_large_tile_impl is not _qk_and_max_large_tile_impl
    module._qk_and_max_large_tile_impl = _qk_and_max_large_tile_impl
    emit("NKILIB_READOUT_COMPAT", dict(installed=True, isolated_upstream_module_only=True,
        intermediate_row_max_readouts=True, extra_sbuf_allocation=False,
        reset_accumulate_commands_unchanged=True, bundled_control_unchanged=True))
