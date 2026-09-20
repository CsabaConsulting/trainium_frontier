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

"""Use upstream automatic SBUF placement with unchanged training arithmetic.

The checked allocator retains original virtual capacity/lifetime/underflow
checks. PSUM banking remains upstream. Every fixture and complete-model gate
is required before treating this compiler-placement hypothesis as a speedup.
"""
import nki
import nki.language as nl

from nki_library_relu2_adapter import token_row_sharding as calculate_sharding
from research.vendor.nki_library_92d11f63.core.mlp.mlp_parameters import (
    MLPParameters, validate_mlp_arguments,
)
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.basic.mlp_cte_basic import (
    _execute_on_shard,
)
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.mlp_cte_sharding import (
    DimShard, ShardedDim, is_launch_grid_valid_for_mlp,
)
from nki_library_compiler_sbuf_allocator import CheckedCompilerSbufManager
from research.vendor.nki_library_92d11f63.core.mlp.mlp_cte.mlp_cte_constants import MAX_AVAILABLE_SBUF_SIZE
from research.vendor.nki_library_92d11f63.core.utils.common_types import (
    ActFnType, NormType, QuantizationType,
)
from research.vendor.nki_library_92d11f63.core.utils.kernel_assert import kernel_assert
from research.vendor.nki_library_92d11f63.core.utils.kernel_helpers import (
    get_program_sharding_info, is_launched_as_spmd,
)
from research.vendor.nki_library_92d11f63.core.utils.logging import get_logger


def compiler_placed_mlp_cte_basic(
    mlp_params: MLPParameters,
    output_tensor_hbm: nl.NkiTensor,
    output_stored_add_tensor_hbm: nl.NkiTensor,
):
    """
    MLP CTE kernel for non-MX quantization types (NONE, ROW, STATIC).
    """
    kernel_assert(
        is_launch_grid_valid_for_mlp(),
        "Launch grid is not valid. MLP CTE only supports sharding on 1 dimension.",
    )

    top_level_interleave_degree = 1  # if mlp_params.hidden_size >= 8192 else 2
    sbm = CheckedCompilerSbufManager(
        sb_lower_bound=0,
        sb_upper_bound=MAX_AVAILABLE_SBUF_SIZE,
        logger=get_logger("mlp_cte"),
        use_auto_alloc=True,
    )
    sbm.open_scope(interleave_degree=top_level_interleave_degree, name='mlp_cte_basic')

    if is_launched_as_spmd():
        _, total_programs, program_id = get_program_sharding_info()

        sharding_info = calculate_sharding(mlp_params)
        for shard_idx in range(len(sharding_info.shards)):
            shard = sharding_info.shards[shard_idx]
            _execute_on_shard(
                shard.shard_mlp_params,
                sharding_info.sharded_dim,
                shard,
                total_programs,
                program_id,
                shard_idx,
                sbm,
                output_tensor_hbm,
                output_stored_add_tensor_hbm,
            )

    else:  # No SPMD
        dim_shard = DimShard(
            dim_offset=0,
            dim_size=mlp_params.batch_size * mlp_params.sequence_len,
            shard_mlp_params=mlp_params,
        )
        _execute_on_shard(
            mlp_params,
            ShardedDim.BATCH_X_SEQUENCE_LENGTH,
            dim_shard,
            1,
            0,
            0,
            sbm,
            output_tensor_hbm,
            output_stored_add_tensor_hbm,
        )

    sbm.close_scope()


@nki.jit
def saved_relu_mlp_compiler_sbuf(x, up, down):
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
    compiler_placed_mlp_cte_basic(params, output, None)
    return output.reshape((tokens, channels)), saved
