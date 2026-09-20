"""Narrow ReLU-squared adapter for the isolated upstream CTE MLP.

Upstream defines ActFnType.SquaredReLU and apply_activation(), but the CTE
projection still calls get_nl_act_fn_from_type(), which explicitly rejects that
two-instruction activation. Install this hook only in our isolated namespace.
The pinned source files and the installed compiler library remain unchanged.
"""
import importlib

import nki.language as nl

from nki_library_probe_common import NAMESPACE

_helpers = importlib.import_module(NAMESPACE + ".core.utils.kernel_helpers")
_assertions = importlib.import_module(NAMESPACE + ".core.utils.kernel_assert")
_types = importlib.import_module(NAMESPACE + ".core.utils.common_types")
apply_activation = _helpers.apply_activation
kernel_assert = _assertions.kernel_assert
ActFnType = _types.ActFnType


def relu2_evict(mlp_params, constants, psum_bank, proj_results_sbuf_list,
                bxs_subtile, int_tile, bias_sbuf, static_scales_sbuf,
                weight_row_scales_sbuf, hidden_scales_sbuf_list):
    kernel_assert(mlp_params.skip_gate_proj, "ReLU2 adapter requires an ungated MLP")
    kernel_assert(mlp_params.activation_fn == ActFnType.SquaredReLU, "ReLU2 only")
    kernel_assert(not mlp_params.quant_params.is_quant(), "BF16 adapter; no quantization")
    kernel_assert(bias_sbuf == None, "No projection bias in this model")
    kernel_assert(static_scales_sbuf == None, "No static scales")
    kernel_assert(weight_row_scales_sbuf == None, "No row scales")
    kernel_assert(hidden_scales_sbuf_list == None, "No input scales")
    dst = proj_results_sbuf_list[bxs_subtile.index][
        :bxs_subtile.size, nl.ds(int_tile.start_offset, int_tile.size)]
    # Upstream's own two-instruction implementation: BF16 ReLU then BF16
    # square. The intermediate rounding matches our native MLP contract.
    apply_activation(dst=dst, data=psum_bank[:bxs_subtile.size, :int_tile.size],
                     act_fn=ActFnType.SquaredReLU)


def install():
    module = importlib.import_module(NAMESPACE + ".core.mlp.mlp_cte.basic.mlp_cte_basic_projection")
    if module._evict_gate_res_tile is not relu2_evict:
        module._evict_gate_res_tile = relu2_evict
