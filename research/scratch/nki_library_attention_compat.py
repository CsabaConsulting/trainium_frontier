"""Materialize one upstream property for the installed NKI0.5 parser.

The source166 failure resolves KTileInfo.num_valid_k to its Python function
instead of the property's integer result. Preserve its exact expression in a
dataclass field. Classification, masks, matmuls and reduction schedules remain
in the unchanged pinned source. Only the isolated module's factory is replaced.
"""
from dataclasses import dataclass
import importlib

import nki.language as nl

from nki_library_probe_common import NAMESPACE, emit


@dataclass
class MaterializedKTileInfo(nl.NKIObject):
    is_prior: bool
    seqlen_k: int
    k_start_pos: int
    unmasked: bool
    num_valid_k: int


def make_tile_info(is_prior, seqlen_k, k_start_pos, unmasked):
    return MaterializedKTileInfo(is_prior, seqlen_k, k_start_pos, unmasked,
                                 seqlen_k - k_start_pos)


def install():
    module = importlib.import_module(NAMESPACE + ".core.attention.attention_cte")
    original = module.KTileInfo
    assert original is not make_tile_info, "Install this process-local adapter only once"
    cases = 0
    for length in (0, 1, 127, 128, 511, 512, 1024, 2048):
        for start in (0, 127, 512, 1536):
            for prior in (False, True):
                for unmasked in (False, True):
                    before = original(prior, length, start, unmasked)
                    after = make_tile_info(prior, length, start, unmasked)
                    for field in ("is_prior", "seqlen_k", "k_start_pos", "unmasked", "num_valid_k"):
                        assert getattr(before, field) == getattr(after, field), field
                    cases += 1
    module.KTileInfo = make_tile_info
    emit("NKILIB_PARSER_COMPAT", dict(passed=True, cpu_factory_cases=cases,
                                     materialized_property="seqlen_k - k_start_pos",
                                     installed_compiler_unchanged=True,
                                     isolated_upstream_module_only=True))
