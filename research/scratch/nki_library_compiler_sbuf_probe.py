#!/usr/bin/env python3
"""Keep every source175 training fixture; change only SBUF placement."""
import hashlib
from pathlib import Path

import nki_library_training_mlp_reuse_transpose_probe as original
from nki_library_compiler_sbuf_mlp import saved_relu_mlp_compiler_sbuf
from nki_library_probe_common import emit


if __name__ == '__main__':
    path = Path(original.__file__)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == '97ae3dbd9032dcfee6a5a9c43c505e7d99cec51a88faac7134fd0e050fccf9f7'
    original.saved_relu_mlp_reuse_transpose = saved_relu_mlp_compiler_sbuf
    emit('NKILIB_COMPILER_SBUF', dict(
        original_probe_sha256=digest,
        original_probe_source_unchanged=True,
        compiler_allocated_sbuf=True,
        logical_allocator_capacity_and_lifetime_checks_retained=True,
        original_tiling_and_arithmetic_retained=True,
        all_original_fixtures_and_bounds_retained=True,
        model_scheduling_hypothesis=True, component_speedup_required=False))
    original.main()
