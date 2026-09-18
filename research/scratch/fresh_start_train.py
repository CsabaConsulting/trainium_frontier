#!/usr/bin/env python3
"""Run the unchanged trainer with explicit, gated scratch integrations.

Evaluation and ordinary checkpoint parameters retain their native definitions.
The entrypoint, selected integration, and full source hashes belong in the run
manifest. This script does not alter the loader, timing hooks, or scorer.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experimental-native-mlp", action="store_true")
    parser.add_argument("--experimental-branch-norm", action="store_true")
    args, remaining = parser.parse_known_args()
    if not (args.experimental_native_mlp or args.experimental_branch_norm):
        parser.error("select an explicit scratch integration")
    original_init = T.GPT.init_weights
    norm_installed = False

    def initialize(model):
        nonlocal norm_installed
        original_init(model)
        assert T.USE_BF16_NORM_OUTPUT and T.COMPUTE_DTYPE == torch.bfloat16
        before_rng = torch.get_rng_state().clone()
        if args.experimental_native_mlp:
            from mlp_fused_integration import install
            install(model, "neuron", T, forward_backend="native")
        if args.experimental_branch_norm and not norm_installed:
            from branch_norm_nki import install as install_norm
            install_norm(T)
            norm_installed = True
        assert torch.equal(before_rng, torch.get_rng_state())
        print("FRESH_START_TRAIN_INTEGRATION " + json.dumps(dict(
            native_forward_custom_mlp_backward=args.experimental_native_mlp,
            nki_branch_norm=args.experimental_branch_norm,
            loader_and_timing_hooks_unchanged=True,
            native_evaluation_and_checkpoint_structure=True)), flush=True)

    T.GPT.init_weights = initialize
    sys.argv = [sys.argv[0], *remaining]
    T.main()


if __name__ == "__main__":
    main()
