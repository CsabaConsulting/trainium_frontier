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
    parser.add_argument("--experimental-adamw-fp32", action="store_true")
    parser.add_argument("--experimental-optimizer-ownership", action="store_true")
    parser.add_argument("--experimental-adaptive-ce", action="store_true")
    args, remaining = parser.parse_known_args()
    if not any(vars(args).values()):
        parser.error("select an explicit scratch integration")
    if args.experimental_adamw_fp32 and args.experimental_optimizer_ownership:
        parser.error("combined ownership and NKI AdamW has not passed its integration gate")
    original_init = T.GPT.init_weights
    original_setup_optimizer = T.GPT.setup_optimizer
    norm_installed = False
    ce_installed = False

    def initialize(model):
        nonlocal norm_installed, ce_installed
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
        if args.experimental_adaptive_ce and not ce_installed:
            from softcap_ce_parallel_integration import install as install_ce
            install_ce(T)
            ce_installed = True
        assert torch.equal(before_rng, torch.get_rng_state())
        print("FRESH_START_TRAIN_INTEGRATION " + json.dumps(dict(
            native_forward_custom_mlp_backward=args.experimental_native_mlp,
            nki_branch_norm=args.experimental_branch_norm,
            nki_adamw_fp32=args.experimental_adamw_fp32,
            optimizer_ownership=args.experimental_optimizer_ownership,
            adaptive_nki_ce=args.experimental_adaptive_ce,
            loader_and_timing_hooks_unchanged=True,
            native_evaluation_and_checkpoint_structure=True)), flush=True)

    def setup_optimizer(model, *setup_args, **setup_kwargs):
        optimizer = original_setup_optimizer(model, *setup_args, **setup_kwargs)
        if args.experimental_adamw_fp32 or args.experimental_optimizer_ownership:
            assert T.USE_FOREACH_OPTIM and T.USE_TENSOR_LR_SCALARS
            assert not T.USE_CAUTIOUS_WD
            if args.experimental_adamw_fp32:
                from adamw_fused_integration import install_optimizer
                install_optimizer(optimizer, T)
                print("FRESH_START_OPTIMIZER_INTEGRATION " + json.dumps(dict(
                    nki_adamw_fp32=True, owner_partitioning=False,
                    fp32_parameters_and_moments=True)), flush=True)
            else:
                import torch.distributed as dist
                assert dist.is_initialized() and dist.get_world_size() == 4
                from optimizer_ownership import install_optimizer
                ownership = install_optimizer(optimizer, T, pad_singleton_muon=True)
                print("FRESH_START_OPTIMIZER_INTEGRATION " + json.dumps(dict(
                    nki_adamw_fp32=False, owner_partitioning=True,
                    rank=dist.get_rank(), **ownership.summary)), flush=True)
        return optimizer

    T.GPT.init_weights = initialize
    T.GPT.setup_optimizer = setup_optimizer
    sys.argv = [sys.argv[0], *remaining]
    T.main()


if __name__ == "__main__":
    main()
