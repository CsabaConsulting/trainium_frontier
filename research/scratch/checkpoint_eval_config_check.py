#!/usr/bin/env python3
"""Check standalone save/load semantics across incompatible architecture flags.

This uses real nonzero model branches and deliberately changes live module
defaults before loading. It catches both strict-load failures and silent changes
to logits. CPU checks do not substitute for native checkpoint evaluation.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", choices=["baseline", "norm", "attention"], default="baseline")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(937)
    T.COMPUTE_DTYPE = torch.float32
    T.EVAL_MIN_SEQ_LEN = 256  # Same table-extension path without two 8192-square CPU masks.
    T.USE_NKI_ROPE_NORM = T.USE_NKI_SOFTCAP_CE = False
    T.USE_ENGRAM = T.USE_OUT_BIGRAM = False
    T.USE_VE = T.USE_LOCAL_CONV = T.TRAIN_LOCAL_CONV = True
    T.LOCAL_CONV_KERNEL = 2
    T.NGRAM_VE_TABLE_MULT = 1
    T.NGRAM_VE_DIM = 16
    T.NGRAM_VE_LAYERS = "0,1"
    T.NGRAM_VE_BF16 = False
    T.NGRAM_VE_TRIGRAM = True
    cfg = T.GPTConfig(sequence_len=128, vocab_size=256, n_layer=2, n_head=1, n_embd=128)
    x = torch.randint(cfg.vocab_size, (2, 128), dtype=torch.int32)
    reports = []
    with tempfile.TemporaryDirectory(prefix="trainium-eval-config-") as folder:
        for tables in (False, True):
            T.USE_NGRAM_VE = T.USE_QK_SHIFT = tables
            T.USE_NKI_LOCAL_CONV = True
            T.USE_BF16_NORM_OUTPUT = args.policy == "norm"
            T.USE_BF16_SDPA_INPUT = args.policy == "attention"
            T.LOGIT_SOFTCAP = 7.0
            model = T.GPT(cfg)
            model.init_weights()
            with torch.no_grad():
                for block in model.transformer.h:
                    block.attn.c_proj.weight.normal_(std=0.02)
                    block.mlp.c_proj.weight.normal_(std=0.02)
                    block.local_conv.weight.normal_(std=0.1)
                    if tables:
                        block.qk_shift_beta.fill_(0.07)
                expected = model.eval()(x)
            saved = T.current_toggles()
            expected_tables = set(model.ngram_embeds)
            assert bool(expected_tables) == tables
            path = Path(folder) / f"tables-{tables}.pt"
            T.save_checkpoint(path, model, {"purpose": "standalone configuration gate"})
            # Opposite parameter layout plus a same-shape silent behavior change.
            T.USE_NGRAM_VE = T.USE_QK_SHIFT = not tables
            T.USE_NKI_LOCAL_CONV = False
            T.USE_BF16_NORM_OUTPUT = not saved["USE_BF16_NORM_OUTPUT"]
            T.USE_BF16_SDPA_INPUT = not saved["USE_BF16_SDPA_INPUT"]
            T.NGRAM_VE_TABLE_MULT = 3
            T.LOGIT_SOFTCAP = 15.0
            loaded = T.load_for_eval(str(path), torch.device("cpu"))
            assert T.current_toggles() == saved
            assert loaded.config.sequence_len == 256
            with torch.no_grad():
                actual = loaded(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # VE exists only on selected alternating layers; a two-layer model
            # need not have two table-bearing sites even when both are requested.
            assert set(loaded.ngram_embeds) == expected_tables
            # New flags absent from a legacy payload fall back to import defaults,
            # never to the current model. Unknown keys cannot replace code globals.
            legacy = torch.load(path, weights_only=True)
            legacy["toggles"].pop("USE_NKI_LOCAL_CONV")
            legacy["toggles"]["norm"] = "must never replace a function"
            legacy_path = Path(folder) / f"legacy-{tables}.pt"
            torch.save(legacy, legacy_path)
            legacy_model = T.load_for_eval(str(legacy_path), torch.device("cpu"))
            assert T.USE_NKI_LOCAL_CONV is False and callable(T.norm)
            with torch.no_grad():
                torch.testing.assert_close(legacy_model(x), expected, rtol=0, atol=0)
            reports.append(dict(policy=args.policy, tables=tables, table_keys=sorted(expected_tables),
                                qk_shift=tables, max_abs=0.0,
                                changed_live_defaults_restored=True,
                                legacy_missing_flag_uses_import_default=True,
                                unknown_key_ignored=True, nki_cpu_fallback_exact=True))
    print(json.dumps(dict(checks=reports, passed=True)), flush=True)


if __name__ == "__main__":
    main()
