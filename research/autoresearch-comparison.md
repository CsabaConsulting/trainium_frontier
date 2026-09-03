# Trainium Autoresearch vs. Karpathy's Original Autoresearch

Comparison of `trainium-frontier/autoresearch/` (+ root `train.py`/`prepare.py`) against the cloned original `autoresearch/` (Karpathy).

## Verdict

Trainium version is a **faithful port** of Karpathy's autoresearch to AWS Trainium 2, not an independent rewrite. Same nanochat-derived model, same competition-loop philosophy, swapped CUDA/flash-attn/torch.compile stack for Neuron/NKI stack.

## Shared core (present in both)

- GPT architecture: `class GPT`, `class Block`, `class CausalSelfAttention`, `class MLP`
- Rotary embeddings
- Muon optimizer
- `class Tokenizer`, `def evaluate_bpb`
- `program.md` skeleton: branch-per-run workflow, `results.tsv` logging, 5-minute fixed time budget, "NEVER STOP" autonomous-loop instructions, simplicity criterion, keep/discard/crash status logging

## Trainium-specific deltas

| Area | Karpathy original | Trainium port |
|---|---|---|
| Device | Hardcoded CUDA | Device abstraction: `torch.device("neuron")`, `neuron_available()`, `detect_device_type()` |
| Attention kernel | `kernels-community/flash-attn3` (FA3) | Removed — eager `F.scaled_dot_product_attention` |
| `torch.compile` | Used (4 refs) | Mostly removed (3 refs, conditional) |
| `value_embed` | 12 refs | 0 refs (dropped) |
| `get_lr` helper | Present | Removed |
| Custom kernel example | None | `_relu_sq_fwd`/`_relu_sq_bwd` — illustrative fused NKI kernel (relu², off by default), teaches 128-partition SBUF tiling + manual autograd wrap |
| CLI | Bare `uv run train.py` | `argparse` + `run.sh` launcher |
| Package mgmt | `uv.lock` / `pyproject.toml` | `requirements.txt` |
| `results.tsv` columns | 5 (incl. `memory_gb`) | 4 (no VRAM column) |
| `prepare.py` size | 389 lines | 861 lines (extra Neuron data/tokenizer plumbing) |
| `train.py` size | 630 lines | 758 lines |
| Constants naming | `TIME_BUDGET`, `MAX_SEQ_LEN` | `TRAIN_TIME_BUDGET_SECONDS`, `TOKENIZER_VOCAB_SIZE` |

## program.md diff highlights

- Platform section added up top: NeuronCore constraints (no CUDA, ~24GB HBM/LNC=2, NEFF recompilation cost on shape changes, bf16 weights/fp32 optimizer state, stochastic rounding).
- Launch command: `bash run.sh` / `NEURON_LOGICAL_NC_CONFIG=2 python3 train.py` instead of `uv run train.py`.
- Data cache path: `~/.cache/neuron_competition_r1/` instead of `~/.cache/autoresearch/`.
- Branch base: `master` → `mainline`.
- Dropped VRAM soft-constraint language (not applicable to fixed-size NeuronCore HBM the same way).
- Timeout tightened: kill at 15 min (trainium) vs 10 min (original).
