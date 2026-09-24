# Reviewer-risk disposition

This is a disclosure, not an approval.

At import time, `train.py` uses `os.environ.setdefault` for two Neuron settings before importing PyTorch:

- When executed as the training program, it supplies the intended full `NEURON_CC_FLAGS` value only if the launcher did not already set one.
- It supplies `/tmp/neuron_cache` as `NEURON_COMPILE_CACHE_DIR` only if the caller did not already select a cache.
- A later `NEURON_CC_FLAGS` fallback is also `setdefault`, so it cannot replace either the launcher value or the training-program value already selected.

These are compiler/cache defaults. They can affect how the training graph is compiled when the launch environment omits values; they do not load data, alter the evaluator or tokenizer, select alternate weights, inspect validation inputs, install hooks, or create an eval-only forward. `load_for_eval` reconstructs the same GPT, strictly loads `model_state`, freezes it, and extends only deterministic nonpersistent rotary buffers for the requested evaluation context.

The full text of both prior portal rejections is unavailable: the authoritative records contain only truncated visible fragments and bind earlier candidate hashes, not `116188cf...`. This package does not infer the missing text. If the human cannot recover an untruncated portal record and exact uploaded-file binding, explicit human acceptance of that unresolved reviewer risk is required before any submission approval.
