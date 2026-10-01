# A3 audit material — human review pending

hashed bigram embedding table (modded-nanogpt / nanochat / Engram lineage), zero-initialised, learned from the provided training data inside the timed loop by RMSprop with gradient accumulation over two steps, saved with the model, identical train/eval forward, a pure function of token ids at positions <= t; harness untouched.

Upload bytes `29e5cd2cfd35bb52e5a955057a1661786239d0812d9a797c85dc0df7fe2d7446` differ from scored `c5ef8414e2df998a9b52eeb2e249b66c0110a3c37869e676719e93975a3432b4` only by the three authorized replacements (two header sentences and argparse freeze default0). `SOURCE.diff` is the exact derivative diff; `SCORED_VS_D2.diff` records the fixed repeated128 map and charged rank diagnostic.

ALL THREE seed42 scores, including the first, have mean 0.9717176447018706 <=0.97225. Same-seed repeatability is not independent-seed robustness or an official score prediction. No worst-run exclusion.

Only `train.py` is the proposed upload payload. All other files are audit evidence, not dependencies; no checkpoint/dataset/SDK payload. Exact command:

```sh
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py
```

`*.exact.json` containers preserve original receipt bytes (including absent trailing newlines) as base64 plus original SHA256/length; decode original_b64 to recover the unchanged archived receipt. ALL3_RESULT.json is the readable verifier result.

Unknown cumulative writes, clean-box/environment and Claude compliance/readability judgments remain unknown or pending. No automatic PACKAGE READY/Claude ACCEPT/upload claim is made. D2 stays untouched and its accepted 6d197b3c reviewer probe remains first for Poojith. Human upload only after the applicable review.
