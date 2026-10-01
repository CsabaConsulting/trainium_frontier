# D2 CONTINUE — table-method framing

Hashed bigram embedding table (modded-nanogpt / nanochat / Engram lineage),
zero-initialised, learned from the provided training data inside the timed
loop by RMSprop with gradient accumulation over two steps, saved with the
model, identical train/eval forward, a pure function of token ids at
positions <= t; harness untouched.

Round69 requested this description. It adds no code change and does not
replace the original audit evidence. Upload payload is
`train_UPLOAD_6d197b3c.py`, to be named `train.py`; original `train.py` is
the frozen scored b758 source and must not be used as the upload payload.
See UPLOAD_README.txt and CLAUDE_AUDIT_RESULT.md. Poojith alone uploads.
