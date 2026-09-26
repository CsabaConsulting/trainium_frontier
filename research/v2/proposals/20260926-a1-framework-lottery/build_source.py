"""Mechanical scalar/CLI-only variant of the frozen FP32 A1 implementation.

No kernel changes; the existing ordinary framework branch is selected.
"""
import ast
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
parent=REPO/'research/v2/lanes/tables-compact-k2/train.py'
raw=parent.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='4b74ed48c772a5e82807d5509529704deaf1c8cd56692705efe9047ea7ae4a7b'
text=raw.decode()
replacements=[
 ('TOTAL_BATCH_SIZE = 131072','TOTAL_BATCH_SIZE = 262144'),
 ('MATRIX_LR = 0.015','MATRIX_LR = 0.021'),
 ('Default off; requires K2 for native training.','Default off; standard framework backward only in this lottery.'),
 ("if args.ngram_ve and not K2_EMBEDDING:\n        parser.error('native A1 table training requires --k2-embedding')",
  "if K2_EMBEDDING:\n        parser.error('K2 is closed; this lottery requires standard framework backward')")]
for before,after in replacements:
    assert text.count(before)==1,before
    text=text.replace(before,after)
ast.parse(text)
# No model, backward, optimizer, checkpoint, or evaluation function changed.
old={n.name:ast.dump(n) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
new={n.name:ast.dump(n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert old.keys()==new.keys()
assert [k for k in old if old[k]!=new[k]]==['main']
target=HERE/'train.py'
data=text.encode()
if target.exists(): assert target.read_bytes()==data,'FROZEN_SOURCE_DRIFT'
else: target.write_bytes(data)
print(hashlib.sha256(data).hexdigest())
