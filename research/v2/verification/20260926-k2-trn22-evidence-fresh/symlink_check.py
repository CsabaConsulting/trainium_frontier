import sys, json
from pathlib import Path
sys.path.insert(0, '/home/poojith-devan/k2-trn22-evidence-review-Mh0Zed/candidate')
from monitor import tree_bytes
root = Path(__file__).resolve().parent/'own-symlink-probe'
root.mkdir()
(root/'cache').mkdir()
(root/'cache'/'debug').mkdir()
(root/'cache'/'debug'/'default_control_fast_table.bin').symlink_to('/does-not-exist-reviewer-probe')
try:
    tree_bytes(root/'cache')
except RuntimeError as exc:
    if str(exc) != 'SCRATCH_SYMLINK':
        raise
    print(json.dumps({'native':False, 'result':'REJECT', 'reason':str(exc), 'test':'Own symlink in cache/debug; actual tree_bytes; no native runtime', 'target':'/does-not-exist-reviewer-probe'}))
else:
    raise RuntimeError('symlink unexpectedly accepted')
