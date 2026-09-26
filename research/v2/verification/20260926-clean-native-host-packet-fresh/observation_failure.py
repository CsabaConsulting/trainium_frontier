"""Independent fault: failed directory observation must not become zero bytes."""
import importlib.util, json, os, sys
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode=True
p=Path('/home/poojith-devan/clean36-native-trn21-packet-fresh-O2iY7F/packet/host_driver.py')
s=importlib.util.spec_from_file_location('driver',p); d=importlib.util.module_from_spec(s); s.loader.exec_module(d)
root=Path(__file__).parent/('scan-optimized' if sys.flags.optimize else 'scan-normal'); root.mkdir()
(root/'receipts').mkdir(); (root/'receipts/.rank0.interrupted.tmp').write_bytes(b'protected partial')
original=os.scandir
def denied(path):
    if Path(path)==root/'receipts': raise PermissionError('independent simulated directory EACCES')
    return original(path)
with patch.object(os,'scandir',denied):
    observation=d.sizes(root)
    d.artifact_gate(observation)
print(json.dumps({'reproduced_defect':observation['artifact_bytes']==0,'observation':observation,'actual_hidden_bytes':len(b'protected partial'),'fault':'PermissionError reading receipts directory silently suppressed by os.walk'},sort_keys=True))
if observation['artifact_bytes']!=0: raise RuntimeError('reproduction changed')
