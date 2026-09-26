"""One read-only repeat of the pinned prior collector, in a new receipt directory."""
import hashlib
from pathlib import Path

prior = Path(__file__).resolve().parent.parent / '20260926-trn22-eval-boundary-0550/collect_once.py'
raw = prior.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'ba238ded8a6160e11e38a26ff1f5ab67540052d6a043ab5c160de41a97eccbdb'
source = raw.decode()
old = "for path in paths:\n"
new = "paths.append(HERE.parent/'20260926-trn22-eval-boundary-0550/trn22.json')\nfor path in paths:\n"
assert source.count(old) == 1
exec(compile(source.replace(old, new), str(Path(__file__).resolve()), 'exec'))
