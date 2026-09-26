"""One read-only startup observation using the already-used P0 collector."""
import hashlib
from pathlib import Path

here = Path(__file__).resolve().parent
original = here.parent / '20260926-final-sprint-p0' / 'observe_once.py'
raw = original.read_bytes()
if hashlib.sha256(raw).hexdigest() != 'acc5067b73fa106b6b5f2d40b7a1f9f422a930b66e73d246f6731ba53f51418d':
    raise RuntimeError('ORIGINAL_OBSERVER_DRIFT')
text = raw.decode()
old = "out=HERE/(host+'-live.json');assert not out.exists()"
new = "out=Path(" + repr(str(here)) + ")/(host+'-live.json');assert not out.exists()"
if text.count(old) != 1:
    raise RuntimeError('OUTPUT_REBIND_NOT_UNIQUE')
text = text.replace(old, new)
# All remote reads and bounds are unchanged; only the local archive destination
# differs. The initial launch receipt is never overwritten.
exec(compile(text, str(original), 'exec'), {'__file__': str(original), '__name__': '__main__'})
