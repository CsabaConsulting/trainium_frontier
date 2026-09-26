"""Use the exact proven read-only observer, changing only its local archive directory."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
assert not Path('/proc/2171819').exists(), 'OLD_MONITOR_STILL_PRESENT'
old = HERE.parent / '20260926-adopted-batch-handoff/observe_light.py'
raw = old.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'dc3aa0706facc38a304d2b37bfd0c0ff56e3057aa85fcc26524fcda7f288e010'
exec(compile(raw, str(old), 'exec'), {'__file__': __file__, '__name__': '__main__'})
