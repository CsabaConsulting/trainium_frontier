"""Finite recovery polling; the previous monitor is absent; no host writes or launches."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
assert not Path('/proc/2171819').exists(), 'OLD_MONITOR_STILL_PRESENT'
old = HERE.parent / '20260926-adopted-batch-handoff/watch_until_1103.py'
raw = old.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '9203171e60778e488920a8816cc11ff04aab7ab1f316ed963f03161be9a14b64'
text = raw.decode()
assert text.count('2026-09-26T11:03:01+00:00') == 1
text = text.replace('2026-09-26T11:03:01+00:00', '2026-09-26T11:15:01+00:00')
ast.parse(text)
exec(compile(text, str(old), 'exec'), {'__file__': __file__, '__name__': '__main__'})
