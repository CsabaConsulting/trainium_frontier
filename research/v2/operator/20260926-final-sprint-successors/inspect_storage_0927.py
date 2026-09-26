"""Same user-requested read-only geometry check, fresh uniquely named receipt."""
import hashlib
from pathlib import Path
old=Path(__file__).resolve().parent/'inspect_storage_identity.py'
raw=old.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='900d8091dcfc49e153b95b0874cf58dc293973b61c4d36aae4668586e4075dd5'
text=raw.decode();assert text.count('-storage-identity-0834.json')==1
text=text.replace('-storage-identity-0834.json','-storage-identity-0927.json')
exec(compile(text,str(old),'exec'),{'__file__':str(old),'__name__':'__main__'})
