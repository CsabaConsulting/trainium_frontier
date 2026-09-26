"""Hash the retained independent review files; no candidate writes."""
import hashlib
from pathlib import Path
root = Path(__file__).resolve().parent
lines=[]
for path in sorted(root.rglob('*')):
    if path.is_file() and path.name != 'MANIFEST.sha256':
        lines.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(root)))
(root/'MANIFEST.sha256').write_text('\n'.join(lines)+'\n')
for name in ('REPORT.md','review_checks.py','raw-results.json','MANIFEST.sha256'):
    print(hashlib.sha256((root/name).read_bytes()).hexdigest()+'  '+name)
