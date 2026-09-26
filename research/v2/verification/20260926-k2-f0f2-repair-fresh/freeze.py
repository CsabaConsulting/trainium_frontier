from pathlib import Path
import hashlib
root=Path(__file__).parent
lines=[]
for path in sorted(root.rglob('*')):
    if path.is_file() and path.name!='MANIFEST.sha256':
        lines.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(root)))
(root/'MANIFEST.sha256').write_text('\n'.join(lines)+'\n')
print(hashlib.sha256((root/'MANIFEST.sha256').read_bytes()).hexdigest())
