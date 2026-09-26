from pathlib import Path
import hashlib,json
p=Path(__file__).parent.parent
out=Path(__file__).parent
for patch,old,new in [('vs-116188.full.patch','vehicle.py','candidate.py'),('vs-research36.full.patch','research.py','candidate2.py')]:
    rows=(p/patch).read_text().splitlines(keepends=True)
    rows=rows[next(i for i,l in enumerate(rows) if l.startswith('@@ '))+1:]
    for name,chars in [(old,' -'),(new,' +')]:
        data=''.join(l[1:] for l in rows if l[:1] in chars)
        (out/name).write_text(data)
        print(name,len(data.splitlines()),hashlib.sha256(data.encode()).hexdigest())
