#!/usr/bin/env python3
"""Rebuild NKI resampling candidates without changing the original arms."""
from pathlib import Path
import hashlib, json
ROOT = Path(__file__).resolve().parents[1]
kernel = (ROOT/'deadline/pair_resampling.py').read_text()
manifest_path = ROOT/'deadline/candidate-manifest.json'
manifest = json.loads(manifest_path.read_text())
for old, name in [('mlp', 'mlpnki'), ('mlp6', 'mlp6nki')]:
    path = ROOT/'candidates'/old/'train.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['candidates'][old]['sha256']
    original = path.read_text()
    source = original.replace('def softcap_cross_entropy(', kernel+'\n\ndef softcap_cross_entropy(', 1)
    start = source.index('        t = x.shape[1]\n        assert t % 2 == 0\n        residual = self.mlp(norm(x[:, self.coarse_parity::2, :]))')
    end = source.index('\nclass GPT', start)
    source = source[:start]+'''        residual = self.mlp(norm(_PairSelectNKI.apply(x, self.coarse_parity)))
        return _PairExpandNKI.apply(residual, self.coarse_parity)
'''+source[end:]
    assert source[source.index('def main()'):source.index('def make_plan(')] == original[original.index('def main()'):original.index('def make_plan(')]
    assert source[source.index('def load_for_eval('):] == original[original.index('def load_for_eval('):]
    dest = ROOT/'candidates'/name/'train.py'; dest.parent.mkdir(exist_ok=True);dest.write_text(source)
    compile(source, str(dest), 'exec')
    manifest['candidates'][name] = dict(mode=old, path=str(dest.relative_to(ROOT)), sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({n:manifest['candidates'][n] for n in ('mlpnki','mlp6nki')}))
