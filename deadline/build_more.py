#!/usr/bin/env python3
"""Pre-register six-block half-resolution arms from the frozen submission."""
import hashlib
import json
from pathlib import Path

from build_candidates import BASE, ROOT, build, replace_once


def main():
    middle = build('middle')
    middle = replace_once(middle, 'if self.config.coarse_middle and i == 2:',
                          'if self.config.coarse_middle and i == 1:')
    middle = replace_once(middle, 'if self.config.coarse_middle and i == 6:',
                          'if self.config.coarse_middle and i == 7:')
    # These sites preserve U-Net skip sizes: 0->7 stays full resolution;
    # 1->6, 2->5 and 3->4 are all coarse. Both end attention layers stay full.
    sources = {'middle6': middle}
    combined = replace_once(middle,
        'self.square_avg = torch.zeros_like(self.table, dtype=torch.bfloat16)',
        'self.square_avg = torch.zeros_like(self.table, dtype=torch.float32)')
    before = BASE.read_text()
    rms = build('rmsfp32')
    start, end = 'def d1_rms_step(', '\n@torch.no_grad()\ndef d2_table_rank_max_abs_diff('
    # Match exactly the named optimizer function, never the main clock or data.
    old_start = before.index(start)
    old_end = before.index(end, old_start)
    new_start = rms.index(start)
    new_end = rms.index(end, new_start)
    combined = replace_once(combined, before[old_start:old_end], rms[new_start:new_end])
    sources['middle6rms'] = combined
    manifest_path = ROOT / 'deadline/candidate-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, source in sources.items():
        assert source[source.index('def main()'):source.index('def make_plan(')] == before[before.index('def main()'):before.index('def make_plan(')]
        path = ROOT / 'candidates' / name / 'train.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        compile(source, str(path), 'exec')
        path.write_text(source)
        manifest['candidates'][name] = dict(mode=name, path=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            coarse_span=[1, 7], fp32_table_moment=name == 'middle6rms')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({name: manifest['candidates'][name] for name in sources}))


if __name__ == '__main__':
    main()
