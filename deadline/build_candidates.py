#!/usr/bin/env python3
"""Generate five isolated deadline experiments from the exact submitted file."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'submission/train.py'
EXPECTED = '2b4b1ac0d7cd39628ee20282122797abd5f88a5bfc30d7d5115b45289d172894'
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == EXPECTED


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def build(mode):
    s = BASE.read_text()
    if mode == 'rmsfp32':
        s = replace_once(s, 'self.square_avg = torch.zeros_like(self.table, dtype=torch.bfloat16)',
                         'self.square_avg = torch.zeros_like(self.table, dtype=torch.float32)')
        s = replace_once(s, '    """BF16 historical operation stages; dynamic FP32 0-D LR/correction inputs."""\n    squared = (gradient * gradient).to(torch.bfloat16)\n    second = torch.lerp(square_avg, squared, 0.001).to(torch.bfloat16)\n    denom = (second / correction).to(torch.bfloat16).sqrt().to(torch.bfloat16)\n    denom = (denom + 1e-10).to(torch.bfloat16)\n    update = (gradient / denom).to(torch.bfloat16)', '    """Keep the slowly changing RMSprop second moment and denominator in FP32."""\n    squared = gradient.float().square()\n    second = torch.lerp(square_avg, squared, 0.001)\n    denom = (second / correction).sqrt() + 1e-10\n    update = gradient.float() / denom')
        return s
    if mode == 'mimetic':
        s = replace_once(s, '    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int=ROPE_BASE):',
                         """        # Structured V/O pair, isolated from all other initializers. A local
        # generator preserves the global RNG stream and all unrelated weights.
        generator = torch.Generator(device='cpu').manual_seed(8675309)
        for block in self.transformer.h:
            if block.attn is not None:
                width = self.config.n_embd
                matrix = torch.randn(width, width, generator=generator, dtype=torch.float32)
                orthogonal, _ = torch.linalg.qr(matrix)
                block.attn.c_v.weight.copy_(orthogonal * INIT_SCALE)
                block.attn.c_proj.weight.copy_(orthogonal.T * (INIT_SCALE * ATTN_CPROJ_INIT_SCALE))

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int=ROPE_BASE):""")
        return s
    s = replace_once(s, 'ATTENTION_LAYERS = (0, 2, 4, 6, 7)',
                     "ATTENTION_LAYERS = (0, 2, 4, 6, 7)\nCOARSE_MODE = " + repr(mode) + "\nCOARSE_MLP_START = " + repr(1 if mode == 'mlp6' else 2) + "\nCOARSE_MLP_END = " + repr(7 if mode == 'mlp6' else 6))
    s = replace_once(s, '    lane_b: bool = False\nCOMPUTE_DTYPE:',
                     '    lane_b: bool = False\n    coarse_middle: bool = False\n'
                     '    coarse_mlp: bool = False\nCOMPUTE_DTYPE:')
    s = replace_once(s,
                    '        self.attn = CausalSelfAttention(config, layer_idx) if layer_idx in ATTENTION_LAYERS else None',
                    '        self.coarse_mlp = config.coarse_mlp and COARSE_MLP_START <= layer_idx < COARSE_MLP_END\n'
                    '        self.coarse_parity = layer_idx % 2\n'
                    '        self.attn = CausalSelfAttention(config, layer_idx) if layer_idx in ATTENTION_LAYERS else None')
    s = replace_once(s, '            return x + self.mlp(norm(x))',
                     '            return x + self.mlp_residual(x)')
    s = replace_once(s, '        x = x + self.mlp(norm(x))\n        return x\n\nclass GPT',
                     '''        x = x + self.mlp_residual(x)
        return x

    def mlp_residual(self, x):
        if not self.coarse_mlp:
            return self.mlp(norm(x))
        # Each shared residual is produced from the current or previous token.
        # Odd-layer parity is left-padded, so it never uses a future position.
        t = x.shape[1]
        assert t % 2 == 0
        residual = self.mlp(norm(x[:, self.coarse_parity::2, :]))
        expanded = residual.repeat_interleave(2, dim=1)
        if self.coarse_parity:
            expanded = torch.cat((torch.zeros_like(expanded[:, :1]), expanded[:, :-1]), dim=1)
        return expanded

class GPT''')
    s = replace_once(s,
                    '        unet_skips: list[torch.Tensor] = []\n        for i, block in enumerate(self.transformer.h):',
                    '''        unet_skips: list[torch.Tensor] = []
        full_base = None
        full_x0, full_cos, full_sin = x0, cos, sin
        for i, block in enumerate(self.transformer.h):
            if self.config.coarse_middle and i == 2:
                assert t % 2 == 0
                full_base = x
                x = x[:, ::2, :].contiguous()
                x0 = full_x0[:, ::2, :].contiguous()
                # Retain original absolute positions 0,2,4,... in RoPE.
                cos = full_cos[:, ::2].contiguous()
                sin = full_sin[:, ::2].contiguous()
            if self.config.coarse_middle and i == 6:
                # Even positions use their actual coarse state. Odd positions
                # retain their full-resolution state plus a past-only delta.
                odd = full_base[:, 1::2, :] + (x - full_base[:, ::2, :])
                x = torch.stack((x, odd), dim=2).reshape_as(full_base)
                x0, cos, sin = full_x0, full_cos, full_sin''')
    s = replace_once(s,
                    'tbl_every=tbl_every, d2_two_site=d2_two_site, tbl_freeze_tokens=tbl_freeze_tokens)\nPOLAR_EXPRESS_COEFFS',
                    "tbl_every=tbl_every, d2_two_site=d2_two_site, tbl_freeze_tokens=tbl_freeze_tokens, coarse_middle=COARSE_MODE == 'middle', coarse_mlp=COARSE_MODE in ('mlp', 'mlp6'))\nPOLAR_EXPRESS_COEFFS")
    # Preserve all timing, loader, optimizer, loss, kernel and evaluation hooks.
    assert s[s.index('def main()'):s.index('def make_plan(')] == BASE.read_text()[BASE.read_text().index('def main()'):BASE.read_text().index('def make_plan(')]
    return s


def main():
    manifest = {'base_sha256': EXPECTED, 'candidates': {}}
    for name, mode in [('middle', 'middle'), ('mlp', 'mlp'), ('mlp6', 'mlp6'), ('rmsfp32', 'rmsfp32'), ('mimetic', 'mimetic'), ('control', 'none')]:
        dest = ROOT / 'candidates' / name / 'train.py'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(build(mode))
        compile(dest.read_text(), str(dest), 'exec')
        manifest['candidates'][name] = {'mode': mode, 'path': str(dest.relative_to(ROOT)),
                                      'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    (ROOT / 'deadline/candidate-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
