#!/usr/bin/env python3
"""Output-head hypotheses, preserving the backbone RNG and training protocol."""
import hashlib
import json
from build_candidates import BASE, ROOT, replace_once


def main():
    base = BASE.read_text()
    anchor = '        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)'
    bias = replace_once(base, anchor, anchor+'\n        self.logit_bias = nn.Parameter(torch.zeros(padded_vocab_size))')
    bias = replace_once(bias, '        scalar_params.extend(self.unet_skip_gates)',
        '        scalar_params.extend(self.unet_skip_gates)\n        scalar_params.append(self.logit_bias)')
    bias = replace_once(bias, '        logits = self.lm_head(x)',
        '        logits = self.lm_head(x)\n        logits = logits + self.logit_bias.to(logits.dtype)')
    dual = replace_once(base, anchor, anchor+'''
        # Constructor and extra initializer cannot perturb the backbone RNG.
        with torch.random.fork_rng(devices=[]):
            self.lm_head2 = Linear(config.n_embd, padded_vocab_size, bias=False)''')
    dual = replace_once(dual, '    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int=ROPE_BASE):', '''        generator = torch.Generator(device='cpu').manual_seed(8675309)
        extra_head = torch.randn(self.lm_head2.weight.shape, generator=generator, dtype=torch.float32) * .001
        self.lm_head2.weight.copy_(extra_head)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int=ROPE_BASE):''')
    dual = replace_once(dual, '        lm_head_params = list(self.lm_head.parameters())',
        '        lm_head_params = list(self.lm_head.parameters()) + list(self.lm_head2.parameters())')
    dual = replace_once(dual, '        logits = self.lm_head(x)', '''        logits = self.lm_head(x)
        extra = self.lm_head2(F.relu(x) * math.sqrt(2.0))
        # Stable log-mean-exp of two unnormalized logit heads. The second sees
        # different features, preventing the two heads from being duplicates.
        a, b = logits.float(), extra.float()
        logits = (torch.maximum(a, b) + torch.log1p(torch.exp(-(a-b).abs())) - math.log(2.0)).to(logits.dtype)''')
    manifest_path = ROOT/'deadline/candidate-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, source in [('headbias', bias), ('dualhead', dual)]:
        assert source[source.index('def main()'):source.index('def make_plan(')] == base[base.index('def main()'):base.index('def make_plan(')]
        path = ROOT/'candidates'/name/'train.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        compile(source, str(path), 'exec'); path.write_text(source)
        manifest['candidates'][name] = dict(mode=name, path=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({n: manifest['candidates'][n] for n in ['headbias', 'dualhead']}))


if __name__ == '__main__':
    main()
