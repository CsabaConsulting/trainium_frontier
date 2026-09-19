#!/usr/bin/env python3
"""Locate the compiled head/CE mismatch without changing its exact gate.

Compare loss-only graphs with graphs exposing the logits, standalone head
outputs, and CE on frozen copies of those outputs. Exposing an intermediate can
affect compilation, so that intervention is measured rather than assumed inert.
This is a diagnostic, not a relaxed numerical gate or a training promotion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
from softcap_ce_bf16_integration import install


def comparison(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    delta = a.double() - b.double()
    return dict(exact=torch.equal(a, b), different=int((a != b).sum()),
                elements=a.numel(), max_abs=float(delta.abs().max()),
                rms=float(delta.square().mean().sqrt()),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-dir', type=Path, required=True)
    args = parser.parse_args()
    args.dump_dir.mkdir(parents=True, exist_ok=False)
    import torch_neuronx
    from torch_neuronx.neuron_dynamo_backend.compile import CompileGraph
    original_compile = CompileGraph.compile
    compile_records = []

    def capture_compile(graph):
        index = len(compile_records)
        text = str(graph.mlir)
        destination = args.dump_dir / f'graph-{index:03d}.mlir'
        destination.write_text(text)
        record = dict(index=index, model=graph.model_name, segment=graph.segment_id,
                      graph=graph.graph_name, file=destination.name,
                      sha256=hashlib.sha256(text.encode()).hexdigest())
        compile_records.append(record)
        result = original_compile(graph)
        record['execution_key'] = result
        return result

    CompileGraph.compile = capture_compile
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 64
    T.USE_WEIGHT_CACHE = True
    install(T)
    generator = torch.Generator().manual_seed(8091)
    results = []
    # Independent small/production-width cases; fresh head each case avoids
    # confusing a changed rounding boundary with stale weight-cache identity.
    for channels in (256, 1024):
        rows = 256 if channels == 256 else 512
        for scale in (.001, .03, .3):
            head = T.Linear(channels, 8192, bias=False).to('neuron')
            x_cpu = torch.randn(rows, channels, generator=generator).bfloat16()
            w_cpu = torch.randn(8192, channels, generator=generator) * scale
            target_cpu = torch.randint(8192, (rows,), generator=generator)
            x, target = x_cpu.to('neuron'), target_cpu.to('neuron')
            with torch.no_grad():
                head.weight.copy_(w_cpu.to('neuron'))
            head.refresh_weight_cache(torch.bfloat16)

            def original(a, t):
                return T.softcap_cross_entropy(head(a).float(), t, 'none')

            def candidate(a, t):
                return T.softcap_cross_entropy(head(a), t, 'none')

            def original_exposed(a, t):
                z = head(a).float()
                return T.softcap_cross_entropy(z, t, 'none'), z

            def candidate_exposed(a, t):
                z = head(a)
                return T.softcap_cross_entropy(z, t, 'none'), z

            def head_float(a):
                return head(a).float()

            def head_bf16(a):
                return head(a)

            functions = dict(original=original, candidate=candidate,
                             original_exposed=original_exposed,
                             candidate_exposed=candidate_exposed,
                             head_float=head_float, head_bf16=head_bf16)
            compiled = {name: torch.compile(fn, backend='neuron', fullgraph=True,
                                            dynamic=False)
                        for name, fn in functions.items()}
            with torch.no_grad():
                eager_z = head(x)
                eager_loss = T.softcap_cross_entropy(eager_z.float(), target, 'none')
                old_loss, new_loss = compiled['original'](x, target), compiled['candidate'](x, target)
                old_exposed, old_z = compiled['original_exposed'](x, target)
                new_exposed, new_z = compiled['candidate_exposed'](x, target)
                only_old_z, only_new_z = compiled['head_float'](x), compiled['head_bf16'](x)
                # Host round trips make frozen CE inputs independent of head
                # fusion; these results are diagnostic and are not timed.
                frozen_old = old_z.cpu().to('neuron')
                frozen_new = new_z.cpu().to('neuron')
                frozen_old_loss = T.softcap_cross_entropy(frozen_old, target, 'none')
                frozen_new_loss = T.softcap_cross_entropy(frozen_new, target, 'none')
                pairs = {
                    'loss_only_original_vs_candidate': (old_loss, new_loss),
                    'expose_original_loss': (old_loss, old_exposed),
                    'expose_candidate_loss': (new_loss, new_exposed),
                    'original_exposed_vs_frozen_loss': (old_exposed, frozen_old_loss),
                    'candidate_exposed_vs_frozen_loss': (new_exposed, frozen_new_loss),
                    'compiled_original_vs_eager_loss': (old_loss, eager_loss),
                    'compiled_candidate_vs_eager_loss': (new_loss, eager_loss),
                    'original_float_logits_vs_bf16_roundtrip': (old_z, old_z.bfloat16().float()),
                    'candidate_logits_vs_rounded_original': (new_z, old_z.bfloat16()),
                    'original_exposed_vs_head_only': (old_z, only_old_z),
                    'candidate_exposed_vs_head_only': (new_z, only_new_z),
                    'original_float_logits_vs_eager': (old_z, eager_z.float()),
                    'candidate_bf16_logits_vs_eager': (new_z, eager_z),
                }
                checks = {name: comparison(*pair) for name, pair in pairs.items()}
                # Sample independent FP64 dots; do not allocate a full CPU
                # production-size double-precision head output.
                sample_rows = torch.randint(rows, (512,), generator=generator)
                sample_cols = torch.randint(8192, (512,), generator=generator)
                truth = (x_cpu[sample_rows].double()
                         * w_cpu.bfloat16()[sample_cols].double()).sum(-1)
                sampled = dict(original_fp32=comparison(old_z.cpu()[sample_rows, sample_cols], truth),
                               candidate_bf16=comparison(new_z.cpu()[sample_rows, sample_cols], truth))
                record = dict(rows=rows, channels=channels, weight_std=scale,
                              comparisons=checks, sampled_fp64_dots=sampled,
                              original_dtype=str(old_z.dtype), candidate_dtype=str(new_z.dtype),
                              production_exact_gate_changed=False,
                              exposure_can_change_graph=True)
                results.append(record)
                print('CE_HEAD_ROUNDING_DIAGNOSTIC ' + json.dumps(record), flush=True)
                assert all(check['finite'] for check in checks.values()), record
                assert torch.equal(head.weight.cpu(), w_cpu)
                assert torch.equal(head._cached_weight.cpu(), w_cpu.bfloat16())
                assert torch.equal(x.cpu(), x_cpu) and torch.equal(target.cpu(), target_cpu)
    report = dict(diagnostic_completed=True, cases=results, compiled_graphs=compile_records,
                  whole_model_promotion=False, original_exact_gate_still_required=True)
    (args.dump_dir / 'diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')
    print('CE_HEAD_ROUNDING_RESULT ' + json.dumps(dict(cases=len(results),
          graphs=len(compile_records), diagnostic_completed=True,
          no_training_promotion=True)), flush=True)


if __name__ == '__main__':
    main()
