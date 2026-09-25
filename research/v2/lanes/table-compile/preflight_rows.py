"""Lane-local synthetic G2 adapter: slice raw rows exactly like idx/targets.

The canonical preflight and all production/evaluator files are left untouched.
This executes the canonical harness with four guarded, auditable substitutions:
dispatch children through this adapter, slice optional rows, record the shim,
and attach provenance.
It is test code only and must never accompany a submitted training file.
"""
from pathlib import Path
import difflib
import hashlib
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORIGINAL = ROOT / "workflow/tools/preflight_dryrun.py"
SUPPORT = ROOT / "research/v2/lanes/recursive-schedule/test_support"
# Import-only stub; sitecustomize is deliberately not imported. Torch 2.13's
# native Gloo AVG is used, with no old Torch-2.6 SUM/div compatibility override.
sys.path.insert(0, str(SUPPORT))
source = ORIGINAL.read_text()
adapted = source


def replace_once(old, new):
    global adapted
    assert adapted.count(old) == 1, f"canonical harness seam changed: {old!r}"
    adapted = adapted.replace(old, new, 1)


replace_once('str(Path(__file__).resolve()), "--worker"',
             f'{str(Path(__file__).resolve())!r}, "--worker"')
replace_once('            out = orig_forward(self, idx, targets, *a, **k)',
             '''            if k.get("ngram_raw_rows") is not None:
                k["ngram_raw_rows"] = tuple(
                    row[: ns.micro_rows, : ns.micro_tokens].contiguous()
                    for row in k["ngram_raw_rows"])
            out = orig_forward(self, idx, targets, *a, **k)''')
marker = "lane-local synthetic adapter slices ngram_raw_rows to the identical idx/targets prefix; eager gather/autograd retained"
replace_once('    shims.append(f"GPT.forward training micro-batch sliced to {ns.micro_rows}x{ns.micro_tokens} tokens")',
             '    shims.append(f"GPT.forward training micro-batch sliced to {ns.micro_rows}x{ns.micro_tokens} tokens")\n'
             f'    shims.append({marker!r})')
provenance = dict(adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  canonical_preflight_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  adapted_preflight_sha256_before_provenance=hashlib.sha256(adapted.encode()).hexdigest(),
                  tuple_slice="row[: ns.micro_rows, : ns.micro_tokens].contiguous()",
                  synthetic_only=True)
replace_once('        "schema": "g2-preflight-v1",',
             f'        "lane_adapter": {provenance!r},\n        "schema": "g2-preflight-v1",')
if sys.argv[1:] == ["--adapter-diff"]:
    print("".join(difflib.unified_diff(source.splitlines(True), adapted.splitlines(True),
                                    fromfile=str(ORIGINAL), tofile="lane-local execution", n=0)), end="")
    sys.exit(0)
exec(compile(adapted, str(ORIGINAL), "exec"),
     {"__name__": "__main__", "__file__": str(ORIGINAL)})
