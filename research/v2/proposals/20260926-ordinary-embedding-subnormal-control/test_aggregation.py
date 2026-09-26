"""Synthetic local wrapper/rank aggregation; no native execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch

import ordinary_subnormal_probe as p
import validate_receipts as v


def main():
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    weight = torch.randn(p.V,p.C)
    here = Path(__file__).resolve().parent
    b = json.loads((here/'NATIVE_BINDING.disabled.json').read_text())
    base = dict(label=p.LABEL, rank=0,world_size=4,error=None,cleanup='DESTROYED',
                original_K2_gate='FAILED_PRESERVED',compiled_callable_count=1,returncode=0,
                routes=[],blobs={},argv=b['checker_args'],checker_sha256=b['support_sha256']['ordinary_subnormal_probe.py'],
                runtime=dict(torch='2.11.0+cpu',nki='0.5.0+28631259367.ga768afa6'),seed=42,
                environment={k:b['environment'][k] for k in ('NEURON_CC_FLAGS','NEURON_LOGICAL_NC_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                native_qualification=False)
    base['weight'] = p.tensor_record(weight,lambda:None,base['blobs'])
    for c in p.CASES:
        for r in p.ROUTES:
            base['routes'].append(p.run_route(lambda w,i:F.embedding(i,w).to(torch.float32),r,c,weight,p.fixture(c),torch.device('cpu'),lambda:None,base['blobs']))
    records = {}
    for rank in range(4):
        item = copy.deepcopy(base)
        item.update(rank=rank,pid=100+rank)
        records['receipts/rank'+str(rank)+'.json'] = item
        records['rank-environment-r'+str(rank)+'.json'] = dict(rank=rank,local_rank=rank,world_size=4,pid=100+rank,
            checker_argv=b['checker_args'],checker_sha256=b['support_sha256']['ordinary_subnormal_probe.py'],environment=b['environment'])
        records['rank-wrapper-exit-r'+str(rank)+'.json'] = dict(rank=rank,returncode=0,error=None)
    with tempfile.TemporaryDirectory(prefix='ordinary-aggregate-') as tmp:
        root = Path(tmp)
        shutil.copyfile(here/'ordinary_subnormal_probe.py',root/'ordinary_subnormal_probe.py')
        (root/'receipts').mkdir()
        for rank in range(4):
            (root/'receipts'/('rank'+str(rank)+'.json')).touch()
        original_read = Path.read_text
        def validate(data):
            def read(path,*a,**kw):
                relative = str(path.relative_to(root))
                return json.dumps(data[relative]) if relative in data else original_read(path,*a,**kw)
            with patch.object(Path,'read_text',read):
                return v.validate(root,b['label'],b['environment'],b['support_sha256'],b['checker_args'])
        p.require(validate(records)['ranks'] == 4,'SYNTHETIC_BASELINE')
        tests = ['synthetic_four_rank_baseline']
        rank0 = 'receipts/rank0.json'
        mutations = {
            'wrong_rank':lambda d:d[rank0].update(rank=1),
            'wrong_leaf_pin':lambda d:d[rank0].update(checker_sha256='0'*64),
            'wrong_runtime':lambda d:d[rank0]['runtime'].update(torch='2.13.0+cpu'),
            'wrong_argv':lambda d:d[rank0].update(argv=[]),
            'nonzero_wrapper':lambda d:d['rank-wrapper-exit-r0.json'].update(returncode=1),
            'bool_wrapper':lambda d:d['rank-wrapper-exit-r0.json'].update(returncode=False),
            'wrong_environment':lambda d:d['rank-environment-r0.json'].update(environment={}),
            'wrong_pid':lambda d:d['rank-environment-r0.json'].update(pid=999),
            'wrong_rankset':lambda d:d.pop('receipts/rank3.json'),
            'qualification':lambda d:d[rank0].update(native_qualification=True),
        }
        for name,mutate in mutations.items():
            changed = copy.deepcopy(records)
            mutate(changed)
            try:
                validate(changed)
            except (ValueError,KeyError,json.JSONDecodeError):
                tests.append(name)
            else:
                raise RuntimeError('NEGATIVE_ACCEPTED:'+name)
    print(json.dumps(dict(status='SELF_CHECK_PASS_SYNTHETIC_NOT_NATIVE',tests=tests,
                         actual_cpu_torch=torch.__version__,runtime_fields='SYNTHETIC_NATIVE_PINS',launch_authorized=False),indent=2))


if __name__ == '__main__':
    main()
