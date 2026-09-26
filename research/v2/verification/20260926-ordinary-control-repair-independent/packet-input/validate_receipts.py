"""Completion authentication for the single ordinary-only diagnostic binding."""
import importlib.util
import json


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(root, label, env, pins, args):
    spec = importlib.util.spec_from_file_location('ordinary_probe_validator', root / 'ordinary_subnormal_probe.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    require(probe.sha((root/'ordinary_subnormal_probe.py').read_bytes()) == pins['ordinary_subnormal_probe.py'], 'LEAF_HASH')
    require(sorted(p.name for p in (root/'receipts').iterdir()) == ['rank0.json','rank1.json','rank2.json','rank3.json'], 'EXACT_RANK_FILES')
    signatures, pids = [], []
    for rank in range(4):
        receipt = json.loads((root/'receipts'/('rank'+str(rank)+'.json')).read_text())
        probe.validate_receipt(receipt,rank)
        require(receipt['label'] == label and receipt['argv'] == args and receipt['checker_sha256'] == pins['ordinary_subnormal_probe.py'], 'LEAF_BINDING')
        require(receipt['runtime'] == dict(torch='2.11.0+cpu',nki='0.5.0+28631259367.ga768afa6') and receipt['seed'] == 42, 'RUNTIME')
        require(receipt['environment'] == {k:env[k] for k in receipt['environment']} and
                set(receipt['environment']) == {'NEURON_CC_FLAGS','NEURON_LOGICAL_NC_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'}, 'LEAF_ENV')
        require(receipt['native_qualification'] is False, 'QUALIFICATION_FORBIDDEN')
        wrapper = json.loads((root/('rank-environment-r'+str(rank)+'.json')).read_text())
        exited = json.loads((root/('rank-wrapper-exit-r'+str(rank)+'.json')).read_text())
        require(type(wrapper['rank']) is int and type(wrapper['local_rank']) is int and wrapper['rank'] == rank and wrapper['local_rank'] == rank and wrapper['world_size'] == 4 and
                wrapper['checker_argv'] == args and wrapper['checker_sha256'] == pins['ordinary_subnormal_probe.py'], 'WRAPPER_BINDING')
        require(wrapper['environment'] == {k:env.get(k) for k in wrapper['environment']} and
                set(wrapper['environment']) == set(env), 'WRAPPER_ENV')
        require(type(exited['rank']) is int and exited['rank'] == rank and type(exited['returncode']) is int and exited['returncode'] == 0 and exited['error'] is None, 'WRAPPER_EXIT')
        require(type(receipt['pid']) is int and receipt['pid'] > 0 and wrapper['pid'] == receipt['pid'], 'PID')
        pids.append(receipt['pid'])
        signatures.append(probe.sha(json.dumps(dict(weight=receipt['weight'],routes=receipt['routes']), sort_keys=True).encode()))
    require(len(set(pids)) == 4 and len(set(signatures)) == 1, 'CROSS_RANK')
    return dict(status='COMPLETE_EXACT_DIAGNOSTIC_PENDING_INDEPENDENT_RESULT_REVIEW', ranks=4,
                cases_per_rank=4, routes_per_case=2, cross_rank_sha256=signatures[0],
                original_K2_gate='FAILED_PRESERVED', native_qualification=False, accepted_speedup=False)
