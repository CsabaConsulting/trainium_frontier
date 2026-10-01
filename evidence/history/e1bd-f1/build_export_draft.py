#!/usr/bin/env python3
"""One-use, local metadata-only DRAFT assembly; no host/model/acceptance action."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
R = H.parents[3]
P = R / 'research/v2/proposals/20260930-r158-e1bd-f1-lncfix'
E = R / 'research/v2/operator/20260930-r158-e1bd-f1-exact-eval-trn21-preparation'
T = R / 'research/v2/operator/20260930-r158-e1bd-f1-native60-trn21-preparation'
U = R / 'research/v2/operator/20260930-r158-e1bd-f1-full-trn21-preparation'
OLD = Path('/home/poojith-devan/Downloads/Trainium-STATE-LIVE/audit-e1bd')
SOURCE = '25c5aec53ad7cc20f4dadf437d3ebe6dfa4f4be0214ab39d638a45df5ec7f936'
PARENT = 'e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310'
COMMAND = 'NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0\n'
copies = {}
references = []
raw_index = []
wrapped_raw = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path, pin=None):
    assert path.is_file() and not path.is_symlink() and path.stat().st_size < 16*1024*1024
    data = path.read_bytes()
    if pin is not None:
        assert sha(data) == pin, ('PIN_DRIFT', str(path))
    return data


def ref(path, pin=None):
    data = read(path, pin)
    try:
        name = str(path.relative_to(R))
    except ValueError:
        name = str(path)
    row = dict(path=name, sha256=sha(data), bytes=len(data))
    references.append(row)
    return row, data


def add(name, path, pin=None):
    row, data = ref(path, pin)
    assert name not in copies and not (H/name).exists()
    data.decode('utf-8')
    if data.endswith(b'\n'):
        copies[name] = data
    else:
        # apply_patch adds a terminal newline. Preserve a non-newline-terminated
        # raw archive exactly as a decoded payload, never silently normalize it.
        assert name.endswith('.json'), ('NO_SOURCE_BYTE_NORMALIZATION', name)
        copies[name] = enc(dict(schema='exact-raw-bytes-envelope-v1',
            original=row, encoding='base64', raw_bytes_b64=base64.b64encode(data).decode()))
        wrapped_raw[name] = row
    return row, data


def enc(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def full_members(path, pin):
    data = read(path, pin)
    outer = json.loads(data)
    if 'files' in outer:
        a = outer  # Original native collector's direct raw-member archive.
    else:
        assert outer['returncode'] == 0 and outer['stderr'] == ''
        a = json.loads(outer['stdout'])
    index = {}
    for name, m in a['files'].items():
        assert not m.get('tail_only', False) and m.get('full_raw', True)
        content = m['text'].encode() if 'text' in m else base64.b64decode(m['b64'], validate=True)
        assert sha(content) == m['sha256'] and len(content) == m['bytes'], name
        index[name] = dict(sha256=m['sha256'], bytes=m['bytes'])
    raw_index.append(dict(path=str(path.relative_to(R)), sha256=pin, members=index))
    return a


def main():
    assert not (H/'EXPORT_MANIFEST.json').exists(), 'SEALED_DRAFT_NO_OVERWRITE'
    _, source = add('train.py', P/'train.py', SOURCE)
    add('audited_parent.py', R/'research/v2/proposals/20260929-r109-ce2-flat-zbundle/train.py', PARENT)
    add('SOURCE.diff', P/'SOURCE.diff', '7a8f733effaf815a25c1e001fc51aaaeaaba486749b0d86814ac08b02cbd2b80')
    add('SOURCE_SPEC.md', P/'SPEC.md', '13aa06bd0316f90336fae4efdb10c5e7eea212c20d161e7f6aec78af7ca2a112')
    add('AUTHORITY_R158.md', E/'AUTHORITY_R158.md')
    add('REJECTION_CHECKLIST.md', OLD/'REJECTION_CHECKLIST.md', '71717878b1c070bb7bce459579ae31f8b26b6339db9c7661859db1f19b9b9612')
    _, command = add('launch-command.txt', U/'artifacts/snapshot/launch-command.txt')
    assert command.decode() == COMMAND
    review_ref, review_bytes = add('source-cpu-review.json',
        R/'research/v2/verification/20260930-r158-e1bd-f1-independent/FINAL_REVIEW.json',
        'e18a626152f3ae41d4fe5cbab105715634e548f831b30e10c050451e1402d29c')
    review = json.loads(review_bytes)
    assert review['status'] == 'ACCEPT_SOURCE_CPU_ONLY' and review['source_sha256'] == SOURCE
    assert not review['native_accepted'] and not review['package_accepted']
    for role in ('author_cpu', 'own_cpu'):
        item = review[role]
        ref(R/item['path'], item['sha256'])
    reader_path = E/'read_root.py'
    ref(reader_path)
    spec = importlib.util.spec_from_file_location('r162_draft_existing_exact21_reader', reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    evals = []
    for case, filename, pin in [
        ('01-f1-lnc1', 'collect-20260930T084026316343Z.json', '7c2ffe78f6bde5cb44e607d2f896ece59144bd973ac8a3e3ad178ac8e9e7468f'),
        ('02-f1-unset', 'collect-20260930T084329340223Z.json', '02c7eff48788894e7ec5d11bf7da3cd34a497c30180860dd49ed5fddc518a8c6')]:
        path = E/'cases'/case/filename
        raw_ref, data = add('raw-exact-eval/'+case+'.json', path, pin)
        a = full_members(path, pin)
        proof = reader.authenticate(case, json.loads(data))
        assert proof['exit_code'] == 0 and proof['source_sha256'] == SOURCE
        result = proof['worker_result']
        assert result['exact_source_execution'] and result['comparison']['absolute_delta'] == 0
        assert result['result']['val_bpb'] == 0.9617131518505532
        assert result['source_unchanged'] and result['checkpoint_unchanged'] and result['evaluator_unchanged']
        evals.append(dict(case=case, raw_archive=raw_ref, proof=proof))
    timing_pin = '6178c1cfa337bd13d786cee6f7c4c1747c9bce523c90d397a4db1212decc4c02'
    timing_ref, _ = add('raw-timing/results-bundle.json', T/'artifacts/results-bundle.json', timing_pin)
    full_members(T/'artifacts/results-bundle.json', timing_pin)
    comparison_ref, comparison_bytes = add('matched60-comparison.json', T/'ACTUAL_COMPARISON.json',
        '589ecf2e587aeb352e082a04968b0a7cd923955bed4437621e36f0c6824f35d6')
    comparison = json.loads(comparison_bytes)
    assert comparison['metadata_valid'] and comparison['scientific_source_sha256'] == SOURCE
    assert comparison['actual']['archive_sha256'] == timing_pin and comparison['within_noise'] is None
    disposition_ref, disposition_bytes = add('Root-noise-disposition.json', U/'ROOT_NOISE_DISPOSITION.r162.json',
        'a960eaabceec4867b8fc59190f4205660fd0ac56342b7848cb32294a1d1d5e3b')
    disposition = json.loads(disposition_bytes)
    assert disposition['operator'] == 'Root' and disposition['within_noise'] is True
    assert disposition['comparison_sha256'] == comparison_ref['sha256']
    packet_ref, packet_bytes = ref(U/'artifacts/packet.json',
        '04d5cc3170b13a6ebba761a25f93d08fff0ee5759ce4ea2c849e85e0e2588a6e')
    packet = json.loads(packet_bytes)
    for name, member in packet['files'].items():
        assert sha(base64.b64decode(member['b64'], validate=True)) == member['sha256'], name
    launch_ref, launch_bytes = add('FULL-launch-only.json', U/'launch-result.json',
        '96aebe4cbef7e6591aff1d4f38ab709765fa54401d89ada46363b64e2dab2c74')
    launch_outer = json.loads(launch_bytes)
    assert launch_outer['returncode'] == 0 and launch_outer['stderr'] == ''
    launch = json.loads(launch_outer['stdout'])
    assert launch['launch']['status'] == 'DETACHED_DISPATCH_STARTED'
    assert launch['launch']['source_sha256'] == SOURCE
    historical_ref, historical_bytes = add('historical-e1bd-cold-evidence.json', OLD/'cold-evidence.json',
        'a836f8852c3c64f07ea4e0c79182f3179928f7b9329ab10bff669163022c687d')
    historical = json.loads(historical_bytes)
    assert historical['source_sha256'] == PARENT and historical['receipt_count'] == 2
    for row in historical['receipts']:
        archive = row['raw_archive']
        ref(Path(archive['path']), archive['sha256'])
    add('historical-independent-receipt-audit.json', R/'research/v2/analysis/20260930-r153-independent-fallback-receipts.json',
        '207803b8a1be56f8b4878bdce9779bd5edb132b114357333559a22f0b1237dd1')
    tree = ast.parse(source.decode())
    definitions, io_env = [], []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.append(dict(name=n.name, line=n.lineno, kind=type(n).__name__, docstring=ast.get_docstring(n)))
        if isinstance(n, ast.Call):
            name = ast.unparse(n.func)
            if any(part in name for part in ('open', 'load', 'save', 'environ', 'getenv', 'tokenizer', 'dataloader', 'time.', 'seed')):
                io_env.append(dict(line=n.lineno, expression=ast.unparse(n)))
    copies['SOURCE_INVENTORY.json'] = enc(dict(source_sha256=SOURCE, scope='AST source occurrences, not runtime trace or compliance verdict',
        definitions=definitions, io_environment_time_seed_calls=io_env))
    copies['CHECKLIST.json'] = enc(dict(schema='r162-e1bd-f1-eight-item-export-draft-v1',
        status='EXPORT_DRAFT_NOT_READY', unknown_is_fail=True,
        items=[dict(item=n, status='FAIL_PENDING_COMPLETE_EVIDENCE_AND_AUDIT') for n in range(1,9)],
        package_accepted=False, claude_verdict=None, submission_authorized=False))
    future_fields = ['archive_path','archive_sha256','terminal_sha256','binding_sha256','source_sha256',
        'command_sha256','host','seed','six_cold_cache_identities','cold_step4_boundary','cold_seconds',
        'charged_through_exit_seconds','completed_steps','processed_tokens','finite_and_rank_proofs',
        'checkpoint_sha256','checkpoint_bytes','checkpoint_stable','fresh_eval_process',
        'public_eval_tokens','public_eval_context','public20m_val_bpb','guardian_released',
        'original_lock_identity','descendants_reaped','protected_inputs_unchanged',
        'final_disk_inventory','cumulative_checkpoint_log_cache_bytes_written','retention_and_backup_state',
        'current_authority','independent_package_review','claude_package_verdict']
    copies['FUTURE_FULL_EVIDENCE.template.json'] = enc(dict(schema='r162-e1bd-f1-future-full-evidence-v1',
        status='PENDING_ACTUAL_FULL_RECEIPT', actual={n:None for n in future_fields},
        full_accepted=False, package_ready=False, submission_authorized=False))
    evidence = dict(schema='r162-e1bd-f1-export-draft-evidence-v1', status='EXPORT_DRAFT_NOT_READY',
        source_sha256=SOURCE, parent_sha256=PARENT, command=COMMAND.strip(), source_cpu=review_ref,
        exact_source_diagnostics=evals, timing=dict(raw_archive=timing_ref, comparison=comparison_ref,
            root_disposition=disposition_ref, exact_source_executed=False,
            timed_source_sha256=comparison['timed_source_sha256'], numerical_noise_threshold=None),
        full=dict(packet=packet_ref, launch_receipt=launch_ref, launch=launch['launch'],
            actual_completed_receipt=None, fresh_public20m_score=None, measured_full_cold_seconds=None,
            measured_full_charged_seconds=None, final_disk_or_total_writes=None),
        historical_parent=dict(evidence=historical_ref, parent_receipt_count=2,
            historical_mean_not_f1_mean=historical['mean_public20m'],
            substitution_scope='Explicit R158 rationale only; not an actual second F1 run'),
        completed_f1_full_receipts_in_this_draft=0, official_crash_cause='UNKNOWN',
        local_numerical_execution=False, host_calls=0, automated_acceptance=False,
        package_accepted=False, claude_verdict=None, upload_performed=False)
    copies['EVIDENCE.json'] = enc(evidence)
    copies['RAW_RECEIPT_INDEX.json'] = enc(dict(schema='r162-e1bd-f1-draft-raw-index-v1', archives=raw_index))
    copies['REFERENCES.json'] = enc(references)
    additions = []
    for name, data in copies.items():
        assert not (H/name).exists() and data.endswith(b'\n')
        additions += ['*** Add File: '+str(H/name)] + ['+'+line for line in data.decode().splitlines()]
    patch = '\n'.join(['*** Begin Patch']+additions+['*** End Patch',''])
    result = subprocess.run(['apply_patch'], input=patch, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    for name,data in copies.items():
        assert (H/name).read_bytes() == data, ('COPY_BYTES_DIFFER',name)
    files = {str(p.relative_to(H)):dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
             for p in sorted(H.rglob('*')) if p.is_file()}
    manifest = enc(dict(schema='r162-e1bd-f1-export-draft-manifest-v1', status='EXPORT_DRAFT_NOT_READY',
        source_sha256=SOURCE, only_proposed_payload='train.py', files=files,
        host_calls=0, model_or_tensor_execution=False, all_copied_bytes_verified=True,
        raw_archives_wrapped_without_byte_normalization=wrapped_raw,
        completed_f1_full_receipt=None, package_ready=False, package_accepted=False,
        claude_verdict=None, submission_authorized=False, upload_performed=False))
    patch = '\n'.join(['*** Begin Patch','*** Add File: '+str(H/'EXPORT_MANIFEST.json')]+
                      ['+'+line for line in manifest.decode().splitlines()]+['*** End Patch',''])
    result = subprocess.run(['apply_patch'],input=patch,text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (H/'EXPORT_MANIFEST.json').read_bytes() == manifest
    print(json.dumps(dict(status='EXPORT_DRAFT_NOT_READY', path=str(H),
        manifest_sha256=sha(manifest), copied_files=len(copies), all_files=len(files),
        authenticated_raw_members=sum(len(x['members'])for x in raw_index),
        host_calls=0, local_model_or_tensor_execution=False)))


if __name__ == '__main__':
    main()
