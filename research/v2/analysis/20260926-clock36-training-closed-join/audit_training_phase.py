"""Local closed-training archive diagnostic; no host, SDK, tensor or data access."""
import base64, csv, hashlib, importlib.util, io, json, re, subprocess
from pathlib import Path

REPO = Path('/home/poojith-devan/Desktop/Trainium')
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('closed_join', HERE.parent / '20260926-clock36-closed-csv-join/join_closed_csv.py')
join = importlib.util.module_from_spec(spec)
spec.loader.exec_module(join)
checks = []
def check(label, ok):
    if not ok:
        raise RuntimeError(label)
    checks.append(label)
def archived(ref, path, digest):
    raw = (REPO / path).read_bytes()
    check('archive hash: ' + path, join.sha(raw) == digest)
    check('immutable Git blob: ' + path, raw == subprocess.check_output(['git', 'show', ref + ':' + path], cwd=REPO))
    return raw
def decode(item):
    raw = item['text'].encode() if 'text' in item else base64.b64decode(item['base64'], validate=True)
    check('per-file hash and size', len(raw) == item['bytes'] and join.sha(raw) == item['sha256'])
    return raw

full_path = 'research/v2/operator/20260926-trn22-startup-clock-full/training-completed/results-bundle.json'
full_sha = '34cf54017543997d7d5673e023601d02ecb6069cfb3a17219970bcf2cee08911'
raw = archived('6b292621', full_path, full_sha)
result = join.analyze(raw, full_sha)
check('final-evaluation gate remains unknown', result['status'] == 'UNKNOWN' and result['unknown_reasons'] == ['FINAL_SUCCESSFUL_FULL_BUNDLE_UNPROVEN'])
files = {name: decode(item) for name, item in join.parse(raw)['files'].items()}
h = join.parse(files['train/rehearsal.json'])
check('training process completed', h['returncode'] == 0 and h['mode'] == 'train' and files['run-exit-code.txt'].strip() == b'0' and b'TRAINING_END_UTC=' in files['training.log'])
check('no final terminal or outer receipt', 'terminal.json' not in files and 'outer-flock-exit.json' not in files)
check('four complete exact graph joins', all(x['all_four_ranks_one_consistent_row'] for x in result['capture_joins']))

cold_path = 'research/v2/operator/20260926-trn22-startup-clock-cold-five/completed/results-bundle.json'
cold_sha = '6c45190b3c1db2242073fdb04a1a5ca997ec2eb51c7abf655e70488b0b0f7e18'
cold = join.parse(archived('e08afdec', cold_path, cold_sha))
cold_files = {name: decode(item) for name, item in cold['files'].items()}
cold_rows = []
for name, value in cold_files.items():
    m = re.fullmatch(r'metrics-training/metric_table_graph_stats_rank([0-3])\.csv', name)
    if m:
        cold_rows.extend(dict(row, rank=int(m[1])) for row in csv.DictReader(io.StringIO(value.decode())))
comparison = []
for entry in result['capture_joins']:
    name = entry['matches'][0]['graph_name']
    matching = [x for x in cold_rows if x['graph_name'] == name]
    check('cold named graph all ranks: ' + name, sorted(x['rank'] for x in matching) == [0, 1, 2, 3])
    check('matching node counts: ' + name, all(int(x['graph_node_count']) == entry['matches'][0]['graph_node_count'] for x in matching))
    comparison.append(dict(graph_name=name, graph_node_count=entry['matches'][0]['graph_node_count'], cold_graph_ids=sorted({x['graph_id'] for x in matching}), full_graph_ids=sorted({x['graph_id'] for x in entry['matches']}), cold_positive_compile_seconds=[dict(rank=x['rank'], seconds=float(x['torch_neuronx_compile_us']) / 1e6) for x in matching if float(x['torch_neuronx_compile_us']) > 0], full_positive_compile_seconds=[dict(rank=x['rank'], seconds=x['positive_compile_seconds']) for x in entry['matches'] if x['positive_compile_seconds'] is not None]))

patterns = [r'persistent.{0,40}cache', r'cache.{0,30}(hit|miss)', r'execution_cache_key', r'(read|writ|load|sav).{0,40}neff', r'neff.{0,40}(read|writ|load|sav)']
def cache_lines(raw):
    return [dict(line=i + 1, text=s) for i, s in enumerate(raw.decode().splitlines()) if any(re.search(p, s, re.I) for p in patterns)]
messages = [dict(artifact='training.log', sha256=join.sha(files['training.log']), retained_bytes=len(files['training.log']), coverage='complete archived training log', matches=cache_lines(files['training.log']))]
for index, cap in enumerate(join.CAPTURES, 1):
    path = f'research/v2/operator/20260926-trn22-clock-full-live-compiler/observation-{index}.json'
    observation = join.parse(archived(cap['commit'], path, cap['observation_sha256']))
    d = observation['directories'][0]
    for f in d['files']:
        content = base64.b64decode(f['base64'], validate=True)
        check('stable capture byte identity', join.sha(content) == f['sha256'] and len(content) == f['bytes_retained'] and f['before'] == f['after'] == f['path_after'])
        if f['name'] == 'module.mlir':
            check('actual module bound to CSV join', join.sha(content) == cap['module_sha256'])
        elif f['name'] == 'log-neuron-cc.txt':
            messages.append(dict(artifact=path + '#log-neuron-cc.txt', sha256=join.sha(content), retained_bytes=len(content), original_bytes_at_capture=f['before']['size'], coverage='prefix only; absent later messages unknown', matches=cache_lines(content)))
out = dict(scope='COMPLETED_TRAINING_PHASE_ONLY_EVALUATION_PENDING', checks=len(checks), bundle_commit='6b292621', bundle_path=full_path, bundle_sha256=full_sha, original_analyzer=result, cold_named_graph_comparison=comparison, explicit_cache_message_search=dict(patterns=patterns, artifacts=messages), conclusions=dict(principal_mapping='Exact raw module SHA equals graph_id prefix on all four ranks.', observed_compilation='Positive per-row compiler durations; not summed across ranks or substituted for wall time.', cache_reuse='UNKNOWN: no explicit matching messages in these retained bytes; no NEFF read/write trace.', source_path_causality='UNPROVEN: key changes and equal graph names/node counts do not isolate the cause.', evaluation='PENDING_IN_THIS_TRAINING_ONLY_ARCHIVE', quality_verdict='NOT_PERFORMED'))
print(json.dumps(out, indent=2, allow_nan=False))
