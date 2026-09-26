"""Fixed stable-source pair checks. Stdlib only; no training imports or SDK calls."""
import csv
import hashlib
import io
import itertools
from importlib.machinery import PathFinder
import json
import math
import os
from pathlib import Path
import re
import stat

def require(value, message):
    if not value:
        raise RuntimeError(message)

def no_symlink(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'NON_ABSOLUTE_OR_PARENT_PATH')
    for item in (path, *path.parents):
        require(not stat.S_ISLNK(item.lstat().st_mode), 'SYMLINK:' + str(item))
    require(path.resolve(strict=True) == path, 'RESOLVED_PATH_CHANGED')
    return path

def identity(path, expected=None, readonly=False, max_bytes=None):
    path = no_symlink(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), 'NOT_REGULAR:' + str(path))
    require(before.st_nlink == 1, 'HARDLINK:' + str(path))
    if readonly:
        require(before.st_mode & 0o222 == 0, 'WRITABLE_IMMUTABLE_FILE:' + str(path))
    if max_bytes is not None:
        require(before.st_size <= max_bytes, 'FILE_TOO_LARGE:' + str(path))
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        opened = os.fstat(stream.fileno())
        require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), 'OPEN_IDENTITY_CHANGED')
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.lstat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode', 'st_uid', 'st_gid', 'st_nlink')
    require(all(getattr(before, k) == getattr(after, k) for k in fields), 'FILE_CHANGED_DURING_HASH')
    require(expected is None or digest == expected, 'SHA_MISMATCH:' + str(path))
    return dict(path=str(path), sha256=digest, **{k: getattr(after, k) for k in fields})

def inputs(plan, phase):
    stable = no_symlink(plan['stable_directory'])
    st = stable.stat()
    require(stat.S_ISDIR(st.st_mode) and st.st_mode & 0o222 == 0, 'STABLE_DIRECTORY_NOT_READONLY')
    require(sorted(x.name for x in stable.iterdir()) == ['prepare.py', 'train.py'], 'STABLE_DIRECTORY_CONTENTS')
    spec = PathFinder.find_spec('prepare', [str(stable)])
    require(spec is not None and spec.origin == str(stable/'prepare.py'), 'EXPECTED_PREPARE_RESOLUTION')
    package = Path(phase['package'])
    require(phase['training_command'][3] == str(stable/'train.py'), 'EXECUTED_SOURCE_PATH')
    require('PYTHONPATH' not in phase['environment'] and 'PYTHONHOME' not in phase['environment'], 'IMPORT_PATH_OVERRIDE')
    checks = {
        'executed_train': identity(stable/'train.py', plan['source_sha256'], True),
        'expected_imported_prepare': identity(stable/'prepare.py', plan['prepare_sha256'], True),
        'package_train': identity(package/'train.py', plan['source_sha256'], True),
        'protected_prepare': identity(plan['prepare_path'], plan['prepare_sha256']),
        'harness': identity(plan['harness_path'], plan['harness_sha256']),
        'launch': identity(package/'launch-command.txt', phase['launch_sha256'], True),
        'provenance': identity(package/'provenance.json', phase['provenance_sha256'], True),
        'stable_directory': {k: getattr(st, k) for k in ('st_dev','st_ino','st_mode','st_uid','st_gid','st_mtime_ns','st_ctime_ns')},
    }
    manifest = json.loads((package/'provenance.json').read_text())
    require(manifest['train_sha256'] == plan['source_sha256'], 'MANIFEST_SOURCE')
    require(manifest['launch_command'] == (package/'launch-command.txt').read_text().strip(), 'MANIFEST_COMMAND')
    return checks

def validate_phase(plan, phase):
    root = Path(phase['root'])
    sandbox = Path(phase['sandbox'])
    receipt = json.loads((sandbox/'rehearsal.json').read_text())
    require(type(receipt['returncode']) is int and receipt['returncode'] == 0 and receipt['mode'] == 'train', 'HARNESS_FAILED')
    obs = receipt['observations']
    require(type(receipt['step_boundaries']) is int and len(obs) == receipt['step_boundaries'] == 5, 'FIVE_BOUNDARIES_REQUIRED')
    require(all(type(v['step']) is int for v in obs) and [v['step'] for v in obs] == list(range(5)), 'STEP_SEQUENCE')
    times = [v['observed_seconds'] for v in obs]
    require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in times), 'INVALID_BOUNDARY_TIME')
    require(times == sorted(times), 'BOUNDARY_ORDER')
    wall = receipt['wall_seconds']
    require(type(wall) in (int,float) and math.isfinite(wall) and wall >= times[-1], 'INVALID_EXIT_WALL')
    allowance = min(times[4], 900.0)
    charged = wall - allowance
    require(all(type(receipt[k]) in (int,float) and math.isfinite(receipt[k]) and receipt[k] >= 0
                for k in ('startup_allowance_observed','charged_through_process_exit')), 'INVALID_ACCOUNTING_NUMBER')
    require(receipt['startup_allowance_observed'] == allowance, 'ALLOWANCE_MISMATCH')
    require(receipt['charged_through_process_exit'] == charged, 'CHARGED_MISMATCH')
    require(receipt['charged_through_process_exit_within_budget'] is True and charged <= 1800, 'SOURCE_BUDGET')
    if phase['name'] == 'S1':
        require(times[4] <= 900, 'S1_COLD_FIFTH_OVER_900')
        require(receipt['fresh_empty_sdk_caches'] == plan['cache_environment'], 'S1_FRESH_CACHE_BINDING')
    else:
        require(receipt['fresh_empty_sdk_caches'] == {}, 'S2_MUST_REUSE_S1_CACHE')
    checkpoint = identity(sandbox/'out/final.pt', receipt['checkpoint_sha256'])
    require(checkpoint['st_size'] == receipt['checkpoint_bytes'] and checkpoint['st_size'] > 0, 'CHECKPOINT_SIZE')
    identity(sandbox/'train.py', plan['source_sha256'], True)
    identity(sandbox/'prepare.py', plan['prepare_sha256'])
    log = (root/'training.log').read_text()
    rows = re.findall(r'^step (\d+) \| loss (\S+) \| lrm (\S+) \| dt (\S+)s \|', log, re.MULTILINE)
    require([int(row[0]) for row in rows] == list(range(5)), 'FIVE_LOSS_ROWS_REQUIRED')
    losses = [dict(step=int(step), loss=float(loss), lrm=float(lrm), step_seconds=float(dt)) for step,loss,lrm,dt in rows]
    require(all(math.isfinite(v[k]) for v in losses for k in ('loss','lrm','step_seconds')), 'NONFINITE_LOGGED_VALUE')
    return dict(status='FIVE_STEP_MECHANICAL_GATES_PASS', fifth_boundary_seconds=times[4],
                process_wall_seconds=wall, charged_through_exit_seconds=charged,
                checkpoint=checkpoint, first_five_logged=losses,
                loss_precision='Existing rank0 log only: loss 4 decimals, lrm 3; not full-precision or per-rank parity',
                actual_import_origin_observed=False, runtime_reuse='UNPROVEN')

def capture_inputs(plan, phase):
    """Bounded read/copy of exact existing compiler bytes, never mutation/decoding."""
    evidence = Path(phase['root'])/'compiler-evidence'
    evidence.mkdir(exist_ok=False)
    graph_rows = {}
    keys = set()
    for rank in range(4):
        file = Path(phase['environment']['TORCH_NEURONX_METRICS_DIR'])/('metric_table_graph_stats_rank'+str(rank)+'.csv')
        entry = identity(file, max_bytes=2*1024**2)
        raw = file.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == entry['sha256'], 'METRICS_CHANGED')
        with (evidence/file.name).open('xb') as stream: stream.write(raw)
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
        principal = [r for r in rows if r.get('model_name') == 'model_default' and
                     ('forward' in r.get('graph_name','') or 'backward' in r.get('graph_name',''))]
        require(any('forward' in r['graph_name'] for r in principal) and any('backward' in r['graph_name'] for r in principal), 'PRINCIPAL_GRAPH_ROWS_MISSING')
        for row in principal:
            require(re.fullmatch(r'[0-9a-f]{64}:[0-9a-f]{64}', row['graph_id']) is not None, 'UNKNOWN_GRAPH_KEY_FORMAT')
            keys.add(row['graph_id'].split(':')[0])
        graph_rows[str(rank)] = dict(file=entry, rows=rows)
    debug = no_symlink(Path(plan['cache_environment']['TORCH_NEURONX_DEBUG_DIR'])/'neuronx-cc')
    children = list(itertools.islice(debug.iterdir(), 129))
    require(len(children) <= 128, 'BOUNDED_DEBUG_DIRECTORY_LIMIT')
    found = {}
    total = 0
    for child in children:
        no_symlink(child)
        if not child.is_dir(): continue
        file = child/'module.mlir'
        if not file.exists(): continue
        entry = identity(file, max_bytes=4*1024**2)
        if entry['sha256'] not in keys: continue
        total += entry['st_size']
        require(total <= 64*1024**2, 'BINARY_EVIDENCE_CAP')
        raw = file.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == entry['sha256'], 'COMPILER_INPUT_CHANGED')
        target = evidence/(entry['sha256']+'.mlir')
        if not target.exists():
            with target.open('xb') as stream: stream.write(raw)
        found.setdefault(entry['sha256'], []).append(entry)
    return dict(graphs=graph_rows, principal_keys=sorted(keys), authenticated_raw_inputs=found,
                missing_keys=sorted(keys-set(found)), exact_input_bytes_complete=keys == set(found),
                input_provenance_scope='Existing exact debug input bytes matching this phase graph keys; S2 may retain S1 files. Not proof of new generation or runtime reuse.',
                sdk_layout_scope='Only debug/neuronx-cc/*/module.mlir, at most128 immediate children; no recursive cache walk',
                cache_hit=False, zero_compile_us_is_hit_proof=False, native_neff_reuse='UNPROVEN')

def compare_losses(s1, s2):
    return dict(scope='Printed rank0 first-five losses only; wallclock LR is a confounder',
                rows=[dict(step=a['step'], s1_loss=a['loss'], s2_loss=b['loss'],
                           loss_delta=b['loss']-a['loss'], s1_lrm=a['lrm'], s2_lrm=b['lrm'])
                      for a,b in zip(s1['first_five_logged'],s2['first_five_logged'])],
                arithmetic_equivalence_proven=False, source_path_only_causality_proven=False)
