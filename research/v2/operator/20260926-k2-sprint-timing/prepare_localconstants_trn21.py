"""Disabled local-only literal rebind; no host access or scientific execution."""
import ast
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')

def sha(value):
    return hashlib.sha256(value).hexdigest()

def encoded(value):
    return {'sha256': sha(value), 'b64': base64.b64encode(value).decode()}

raw = (HERE / 'trn21-on-prepared.json').read_bytes()
assert sha(raw) == '305f2827a638694cb339687f87844e9ee56b269eb5a2c512834524216bcb75f8'
old = json.loads(raw)
old_label = old['binding']['label']
new_label = 'fs-20260926-k2-localconstants-on-parentcontrol-trn21-r1'
old_source = old['binding']['source_sha256']
new_source = '662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00'
source = (REPO / 'research/v2/lanes/k2-compiler-local-constants/train.py').read_bytes()
assert sha(source) == new_source
proposal_path = 'research/v2/proposals/20260926-k2-compiler-local-constants-timing/EXPERIMENT_PROPOSAL.json'
proposal = (REPO / proposal_path).read_bytes()
assert sha(proposal) == '8f0b1201e0be4e4171835614cc21ec527e52a4708d5aceb3d9eabce68606f370'
p = json.loads(proposal)
assert p['candidate']['source_sha256'] == new_source
assert p['dispatch']['stage_authorized'] is False
assert p['dispatch']['launch_authorized'] is False

binding = json.loads(json.dumps(old['binding']).replace(old_label, new_label).replace(old_source, new_source))
files = {name: base64.b64decode(value['b64']) for name, value in old['files'].items()}
for name, value in files.items():
    assert sha(value) == old['files'][name]['sha256'], name
prior_provenance = files['package/provenance.json']
provenance = json.loads(prior_provenance.decode().replace(old_label, new_label).replace(old_source, new_source))
provenance['scientific_causal_change'] = 'Exact frozen662c K2 ON; six fixed kernel constants localized after the prior dtype repair; no Operator source edit'
files['package/provenance.json'] = (json.dumps(provenance, indent=2) + '\n').encode()
prior_driver = files['host_driver.py']
driver = prior_driver.decode().replace(old_label, new_label).replace(old_source, new_source).replace(sha(prior_provenance), sha(files['package/provenance.json']))
assert driver.replace(new_label, old_label).replace(new_source, old_source).replace(sha(files['package/provenance.json']), sha(prior_provenance)) == prior_driver.decode()
ast.parse(driver)
files['host_driver.py'] = driver.encode()
files['package/train.py'] = source
files['HISTORICAL_MECHANICS_PROPOSAL.json'] = files['EXPERIMENT_PROPOSAL.json']
files['EXPERIMENT_PROPOSAL.json'] = proposal
binding.update(
    driver_sha256=sha(files['host_driver.py']),
    independent_review_sha256=None,
    operator_assignment_sha256=None,
    completed_proposal_path=proposal_path,
    proposal_snapshot_sha256=sha(proposal),
    completed_proposal_sha256=None,
    preparation_status='DISABLED_PENDING_FINAL_PROPOSAL_COMPILE_GATES_CONTROL_AND_SUPERVISOR_RELEASE',
    native_short_timing='One proposed new-source ON45; no additional OFF, quality run or eval',
    timing=p['timing'],
    parent_comparison_precommitted=True,
    comparison_control={
        'host': 'trn21',
        'label': 'fs-20260926-sprint-e1e2-control-trn21-r1',
        'source_sha256': 'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43',
        'kind': 'control',
        'primary_statistic': '35*131072/(t44-t9)',
        'control_window_selection': 'Same-host genuine control prospectively selected before candidate execution; no best-window or cross-host substitution',
        'training_handoff_raw_sha256': '642118a200368f49642f6ba8f612bc4c05227442d82857aad02d0b01abcbda91',
        'completed_bundle_sha256': None,
        'actual_terminal_and_timing_pending': True,
        'attached_public20m_and_quiescence_pending': True,
    },
    additional_human_compile_gate={
        'production_shape_forward_backend_compile_accepted': False,
        'production_shape_backward_backend_compile_accepted': False,
        'simulator_numerics_if_available': 'Pending Supervisor acceptance; no Operator inference from parser-only PASS',
        'one_local_process_at_a_time': True,
    },
    native_gate_conditions=[
        'Exact662c source, final completed proposal and one fresh independent source review authenticated by Supervisor.',
        'New direct human requirement: real production-shape forward AND backward backend compilation accepted; simulator numerics if available; parser-only success does not qualify.',
        'Current trn21 genuine e1e2 control AND attached fresh20M naturally terminal and retrieved; exact matched t9..t44/source/argv/runtime accepted.',
        'Explicit all-observed descendant/compiler/device/original-lock quiescence, then fresh original596823 lock plus ps/neuron-ls/fuser/STOP/hash/resource checks.',
        'Six fresh private cold cache paths absent then created empty; actual four-rank environments verified.',
        'Separate exact Supervisor native release required. This packet creates no RELEASE and grants no stage or launch authority.',
        'ON45 only; no extra OFF, eval, full run, retry, adoption or dependent-table launch. Exact unrounded same-host rate>=1.03 remains required.',
    ],
)
assert p['candidate']['launch_command'] == binding['launch_command']
assert files['package/launch-command.txt'] == base64.b64decode(old['files']['package/launch-command.txt']['b64'])
for name in ('dispatch_driver.py', 'idle_guard.py', 'lifetime_guard.py'):
    assert files[name] == base64.b64decode(old['files'][name]['b64'])
assert binding['lock_identity'] == [66305, 596823]
assert binding['resource_reserve']['initial_floor_bytes'] == 14495514624
assert binding['resource_reserve']['runtime_floor_bytes'] == 11274289152
assert len(set(binding['fresh_cache_directories'])) == 6
assert all(path.startswith(binding['root'] + '/cold-training-caches/') for path in binding['fresh_cache_directories'])
assert binding['steps'] == 45 and binding['no_eval'] is True
assert all(binding[key] is False for key in ('stage_authorized', 'launch_authorized', 'host_reserved', 'release_file_created', 'source_adoption', 'submission_authorized'))
files['NATIVE_BINDING.json'] = (json.dumps(binding, indent=2) + '\n').encode()
packet = {
    'binding': binding,
    'files': {name: encoded(value) for name, value in files.items()},
    'mechanical_derivation': {
        'prior_packet_sha256': sha(raw),
        'prior_driver_sha256': sha(prior_driver),
        'literal_driver_changes': ['label/root/private-cache namespace', 'exact new source SHA', 'corresponding provenance SHA'],
        'reverse_literal_derivation_exact': True,
        'driver_logic_scientific_argv_guardian_dispatch_unchanged': True,
        'source_copied_unmodified_not_executed': True,
        'historical_driver_docstrings_not_scientific_authority': True,
    },
    'local_only': True,
    'host_calls': 0,
    'native_started': False,
    'disabled': True,
}
out = HERE / 'trn21-localconstants-prepared.disabled.json'
with out.open('x') as handle:
    json.dump(packet, handle, indent=2)
    handle.write('\n')
print(json.dumps({'packet_sha256': sha(out.read_bytes()), 'driver_sha256': binding['driver_sha256'], 'proposal_snapshot_sha256': sha(proposal), 'source_sha256': new_source, 'label': new_label, 'disabled': True, 'host_calls': 0}))
