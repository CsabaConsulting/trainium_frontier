"""Exact narrow successor proof; stdlib/Git objects only."""
import ast
import hashlib
import json
import subprocess

REPO = '/home/poojith-devan/Desktop/Trainium'
P = 'research/v2/proposals/20260926-p-offline-decoder-trn21/'
OLD = '8429c1a2033e46936a40a7be4195419724971e48'
NEW = 'b529468f29d6a8f763cf772d9a607b3b6ad12286'
checks = []


def need(value, name):
    if not value:
        raise RuntimeError(name)
    checks.append(name)


def raw(commit, name):
    return subprocess.check_output(['git', 'show', commit + ':' + P + name], cwd=REPO)


def sha(value):
    return hashlib.sha256(value).hexdigest()


addition = '''                unexpected = {row['pid']: row for event in result['termination']
                              for row in event['identities'] if row['pid'] != child.pid}
                if unexpected:
                    result['final_cleanup_unexpected_descendants'] = list(unexpected.values())
                    result['final_cleanup_error'] = 'UNEXPECTED_DECODER_DESCENDANT_AT_FINAL_CLEANUP'
                    result.setdefault('error', result['final_cleanup_error'])
                    result['status'] = 'REFUSED_OR_INCOMPLETE'
'''.encode()
old_helper, new_helper = raw(OLD, 'decoder_support.py'), raw(NEW, 'decoder_support.py')
need(new_helper.count(addition) == 1, 'one exact seven-line addition')
need(new_helper.replace(addition, b'', 1) == old_helper, 'all other helper bytes unchanged')
need(raw(OLD, 'host_driver.py').replace(sha(old_helper).encode(), sha(new_helper).encode()) == raw(NEW, 'host_driver.py'), 'driver only helper hash')
need(raw(OLD, 'EXPERIMENT_PROPOSAL.json') == raw(NEW, 'EXPERIMENT_PROPOSAL.json'), 'all proposal inputs/argv/limits unchanged')
old_binding, new_binding = (json.loads(raw(commit, 'OFFLINE_BINDING.disabled.json')) for commit in (OLD, NEW))
need(new_binding.pop('repair')['original_frozen_commit'] == OLD, 'original history retained')
for name in ('host_driver.py', 'decoder_support.py'):
    old_binding['runtime_sha256'][name] = sha(raw(NEW, name))
old_binding['test_sha256'] = sha(raw(NEW, 'test_packet.py'))
need(old_binding == new_binding, 'binding only matching pins and repair metadata')
pins_node = next(node for node in ast.parse(raw(NEW, 'host_driver.py')).body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'PINS')
pins = ast.literal_eval(pins_node.value)
for name, expected in pins.items():
    path = P + name if name not in ('idle_guard.py', 'dispatch_driver.py') else 'research/v2/operator/20260926-common-host-idle-guard/' + name
    value = subprocess.check_output(['git', 'show', NEW + ':' + path], cwd=REPO)
    need(sha(value) == expected, 'actual driver pin:' + name)
for name in ('checks-normal.json', 'checks-optimized.json', 'preparation-pipe-warning.json', 'FINAL_RECHECK.json'):
    need(raw(OLD, name) == raw(NEW, name), 'original evidence retained:' + name)
print(json.dumps(dict(status='PASS_NARROW_REPAIR_SCOPE', old_commit=OLD, new_commit=NEW,
                      checks=len(checks), check_names=checks,
                      hashes={name:sha(raw(NEW,name)) for name in ('host_driver.py','decoder_support.py','OFFLINE_BINDING.disabled.json','MANIFEST.sha256.json')},
                      host_or_ingestion_authority=False), indent=2, sort_keys=True))
