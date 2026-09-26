import sys, json, hashlib
from pathlib import Path
from fractions import Fraction
from functools import lru_cache

B = Path('/home/poojith-devan/k2-trn22-evidence-review-Mh0Zed')
sys.path.insert(0, str(B/'candidate'))
import oracle, host_driver, evidence

def check(ok, message):
    if not ok:
        raise RuntimeError(message)

@lru_cache(None)
def value(word):
    exponent = (word >> 23) & 255
    mantissa = word & 0x7fffff
    if exponent:
        mantissa += 1 << 23
    power = exponent - 150 if exponent else -149
    return Fraction(mantissa) * Fraction(2) ** power

def reference_add(a, b):
    total = (-1 if a >> 31 else 1)*value(a & 0x7fffffff) + (-1 if b >> 31 else 1)*value(b & 0x7fffffff)
    if total == 0:
        return 0x80000000 if a == b == 0x80000000 else 0
    sign = 0x80000000 if total < 0 else 0
    total = abs(total)
    if total >= value(0x7f7fffff) + Fraction(2)**103:
        return sign | 0x7f800000
    lo, hi = 0, 0x7f7fffff
    while lo < hi:
        mid = (lo + hi + 1)//2
        if value(mid) <= total:
            lo = mid
        else:
            hi = mid - 1
    if lo == 0x7f7fffff:
        return sign | lo
    lower, upper = total-value(lo), value(lo+1)-total
    nearest = lo if lower < upper or (lower == upper and lo % 2 == 0) else lo+1
    return sign | nearest

raw = {'native': False, 'scope': 'OWN_SCALAR_REFERENCE_AND_DISABLED_BINDING_ONLY', 'checks': {}}
f0 = []
for channel in range(1024):
    ref = got = 0
    for _ in range(4):
        word = (0x3800+channel) << 16
        ref = reference_add(ref, word)
        got = oracle.add32(got, word)
        check(ref == got, 'F0 mismatch')
    f0.append('%08x' % got)
raw['checks']['f0_channels'] = len(f0)
raw['f0_final_words'] = f0
timelines = [[0x4b80, 0x3f80, 0xcb80], [0x3f80, 0x3380, 0xbf80],
             [0x0081, 0x8080], [0x8081, 0x0080], [0x0001], [0x8001],
             [0x8000, 0x0000], [0x8000, 0x8000]]
f2 = []
for line in timelines:
    ref = 0
    for word in line:
        ref = reference_add(ref, word << 16)
    got = oracle.recurrence(line)
    check(got == ref, 'F2 mismatch')
    f2.append({'bf16_words': line, 'result': '%08x' % got})
raw['f2_scalar_witnesses'] = f2
negative = {}
def reject(name, function, expected):
    try:
        function()
    except (RuntimeError, ValueError) as exc:
        check(str(exc) == expected, name + ': wrong failure ' + str(exc))
        negative[name] = str(exc)
    else:
        raise RuntimeError(name + ': did not reject')
for bad in [-1, 65536, True, 1.5]:
    reject('bf16_'+repr(bad), lambda b=bad: oracle.recurrence([b]), 'BF16_BITS_REQUIRED')
reject('disabled_entrypoint', host_driver.main, 'DISABLED_NO_SEPARATE_HOST_RELEASE')
host_driver.require_original_lock({'device':66305, 'inode':832589})
for lock in [{}, {'device':66305, 'inode':596823}, {'device':66306, 'inode':832589}]:
    reject('lock_'+str(lock), lambda l=lock: host_driver.require_original_lock(l), 'ORIGINAL_LOCK_IDENTITY_MISMATCH')
saved = dict(host_driver.BINDING)
try:
    host_driver.BINDING['launch_authorized'] = True
    reject('flag_only_enable', host_driver.main, 'WRONG_BOUND_PACKET_ROOT')
    old_here = host_driver.HERE
    host_driver.HERE = Path(host_driver.BINDING['packet_root'])
    reject('flag_and_path_enable', host_driver.main, 'INCOMPLETE_BOUND_PROJECT_OR_STAGE')
    host_driver.HERE = old_here
finally:
    host_driver.BINDING.clear()
    host_driver.BINDING.update(saved)
inv = evidence.Invocation(0, 0, 'challenge', {}, ['gradient'], {})
reject('missing_inputs', lambda: inv.call_start({'event':'call_start','nonce':'challenge','ordinal':0}), 'CALL_WITHOUT_INPUTS')
reject('fabricated_start', lambda: inv.call_start({'event':'call_start','nonce':'wrong','ordinal':0}), 'FABRICATED_CALL_START')
reject('missing_route', inv.complete, 'MISSING_ROUTE_OR_ASSOCIATION')
env = host_driver.environment(Path(saved['output_root']), '<authenticated inherited FD>')
check(env == json.loads((B/'spec/ENVIRONMENT_TEMPLATE.json').read_text()), 'environment mismatch')
check(saved['stage_authorized'] is False and saved['launch_authorized'] is False and saved['host_reserved'] is False, 'authority flags')
check(saved['project_commit'] == '65155c668efaa215a2136f67617b44ed32d4c9f6', 'project commit')
donor = json.loads((B/'receipts/DONOR_SOURCE_PINS.json').read_text())
pins = json.loads((B/'candidate/SOURCE_PINS.json').read_text())
unchanged = []
for name in ['consumer_nki.py','worker.py','coordinator.py','oracle.py','fixtures.py','generate.py','evidence.py','artifacts.py','monitor.py','idle_guard.py']:
    actual = hashlib.sha256((B/'candidate'/name).read_bytes()).hexdigest()
    check(actual == donor[name] == pins[name], 'unchanged pin '+name)
    unchanged.append({'name':name, 'sha256':actual})
for name, new, old in [('common.py','fs-20260926-k2-realshape-f0f2-trn22-evidence-r1','fs-20260926-k2-realshape-f0f2-trn21-r1'), ('host_driver.py','e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43','36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb')]:
    content = (B/'candidate'/name).read_text()
    check(content.count(new) == 1, 'inverse unique '+name)
    check(hashlib.sha256(content.replace(new,old).encode()).hexdigest() == donor[name], 'inverse mismatch '+name)
raw.update(negative_checks=negative, unchanged_sources=unchanged, inverse_identity=True, environment_exact=True,
           input_manifest_sha256=hashlib.sha256((B/'SHA256SUMS').read_bytes()).hexdigest())
print(json.dumps(raw, indent=2, sort_keys=True))
