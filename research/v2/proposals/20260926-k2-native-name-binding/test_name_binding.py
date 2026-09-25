#!/usr/bin/env python3
"""Own stdlib scope fixture; never imports real NKI/Torch or runs the kernel."""
import ast
import dis
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[4]
LANE = 'research/v2/lanes/k2-resident-consumer/'
OLD = '18b9365aabb63fa85e1720fdefc11177ca8e7853'
PATTERN = b'def build_kernel(*, enabled=False):\n'
ADDITION = b'    global nl, nisa\n'
checks = []


def check(name, condition):
    if not condition:
        raise RuntimeError(name)
    checks.append(name)


def blob(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=ROOT)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parts(raw):
    tree = ast.parse(raw)
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    kernel = next(n for n in factory.body if isinstance(n, ast.FunctionDef))
    return tree, factory, kernel


def main():
    old = blob(OLD, LANE + 'consumer_nki.py')
    new = (ROOT / LANE / 'consumer_nki.py').read_bytes()
    check('old source authenticated', sha(old) == '57b5b118e86eb9d78e26a011b276904aed9065868b8a12f39169e1257c3a1843')
    check('one exact added outer declaration', old.count(PATTERN) == 1 and new == old.replace(PATTERN, PATTERN + ADDITION, 1))
    old_tree, old_factory, old_kernel = parts(old)
    new_tree, new_factory, new_kernel = parts(new)
    check('exact kernel AST unchanged', ast.dump(old_kernel) == ast.dump(new_kernel))
    # Include its decorator and every comment/blank line through EOF, not just AST.
    old_body = old[old.index(b'    @nki.jit\n'):]
    new_body = new[new.index(b'    @nki.jit\n'):]
    check('exact kernel and return bytes unchanged', old_body == new_body)
    check('signature unchanged', ast.dump(old_factory.args) == ast.dump(new_factory.args))
    check('new declaration only globals', isinstance(new_factory.body[0], ast.Global) and new_factory.body[0].names == ['nl', 'nisa'])
    del new_factory.body[0]
    check('entire remaining module AST unchanged', ast.dump(old_tree) == ast.dump(new_tree))
    for name, expected in {
        'check_native.py': '89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850',
        'reference.py': '14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f',
        'test_reference.py': '972cb83fe1d31a4a5cd328d36b2e196123b8ef74bbcdff081981373c257f90d4',
        'test_reporting.py': '8dd3fe2e064cf17b7403576c8c205b534dbde1d66b72c0ee5e53c6862906abd3',
    }.items():
        current = (ROOT / LANE / name).read_bytes()
        check(name + ' unchanged bytes and pin', current == blob(OLD, LANE + name) and sha(current) == expected)

    imported = []
    def refuse_import(name, *args, **kwargs):
        imported.append(name)
        raise RuntimeError('FIXTURE_IMPORT_FORBIDDEN:' + name)

    # This explicit unoptimized compile retains dtype assertions even under -O.
    codes = {name: compile(raw, name + '_consumer.py', 'exec', optimize=0)
             for name, raw in [('old', old), ('new', new)]}
    for name, code in codes.items():
        namespace = {'__builtins__': {**vars(__import__('builtins')), '__import__': refuse_import}}
        exec(code, namespace)
        check(name + ' module import has no API bindings', 'nl' not in namespace and 'nisa' not in namespace)
        for kwargs in ({}, {'enabled': False}):
            try:
                namespace['build_kernel'](**kwargs)
            except RuntimeError as error:
                check(name + ' disabled refusal', str(error) == 'EXPERIMENT_DISABLED: no native module imported')
            else:
                raise RuntimeError(name + ' default must refuse')
        check(name + ' disabled makes no import', not imported)

    fake_nki = types.ModuleType('nki')
    fake_nki.__path__ = []
    fake_nl = types.ModuleType('nki.language')
    fake_isa = types.ModuleType('nki.isa')
    fake_nl.bfloat16 = object()
    decorated = []
    def identity_jit(function):
        decorated.append(function)
        return function
    fake_nki.jit = identity_jit
    fake_nki.language = fake_nl
    fake_nki.isa = fake_isa
    fixtures = {'nki': fake_nki, 'nki.language': fake_nl, 'nki.isa': fake_isa}
    check('no real framework imported before fixture', not any(n == 'nki' or n.startswith('nki.') or n == 'torch' or n.startswith('torch.') for n in sys.modules))
    result = {}
    try:
        sys.modules.update(fixtures)
        for name, code in codes.items():
            namespace = {}
            exec(code, namespace)
            fn = namespace['build_kernel'](enabled=True)
            check(name + ' decorator only returned function', fn is decorated[-1] and fn.__name__ == 'resident_consumer')
            check(name + ' fixed five-input API', fn.__code__.co_argcount == 5)
            aliases = {i.argval: i.opname for i in dis.get_instructions(fn) if i.argval in ('nl', 'nisa') and i.opname.startswith('LOAD_')}
            result[name] = {'freevars': list(fn.__code__.co_freevars), 'loads': aliases,
                            'module_aliases_present': all(alias in fn.__globals__ for alias in ('nl', 'nisa'))}
            if name == 'old':
                check('old aliases closure-only', set(fn.__code__.co_freevars) == {'nl', 'nisa'} and not result[name]['module_aliases_present'])
                check('old namespace loads dereference', aliases == {'nl': 'LOAD_DEREF', 'nisa': 'LOAD_DEREF'})
            else:
                check('new aliases not closure cells', fn.__code__.co_freevars == ())
                check('new aliases in module globals at decoration', fn.__globals__['nl'] is fake_nl and fn.__globals__['nisa'] is fake_isa)
                check('new namespace loads global', aliases == {'nl': 'LOAD_GLOBAL', 'nisa': 'LOAD_GLOBAL'})
                check('global dotted dtype lookup resolves in fixture', fn.__globals__['nl'].bfloat16 is fake_nl.bfloat16)
                again = namespace['build_kernel'](enabled=True)
                check('repeat factory retains API identities', again.__globals__['nl'] is fake_nl and again.__globals__['nisa'] is fake_isa)
    finally:
        for name in fixtures:
            sys.modules.pop(name, None)
    check('fake namespaces removed and no real framework imported', not any(n == 'nki' or n.startswith('nki.') or n == 'torch' or n.startswith('torch.') for n in sys.modules))
    print(json.dumps({'status': 'PASS_STDLIB_SCOPE_ONLY', 'checks': len(checks),
                      'kernel_sha256': sha(new), 'old_kernel_sha256': sha(old),
                      'unchanged_kernel_tail_sha256': sha(new_body), 'namespace_fixture': result,
                      'checks_passed': checks, 'real_native_imported': False,
                      'kernel_body_executed': False, 'tensor_allocated': False,
                      'native_correctness': 'UNPROVEN', 'launch_authorized': False}, indent=2))


if __name__ == '__main__':
    main()
