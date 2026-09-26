"""One F0/F2 preparation and authenticated four-rank coordinator, no host release."""
import concurrent.futures
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import numpy as np
from common import *
from evidence import Writer, Invocation, blob_from_worker
from fixtures import V, T, C, token_id, validate_metadata
from generate import prepare_fixture, OracleBudget


def authenticate_worker(info, hello, launcher, seen, pins, root, worker_path, python_path):
    """Kernel peer + distinct process + live ancestry + actual exec environment."""
    rank = hello.get('rank')
    key = (info['pid'],info['start'])
    require(type(rank) is int and rank in range(4), 'RANK_IDENTITY')
    require(info['pid'] != launcher['pid'], 'LAUNCHER_IS_NOT_WORKER')
    require(key not in seen and all(pid != info['pid'] for pid,start in seen), 'DUPLICATE_WORKER_PROCESS')
    require(hello.get('pid') == info['pid'] and hello.get('start') == info['start'] and
            hello.get('world') == 4 and hello.get('pins') == pins, 'PEER_SOURCE_IDENTITY')
    proc = Path('/proc/%d' % info['pid'])
    with (proc/'environ').open('rb') as stream:
        raw_env = stream.read(65537)
    require(len(raw_env) <= 65536, 'WORKER_ENV_CAP')
    wanted = {'RANK':str(rank),'LOCAL_RANK':str(rank),'WORLD_SIZE':'4','NEURON_LOGICAL_NC_CONFIG':'2'}
    actual = {}
    for pair in raw_env.split(b'\0'):
        name, sep, value = pair.partition(b'=')
        if name in (b'RANK',b'LOCAL_RANK',b'WORLD_SIZE',b'NEURON_LOGICAL_NC_CONFIG'):
            name = name.decode('ascii')
            require(name not in actual, 'DUPLICATE_WORKER_ENV_KEY')
            actual[name] = value.decode('ascii')
    require(actual == wanted, 'ACTUAL_WORKER_ENVIRONMENT')
    del raw_env  # Never persist unrelated environment/credentials.
    with (proc/'cmdline').open('rb') as stream:
        raw_argv = stream.read(4097)
    require(len(raw_argv) <= 4096, 'WORKER_ARGV_CAP')
    argv = raw_argv.rstrip(b'\0').decode().split('\0')
    require(argv and Path(argv[0]).parent == Path(python_path).parent and
            os.path.samefile(argv[0],python_path), 'ACTUAL_WORKER_INTERPRETER')
    expected = [argv[0],str(worker_path),str(root)]
    require(argv in (expected,[expected[0],'-u',*expected[1:]]), 'ACTUAL_WORKER_ARGV')
    require(os.path.samefile(proc/'exe',python_path), 'ACTUAL_WORKER_EXECUTABLE')
    ancestry, ancestor, visited = [], info, set()
    while ancestor['pid'] != launcher['pid']:
        require(ancestor['pid'] not in visited and ancestor['ppid'] > 1, 'PEER_NOT_OWNED')
        visited.add(ancestor['pid'])
        ancestry.append(ancestor)
        ancestor = identity(ancestor['ppid'])
    require(ancestor['start'] == launcher['start'] and ancestor['state'] != 'Z', 'TORCHRUN_REUSED_OR_EXITED')
    require(identity(info['pid'])['start'] == info['start'], 'WORKER_IDENTITY_CHANGED')
    seen.add(key)
    return dict(peer=info, actual_rank_environment=actual, actual_argv=argv,
                authenticated_ancestry=ancestry, launcher=launcher)


def canonical(writer, deadline):
    budget, cases = OracleBudget(), {}
    for name in ('F0', 'F2'):
        cases[name] = prepare_fixture(name, writer, deadline, budget)
    def rows(ids, bf16):
        channels = np.arange(C, dtype=np.int32)
        for row in ids:
            remaining(deadline)
            words = ((((17 * row + 3 * channels) % 257) - 128).astype('<f4') / 128).view('<u4')
            yield (words >> 16).astype('<u2').tobytes() if bf16 else words.tobytes()
    weight = writer.put(rows(range(V), False))
    for name, case in cases.items():
        case['weight'] = weight
        case['forward'] = writer.put(rows((token_id(name, t) for t in range(T)), True))
    writer.record('canonical.json', dict(cases=cases, oracle_additions=budget.additions,
                                       frozen_before_native=True))
    return cases


def inputs(case, ordinary):
    values = dict(tangent=(case['tangent'], 'torch.bfloat16', [T, C]),
                  ids=(case['ids'], 'torch.int32', [T]))
    if ordinary:
        values['weight'] = (case['weight'], 'torch.float32', [V, C])
    else:
        for name, ref, shape in zip(('counts', 'starts', 'positions', 'maximums'), case['metadata'],
                                    ([V, 1], [V, 1], [T, 1], [64, 1])):
            values[name] = (ref, 'torch.int32', shape)
    return values


def revalidate(writer, returned, ordinary):
    ids = np.fromfile(writer.check(returned['ids']), dtype='<i4').tolist()
    if not ordinary:
        metadata = [np.fromfile(writer.check(returned[name]), dtype='<i4').tolist()
                    for name in ('counts', 'starts', 'positions', 'maximums')]
        validate_metadata(ids, metadata)


def route(sock, writer, rank, peer_info, ordinal, case, f0, route_deadline, pins):
    fixture, kind = ROUTES[ordinal]
    ordinary = kind in ('eager', 'compiled')
    domains = inputs(case, ordinary)
    invocation = Invocation(rank, ordinal, secrets.token_hex(24), peer_info, domains, pins['worker.py'])
    receipt = dict(rank=rank, ordinal=ordinal, fixture=fixture, route=kind, status='FAILED', blobs={})
    try:
        sock.settimeout(remaining(route_deadline))
        # Worker sees canonical input blobs only, never an expected/output path.
        send(sock, dict(event='route', ordinal=ordinal, fixture=fixture, kind=kind,
                        nonce=invocation.nonce, inputs={k: {'blob':v[0], 'dtype':v[1], 'shape':v[2]}
                                                       for k,v in domains.items()}))
        for phase in ('input_before', 'input_after'):
            returned = {}
            if phase == 'input_after':
                message = receive(sock)
                invocation.call_start(message)
                send(sock, {'invoke': True})
                invocation.call_end(receive(sock))
                if ordinal in (1, 4, 5, 6):
                    require(invocation.ended - invocation.started <= 60, 'WARM_CALL_LIMIT')
                send(sock, {'capture': True})
                blob_from_worker(sock, writer, invocation, 'output', 'gradient', case['expected'],
                                 'torch.float32', [V, C], fp32=True)
                if ordinary:
                    blob_from_worker(sock, writer, invocation, 'forward', 'forward', case['forward'],
                                     'torch.bfloat16', [T, C])
            for name, (reference, dtype, shape) in domains.items():
                returned[name] = blob_from_worker(sock, writer, invocation, phase, name, reference, dtype, shape)
            revalidate(writer, returned, ordinary)
            receipt['blobs'][phase] = returned
        if ordinal in (1, 4):
            receipt['retained_f0'] = blob_from_worker(sock, writer, invocation, 'retained', 'f0',
                f0['expected'], 'torch.float32', [V, C], fp32=True)
        done = receive(sock)
        require(done.get('event') == 'done' and done.get('nonce') == invocation.nonce and
                done.get('tensor_peak_bytes', 2**60) <= 448*MIB, 'ROUTE_DONE_OR_TENSOR_CAP')
        receipt.update(invocation.complete(), status='EXACT_OBSERVED', done=done)
        send(sock, {'route_complete': True})
        return receipt
    except BaseException as exc:
        receipt['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        receipt['partial_actual_blob'] = invocation.output
        writer.record('rank%d-route%d.json' % (rank, ordinal), receipt)


def main():
    pins = source_pins()
    root, started = Path(sys.argv[1]), float(sys.argv[2])
    work_deadline = started + 1100
    writer = Writer(root / 'artifacts')
    terminal = dict(status='FAILED_OR_INCOMPLETE', routes=[], unknown_precision=UNKNOWN,
                    remaining_routes='NOT_EXECUTED', native_qualification=False)
    listener, child, clients = None, None, {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    try:
        phase_start = time.monotonic()
        cases = canonical(writer, min(work_deadline, phase_start + 180))
        generation_seconds = time.monotonic() - phase_start
        require(generation_seconds <= 180, 'GENERATION_LIMIT')
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        address = root / 'control.sock'
        listener.bind(str(address))
        os.chmod(address, 0o600)
        listener.listen(4)
        listener.settimeout(min(80, remaining(work_deadline)))
        child = subprocess.Popen(['/home/ubuntu/workspace/native_venv/bin/torchrun', '--standalone',
            '--nproc_per_node=4', str(HERE / 'worker.py'), str(root)],
            stdin=subprocess.DEVNULL, pass_fds=(int(os.environ['K2_LOCK_FD']),))
        launcher = identity(child.pid)
        seen_workers = set()
        for _ in range(4):
            sock, _ = listener.accept()
            sock.settimeout(min(80, remaining(work_deadline)))
            info, hello = peer(sock), receive(sock)
            rank = hello.get('rank')
            require(type(rank) is int and rank in range(4) and rank not in clients, 'RANK_IDENTITY')
            try:
                authenticated = authenticate_worker(info,hello,launcher,seen_workers,pins,root,
                    HERE/'worker.py',Path('/home/ubuntu/workspace/native_venv/bin/python'))
            except BaseException:
                sock.close()
                raise
            clients[rank] = sock, info
            writer.record('rank%d-environment.json' % rank, dict(hello=hello,authenticated=authenticated))
        require(len(seen_workers) == 4, 'FOUR_DISTINCT_WORKER_PROCESSES_REQUIRED')
        for sock,info in clients.values():
            current=identity(info['pid'])
            require(current['start']==info['start'] and current['state']!='Z', 'ADMITTED_WORKER_EXITED_OR_REUSED')
        first_used, later_used, first_receipts, compiled_receipts = 0.0, 0.0, {}, {}
        for ordinal, (fixture, kind) in enumerate(ROUTES):
            began = time.monotonic()
            first = ordinal in (0, 2, 3)
            allowance = 600 - first_used if first else min(60, 240 - later_used)
            deadline = min(work_deadline, began + allowance)
            remaining(deadline)
            futures = [executor.submit(route, sock, writer, rank, info, ordinal, cases[fixture], cases['F0'], deadline, pins)
                       for rank, (sock, info) in sorted(clients.items())]
            done, pending = concurrent.futures.wait(futures, timeout=remaining(deadline),
                                                    return_when=concurrent.futures.FIRST_EXCEPTION)
            for future in done:
                if future.exception() is not None:
                    raise future.exception()
            require(not pending, 'ROUTE_RENDEZVOUS_DEADLINE')
            results = [f.result() for f in futures]
            require(len(results) == 4 and all(r['status'] == 'EXACT_OBSERVED' for r in results), 'FOUR_RANK_ROUTE')
            for receipt in results:
                rank = receipt['rank']
                if ordinal == 0:
                    first_receipts[rank] = receipt
                else:
                    original = first_receipts[rank]
                    require(receipt['worker_call']['kernel_object_identity'] == original['worker_call']['kernel_object_identity'], 'KERNEL_CALLABLE_CHANGED')
                    require(receipt['worker_call']['retained_f0_object_identity'] == original['worker_call']['output_object_identity'], 'RETAINED_STORAGE_IDENTITY')
                    if ordinal == 1:
                        require(receipt['done']['input_object_identities'] == original['done']['input_object_identities'], 'REPEAT_INPUT_IDENTITY')
                if ordinal == 3:
                    compiled_receipts[rank] = receipt
                if ordinal == 6:
                    require(receipt['worker_call']['compiled_object_identity'] == compiled_receipts[rank]['worker_call']['compiled_object_identity'], 'COMPILED_CALLABLE_CHANGED')
            elapsed = time.monotonic() - began
            if first:
                first_used += elapsed
                require(first_used <= 600, 'COMBINED_FIRST_USE_LIMIT')
            else:
                later_used += elapsed
                require(later_used <= 240, 'LATER_ROUTE_LIMIT')
            terminal['routes'].append(dict(ordinal=ordinal, first_use=first, seconds=elapsed))
            require(time.monotonic()-started-generation_seconds-first_used-later_used <= 80, 'OTHER_WORK_RESERVE')
        for sock, info in clients.values():
            send(sock, {'event': 'finish'})
        require(child.wait(timeout=min(10, remaining(work_deadline))) == 0, 'WORKER_EXIT')
        # Independent retained-blob verification; all seven receipts per rank required.
        for rank in range(4):
            for ordinal in range(7):
                receipt = json.loads((root/'artifacts'/('rank%d-route%d.json' % (rank, ordinal))).read_text())
                require(receipt['status'] == 'EXACT_OBSERVED', 'MISSING_ROUTE')
                writer.check(receipt['actual_blob'])
        inventory = []
        for path in sorted(writer.root.iterdir()):
            remaining(work_deadline)
            reference = dict(sha256=path.name, bytes=path.stat().st_size)
            writer.check(reference)
            inventory.append(reference)
        writer.record('complete-blob-inventory.json', inventory)
        require(time.monotonic()-started-generation_seconds-first_used-later_used <= 80, 'FINALIZATION_WORK_RESERVE')
        terminal.update(status='PARTIAL_DIAGNOSTIC_F0_F2_EXACT_OBSERVED_PENDING_POSTFLIGHT',
                        remaining_routes=[], first_use_seconds=first_used, later_seconds=later_used)
        terminal['generation_seconds'] = generation_seconds
    except BaseException as exc:
        terminal['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        for sock, _ in clients.values():
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if listener:
            listener.close()
        executor.shutdown(wait=False, cancel_futures=True)
        writer.record('coordinator-terminal.json', terminal)
    return 0 if terminal['status'].endswith('PENDING_POSTFLIGHT') else 1


if __name__ == '__main__':
    raise SystemExit(main())
