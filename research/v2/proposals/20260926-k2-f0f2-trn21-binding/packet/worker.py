"""Fixed native seven-route worker. No expected gradient/reference imports."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
from common import *


class TensorLedger:
    """Harness-owned logical storage only; no claim about runtime intermediates."""
    def __init__(self):
        self.live, self.peak, self.events = {}, 0, []

    def reserve(self, name, size):
        require(name not in self.live and size >= 0, 'TENSOR_LEDGER_DUPLICATE')
        require(sum(self.live.values()) + size <= 448*MIB, 'TENSOR_STORAGE_CAP')
        self.live[name] = size
        self.peak = max(self.peak, sum(self.live.values()))
        self.events.append(dict(name=name, allocate_bytes=size, at=time.monotonic()))

    def drop(self, name):
        if name in self.live:
            self.events.append(dict(name=name, release_bytes=self.live.pop(name), at=time.monotonic()))


def capture(sock, command, role, name, tensor, ledger):
    # Reject a supplied CPU oracle/result even if its bytes are right. The only
    # accepted object here comes directly from the pinned native call below.
    require(tensor.device.type == 'neuron', 'NON_NATIVE_CAPTURE_OBJECT')
    size = tensor.numel() * tensor.element_size()
    header = dict(event='blob', role=role, name=name, nonce=command['nonce'],
                  ordinal=command['ordinal'], bytes=size, dtype=str(tensor.dtype),
                  shape=list(tensor.shape), device=tensor.device.type)
    send(sock, header)
    require(receive(sock).get('ready') is True, 'NO_WRITE_RESERVATION')
    ledger.reserve('roundtrip_cpu', size)
    cpu = tensor.detach().cpu().contiguous()
    raw = cpu.view(torch.uint8).reshape(-1).numpy()
    for first in range(0, size, MIB):
        sock.sendall(memoryview(raw[first:first+MIB]))
    del raw, cpu
    ledger.drop('roundtrip_cpu')
    answer = receive(sock)
    require('stored' in answer, 'NO_RETAINED_BLOB')
    comparison = answer.get('comparison')
    require(comparison == 'FULL_BYTES_EXACT' or
            isinstance(comparison, dict) and comparison.get('verdict') == 'EXACT_FINITE', 'HOST_COMPARISON_FAILED')


def load_tensor(root, item, name, ledger):
    ref = item['blob']
    key = ref['sha256']
    require(len(key) == 64 and all(c in '0123456789abcdef' for c in key), 'BLOB_PATH')
    path = root/'artifacts'/'blobs'/key
    require(not path.is_symlink() and path.stat().st_size == ref['bytes'] and digest(path) == key, 'CANONICAL_BLOB')
    dtype = {'torch.int32':torch.int32, 'torch.float32':torch.float32, 'torch.bfloat16':torch.bfloat16}[item['dtype']]
    ledger.reserve('load_cpu', ref['bytes'])
    cpu = torch.from_file(str(path), shared=False, size=math.prod(item['shape']), dtype=dtype).reshape(item['shape'])
    ledger.reserve(name, ref['bytes'])
    native = cpu.to(torch.device('neuron'))
    torch.neuron.synchronize()
    del cpu
    ledger.drop('load_cpu')
    return native


def main():
    global torch
    pins = source_pins()
    root = Path(sys.argv[1])
    rank = int(os.environ.get('RANK', '-1'))
    require(rank in range(4) and os.environ.get('LOCAL_RANK') == str(rank) and
            os.environ.get('WORLD_SIZE') == '4' and os.environ.get('NEURON_LOGICAL_NC_CONFIG') == '2', 'RANK_ENV')
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    import torch_neuronx
    from consumer_nki import build_kernel
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(1100)
    sock.connect(str(root/'control.sock'))
    own = identity(os.getpid())
    keys = ('NEURON_LOGICAL_NC_CONFIG', 'NEURON_CC_FLAGS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NKI_VENV_PATH', 'TRN_EXP_LABEL', 'TMPDIR', 'TORCH_NEURONX_ENABLED_METRIC_TABLES')
    send(sock, dict(rank=rank, pid=own['pid'], start=own['start'], world=4, pins=pins,
                    torch=torch.__version__, nki=importlib.metadata.version('nki'),
                    python=sys.version, package_versions={d.metadata['Name']:d.version
                        for d in importlib.metadata.distributions()
                        if any(key in d.metadata.get('Name','').lower() for key in ('neuron','nki','torch','numpy'))},
                    environment={key:os.environ.get(key) for key in keys}))
    # SDK initialization is donor-established. No device collectives are used
    # for route rendezvous; a blocked SDK init is stopped by the independent host.
    dist.init_process_group('neuron')
    kernel = build_kernel(enabled=True)
    def ordinary(weight, ids):
        return F.embedding(ids, weight).to(torch.bfloat16)
    compiled = torch.compile(ordinary, backend='neuron', fullgraph=True, dynamic=False)
    ledger, tensors = TensorLedger(), {}
    fixture, retained_f0, output, forward = None, None, None, None
    for ordinal in range(7):
        command = receive(sock)
        require(command.get('event') == 'route' and command.get('ordinal') == ordinal and
                (command['fixture'], command['kind']) == ROUTES[ordinal], 'ROUTE_ORDER')
        output, forward = None, None
        ledger.drop('output')
        ledger.drop('forward')
        if 'weight' in tensors:
            tensors['weight'].grad = None
        if command['fixture'] != fixture:
            for name in list(tensors):
                if name != 'weight':
                    del tensors[name]
                    ledger.drop(name)
            fixture = command['fixture']
        for name, item in command['inputs'].items():
            if name not in tensors:
                tensors[name] = load_tensor(root, item, name, ledger)
                if name == 'weight':
                    tensors[name].requires_grad_(True)
        # F0 repeat reuses exact objects, callable, and storage. Metadata is not
        # reconstructed between repeated calls. F2 replaces all nonweight input.
        torch.neuron.synchronize()
        for name in command['inputs']:
            capture(sock, command, 'input_before', name, tensors[name], ledger)
        send(sock, {'event':'call_start', 'nonce':command['nonce'], 'ordinal':ordinal})
        require(receive(sock).get('invoke') is True, 'NO_INVOCATION_CHALLENGE')
        started = time.monotonic()
        ledger.reserve('output' if ordinal else 'retained_f0', 32*MIB)
        if command['kind'] in ('direct', 'repeat'):
            output = kernel(tensors['tangent'], tensors['counts'], tensors['starts'],
                            tensors['positions'], tensors['maximums'])
        else:
            ledger.reserve('forward', 64*MIB)
            fn = ordinary if command['kind'] == 'eager' else compiled
            forward = fn(tensors['weight'], tensors['ids'])
            require(forward.dtype == torch.bfloat16, 'FORWARD_NOT_BF16')
            forward.backward(tensors['tangent'])
            output = tensors['weight'].grad
        torch.neuron.synchronize()
        ended = time.monotonic()
        require(output is not None and output.device.type == 'neuron', 'MISSING_ACTUAL_NATIVE_OUTPUT')
        if ordinal == 0:
            retained_f0 = output
        send(sock, dict(event='call_end', nonce=command['nonce'], ordinal=ordinal,
                        output_device=output.device.type, output_dtype=str(output.dtype),
                        output_shape=list(output.shape), tangent_dtype=str(tensors['tangent'].dtype),
                        started=started, ended=ended, elapsed=ended-started,
                        kernel_object_identity=id(kernel), compiled_object_identity=id(compiled),
                        output_object_identity=id(output), retained_f0_object_identity=id(retained_f0)))
        require(receive(sock).get('capture') is True, 'NO_CAPTURE_CHALLENGE')
        capture(sock, command, 'output', 'gradient', output, ledger)
        if forward is not None:
            capture(sock, command, 'forward', 'forward', forward, ledger)
        for name in command['inputs']:
            capture(sock, command, 'input_after', name, tensors[name], ledger)
        if ordinal in (1, 4):
            capture(sock, command, 'retained', 'f0', retained_f0, ledger)
        send(sock, dict(event='done', nonce=command['nonce'], tensor_peak_bytes=ledger.peak,
                        tensor_live=ledger.live, tensor_lifetime_events=ledger.events,
                        input_object_identities={name:id(tensors[name]) for name in command['inputs']}))
        require(receive(sock).get('route_complete') is True, 'NO_ROUTE_COMPLETION')
    require(receive(sock) == {'event':'finish'}, 'NO_SUITE_COMPLETION')
    dist.destroy_process_group()
    sock.close()


if __name__ == '__main__':
    main()
