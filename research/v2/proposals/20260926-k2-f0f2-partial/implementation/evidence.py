"""Single coordinator owns artifact writes and route state; no rank writes files."""
import json
import os
from pathlib import Path
import threading
import time
from artifacts import Blobs, compare_fp32
from common import GIB, MIB, require, exact_files, receive, receive_bytes, send
from monitor import tree_bytes


class Writer(Blobs):
    def __init__(self, root):
        super().__init__(Path(root) / 'blobs', ceiling=2 * GIB)
        self.artifact_root = Path(root)
        self.lock = threading.RLock()

    def reserve(self, size):
        require(type(size) is int and size >= 0, 'RESERVATION_SIZE')
        used, _ = tree_bytes(self.artifact_root)
        require(used + size <= 2 * GIB, 'ARTIFACT_RESERVATION_CAP')

    def put(self, chunks):
        with self.lock:
            def reserved():
                for block in chunks:
                    self.reserve(len(block))
                    yield block
            return super().put(reserved())

    def record(self, name, value):
        require(Path(name).name == name, 'RECORD_NAME')
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
        with self.lock:
            self.reserve(len(raw))
            with (self.artifact_root / name).open('xb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())

    def incoming(self, sock, expected_size):
        # Entire incoming size is reserved while holding sole writer lock.
        # Socket reader yields <=1 MiB; no full-size coordinator tensor is built.
        with self.lock:
            self.reserve(expected_size)
            send(sock, {'ready': True})
            def chunks():
                left = expected_size
                while left:
                    block = receive_bytes(sock, min(MIB, left))
                    left -= len(block)
                    yield block
            return self.put(chunks())


class Invocation:
    """Provenance is process/source/channel authenticated, not hardware tracing.

    Reference-blob IDs are never accepted as outputs: only streamed bytes after
    exact staged call messages on the authenticated channel. Device-object
    checks and direct call/capture are additionally fixed in the pinned worker.
    """
    def __init__(self, rank, ordinal, nonce, peer, input_names, source_sha):
        self.rank, self.ordinal, self.nonce = rank, ordinal, nonce
        self.peer, self.input_names, self.source_sha = peer, set(input_names), source_sha
        self.pre, self.post, self.output, self.started, self.ended = set(), set(), None, None, None

    def call_start(self, message):
        require(message == {'event': 'call_start', 'nonce': self.nonce, 'ordinal': self.ordinal}, 'FABRICATED_CALL_START')
        require(self.pre == self.input_names and self.started is None, 'CALL_WITHOUT_INPUTS')
        self.started = time.monotonic()

    def call_end(self, message):
        require(message.get('event') == 'call_end' and message.get('nonce') == self.nonce and
                message.get('ordinal') == self.ordinal and self.started is not None and self.ended is None,
                'FABRICATED_CALL_END')
        require(message.get('output_device') == 'neuron' and message.get('tangent_dtype') == 'torch.bfloat16' and
                message.get('output_dtype') == 'torch.float32' and message.get('output_shape') == [8192, 1024],
                'ACTUAL_OUTPUT_DOMAIN')
        require(message.get('elapsed', -1) >= 0, 'CALL_CLOCK')
        self.ended = time.monotonic()
        self.worker_call = message

    def complete(self):
        require(self.started is not None and self.ended is not None and self.output is not None and
                self.pre == self.post == self.input_names, 'MISSING_ROUTE_OR_ASSOCIATION')
        return dict(rank=self.rank, ordinal=self.ordinal, nonce=self.nonce,
                    authenticated_peer=self.peer, source_sha=self.source_sha,
                    parent_call_start=self.started, parent_call_end=self.ended,
                    worker_call=self.worker_call, actual_blob=self.output)


def blob_from_worker(sock, writer, invocation, role, name, reference, dtype, shape, *, fp32=False):
    header = receive(sock)
    require(header == dict(event='blob', role=role, name=name, nonce=invocation.nonce,
                           ordinal=invocation.ordinal, bytes=reference['bytes'], dtype=dtype,
                           shape=shape, device='neuron'), 'BLOB_ASSOCIATION_OR_DOMAIN')
    if role == 'output':
        require(invocation.ended is not None and invocation.output is None, 'OUTPUT_WITHOUT_INVOCATION')
    actual = writer.incoming(sock, reference['bytes'])
    writer.record('blob-r%d-i%d-%s-%s.json' % (invocation.rank, invocation.ordinal, role, name),
                  dict(header=header, actual=actual, expected=reference,
                       authenticated_peer=invocation.peer, source_sha=invocation.source_sha,
                       received_at=time.monotonic()))
    if role == 'output':
        invocation.output = actual
    if fp32:
        comparison = compare_fp32(writer.check(actual), writer.check(reference), reference['bytes'] // 4)
        send(sock, {'stored': actual, 'comparison': comparison})
        require(comparison['verdict'] == 'EXACT_FINITE', 'OUTPUT_NOT_EXACT_FINITE')
    else:
        exact_files(writer.check(actual), writer.check(reference))
        send(sock, {'stored': actual, 'comparison': 'FULL_BYTES_EXACT'})
    if role == 'input_before':
        invocation.pre.add(name)
    elif role == 'input_after':
        invocation.post.add(name)
    return actual
