"""Stdlib model of two feature owners; never imports a native backend."""
from reference import C, P, T, V, fp32, validate_metadata, require
import math


def symmetric(ids, gradients, meta):
    validate_metadata(ids, meta)
    require(len(ids) == len(gradients), 'input length')
    require(all(len(g) == C and all(math.isfinite(x) and fp32(x) == x for x in g)
                for g in gradients), 'finite FP32 shape/input required')
    counts, starts, positions, maximums = meta
    output = [[0.0] * C for _ in range(V)]
    owners = bytearray(V * C)
    traces = []
    for program in range(2):
        feature_first = program * 64
        trace = dict(program=program, group_rounds=[], schedule=[], visits=[],
                     stores=0, active_gradient_elements=0)
        for group in range(2):
            first = group * P
            maximum = maximums[group]
            trace['group_rounds'].append(maximum)
            accumulator = [[0.0] * 64 for _ in range(P)]
            pointer = starts[first:first + P].copy()
            remaining = counts[first:first + P].copy()
            for iteration in range(maximum):
                trace['schedule'].append((group, iteration))
                for lane in range(P):
                    active = remaining[lane] > 0
                    address = pointer[lane] if active else T
                    token = positions[address] if address < T else T
                    token = token if active else T
                    if active:
                        require(token < len(ids) and ids[token] == first + lane, 'invalid gather')
                        trace['visits'].append((first + lane, token))
                        for local_feature in range(64):
                            # Explicit full-base access-pattern index, no grouped reduction.
                            feature = feature_first + local_feature
                            flat_index = 128 * token + feature_first + local_feature
                            require(divmod(flat_index, 128) == (token, feature), 'AP offset')
                            accumulator[lane][local_feature] = fp32(
                                accumulator[lane][local_feature] + gradients[token][feature])
                            trace['active_gradient_elements'] += 1
                    pointer[lane] += 1
                    remaining[lane] -= 1
            for lane in range(P):
                for local_feature in range(64):
                    feature = feature_first + local_feature
                    index = (first + lane) * C + feature
                    owners[index] += 1
                    output[first + lane][feature] = accumulator[lane][local_feature]
                    trace['stores'] += 1
        traces.append(trace)
    require(all(owner == 1 for owner in owners), 'missing or duplicate writer')
    require(traces[0]['schedule'] == traces[1]['schedule'], 'asymmetric control')
    require(traces[0]['visits'] == traces[1]['visits'], 'asymmetric token routing')
    return output, traces, owners
