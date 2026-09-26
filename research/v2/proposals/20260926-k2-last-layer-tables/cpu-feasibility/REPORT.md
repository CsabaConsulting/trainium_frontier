# BF16 table-backward numerical falsifier

Result: **sequential FP32 accumulation followed by one BF16 cast is not equal
to the tested ordinary BF16 advanced-indexing backward.** Do not reuse the
current FP32 K2 recurrence unchanged for the proposed BF16 tables.

Observed using PyTorch2.13.0+cpu, one CPU thread, BF16 table[5,4], tiny duplicate/
interleaved row IDs. Largest tensor has48 elements. No native, model candidate,
training data, SDK, host or performance operation was performed. This is a
design falsifier, not a candidate gate or universal backend specification.

Every target-row occurrence was separated by row3, whose own duplicate gradient
was also accumulated; rows0,2,4 were unused and remained zero. Four channels
scale the listed witness by [1,-1,2,.5]. All four mismatch coordinates and raw
BF16 hexadecimal outputs are preserved in result.json.

First target channel, exact representable values:

| Microstep grouping | Actual indexing and F.embedding | Per-call BF16 additions, then BF16 grad accumulation | Per-call FP32 sum/cast, then BF16 grad accumulation | One global FP32 sum/cast |
|---|---:|---:|---:|---:|
| [1, 2^-8, -1] | 0 | 0 | 0.00390625 | 0.00390625 |
| [1, -1, 2^-8] | 0.00390625 | 0.00390625 | 0.00390625 | 0.00390625 |
| [1, 2^-8] ; [-1] | 0 | 0 | 0 | 0.00390625 |
| [1] ; [2^-8, -1] | 0.00390625 | 0.00390625 | 0.00390625 | 0.00390625 |
| [1, 2^-8, -1] ; [1, 2^-8, -1] | 0 | 0 | 0.0078125 | 0.0078125 |

Here `;` separates actual backward calls without resetting `.grad`. At1.0,
1+2^-8 is halfway between BF16 representable neighbors and rounds to1.0; the
subsequent -1 therefore produces0. The FP32 recurrence retains2^-8.

Microstep boundaries matter independently. `[1] ; [2^-8,-1]` gives2^-8 after
separately computing the second BF16 table-gradient and accumulating it into
the first. Concatenating these tokens into a single BF16 recurrence instead
gives0. `[1,2^-8] ; [-1]` independently falsifies maintaining one FP32 accumulator
across microsteps. The two operators matched each other in every tested call;
forward gather equivalence alone did not establish either reduction contract.

## Recommendation for a later authorized table specialization

Use a separate BF16-table backward contract. For these CPU fixtures it must
start fresh each call, visit each row's original token order, round after every
addition to BF16, return BF16 dW, and leave ordinary parameter-gradient
accumulation across microsteps intact. Do not alter WTE/VE's FP32 recurrence,
promote table parameters/state, substitute F.embedding, concatenate microsteps,
or cast only at the final store to claim equality.

This evidence specifies a necessary CPU behavior, not a qualified NKI algorithm.
An eventual implementation must test its actual rounding and native ordinary
reference under the existing single patch review/timing workflow. No P2 model,
kernel or optimizer source was implemented here; native support and performance
remain unknown. K2 native timing remains the separate P2 prerequisite.

Reproduce with the existing local venv:

```
/home/poojith-devan/venv/bin/python -B probe.py
```

`probe.py` imports only Torch and standard-library utilities. `result.json`
contains per-call actual gradients, both ordered references, accumulated
gradients, all comparison results, exact hexadecimal BF16 representations,
version/thread details and probe SHA256. No additional reviewer requested.
