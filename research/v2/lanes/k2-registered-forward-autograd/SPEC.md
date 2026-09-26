# Registered forward / registered derivative: neutral component specification

Selected proposal: `20260926-k2-registered-forward-autograd`, canonical
`a6ebb4c2a8e165b4e11bce684f4b9dc544992bd4`. This is a standalone diagnostic,
disabled by default, not a production training change or native launch binding.

`adapter.py` accepts the prebuilt, unchanged
`consumer_nki.build_kernel(enabled=True)` result and one original output dtype.
The native caller must construct that exact protected callable; the adapter
does not rewrap it. The selected registration factory defaults to installed
`torch_neuronx.nki_op`; the explicit dependency argument is solely for the
local archived-AST test. Instantiate once per dtype in a fresh process.

There is one six-Tensor forward operator for each of FP32 and BF16, named
`trainium_k2_registered_v1::embedding_fp32` and `::embedding_bf16`. The body is
exactly `F.embedding(ids, weight).to(output_dtype)`, with `mutates_args=()`.
Registered setup saves only counts, starts, positions and maximums. Registered
backward calls the supplied kernel with the contiguous upstream and these
four saved tensors; it returns FP32 dW followed by five None values.
No explicit NKI wrap, second consumer operator, tensor cache, kernel arithmetic
change, shape change or altered dtype/rounding contract is permitted.

Protected kernel/reference hashes are respectively
`2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae` and
`14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f`.
The kernel remains `resident_consumer[2]` with native V256/T256/C128. Original
Function checker hash is
`bd2bca7c29615df599094d3c618f5cdf0e4f4381e6d19213dc022691214ce788`.
Its authenticated native zero-gradient failure is not reversed by CPU results.

CPU contract: one thread, V8/T8/C4, FP32 weight and dW; FP32 then BF16 output
and upstream; strict finite shape/dtype/byte equality, including signed zero.
The consumer substitute uses supplied runtime metadata and one FP32 addition
per token occurrence. It is a test double, not an NKI simulator or native proof.
Compare registered eager/AOT, original Function eager/AOT and ordinary eager
embedding to an independent scalar FP32 token-ordered recurrence. Reuse each
compiled registration with changed tangents and changed valid metadata; retain
all seven input hashes before/after every route. Cover unique first, duplicate
rows/unused rows, order-sensitive cancellation, empty metadata and signed zero.

Separately supplied wrong zero/stale consumers must change actual registered
eager and AOT dW and be refused; merely injecting wrong values after execution
is insufficient. Validator negatives must reject NaN, both infinities,
missing outputs, wrong dtype and signed-zero differences. Local API private
registrations must show the real installed decorator's fake and functional
decomposition handlers plus registered setup/backward. The extracted SDK AST
must be unchanged and hash-pinned. No native module may be imported by these
tests, and the disabled path must not import Torch either.

No native correctness, exact automatic bridge capture, higher-order gradients,
full-model routing, speed or quality is established by this component. A fresh
independent source review and separate disabled native packet are still needed;
sentinel success alone cannot replace all 20 original cases on all four ranks.
