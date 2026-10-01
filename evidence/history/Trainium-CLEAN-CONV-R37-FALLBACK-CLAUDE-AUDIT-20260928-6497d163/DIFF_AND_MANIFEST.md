# Exact diff and SHA256 manifest

This new, five-file Wednesday fallback folder contains exact cleanup source 6497 and evidence documents. Existing accepted folders are preserved. train.py is byte-identical to the frozen canonical proposal and the native Round37 archive member; no scientific source edit was made during packaging.

Candidate: `6497d163d930275dfbc0bb494fbb546456478e3751119dcc1baf918f5638f9d3` (84,679 bytes).
Accepted parent: `b585d0a647962009ae13068fd0f823e17fd19392b5d0dbf1087e6e5f40e48742` (84,085 bytes).
Canonical source: `research/v2/proposals/20260928-clean-conv-eval-path/train.py`.
Archived native source: `research/v2/operator/20260928-r37-saved-checkpoint-eval/artifacts/results-bundle.json#files.package/train.py`.

Payload and evidence file manifest (complete SHA256 values, final bytes):

| File | Bytes | SHA256 |
|---|---:|---|
| train.py | 84679 | `6497d163d930275dfbc0bb494fbb546456478e3751119dcc1baf918f5638f9d3` |
| R37_IDENTITY_EVIDENCE.md | 5109 | `c7431b1e4abd527cfade2c7e86ed1362fd57b9bf61ddcd577c4f1108a6e32ab5` |
| cold-evidence.json | 63040 | `2d55a45688ab9ecf8b1fb94b04c2f1f10bed99878a67ce81daa79b280bef8b48` |
| AUDIT_BRIEF.md | 7369 | `37aa42d7866427ddd8919b3a5591cc8d1a7f5752a6725ac68c92f8468444f2d5` |

This manifest is the fifth file. Its own SHA256 is supplied in the final handoff rather than embedded into itself, avoiding an impossible self-referential file digest. The table hashes every other package file. There is no hidden sixth file or runtime dependency.

Exact command, unchanged from parent and Round37 (77 bytes including newline):

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py
```

Archived command-file SHA256 `61a3348a6dc5795ebd8f1a0da90262b19a9a33ba4cb941c09f7c7aa5828fc7db`. It is carried as text in the brief/evidence, not an added package file.

Complete unified diff against the exact accepted b585 file follows. The parent is not copied into this folder. Removal of the self.training test changes supported BF16 evaluation routing to the pre-existing shared MLP path. With self.training=True, routes and arithmetic are unchanged. Added docstrings can affect compiler source/cache keys; these edits do not prove candidate cold startup.

```diff
--- accepted-b585/train.py
+++ fallback-6497/train.py
@@ -466,6 +466,7 @@
     from torch_neuronx import wrap_nki as _wrap_conv2_nki
 
     def _conv2_taps(wt, channels):
+        """Broadcast the previous/current [2, C] taps to [128, C] SBUF tiles."""
         prev = nl.ndarray((128, channels), dtype=wt.dtype, buffer=nl.sbuf)
         curr = nl.ndarray((128, channels), dtype=wt.dtype, buffer=nl.sbuf)
         pattern = [[0, 128], [1, channels]]
@@ -474,6 +475,7 @@
         return (prev, curr)
 
     def _conv2_fwd_tile(x, out, prev_w, curr_w, batch, start, channels, first):
+        """Write one causal 128-token tile, zero-padding only the first previous row."""
         prev = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
         curr = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
         nisa.dma_copy(dst=curr, src=x[batch, start:start + 128, :])
@@ -492,6 +494,7 @@
 
     @nki.jit
     def _conv2_forward(x, wt, lanes: int=1):
+        """Return causal two-tap [B, T, C] outputs with disjoint batch ownership per lane."""
         batch, length, channels = x.shape
         assert length % 128 == 0 and channels % 128 == 0
         assert wt.shape == (2, channels)
@@ -508,6 +511,7 @@
         return out
 
     def _conv2_dw_partial(xpart, grad, ones, partials, batch, tile, tap, channels):
+        """Store unchanged FP32 128-token dW reductions in one contiguous channel row."""
         prod = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
         nisa.tensor_tensor(dst=prod, data1=xpart, data2=grad, op=nl.multiply)
         packed = nl.ndarray((1, channels), dtype=nl.float32, buffer=nl.sbuf)
@@ -524,6 +528,7 @@
         nisa.dma_copy(dst=partials.ap(pattern=[[channels, 1], [1, channels]], offset=offset), src=packed)
 
     def _conv2_bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones, batch, tile, length, channels, first, last):
+        """Write one dX tile and both FP32 dW partials, respecting sequence boundaries."""
         start = tile * 128
         xp = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
         xc = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
@@ -553,6 +558,7 @@
 
     @nki.jit
     def _conv2_backward(x, grad, wt, lanes: int=1):
+        """Return dX and FP32 [B, T/128, 2, C, 1] partials for the later sum/casts."""
         batch, length, channels = x.shape
         assert length % 128 == 0 and channels % 128 == 0
         assert grad.shape == x.shape and wt.shape == (2, channels)
@@ -775,7 +781,8 @@
     def forward(self, x: torch.Tensor) -> torch.Tensor:
         tokens = x.numel() // x.shape[-1]
         tile = min(512, tokens // 2)
-        if not self.training or x.dtype != torch.bfloat16 or (tile < 128) or tile % 128 or tokens % (2 * tile):
+        # Keep the eager fallback for non-BF16 inputs or unsupported token tiling.
+        if x.dtype != torch.bfloat16 or (tile < 128) or tile % 128 or tokens % (2 * tile):
             h = self.c_fc(x)
             return self.c_proj(F.relu(h).square())
         up = _linear_weight(self.c_fc, x.dtype)
```

Origin-document manifest:

| Source document | Bytes | SHA256 |
|---|---:|---|
| research/v2/verification/20260928-r37-eval-path-independent/REVIEW.md | 6415 | `0597b9c0595136123c9b30db5ddbf2f5b64f6c8cfdd26b2765d1d2c9c40f3ea2` |
| research/v2/verification/20260928-r37-eval-path-independent/RECEIPT.json | 1462 | `aa4f1705104352d86a9f9bef31d67ef32d806291a164b58369424b9af3d21414` |
| research/v2/verification/20260928-r37-eval-path-independent/NATIVE_IDENTITY_REVIEW.md | 4063 | `ac08da5446806555cd13c05cd51e0ddc77398692762d7e907cc946b8650893a3` |
| research/v2/verification/20260928-r37-eval-path-independent/NATIVE_IDENTITY_VERDICT.json | 1718 | `9fc25ab1aec025a9f711a58a64434200345990ae9ad85ab746e7558ec2be7d86` |
| research/v2/operator/20260928-r37-saved-checkpoint-eval/artifacts/packet.released.json | 200672 | `1e809c3e1cc07d0ce78440522cacf2a6a81dc7814e1788b17a5401958a6ebf7d` |
| workflow/ROUND48_2026-09-28.md | 1816 | `5e1c043aeb5068e5631776e1ac2a73f327ddf0aebd9685e5cedebb2a4c2af833` |
| workflow/REJECTION_CHECKLIST_2026-09-28_0315.md | 1715 | `71717878b1c070bb7bce459579ae31f8b26b6339db9c7661859db1f19b9b9612` |

R37 results archive SHA256: `8f9e6f172ba31f7f8e001573fede97d2ee0843970c8a3faf0f7d69392618bd31`.
Parent source audit SHA256: `b37601233f9ce326273f7ee34c226ff3e8cb25f02123deac7c3bcae520d13d22`.
Parent cold-evidence.json SHA256: `29045b688ae00d12429a2c3ed602495eaad2a8d5635528d6fe8843759c20e6fe`.
All 33 R37 member hashes and all eight parent archive/source/log/checkpoint/cache identities are retained in cold-evidence.json. Its protected-existing-package manifest records the 11 files in the two prior b585 folders; their bytes were checked unchanged after copying this package.

Scope: source/document assembly and local SHA256/JSON/syntax checks only. No model execution, native training/evaluation, host access, data/checkpoint copy, cleanup, cloud action, audit invocation or upload. This manifest does not assert Claude acceptance of 6497, a newly measured 6497 cold run, new two-run quality qualification or submission authorization. The original b585 acceptance applies only to its exact bytes. See AUDIT_BRIEF.md for the eight-item checklist and remaining evidence gaps.
