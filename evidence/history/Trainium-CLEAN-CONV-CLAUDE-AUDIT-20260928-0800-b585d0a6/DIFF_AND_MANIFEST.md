# Exact CLEAN→CLEAN+CONV diff and manifest

External Claude PENDING; evidence-only, NOT SUBMISSION READY; human upload only.

- `train.py` SHA256 `b585d0a647962009ae13068fd0f823e17fd19392b5d0dbf1087e6e5f40e48742`, 84085 bytes.
- `audited_parent.py` SHA256 `881a4f491d256fb89af33246a3b568fc3ffead9616070eea1860d7d8f7f9f749`, 83562 bytes.
- `AUDIT_BRIEF.md` SHA256 `107aacb3cd4f7f8a36918d075ec57774c0df23440ac9167ad644ee7e38177394`, 12269 bytes.
- `cold-evidence.json` SHA256 `29045b688ae00d12429a2c3ed602495eaad2a8d5635528d6fe8843759c20e6fe`, 36232 bytes.

Manifest excludes its own hash; the external preview hashes all five files. Complete diff:

```diff
--- audited_parent.py (CLEAN881a)
+++ train.py (CLEAN+CONV b585)
@@ -510,12 +510,18 @@
     def _conv2_dw_partial(xpart, grad, ones, partials, batch, tile, tap, channels):
         prod = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
         nisa.tensor_tensor(dst=prod, data1=xpart, data2=grad, op=nl.multiply)
+        packed = nl.ndarray((1, channels), dtype=nl.float32, buffer=nl.sbuf)
         for c in nl.affine_range(channels // 128):
             summed = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
             result = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
             nisa.nc_matmul(dst=summed, stationary=prod[:, c * 128:(c + 1) * 128], moving=ones, accumulate=False)
             nisa.tensor_copy(dst=result, src=summed)
-            nisa.dma_copy(dst=partials[batch, tile, tap, c * 128:(c + 1) * 128, :], src=result)
+            # Repack the unchanged FP32 reduction; no new arithmetic or HBM scratch.
+            transposed = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
+            nisa.nc_transpose(dst=transposed, data=result, engine=nisa.engine.tensor)
+            nisa.tensor_copy(dst=packed[:, c * 128:(c + 1) * 128], src=transposed, engine=nisa.engine.vector)
+        offset = ((batch * partials.shape[1] + tile) * 2 + tap) * channels
+        nisa.dma_copy(dst=partials.ap(pattern=[[channels, 1], [1, channels]], offset=offset), src=packed)
 
     def _conv2_bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones, batch, tile, length, channels, first, last):
         start = tile * 128
```
