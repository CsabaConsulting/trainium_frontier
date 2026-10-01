# Exact source and evidence identities

No Claude ACCEPT or submission-ready claim. Manifest hashes every preceding file and excludes itself; preview hashes all files.

- `ACTUAL_CPU_RECEIPT.json`: SHA256 `d9f86d88422cb6752610f6d96e52d8fee17bca125661e7cd6843d8f95e8a3e8e`, 3713 bytes.
- `ALL3_RESULT.exact.json`: SHA256 `e1e5fdf91c73b55d262b487d013745a0d7a7605c1824cdd5bb0baa690a5c6b1e`, 5685 bytes.
- `ALL3_RESULT.json`: SHA256 `3ffd364f83f4a6c2c61b97d70d05f849c229274c3ce043994cbf9bba49b8af9b`, 4060 bytes.
- `ALL3_verifier.py`: SHA256 `35f75a833b909d78e7d3e2010939544eefdf4ca78db6ad712f552e82110f4ebe`, 7178 bytes.
- `AUDIT_BRIEF.md`: SHA256 `cc35373fe13150ed14c18251c8a41aa297448fe78cdffe664e13d4f080deeafe`, 7435 bytes.
- `INDEPENDENT_REFERENCE.json`: SHA256 `49fc04ada06a47a5a1bfcd2543dd6b7b68f46bf1e32e11968462c67257b9cdde`, 492 bytes.
- `INDEPENDENT_REVIEW_RECEIPT.json`: SHA256 `77d510912a16cfcfe6315446ddd5b9c81192f4ba754dd6b85df47041a8e0e5bc`, 9153 bytes.
- `MANIFEST.json`: SHA256 `62a783456e1f09125b2b5ac8324be32602ca705084335a10589ea4aaef8da852`, 4647 bytes.
- `PACKAGE_PLAN.original.md`: SHA256 `1dcbd374f20a4a12573ef4749ee613a7f821946b85d715d405deb5765d0f6a58`, 4073 bytes.
- `Q2_AUTHORITY.original.md`: SHA256 `566747fa861eb2d9369bfdfd302e29148190f46897f796de412fb23ba7f68f8b`, 14922 bytes.
- `Q2_EFFECTIVE_AUTHORITY.json`: SHA256 `0bb5e0c0c05b70379d0cb87bb5ffd66a14a4b22cbec8189433d815dcb4c58bc1`, 977 bytes.
- `R73_AUTHORITY.md`: SHA256 `737580737727d3082231496c557b627a61481afcb56bea7e8f039d1879c44d69`, 1854 bytes.
- `README.md`: SHA256 `990c8b7e505ca14e619006a0768ac803ee92f671e3ab5b044f8376adcd750b12`, 1774 bytes.
- `SCORED_EXPERIMENT_PROPOSAL.json`: SHA256 `a690a2fa03699456b557a0748f330add7eac0be20fe018a922a4391fbc7470ce`, 6401 bytes.
- `SCORED_MANIFEST.json`: SHA256 `5437a99568758d3826ceab8ac8b2ac06026080c9bc0adf9a93d9776c0905aed0`, 1352 bytes.
- `SCORED_SPEC.md`: SHA256 `90dd5e996131f697d9239b3a1c1d9fa30a09490f6bd894eacd95ee0a4ec57100`, 7111 bytes.
- `SCORED_VS_D2.diff`: SHA256 `f3b9dc6b3ed04e9a3ce22ebb29456316d901e1a6f6b62984af3e4b62b55d63c6`, 4682 bytes.
- `SOURCE.diff`: SHA256 `8fe489eef21ad9d3660106c4fb9ad42a7f8d317d51fc8d5cacf00c66859eecda`, 2283 bytes.
- `STRUCTURAL_RECEIPT.json`: SHA256 `813bb4bf193738f86c98286a147ba3f445fffa06462e541ebc72dce12f2e5b7a`, 1544 bytes.
- `STRUCTURAL_checker.py`: SHA256 `fde48fedf0c95228b282c482a811f8bd719c37ec4aff377cd29bcdc942193deb`, 5521 bytes.
- `UPLOAD_EDITS.json`: SHA256 `25bdc93782c11dc217b91c43aa2f559ede4f4c8312bfc0e1a1200e61044de620`, 486 bytes.
- `cold-evidence.json`: SHA256 `e94ed345abbe6eb56a34f32e117bd3b05373dd6a370faa776b9bd9516114056b`, 6653 bytes.
- `cpu_actual_complete_r2.exact.json`: SHA256 `f08910e29d64f952843eb344b17ebf32bff768a138392d127d89526451c8f86c`, 20177 bytes.
- `cpu_controlled_diagnostic.exact.json`: SHA256 `2a9cc4fd26f18a8e5bd5fabee68dc2a36d333257e0e966f0edf7bf769690a575`, 88918 bytes.
- `cpu_independent_reference.exact.json`: SHA256 `31d43f6d0302d4f4c1f7f124fc0b98259ccd07088c62408b96f779d83d784403`, 20944 bytes.
- `cpu_preserved_original_author_failure.exact.json`: SHA256 `527eae4d27fc8b7d556b2c1aeae1b0b2ebfb9428a5bf81365a2feb3c2de10ad6`, 11571 bytes.
- `launch-command.txt`: SHA256 `61a3348a6dc5795ebd8f1a0da90262b19a9a33ba4cb941c09f7c7aa5828fc7db`, 77 bytes.
- `receipt_first.exact.json`: SHA256 `471d1f1638540e5b83e7a672e5180544338b6296d849b25c8052a9d9f891705a`, 1433368 bytes.
- `receipt_repeat21.exact.json`: SHA256 `ff9e3a5abcc054967531951cfd28fe46ac908d926617cf9c97d13109000b16a3`, 1445339 bytes.
- `receipt_repeat22.exact.json`: SHA256 `53597a6dad0b83aee3831a0c97c1b0a8678c21f9aeb92d2dca67395c8f4b4f34`, 1438920 bytes.
- `scored_train_c5ef8414.py`: SHA256 `c5ef8414e2df998a9b52eeb2e249b66c0110a3c37869e676719e93975a3432b4`, 100757 bytes.
- `train.py`: SHA256 `29e5cd2cfd35bb52e5a955057a1661786239d0812d9a797c85dc0df7fe2d7446`, 100795 bytes.

## Upload derivative: exact three replacements

```diff
--- immutable-c5ef/train.py
+++ r73-a3-upload/train.py
@@ -8,8 +8,8 @@
 
 Native NKI kernels implement rotary RMS normalization, causal convolution and
 softcap cross-entropy, including their backward passes. The MLP's custom
-autograd uses PyTorch matrix operations. Parameters and optimizer state are
-FP32; normalized branch inputs and the residual stream are BF16 on Neuron,
+autograd uses PyTorch matrix operations. Base parameters and optimizer state are
+FP32; the table and its RMS state are BF16; normalized branch inputs and the residual stream are BF16 on Neuron,
 with explicit FP32 normalization/loss/optimizer statistics as implemented below.
 Muon updates matrix weights; AdamW updates embeddings, scalar gates, the 2-D
 value gate and convolutions. Four LNC2 ranks average then clip gradients and
@@ -17,8 +17,8 @@
 32 rows per rank, a 131072-to-262144 token batch ramp, the existing wall-clock
 schedules and the parameter-free next-two-token auxiliary objective.
 
-The main entrypoint trains from seed 42 and saves out/final.pt (the output
-directory is the only CLI option). It never reads validation data. The scorer
+The main entrypoint trains from seed 42 and saves out/final.pt (CLI options:
+--out-dir and --tbl_freeze_tokens). It never reads validation data. The scorer
 calls load_for_eval to load the same causal model with strict saved weights.
 Compiler defaults below preserve the verified gate-fix compilation policy;
 they do not select a different model or inspect scoring inputs.
@@ -1574,7 +1574,7 @@
 def main() -> None:
     parser = argparse.ArgumentParser(description='CLEAN-GATEFIX fixed Phase 1 training recipe')
     parser.add_argument('--out-dir', type=str, default=os.environ.get('NEURON_COMPETITION_R1_OUT_DIR', 'out'), help='Checkpoint output directory.')
-    parser.add_argument('--tbl_freeze_tokens', type=int, default=286000000, help='Freeze D2 table after the completed global-token boundary; 0 continues updates.')
+    parser.add_argument('--tbl_freeze_tokens', type=int, default=0, help='Freeze D2 table after the completed global-token boundary; 0 continues updates.')
     args = parser.parse_args()
     device_type = detect_device_type()
     ddp, rank, _local_rank, world_size, device = init_runtime(device_type, seed=42)
```

## Scored A3 versus D2

```diff
--- D2-CONTINUE-b758/train.py
+++ R69A3-NOPROJ-ID/train.py
@@ -858,6 +858,8 @@
             raise ValueError('D2 boolean and nonnegative integer token-freeze configuration required')
         if config.d2_two_site and (not config.d1_detached or config.n_embd < 64 or config.n_kv_heads not in (0, config.n_head)):
             raise ValueError('D2 requires enabled D1, width>=64 and full query/value heads')
+        if config.d2_two_site and config.n_embd != 128 * config.n_head:
+            raise ValueError('D2 NOPROJ-ID requires 128 channels per head')
         self.__dict__['_d1_compiled_core'] = None
         padded_vocab_size = (config.vocab_size + 63) // 64 * 64
         self.transformer = nn.ModuleDict({'wte': nn.Embedding(padded_vocab_size, config.n_embd), 'h': nn.ModuleList([Block(config, i) for i in range(config.n_layer)])})
@@ -878,8 +880,6 @@
             self.value_bigram_gate_logit = nn.Parameter(torch.zeros((), dtype=torch.float32))
             with torch.random.fork_rng(devices=[]):
                 self.value_bigram_proj = Linear(128, config.n_embd, bias=False)
-                if config.d2_two_site:
-                    self.value_bigram_proj7 = Linear(128, config.n_embd, bias=False)
 
     @torch.no_grad()
     def init_weights(self) -> None:
@@ -913,7 +913,6 @@
             with torch.random.fork_rng(devices=[]):
                 torch.nn.init.uniform_(self.value_bigram_proj.weight, -s, s)
                 if self.config.d2_two_site:
-                    torch.nn.init.uniform_(self.value_bigram_proj7.weight, -s, s)
                     self.transformer.h[7].attn.bigram_gate.weight.zero_()
             self.value_bigram_gate_logit.zero_()
 
@@ -951,8 +950,6 @@
             matrix_params.extend(block.mlp.parameters())
         if self.config.d1_detached:
             matrix_params.append(self.value_bigram_proj.weight)
-            if self.config.d2_two_site:
-                matrix_params.append(self.value_bigram_proj7.weight)
         gate_params = [p for name, p in self.named_parameters() if p.ndim == 2 and 'gate' in name]
         if self.config.d1_detached:
             gate_params.append(self.value_bigram_gate_logit)
@@ -1025,7 +1022,7 @@
             rows0 = rows[..., :128] if self.config.d2_two_site else rows
             projected = self.value_bigram_proj(rows0)
             if self.config.d2_two_site:
-                bigram_ve7 = self.value_bigram_proj7(rows[..., 128:256])
+                bigram_ve7 = rows[..., 128:256].repeat(1, 1, self.config.n_head)
             gate = (2 * torch.sigmoid(self.value_bigram_gate_logit)).to(x.dtype)
             x = x + gate * projected
         x = norm(x)
@@ -1116,6 +1113,19 @@
     update = (gradient / denom).to(torch.bfloat16)
     updated = (table.float() - lr * update.float()).to(torch.bfloat16)
     return updated, second, torch.zeros_like(gradient, dtype=torch.float32)
+
+@torch.no_grad()
+def d2_table_rank_max_abs_diff(table: torch.Tensor) -> float:
+    """End-run diagnostic on a clone; never changes a trained table or master."""
+    if not (dist.is_available() and dist.is_initialized()):
+        return 0.0
+    reference = table.detach().clone()
+    dist.broadcast(reference, src=0)
+    difference = table.detach().cpu().float()
+    difference.sub_(reference.cpu().float()).abs_()
+    maximum = torch.tensor(float(difference.max().item()), dtype=torch.float32, device=table.device)
+    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
+    return float(maximum.item())
 
 class DetachedTableUpdate:
     """Training-only replicated state; never a model Parameter/module/checkpoint key.
@@ -1719,6 +1729,13 @@
         if table_update.pending_tokens:
             raise RuntimeError('unfinished every2 group; refuse dropping gradients or post-loop/post-budget update')
         print0(f'D1_TABLE_COMPLETE updates={table_update.update_count} pending_tokens=0; all updates inside recorded steps')
+        if table_update.d2_two_site:
+            rank_check_started = time.monotonic()
+            table_max_abs_diff = d2_table_rank_max_abs_diff(table_update.table)
+            rank_check_seconds = time.monotonic() - rank_check_started
+            print0(f'D2_TABLE_RANK_CONSISTENCY world_size={world_size} max_abs_diff_vs_rank0={table_max_abs_diff:.9g} method=cloned_rank0_broadcast_host_fp32_max_scalar_allreduce diagnostic_seconds={rank_check_seconds:.6f}')
+            if not math.isfinite(table_max_abs_diff):
+                raise RuntimeError('nonfinite end-run table rank-consistency diagnostic')
     checkpoint_path = Path(args.out_dir) / 'final.pt'
     budget_elapsed_before_save = time.monotonic() - budget_started
     if rank == 0:
```
