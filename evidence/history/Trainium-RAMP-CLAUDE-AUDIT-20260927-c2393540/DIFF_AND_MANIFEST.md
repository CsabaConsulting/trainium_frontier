# Exact diff and SHA-256 manifest

```diff
--- audited_parent.py
+++ train.py
@@ -155,10 +155,11 @@
     mlp_ratio: int = 4
     attn_scale: float = 0.0
     n_kv_heads: int = 0
+    mtp: bool = True
 COMPUTE_DTYPE: torch.dtype = torch.bfloat16
 
 def norm(x: torch.Tensor) -> torch.Tensor:
-    return F.rms_norm(x.float(), (x.size(-1),), eps=_RMS_EPS).to(torch.bfloat16)
+    return F.rms_norm(x, (x.size(-1),), eps=_RMS_EPS).to(torch.bfloat16)
 _NKI_PARTITION = 128
 _RMS_EPS = 1.1920928955078125e-07
 try:
@@ -592,7 +593,7 @@
     unreduced per-token loss; the mean/none reduction is applied here exactly as F.cross_entropy
     would apply it. Other inputs use the eager chain.
     """
-    if _HAS_NKI and logits.device.type == 'neuron' and (loss_reduction in ('mean', 'none')) and (logits.dtype == torch.float32) and logits.is_contiguous():
+    if _HAS_NKI and logits.device.type == 'neuron' and (loss_reduction in ('mean', 'none')) and (logits.dtype in (torch.float32, torch.bfloat16)) and logits.is_contiguous():
         z = logits.reshape(-1, logits.shape[-1])
         if z.shape[0] % _NKI_PARTITION == 0:
             tgt = targets.reshape(-1, 1).to(torch.float32)
@@ -602,13 +603,40 @@
             else:
                 out = per_token.mean()
             return out
-    capped = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
+    capped = LOGIT_SOFTCAP * torch.tanh(logits.float() / LOGIT_SOFTCAP)
     loss = F.cross_entropy(capped.reshape(-1, capped.size(-1)), targets.reshape(-1), reduction=loss_reduction)
     if loss_reduction == 'none':
         out = loss.view_as(targets)
     else:
         out = loss
     return out
+
+
+def mtp_auxiliary_weight(progress: float) -> float:
+    # Host-side schedule only; every model invocation still receives a[2] tensor.
+    return 0.5 * max(0.0, 1.0 - max(0.0, progress) / (2.0 / 3.0))
+
+def mtp_cross_entropy(logits, targets, weights, reduction):
+    """Parameter-free k1/k2 objective, inspired by modded-nanogpt PR178.
+
+    https://github.com/KellerJordan/modded-nanogpt/pull/178 (MIT).
+    Independently implemented. No extra head or eval change. Do not branch on
+    weights: both same-shape CE calls remain when k2 weight reaches zero.
+    Per-row last-position k2 has no valid target, and is masked, never wrapped.
+    """
+    assert targets.ndim == 2 and targets.shape[1] >= 2
+    assert weights.shape == (2,)
+    first = softcap_cross_entropy(logits, targets, 'none')
+    shifted = torch.cat((targets[:, 1:], torch.zeros_like(targets[:, :1])), dim=1)
+    second = softcap_cross_entropy(logits, shifted, 'none')
+    mask = torch.cat((torch.ones_like(second[:, :-1]), torch.zeros_like(second[:, :1])), dim=1)
+    second = second * mask * (targets.shape[1] / (targets.shape[1] - 1))
+    per_token = (weights[0] * first + weights[1] * second) / weights.sum()
+    if reduction == 'none':
+        return per_token
+    if reduction == 'mean':
+        return per_token.mean()
+    raise ValueError('MTP supports mean/none only')
 
 class Linear(nn.Linear):
     """Linear without bias that casts the weight to the input dtype. With USE_WEIGHT_CACHE the cast copy is
@@ -905,7 +933,7 @@
                 group['initial_betas'] = tuple(group['betas'])
         return opt
 
-    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None=None, loss_reduction: str='mean'):
+    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None=None, loss_reduction: str='mean', mtp_weights: torch.Tensor | None=None):
         _b, t = idx.shape
         assert t <= self.config.sequence_len
         cos = self.cos[:, :t]
@@ -922,9 +950,13 @@
             if i < self.unet_n_enc:
                 unet_skips.append(x)
         x = norm(x)
-        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
+        logits = self.lm_head(x)[..., :self.config.vocab_size]
         if targets is None:
-            return LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
+            return LOGIT_SOFTCAP * torch.tanh(logits.float() / LOGIT_SOFTCAP)
+        if self.config.mtp:
+            if mtp_weights is None:
+                mtp_weights = torch.tensor([1.0, 0.0], dtype=torch.float32, device=logits.device)
+            return mtp_cross_entropy(logits, targets, mtp_weights, loss_reduction)
         return softcap_cross_entropy(logits, targets, loss_reduction)
 
 def build_config() -> GPTConfig:
@@ -1327,11 +1359,29 @@
     m = (1 - cd) * m + cd * MUON_MOMENTUM_FINAL
     return m
 
+def _batch_ramp_accum(progress: float, previous: int=1) -> int:
+    """One-way 1->2 accumulation transition; never changes microbatch shapes."""
+    return max(previous, 1 if progress < (1.0 / 3.0) else 2)
+
+
+def _synchronized_batch_ramp_accum(progress: float, previous: int, device) -> int:
+    # Rank zero's charged-time sample is authoritative. No rank may independently
+    # change the number of forward/backward calls or gradient collectives.
+    proposed = _batch_ramp_accum(progress, previous) if dist.get_rank() == 0 else 0
+    phase = torch.tensor(proposed, dtype=torch.int32, device=device)
+    dist.broadcast(phase, src=0)
+    result = int(phase.item())
+    if result not in (1, 2):
+        raise RuntimeError('invalid synchronized batch-ramp phase')
+    return result
+
 def main() -> None:
     global PACK_FACTOR
     global loss_scale_clean, loss_scale_skips
     global N_EMBD, SEQ_LEN
     parser = argparse.ArgumentParser(description='Neuron Competition R1 baseline')
+    parser.add_argument('--batch-ramp', action='store_true', default=True)
+    parser.add_argument('--mtp', action=argparse.BooleanOptionalAction, default=True)
     parser.add_argument('--device-type', type=str, default='')
     parser.add_argument('--num-steps', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_NUM_STEPS', NUM_STEPS)))
     parser.add_argument('--max-train-seconds', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS', MAX_TRAIN_SECONDS)))
@@ -1390,6 +1440,7 @@
     global BOS_TOKEN_ID
     BOS_TOKEN_ID = tokenizer.get_bos_token_id()
     config = build_config()
+    config.mtp = args.mtp
     print0(f'model_config={json.dumps(asdict(config), indent=2)}')
     model = GPT(config)
     model.init_weights()
@@ -1406,6 +1457,9 @@
     if TOTAL_BATCH_SIZE % world_tokens_per_micro != 0:
         raise ValueError(f'TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} must divide micro tokens={world_tokens_per_micro}')
     grad_accum_steps = TOTAL_BATCH_SIZE // world_tokens_per_micro
+    if args.batch_ramp and (world_tokens_per_micro != 131072 or grad_accum_steps != 2 or args.max_train_seconds <= 0):
+        raise ValueError('--batch-ramp requires fixed micro131072 / total262144 and positive time budget')
+    ramp_accum = 1
     print0(f'micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps} pack_factor={PACK_FACTOR}')
     loader_seq_len = PACK_FACTOR * SEQ_LEN
     train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, 'train', device)
@@ -1435,15 +1489,27 @@
         grad_accum_now = grad_accum_steps
         synchronize(device)
         t0 = time.monotonic()
+        if args.batch_ramp:
+            ramp_progress = startup_adjusted_schedule_elapsed(step, elapsed, budget_started) / args.max_train_seconds
+            ramp_accum = _synchronized_batch_ramp_accum(ramp_progress, ramp_accum, device)
+            grad_accum_now = ramp_accum
+            # Fixed-shape runtime backward seed, not Python-scalar graph specialization.
+            ramp_backward_seed = torch.tensor(LOSS_SCALE / grad_accum_now, dtype=torch.float32, device=device)
         if step == 0:
             for m in cached_linears:
                 m.refresh_weight_cache(COMPUTE_DTYPE)
         optimizer.zero_grad(set_to_none=True)
         last_loss = None
+        if args.mtp:
+            mtp_progress = startup_adjusted_schedule_elapsed(step, elapsed, budget_started) / args.max_train_seconds
+            mtp_weights = torch.tensor([1.0, mtp_auxiliary_weight(mtp_progress)], dtype=torch.float32, device=device)
         for micro_index in range(grad_accum_now):
-            loss = model(x, y)
+            loss = model(x, y, mtp_weights=mtp_weights) if args.mtp else model(x, y)
             last_loss = loss.detach()
-            (loss / grad_accum_steps * LOSS_SCALE).backward()
+            if args.batch_ramp:
+                loss.backward(gradient=ramp_backward_seed)
+            else:
+                (loss / grad_accum_steps * LOSS_SCALE).backward()
             x, y, loader_state = fetch()
         clip_and_sync_gradients(model, clip_main_params, max_norm=GRAD_CLIP)
         schedule_elapsed = startup_adjusted_schedule_elapsed(step, elapsed, budget_started)
```

train.py: c23935404ac44314e333ddf3608a6c671f4526263a3348eb901bb3f8d4ca05b0
audited_parent.py: 54c835df60eb5d17ff49c6b79aa05aafd277027313fe6a07ed2955bdfaaebe3b
AUDIT_BRIEF.md: 3dfdc643f6a37d20aa16c9ee02b398df34499e1e572d459952594f8bc250944d
cold-evidence.json: b7fb0b909d83ac6a56cca49520a73fcf3b57b5c89a816e6b0b87d813c673a641
