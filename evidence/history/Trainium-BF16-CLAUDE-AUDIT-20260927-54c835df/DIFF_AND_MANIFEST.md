# Exact diff and SHA-256 manifest

```diff
--- accepted_cd0c4aa5.py
+++ train.py
@@ -770,7 +770,7 @@
         self.local_conv = CausalConv(config.n_embd, LOCAL_CONV_KERNEL) if self.attn is not None else None
 
     def forward(self, x: torch.Tensor, x0: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
-        x = self.resid_lambda * x + self.x0_lambda * x0
+        x = self.resid_lambda.to(x.dtype) * x + self.x0_lambda.to(x.dtype) * x0
         if self.attn is None:
             return x + self.mlp(norm(x))
         q_in = k_in = v_in = x
```

train.py: 54c835df60eb5d17ff49c6b79aa05aafd277027313fe6a07ed2955bdfaaebe3b
accepted_parent.py: cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d
AUDIT_BRIEF.md: 8eba40779fd3485a2bc07677c28ead106af7638084e1b4bb243b852d14b23262
cold-evidence.json: cd0809e3fda0bac4c98c1e9fccffe7c7ae4e16a40316c77de659a49df8cf2661
