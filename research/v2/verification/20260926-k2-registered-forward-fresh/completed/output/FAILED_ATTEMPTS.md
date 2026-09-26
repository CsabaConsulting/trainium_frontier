The complete raw log of independent attempt 1 is `own-ordered-attempt1.log`.
The exact original source is retained as `review_cpu_attempt1.py`.
It exited 1. The source hash was
`888239dc80d861bdbab57dd9e412d2c02e10d64015cc88e9e6ae1eafe3c65998`.

Exact error:

```text
IndexError: index 8 is out of bounds for dimension 0 with size 8
```

This occurred in the reviewer's CPU consumer substitute when evaluating the
logical-empty case. There were 30 passing FP32 route records before it.
The substitute padded an empty gradient to eight rows but allowed the padding
sentinel index eight. The correction is nine padding rows; no candidate source
was edited. The first source is exactly reconstructible from `review_cpu.py`
by reversing the following two-line patch:

```diff
-        padded = torch.cat((gradient.to(torch.float32), gradient.new_zeros((8, 4)).float()), dim=0)
+        padded = torch.cat((gradient.to(torch.float32), gradient.new_zeros((9, 4)).float()), dim=0)
-         maximum_tensor_shape=[16, 4], tiny_tensor_allocation_bound='under 64 MiB, no model',
+         maximum_tensor_shape=[17, 4], tiny_tensor_allocation_bound='under 64 MiB, no model',
```

The developer's separate earlier failed stale-Tensor attempt is preserved as
authenticated packet input `../raw/cpu-stale-attempt1.txt`. Its exact error was
`AssertionError: Please convert all Tensors to FakeTensors first or instantiate
FakeTensorMode with 'allow_non_fake_inputs'. Found in
aten.clone.default(tensor([...], size=(8, 4)))`. That was not executed by this
reviewer and is not used as passing evidence.
