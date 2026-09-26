# One ordinary-embedding control diagnostic — disabled

No scientific/K2 source changes or native action are made here. This proposal specifies a future standalone ordinary eager/compiled FP32 embedding probe, four ranks/LNC2 atV256/T256/C128, under the existing single guarded diagnostic machinery. Original K2 gate **FAILED** remains untouched.

The independent report (`research/v2/verification/20260926-k2-subnormal-independent/REPORT.md`, SHA7130de06a66b7ee984504827aa9163b2ef12b9079cece5859293b57db9062ad7) authenticates all31 raw30630c84 artifacts. Candidate routes were exact; the sole required mismatch per rank was the ordinary compiled gradient for FP32 subnormals. That distinction does not pass the failed full gate. The independently executed integer oracle (`analyze.py`, SHA82d93cb95e1fa94092854984f7f36e3e34cb8026dcd283b3a41f7545bf631532; raw-analysis.log SHA094e88a81a427e571bf9ad5857a560a11379fbf4f50afff32aa13361feb377f1) proves the expected1408 nonzero entries and exact hash. No tolerance change is justified.

## Exact minimal observation

Use ordinary `F.embedding(index, weight).to(torch.float32)` only, eagerly and with one reused `torch.compile(..., backend='neuron', fullgraph=True, dynamic=False)` callable. No K2 adapter, custom autograd, registration, kernel, reference imports or metadata routing.

Warm the callable with the original nonzero unique-ID FP32 sentinel. Then paired eager/compiled routes observe only:

1. Original fixture: `ids[t]=t%11`, tangent±2^-149, negative iff `(t+c)%3==0`.
2. Same ID/sign pattern with tangent±2^-126, constructed on CPU (normal witness).
3. Negated original subnormal tangent through the same callable (stale-capture witness).

Use fresh weight leaves and no accumulated gradient. One seeded finite CPU weight is cloned unchanged for every route; its hash is recorded. This preserves exact ID/tangent fixture bytes but intentionally does **not** claim the late original mixed-suite random weight or K2/BF16 history was replayed. Ordinary embedding's weight-gradient oracle is independent of weight values; forward must still exactly match this control's CPU gather. Not reproducing the failure would leave history/order interactions unresolved.

Compute expected gradients independently with signed integer coefficient accumulation and explicit IEEE754 encoding, not a device reduction or K2 oracle. Original active coefficients are7(683 entries),8(384),9(341);31360 untouched entries retain+0. Scale integer coefficients exactly for the normal witness and negate coefficients, not untouched zero bits, for the negative witness. Pin original IDs/upstream/expected-gradient hashes in the proposal. Read back inputs before forward, immediately before backward and after, and preserve actual forward/gradient bytes/hashes, counts, errors and missing/nonfinite records.

Keep strict bytes/finite/shape/dtype/zero tolerances. Complete all safe witnesses after mismatches and preserve each rank before exit/cleanup; aggregate exactly4 ranks. Exit0 only if every ordinary check is exact, otherwise nonzero. A reproduced control failure is useful diagnostic evidence but never candidate acceptance. No performance or BPB measurement.

## Preparation and release boundary

The completed JSON contains exact intended trn21 argv and unique prospective label; it is not runnable now because no leaf, frozen support manifest or host binding has been created. A distinct Implementer must prepare the disabled leaf and minimal binding; an independent Verifier must execute its own tiny integer/CPU exactness and negative controls, authenticate all pins and lifecycle, then Supervisor may release one host to the sole Operator. This is not a new generic launcher.

Prefertrn21 only after the current attached20M finishes and artifacts/quiescence are proved. Idletrn22 is an alternative only if CPU schedule qualification is not released, with an explicit separately reviewed exact rebind and unique label. No automatic alternate-host dispatch. Use the original host lock (trn21:66305/596823; trn22:66305/832589), freshps plusneuron-ls under lifetimeflock,600+60s, disk>=10.5GiB, private diagnostic caches, one thread, bounded outputs and complete descendant/compiler/device postflight. No stage/reservation/install/cleanup now.

Compiled-only subnormal loss without K2 would support an ordinary platform/control-route limitation, without proving its exact mechanism. An exact isolated control would instead motivate separate process/order diagnosis. Either outcome preserves the original K2 failure and all strict requirements; neither authorizes a candidate retry, validator waiver, native qualification, adoption or timing experiment.
