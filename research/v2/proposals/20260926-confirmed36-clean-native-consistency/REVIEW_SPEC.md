# Clean36 native ordinary-loader consistency diagnostic

Completed Researcher specification, based on canonical5ace9b8c. No source,
checkpoint, host or main-branch mutation was performed to prepare it. The
companion proposal is complete as an implementation specification and remains
disabled. It is not a runnable packet or a source/numeric acceptance receipt.

## Decision and exact state

Use the accepted retained full113,256,724-parameter,65-tensor CPU G2 pair on
trn21, each through its own unpatched loader. Reference is
7419927c7417b5666a984ec9c2b849feadabaaa9, research36 SHA
36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb.
Candidate is7b9c20c696237fbb2abef57135bf777e6a801365, clean SHA
e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43.

Frozen INPUTS SHA0cc0569936b0b1653bcc946edcdedd1bf91d3f671fabbc205a0a7b883bf9a15e
binds17 actual inputs. Reference checkpoint is
`/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/reference/tmp/g2-gsoipfis/out/final.pt`,
453048594B, SHA623577bb7e6f60d183a1944b28a5624cd55767c16fcf03a9248fd4573f6a6762.
Clean checkpoint is
`/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/candidate/tmp/g2-2xlf5hc1/out/final.pt`,
453044626B, SHA3c7efed1e053cf7607e2099d52643cb368e8a1f14aebe2a36ba3f1444b63c9c9.
Different checkpoint-file hashes reflect their different envelopes. Acceptance
requires matching complete saved parameter bytes, keys, dtypes and shapes,
not matching file hashes or truncated G2 digests. The accepted188-check CPU
result independently compared all65 full parameter records and accepted the
reviewed helper's loaded/saved full-byte checks; it is not a native result.

Before native inference, reauthenticate all17 files and independently compare
the two payloads' complete65 model tensors. Both actual loaders must reproduce
their saved tensors exactly on device, checked after synchronization and CPU
copy. Preserve each file's actual saved config/toggles, strict loading,
weights_only behavior and8192 eval table floor. Do not set globals, substitute
compute_dtype_for, patch forward, attach hooks, refresh caches by hand, normalize
parameters or manufacture a checkpoint. Check the documented eight-layer,
1024-width,8192-vocab architecture, attention/convolution layers0,2,4,6,7,
four scalar skips and VE only7 from actual restored models/configuration.

The native clean cold-five checkpoint is separately real and retained:
`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-cold-trn22-r2-packet/train/out/final.pt`,
453044626B, SHA8e867c7a541f799e96502001e91ea06388e46fe0c2159a7c07c684d947b9dbde.
Its uncapped fifth boundary is859.1679093550192s. The completed bundle
c03997f87f4bb95f539b0e4ac10bfeff012e1ac0dc12b3ead27b3dcc25aa9c87
contains stable checkpoint_after_terminal identity. The path is mapped by its
root plus the actual remote_snapshot.py line53 `train/out/final.pt` expression.
The separately named retained-input-identities.json contains only source,
prepare and harness; it must not be cited as a checkpoint path receipt. Any
future use of this checkpoint requires a fresh explicit path/hash/size binding.

It cannot be the unchanged research36 parity input. Clean save_checkpoint
stores model_state/model_config/meta only. Research restore_eval_toggles takes
missing settings from defaults: USE_UNET_SKIP=False, USE_PARTIAL_ATTN_8=False,
USE_QK_SHIFT=True, USE_NKI_LOCAL_CONV=False, USE_BF16_NORM_OUTPUT=False and
COMPILE_SDPA_DIRECT=False. This is an incompatible reconstruction, not merely
different metadata. Do not silently add toggles or patch the loader to make it
work. Nor compare its five native updates against the three CPU G2 updates.
The paired CPU checkpoints avoid both errors. Giving the research CPU file to
both loaders is a possible separately proved simplification, but adds a new
cross-load without reducing native shape compilation; it is not this packet.

## Smallest bounded implementation

Create one proposal-specific helper and tiny tests, plus a disabled binding.
Use the accepted65e7eb1a21b0e880ee2362461ff3664c4dbd52933816e5ec0684645c92cd75e8
helper as the numerical/state/input foundation. Reuse only the necessary native
initialization/synchronization/receipt patterns from old native probe
d21d35c68440e6a2b62abf5338603f0c8a4891a454bb61d3df0fdc420d7682a1.
Retain an exact diff against these inputs. Do not transplant the older probe's
old source pins, T1024/B1/B2 matrix, end-only state checks or unsafe launcher
lifetime assumptions. No general diagnostic framework, new optimizer, training
runner or evaluation adapter belongs in this change.

Native process initialization follows the installed native evaluation pattern:
import torch_neuronx, require WORLD_SIZE4 and LNC2, initialize Neuron process
group, use torch.device('neuron'), require the synchronization API. Use four
workers on the full chip; models are reference then clean serially in each
worker. Teardown and release references before constructing the next model.
Destroy an initialized process group on completion/error; teardown failure is
UNRESOLVED. Model forward must contain no new collective, wrapper, instrumentation
hook or rank-dependent arithmetic. Compare rank receipts outside forward.

Use only synthetic integer tokens1..8191. A private CPU generator is reset to
seed42 for each shape; every rank gets identical inputs. Training seed remains42.
This deliberately records that old CPU helper fixtures used seed7; no native
comparison is claimed to reproduce seed7 outputs. Save exact small token lists
and their complete byte hashes. No tokenizer, dataloader, public evaluation,
scoring function or data read is invoked.

Keep the accepted six shapes and call sequence:

| Shapes, in order | Calls per shape | Checks |
| --- | --- | --- |
| B1/B2/B3 x T256, then B1/B2/B3 x T257 | original, repeat, every independent singleton row, reversed suffix | finite expected logits; repeat and concatenated-row exactness; prefix causality; state/input/mode purity around every call; original cross-source byte equality |

This is30 top-level calls per source per rank,60 total; no backward or optimizer.
Do not add another permutation/duplicate matrix or extra context sweep merely
because the old diagnostic had one. Existing rowwise comparison already measures
the specified supplied-batch route against independently evaluated identical rows.
All original, repeat, row and suffix call records must also agree across sources
and ranks by complete output hashes; original cross-source output equality is
recomputed from full tensors, as in65e7. Keep the exact complete case list in the
validator so missing, duplicate or partial cases cannot be PASS.

T256 is required: width1024 and T divisible by128 activate clean native two-tap
convolution. Its source selects two lanes for even B2 and one for odd B1/B3.
The research wrapper's internal B is always1. T257 is required here because
the unchanged convolution and RoPE predicates fall back when T%128 !=0; the
same route discrepancy must not be declared qualified only on one branch.
This adds no model/weight change. Static reviewer tests must independently
check the exact source predicates and lane choice; actual receipts must record
_HAS_NKI, native device, compute dtype and restored flags. If required native
dispatch cannot be established, describe coverage as unresolved. Do not claim
that top-level B shape alone proves every compiled kernel that executed.

## Numerical and purity contract

All repeats, rowwise outputs and original cross-source outputs require equal
dtype, shape and complete raw bytes, including signed zeros. These criteria
come from65e7 and the prior native diagnostic. No new tolerance is justified by
the clean REVIEW_SPEC or current evidence. Any legitimate native rounding
difference is a REFUSE of this exactness claim, with magnitude and location
reported; it cannot be turned into PASS by retrospectively choosing a BF16
tolerance. Preserve max absolute difference and relative RMS with an explicit
zero-denominator treatment, actual/expected hashes, and useful mismatch location.

Only the existing synthetic causality criterion uses a tolerance: take
cut=max(2,T//2), reverse each row's suffix[cut:], require every row changes,
and compare FP32 logits[:,:cut-1] at maximum absolute error<=1e-2. Also report
exact prefix byte equality. This is the unchanged prepare.py criterion as
reimplemented in the reviewed diagnostic, not a call to the organizer's data
loader. It never relaxes repeat, row or cross-source equality. No nonfinite
output or metric may accidentally pass through NaN comparison semantics.

Before and after every call authenticate all model parameter and persistent/
nonpersistent buffer bytes, requires_grad flags, all module eval modes and hook
dictionaries. Include direct tensor attributes, such as Linear._cached_weight,
when present, with no broad claim about arbitrary nested Python state. Check
both the actual device input tensor and its original CPU fixture, not only the
untouched host source from which it was copied. Synchronize before observation.
Copy outputs to CPU and clone immediately before another forward to defeat
scratch-buffer aliasing, preserving source dtype. Tests must reject aliased
output corruption. These observations are finite-scope evidence, not a universal
proof of statelessness for every token/context/rank/world configuration.

Only research's mask_full/mask_short get the existing independent full causal/
window formula validation and legitimate negative-infinity exemption. Clean
must be type(model) is its module.GPT with cos/sin buffers only; they and all
other state must be finite. Research remains its unmodified _SubmissionEvalGPT.
Keep the mask floor8192 and restored short window4096; never shrink it for cost.
The helper's native adaptation must copy state/masks to CPU before finiteness,
mask formula, byte hashing and comparison arithmetic. Avoid creating new native
isfinite/reduction/comparison graphs for each differently sized parameter.
Validating each complete mask once and rechecking its full raw bytes on every
call is sufficient; no sampled mask/state checks may replace full coverage.

## Cost limits and interpretation

Use one native invocation with a900-second child deadline covering import,
init, checkpoint loading, both models, compilation, all checks and teardown;
15-second TERM-to-KILL grace is termination time, not accepted extra work.
These limits reuse the older native diagnostic; they are safety caps, not
measured sufficiency. The observed859.17s clean training startup does not
predict this inference run's cost. Timeout preserves partial receipts and is
UNRESOLVED. No automatic retry, timeout increase or graph-pruning after results.

Preflight requires64GiB available host RAM and10.5GiB free disk plus3GiB
projected new footprint, at staging and lifetime-lock acquisition. The latter
inherits existing conservative packet reserves, not a measured cache quota.
Use only new private six-cache paths under the exact new root and record all
effective compiler/cache settings. NEURON_CC_FLAGS must include optlevel1;
freeze the same native eval optlevel1 setting for both arms. Do not claim warm
hits or cold performance. Monitor actual free space/cache size; crossing the
10.5GiB residual floor ends the owned job through the accepted guardian. No
checkpoint or cache deletion is authorized.

Retained six reference original outputs total100859904B/rank if FP32 (about
96.19MiB); copies, current outputs, state snapshots, masks, checkpoints and
compiler memory add substantial unmeasured overhead. CPU helper threads1 per
worker. Full113M allocation is host-only, never the workstation. Only tiny
reviewer fixtures execute locally. Research has two base-GPT input shapes,
clean six: eight source/shape signatures per worker. This is not an upper bound
on compiled graphs. Retrieve actual per-rank graph_stats and elapsed/RSS/disk
observations; compilation/cache cost and partial completion must be honest.

Emit zero full tensor files, at most16MiB of scalar/hash/token receipts per
rank and256MiB total emitted logs/metrics/receipts/packet artifacts. Private
SDK caches are outside that artifact limit and are reported separately.
Existing cap polling may overshoot; any breach fails acceptance. Do not invent
a quota guarantee. The guardian retains the lock until every owned descendant
is gone even if the diagnostic has already exceeded its time or disk cap.

## Required fresh review and launch hold

A separate Implementer produces the tiny change; this Researcher cannot verify
or operate it. A genuinely fresh Astra session receives only exact diff, this
specification and factual gate receipts, without inherited Implementer context,
notes or branch. It independently executes its own tiny CPU reference and
adversarial tests, normal and optimized Python, rather than only replaying
supplied tests. Archive input hashes, session/model, test source/output and
literal scope/verdict. Default REFUSE until complete.

Tests must catch mismatched/changed/symlink source/checkpoint inputs; absent or
wrong research toggles; wrong65-tensor/count/config; silent dtype overrides;
wrong masks/NaN/+inf; signed-zero and one-byte output differences; state/input/
mode mutation; output aliasing; degenerate suffix fixtures and future-token
leaks; missing/duplicate rank/case and hash disagreement; missing synchronization,
native init failure and cleanup failure. No new full training G2 is needed:
science is unchanged and the accepted full pair is an explicit prerequisite.
Full113M/native execution during local reviewer tests is prohibited.

Separately, the current lifetime-guard REFUSE at
research/v2/verification/20260926-clean-fullcold-mechanics-fresh/REPORT.md
blocks this packet. Older native mechanical acceptance does not waive it.
Only an independently fault-tested fixed-job guardian can release this job.
It must pin trn21 original device66305/inode596823 at acquisition, hold the
same /home/ubuntu/.cache/trainium_freshstart/device0.lock through all owned
descendant/compiler quiescence, and survive timeout/signal/child crash safely.
Fresh ps AND neuron-ls, fuser, STOP/breaker, exact host, clean source/pins and
resource checks occur inside the retained lock. No lock replacement, bypass,
automatic successor or interruption of another job is allowed. Guardian repair
is a separate workstream; this helper must not become a new launcher framework.

Only the sole Operator may receive a separate exact release after source/CPU
and guardian/mechanics acceptance. Preserve live trn21 training and attached
eval until natural completion, retrieval and explicit all-process quiescence.
Before dispatch freeze helper/binding/guardian/environment hashes and actual
argv. After every outcome retrieve source/input identities, stdout/stderr,
per-rank complete or partial receipts, every actual PID/starttime/PPID/PGID/SID
identity, timeout/signals/exit, graph stats and fresh process/device/lock
postflight. A parent exit or lock competitor alone cannot prove quiescence.

Independent result acceptance may close only native synthetic consistency on
this CPU-trained fullwidth state and the six tested shapes. It does not validate
the8e867 native-trained checkpoint, T1024 public evaluation, long-context/general
causality, arbitrary world sizes, native training-trajectory equivalence, a
quality score or submission readiness. The clean cold full1800, fresh unchanged
public20M/context1024 and Claude compliance audit via Poojith remain separate;
human upload stays human-only. If any exactness criterion fails, preserve it
and park the claim. This proposal authorizes no tolerance revision or adapter.
