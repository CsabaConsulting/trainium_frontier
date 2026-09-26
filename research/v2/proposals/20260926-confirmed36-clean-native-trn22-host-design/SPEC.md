# Clean36 native consistency: trn22 host-only exception design

Completed Researcher proposal, based on canonical `f276169559faa870a00b0b0e122088ae0654517b`
plus later Operator evidence `a4a041e3fb11130aef604d09c89c7656041b594f`.
**HOLD: the byte-unchanged route is empirically blocked. The exact minimum
mechanical exception below is proposed for Supervisor selection, not approved
or implemented. No launch, transfer or host reservation is authorized.**

## Concrete conclusion

The05:43:58.247428UTC Operator observation confirms that
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/prepare.py`
is **absent** on trn22. Original helper937e2/foundation65e7/manifest0cc05699
cannot run unchanged there: the foundation fixes PREPARE, manifest validation
requires that absolute key, and the helper checks both foundation/manifest hashes
and imported prepare.__file__. Merely changing PYTHONPATH fails. Do not create,
copy, edit, symlink, hard-link or mount a protected prepare replacement, and do
not keep treating this unchanged route as a future-ready job.

The smallest useful exception is **one manifest path key, one foundation
PREPARE constant, and three native helper constants (HELPER_SHA, INPUTS_SHA,
LABEL)**. Use existing trn22 project prepare.py at
`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/prepare.py`,
unchanged SHA5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb,
35,467B. This prior real trn22 binding is not a substitute for a fresh
regular-file/no-symlink/hash check after its current evaluation completes.
All16 non-protected input paths remain their original absolute names; both
scientific source and checkpoint bytes remain unchanged. No new G2, checkpoint,
loader, shape, seed, tolerance, numerical behavior or data access is proposed.

At this latest observation free space was17,294,938,112B. Protected training
had completed3628 steps (859.473523823s uncapped startup;
1789.768582730s charged), but the attached fresh20M evaluation was alive from
05:38:55.397878UTC and the original lock66305:832589 remained held.
The full replay/eval/retrieval/descendant-and-device-quiescence sequence must
finish before any separate future host selection. K2 ready work and clean
submission retain priority. No preemption, host slot, queue entry or cleanup.

## Five exact payload edits for separate selection

These are proposed edits to **future packet-local copies**, not to canonical
accepted sources/receipts or protected prepare. They are not implemented here.

| Future file | Exact permitted change | Everything else |
| --- | --- | --- |
| `INPUTS.json` | Replace its one `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/prepare.py` key with `/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/prepare.py` | Every other byte remains original, including all17 payload SHA/size pins, root, arms and16 other paths |
| `proposals/20260926-confirmed36-clean-checkpoint-cpu/compare_checkpoints.py` | PREPARE path literal changes from old path to new path above | ROOT, PREPARE_SHA, SOURCES, REQUESTS, all functions/control flow/checks unchanged |
| `proposals/20260926-confirmed36-clean-native-consistency/probe.py` | HELPER_SHA becomes actual full SHA of the one-constant adapted foundation | Exact new foundation SHA is a future implementation/review pin, not old65e7 |
| same native helper | INPUTS_SHA becomes `707a455c45e31e49621e7c9f8e3d6e0e4d57acf837137657e21089211652e8d9` | No manifest bypass or skipped authentication |
| same native helper | LABEL becomes `fs-20260926-confirmed36-clean-native-consistency-trn22-host-r1` | No other helper behavior changes; no historical receipt relabeling |

The original INPUTS bytes are4295B, SHA
`0cc0569936b0b1653bcc946edcdedd1bf91d3f671fabbc205a0a7b883bf9a15e`. The exact one-key UTF-8 text substitution,
preserving all other bytes and the existing trailing newline, is4293B, projected
SHA `707a455c45e31e49621e7c9f8e3d6e0e4d57acf837137657e21089211652e8d9`. That projection was computed as a read-only
text stream; no new INPUTS, helper or foundation file was created. The future
Implementer and independent Verifier must reproduce the exact digest.
Original foundation SHA is
`65e7eb1a21b0e880ee2362461ff3664c4dbd52933816e5ec0684645c92cd75e8`; original native helper SHA is
`937e2f239ad90b5aa61e0b40ccf067b19ac02febe246492735e948a35c6b86af`. **Both helper file hashes must change.**
Their new hashes remain null until implementation and independent review.
Do not claim source-byte-identical helpers or automatically transfer prior
helper acceptance to the adapted copies.

Additional mechanical binding work is limited to host/hostname, original lock
identity, unique packet/receipt/metric/cache roots, label, PYTHONPATH and the
resulting support-manifest hashes. Preserve helper/foundation sibling layout.
The selected future host driver, disabled binding, packet manifest and exact
transfer/retrieval transport need separately frozen hashes and fresh review.
No general launcher, wrapper, import shim, namespace workaround or model adapter.

Fresh reviewer must independently show that the proposed payload diff is
exactly these five edits; authenticate all scientific bytes; repeat existing
tiny positive/negative tests normal and -O; add old-vs-new prepare origin,
wrong manifest/foundation pin, missing/symlink/path collision and unchanged
fixture/hash failures. Existing numerical/state/mask/receipt negatives remain.
New reviewer acceptance applies only to the changed packet; prior CPU result
remains provenance. No local full113M allocation or native call is allowed.

## Immutable scientific contract and17 existing inputs

The complete controlling science remains
`research/v2/proposals/20260926-confirmed36-clean-native-consistency/REVIEW_SPEC.md`,
SHA1b5a311e04904872dbdf614037a2def9ef45a5362c173d555c058e66eb7f599d, except
the explicit prospective host/path/pin/label changes above, which still require
selection. Reference source commit7419927c7417b5666a984ec9c2b849feadabaaa9 is
SHA36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb; clean
commit7b9c20c696237fbb2abef57135bf777e6a801365 is
SHAe1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43.
Keep full113,256,724 parameters/65 tensors and both distinct checkpoint envelopes.
Do not regenerate, add metadata, cross-load a different envelope or substitute
the native trn22 checkpoint.

Original manifest source is
`research/v2/proposals/20260926-confirmed36-clean-checkpoint-cpu/INPUTS.json`.
These are historically retained source files on trn21, not an assertion of
destination presence. All16 non-protected entries require fresh stable source
and destination identities. Entry13 is never transferred: its future manifest
destination is the existing trn22 prepare path specified above, same SHA/size.

| # | Original source path | Bytes | SHA256 | Destination treatment |
| --- | --- | ---: | --- | --- |
| 1 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/reference/tmp/g2-gsoipfis/out/final.pt` | 453048594 | `623577bb7e6f60d183a1944b28a5624cd55767c16fcf03a9248fd4573f6a6762` | Same absolute destination path; byte-identical transfer |
| 2 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/reference/request.json` | 2716 | `38160400738af45274aad1c96f9a54e0c5ec1343ab9eb87f1279b46e4f5ba354` | Same absolute destination path; byte-identical transfer |
| 3 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/reference/g2.json` | 17875 | `2b2c647ee88a9ddcad57fdacb6f667a89897ea4f1b399ff5069d9329a50c633c` | Same absolute destination path; byte-identical transfer |
| 4 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/reference/terminal.json` | 331 | `aa4c769ae1ab0dae90bca6b6ff0ff557a49042a611ffab989fa20b3dcb589133` | Same absolute destination path; byte-identical transfer |
| 5 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/reference/checkpoint-identity.json` | 320 | `f57cb5b2157349c190eff74b32d9faed8c5dd66133eef8cc7202a2a4c91976af` | Same absolute destination path; byte-identical transfer |
| 6 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/reference/train.py` | 324457 | `36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb` | Same absolute destination path; byte-identical transfer |
| 7 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/candidate/tmp/g2-2xlf5hc1/out/final.pt` | 453044626 | `3c7efed1e053cf7607e2099d52643cb368e8a1f14aebe2a36ba3f1444b63c9c9` | Same absolute destination path; byte-identical transfer |
| 8 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/candidate/request.json` | 2142 | `64d0b756784eb8a8306d1fcd739c09c355d7611479f8da9f64bf2fb7ffc4eca3` | Same absolute destination path; byte-identical transfer |
| 9 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/candidate/g2.json` | 15221 | `3b838553241458abaed25cad6e79554fb4a36b93c2ceca743f176742804d6518` | Same absolute destination path; byte-identical transfer |
| 10 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/candidate/terminal.json` | 331 | `01f4b2c0168e2dd33e30bffd908ee7d61aaf47034c314bbdd8be68361addcefa` | Same absolute destination path; byte-identical transfer |
| 11 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/candidate/checkpoint-identity.json` | 320 | `17174bf781279b1dc0cc6fff2778d391c052f1960ed1bdc430c180d1b447253e` | Same absolute destination path; byte-identical transfer |
| 12 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/candidate/train.py` | 87986 | `e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43` | Same absolute destination path; byte-identical transfer |
| 13 | `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/prepare.py` | 35467 | `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` | Do not transfer; map to existing trn22 pin below |
| 14 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/preflight_dryrun.py` | 23550 | `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8` | Same absolute destination path; byte-identical transfer |
| 15 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/standard-compare.json` | 537 | `c490f2ac3fe51361da4185f5ef1d3226984142404b4d7c204ab4b519563a6a02` | Same absolute destination path; byte-identical transfer |
| 16 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/standard-compare-exit.json` | 66 | `08facec4a1f1beaed069b6d7cf3b673dbf3b1f7434df426d8e8928d780704119` | Same absolute destination path; byte-identical transfer |
| 17 | `/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1/terminal.json` | 183 | `ce3881db6227aeed08353d94bf6e13f8092ca23f3dd6509cd38e82f4bbd8e442` | Same absolute destination path; byte-identical transfer |

The pair totals906,093,220B; all17 payloads906,604,722B;
16 transferable payloads906,569,255B. Original INPUTS is retained as provenance;
the prospective one-key manifest is a distinct binding, never mislabeled0cc05699.
All request/G2/terminal/compare/source/checkpoint bytes stay exact, including
their historical trn21 provenance. Only the protected-file address changes.

## Resource and transfer contract

| Quantity | Bytes | Meaning |
| --- | ---: | --- |
| Retained dual CPU checkpoints | 906093220 | Both complete files retained on both hosts |
|16 transferred input files | 906569255 | Pair plus14 small source/evidence files |
| Transfer + static packet allowance | 1073741824 |1GiB, including filesystem/metadata/packet overhead |
| Headroom after16 input payloads | 167172569 | Allocation/metadata/static packet, not native budget |
| Unchanged retained-free floor | 11274289152 |10.5GiB |
| Entire original native projected growth | 3221225472 |3GiB, not consumed by checkpoint staging |
| Required after transfer and under launch lock | 14495514624 |13.5GiB free |
| Required before any transfer | 15569256448 |14.5GiB free |
| Latest05:43 trn22 observed free | 17294938112 |Protected evaluation active; final free unknown |
| Latest snapshot margin over14.5GiB | 1725681664 |About1.607GiB, no reservation or guarantee |

Earlier05:37 free17,760,641,024B and2,191,384,576B margin are superseded by
the smaller05:43 observation, not usable as current capacity. Exact payload
lower-bound arithmetic13.5GiB +906,569,255B =15,402,083,879B excludes overhead
and is **not** the resource contract. Preserve the rounded14.5GiB before-transfer
requirement. Do not search for lower reserves. Other CPU proposals'12.5GiB
staging floor does not authorize this native screen. Original trn21 free
13,067,235,328B is already1,428,279,296B below native13.5GiB; human cleanup
is unresolved and unauthorized.

Future post-eval preflight must measure >=14.5GiB on the actual destination
filesystem and MemAvailable>=64GiB before transfer. Authenticate existing
trn22 prepare in place and all16 trn21 sources with full hash/size/stable
pre/post stat. The input root and unique packet root must be absent; collision
is HOLD, never overwrite. Preserve the16 original absolute input paths outside
the native capped output root. Use a separately reviewed exact16-file allowlist,
serial bounded <=256KiB opaque streaming and exclusive destination creation.
No archive/compression, recursive project/data/cache copy, workstation checkpoint
copy, source-side checkpoint duplicate or second checkpoint-sized temporary.
A future sole-Operator authenticated connection may relay opaque chunks without
copying credentials; exact transport argv/pins are a separate HOLD.

Authenticate each full destination then publish without overwrite; no source
modification. A failed transfer retains partial files and evidence and returns
HOLD, never auto-cleanup/retry. Actual transfer/static allocation stays<=1GiB.
After staging and again under original lifetime lock require >=13.5GiB free
and >=64GiB RAM, all17 prospective-manifest pins, helper/foundation/packet pins,
fresh STOP/breaker and idle-machine evidence. Do not assume process exit frees
retained checkpoints or caches. Final post-eval measured space controls.

Native max16MiB per-rank receipts and256MiB total packet/log/metric/receipt
artifacts stay unchanged. The16 retained inputs are outside this artifact cap
and accounted in the separate1GiB transfer reserve; never place them beneath
the recursively capped packet root. Static packet bytes conservatively count
in both staging and artifact accounting, not as a credit. SDK caches remain
outside the artifact cap but consume the original3GiB growth reserve and are
governed by the10.5GiB runtime free floor. Monitor at most0.25s apart absent
blocking observation. Any missing/unreadable directory, os.walk/stat/hash/RSS
failure, cap/floor breach or timeout fails; request only owned-job cleanup via
accepted guardian. Polling can overshoot. No reduction, cleanup or budget increase.

## Prospective fixed binding; not a launch packet

Future host trn22 / hostname `ip-172-31-32-253`; original lock
`/home/ubuntu/.cache/trainium_freshstart/device0.lock`, device66305/inode832589.
Unique label `fs-20260926-confirmed36-clean-native-consistency-trn22-host-r1`;
unique root `/home/ubuntu/trainium_confirmed36_clean_native_consistency_20260926_trn22_host_r1`.
Retained input root stays
`/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1`; the old host
word is provenance only. Child --label and TRN_EXP_LABEL match the new helper
LABEL. PYTHONPATH points solely to the existing trn22 prepare directory above.
No shared lock replacement, symlink or trn21 inode596823 use on trn22.

Companion JSON freezes prospective argv/environment: one native_venv torchrun
--standalone --nproc_per_node=4; future path/pin/label-only adapted helper and
manifest; unique receipt root; LNC2; --optlevel=1; seed42; one CPU thread per
worker; graph_stats; six private caches/neff,local,hlo,inductor,compiler,debug;
compiler URL equals compiler directory. These arrays describe a future packet,
not executable authority. Eight runtime executable identities/package inspection,
adapted helper/foundation/driver and final packet hashes remain null/HOLD.
No RELEASE.json or executable/transport implementation is part of this commit.

The current fixed trn21 host packet's os.walk observation-failure repair is
under review, not accepted. Require its fresh independent acceptance, then
review the exact trn22 mechanical/transport diff. Reuse the original reviewed
guardian e461ec5c76c784fd51471c9e8c268cdde42fe1def68b31bfc9524480f980c312,
idle guard adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737
and dispatch c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a
only if the future accepted packet still pins those exact bytes. No acceptance
inferred from author tests. Resource observation failures, unrelated-process
survival, setsid/TERM-ignoring descendants, deadline/caps and original-lock
retention need independent CPU fault evidence normal/-O.

## Unchanged native checks and retained postflight

Reference then candidate are serial in each of four native workers. Keep
all full65 saved/loaded parameter byte comparisons, true ordinary loaders,
config/toggles/weights_only behavior and8192 eval floor. Synthetic seed42,
tokens1..8191, B1/B2/B3 atT256 thenT257, original/repeat/each singleton
row/reversed-suffix calls remain30/source/rank. Preserve exact raw-byte
repeat/row/cross-source checks including signed zero, complete all-rank hashes,
finiteness and before/after input/state/mode/hook purity. Only causal-prefix
maxabs tolerance0.01 is allowed. No BF16 relaxation, dtype shim, hook,
cache refresh, extra shape, training/backward/optimizer/evaluation/data read.

One900-second cap covers imports/init/loading/compile/checks/teardown; timeout
is UNRESOLVED.15s is termination grace, not extra work. Guardian-owned pidfd
lineage/subreaper ECHILD proof holds the same original lock until every owned
descendant exits. Audit failure/unknown quiescence parks with lock. Never signal
unrelated work, protected replay/eval or K2; no name-selected group kill.

Retain all16 transferred files and existing prepare unchanged on destination;
retain all source files. Authenticate both source16 after transfer and all17
destination pins before/after native, including failure. Retrieve every
non-cache packet artifact, hidden/partial/temp rank receipt, stdout/stderr,
old/new pin+mapping receipts, exact sources/binding/transport, runtime/controller
observations, graph_stats, resource-progress/temp, lineage/signals/exits,
original lock and device postflight. Report cache sizes separately. Retrieve
checkpoint identities, not full checkpoint copies to the workstation.
A cap breach is a failure, not permission to truncate/omit. No full output tensors.

Under-lock postflight includes full ps AND neuron-ls, fuser, original lock
identity, no owned descendants, stable17 hashes, both retained checkpoint
paths/sizes and >=10.5GiB free. Repeat clean post-release observations.
Offline four-rank validator and separate independent result review required;
exit0 is not acceptance. PASS would close only this CPU-trained pair's native
six-shape synthetic consistency, not native training trajectory, T1024 eval,
fullcold/public20M quality, source adoption or submission.

## Remaining gates and roles

1. Attached fresh20M eval terminal, full protected artifact retrieval and
   descendant/device quiescence, then fresh Supervisor scheduling decision.
2. Explicit selection of the five-edit exception; separate Implementer changes
   and fresh independent Verifier review, including original CPU provenance.
3. Fresh existing trn22 prepare pin, all16 source/destination identities,
   unused roots, actual post-eval disk/RAM and exact executable/package pins.
4. Accepted os.walk repair plus accepted trn22 mechanics/transport with frozen
   new helper/foundation/manifest/driver/packet hashes.
5. Separate sole-Operator release. Controller STOP/breaker receipt <=60s old
   before staging and under lock; remote STOP, ps AND neuron-ls, fuser,
   original lock, clean pins and resources all pass normal machine protocol.
6. Actual native evidence, complete retained retrieval/postflight and independent
   result acceptance. No automatic next job or promotion.

Researcher writes only these proposal docs. Implementer, Verifier, Operator and
Analyst remain separate. No SSH/host observation, code edit, model allocation,
transfer, launch, cleanup, reservation, protected-file change or canonical-state
change was performed. Submission remains human-only.

Latest evidence:
`a4a041e3fb11130aef604d09c89c7656041b594f:research/v2/operator/20260926-trn22-training-handoff-0543/trn22.json`,
SHA `f701bc30e2b16c7ac26c40e4fe9bb4b464c41069e8dcd365dd6e73c2cce20075`.
Its completed_parent_metadata_only records old prepare missing; remote.free_bytes
and original held lock are explicit. Earlier05:37 snapshot SHA
42c3441cc40d7e225ed8c1506785ff8d3b056c4927810f9acc0d0c5a7ecb3293 and
trn21 snapshot SHA052080a1de4924cc7f21c8c34c54699cc6b10c8d8e287e1e8b8e7425ad9396f1
remain background evidence only. Existing trn22 real prepare binding is in
`research/v2/proposals/20260926-confirmed36-clean-cold-full-trn22-r2/NATIVE_BINDING.disabled.json`;
future fresh authentication is still mandatory.
