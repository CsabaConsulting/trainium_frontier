# Independent static review: trn21 c5c A0 CPU G2 stage-r2 HOLD draft

**Verdict: PASS for non-dispatchable offline planning only. HOLD for any CPU G2,
host operation, native run, evaluation, or submission.** This review is not a
Supervisor assignment, not an Operator request, and not a review of a future
enabled request. No SSH, AWS, host, CPU G2, native, or evaluation action was
performed for this review.

## Exact reviewed object

- Source branch: `codex/20260925-trn21-c5c-a0-g2-stage-r2-hold`.
- Source commit: `2fec2115f287c406b4135a787f4c40dcae4598ee`, clean at review.
- Package: `research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2/`.
- `GATE_CONTRACT.json`: SHA-256
  `9867b33cb250c6851f46c00bada89e7b4b730c0c861770dab182e994240db2f1`.
- `RUNBOOK.md`: SHA-256
  `0a2b0f33988a9f5f62d782977c8c8e4072dd389f6fac0883c86e305c31da9e9c`.
- `build_g2_argv.py`: SHA-256
  `74b1d49cda3b5898ccb66539239f7d3ab78f489e1ef23e80fbd7b616cc65f5eb`.
- `test_g2_argv.py`: SHA-256
  `0ee3965a19edc21753f9bd2b5c3d7d2e3c207ad78ff697e645826bb37d40ed8b`.

The draft is explicitly disabled: `cpu_g2_dispatch_authorized`,
`native_training_authorized`, `evaluation_authorized`, and
`submission_authorized` are all `false`; the checkout-result gates are null;
and direct execution of the offline builder exits `DISPATCH HOLD`.

## Bindings checked

- Host `trn21`, expected hostname `ip-172-31-34-85`. The *only* candidate
  source checkout is the new persistent-ext4
  `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`.
  It remains unverified. The old r1 checkout and `/dev/shm` capsule are not
  source fallbacks. The historical r1 path in `CACHE_ROOT` is solely the
  pinned warm-cache location, not the source or CPU scratch root.
- Frozen e65 source commit
  `e65a3097a5b00f6178188a935bb2acb9a5042c6c`; `train.py` SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
  unmodified `prepare.py` SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The immutable 15-element training suffix, seed 42, LNC 2, bf16 compiler
  flags, and 1800-second setting match the ratified A0 control and prior
  trn22 c5c CPU anchor. This is a proposed cross-host comparison, not a
  claim that trn21's numerical result will match.
- The two persistent ext4 CPU tools are bound by exact paths and SHAs:
  `/home/ubuntu/trainium-v2-g2-preflight-dryrun-20260925-trn21-stage-r2.py`
  (`6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`)
  and `/home/ubuntu/trainium-v2-g2-contained-20260925-trn21-stage-r2.py`
  (`0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d`).
  The contract also pins their prior UID, device, mode, inode, size, and
  single-link status, all requiring future no-follow live verification.
- The unique CPU label is
  `fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2`; the previously absent
  ext4 scratch root must be
  `/home/ubuntu/trainium-v2-g2-20260925-trn21-stage-r2`, mode `0700` if
  later created. `TMPDIR`, contained process files, receipt, and step log
  point there, outside the clean checkout. The preflight options are exactly
  `--steps 3 --nproc 4 --timeout 900 --keep`; synthetic CPU/no-compile data
  only, without real data or public evaluation.
- Offline construction produced the claimed 64-word launch argv and exact
  compact-JSON-array SHA-256 digests: launch
  `1810cdcde05dbda9dea063f160ceb25563d01bc2eb61b70623163d945e432eeb`,
  wait `3a416123b7264585e23fbb3fcc75a1501a993d26fac9fe0580f7f47b2516c380`,
  failure stop `9216b16faa88664f8d91bf80752351af25287134fb5ad635f06f960b55e17e10`,
  and quiesce
  `c959acbee53e6743cf7f2a300d8d8ddb4477c0514c8313aa4596d9d1545e0edf`.
  These hashes are bindings for review, not authority to run them.
- The package binds the completed transfer receipt
  `9ab90252e5e21812f5aad2af7fd0bfd1e71205e2f8f740f6f41faf6606fe6a2e`,
  its independent result review
  `bfafcb56ff449708dc31a1ebbe9023448018ea505e37a15847ae78bc3a3a2916`,
  the frozen extraction code
  `e66afb7a1e3368ed7854eaa679e2bad9a6b8529e3b793e0f8fff2c69c496785d`,
  and its independent implementation review
  `1a05c6dab4cbae6f49bd7e58760155d26562ba39df6a85524fe4a50289679df4`.
  The earlier checkout request `5fc762f851534a61fd29b8ce2ec5af07ac9ca5cb`
  has an independent **HOLD** receipt with SHA-256
  `8656f59af4ca658d558525b69ed4c2a3b8d54a382ccc88d8603f8c20a0bb1678`.
  This reviewed G2 branch does not itself contain that receipt; its hash was
  verified in the separate pushed review branch.

## Local checks and adversarial finding

`git diff --check` passed. The package's six synthetic unit tests passed in
this independent worktree with bytecode writing disabled. Direct builder
execution was tested to refuse dispatch. The offline plan rebuilt with the
four exact argv hashes above. No train process was started.

Adversarial in-memory contract mutations showed that `build_plan()` still
accepts `one_attempt_per_label=false`, `persistent_cpu_tools.required_mode`
changed to `0777`, or `preflight_inode=0`, while the argv hashes continue to
match. The current artifact is safe to archive as a *disabled* plan, because
it grants no action and its runbook requires separate live checks. **Do not
reuse this builder as the sole safety certificate for an enabled request.**
The final request/reviewer must validate these non-argv safety pins and every
fresh host property explicitly, or harden the builder and retest it.

## Conditions still missing

The repaired exact checkout request needs a separate independent PASS; the
old `5fc762f8` request is HOLD. Then an Operator must complete a one-attempt
checkout-only transfer with full terminal receipt and read-only postflight,
followed by independent checkout-result PASS. Only after those prerequisites
may an Implementer create a **new final CPU-only request**, with its own
independent static review, fresh full host preflight, and a separate
Supervisor one-attempt assignment. Even a future CPU G2 PASS would not
authorize native training or evaluation. On any failure, preserve artifacts
and HOLD; no cleanup, substitution, wrong-process signal, or same-label retry.
