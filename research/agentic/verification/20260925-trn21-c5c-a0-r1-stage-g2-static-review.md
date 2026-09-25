# Independent Verifier: trn21 c5c A0 source stage / CPU G2 package

Reviewed exact Implementer commit `9372ce733ca3b2a81304bceae4ad47d2b2e06342`.
`GATE_CONTRACT.json` SHA-256 is
`fa1379c0f040992cd3fa6195a46dacba8de2608857b5847ac2e946ad32fb1c87`;
`RUNBOOK.md` SHA-256 is
`e1aa14f37da72554b5e700a953ab3176121b14009d56084ca00f2a2305e49fc5`.

**Verdict: STATIC PACKAGE PASS; ALL HOST DISPATCH HOLD.** The package is
suitable for a *separate* Supervisor assignment to perform a new exact-source
stage and, after independent stage review, a distinct CPU-only G2 probe. This
is neither a source-stage receipt nor a G2 PASS, native training authority,
BPB result, adoption, evaluation, or submission approval. I did not SSH to or
mutate a host, and did not run the CPU or native probe.

## Checks

- The two-file diff passes `git diff --check`; the gate contract parses and
  keeps source stage, CPU probe, native training, and fresh evaluation
  unauthorized, with all live safety gates false. Its ratification, science
  proposal/contract, trn21 G1 Operator/independent PASS, and prior c5c CPU
  anchor hashes match their bound files. Its train argv and four effective
  environment values exactly equal the trn21 science contract.
- The package predates the A0 read-only feasibility receipt and its review, so
  the package branch lacks the receipt and says `PENDING_INDEPENDENT_VERIFIER`.
  On the canonical branch at `2d009b47`, the cited feasibility receipt has
  its exact bound SHA-256
  `90b77b5f220da9e72dbe1932b27b87b953ba85865793ea9ebc3318997f330df1`,
  and its independent **planning-only PASS / dispatch HOLD** review is SHA-256
  `ef1df8346555088b14fd575f85699b08aa9163efc68e10b582e41f2ee3099c40`.
  That historical pending field needs no mutation of the reviewed package:
  the later Supervisor assignment must explicitly bind both canonical
  artifacts and this static review. It must not treat the field as launch
  authorization or the old host snapshot as a fresh preflight.
- The local transfer capsule is a regular mode-0600 file of exactly
  `104,005,787` bytes with SHA-256
  `32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264`.
  All 6,338 tar members are relative regular files/directories; there are no
  traversal names, links, special files, duplicate paths, PAX overrides, or
  setuid/setgid bits. An isolated local extraction yielded exact HEAD
  `e65a3097a5b00f6178188a935bb2acb9a5042c6c`, one reachable and one
  total commit object, shallow repository, clean tracked/untracked tree,
  no extra ignored files, no remotes/alternates/active hooks/filters, and
  `git fsck --full --strict` PASS. Root `train.py` is exact c5c SHA
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
  unchanged root `prepare.py` is exact SHA
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- The archive contains no standalone training shards, model checkpoints, or
  credential-like files. `checkpoints/` has only its README. The sole
  tokenizer ZIP is the organizer-provided artifact and matches its pinned
  SHA-256 `12fa2c4a56d9f26d46e8ea0a57f69d78145930eecb0a765c9bfc085176af8a54`.
  A high-confidence key-marker scan of approximately 99 MB of text found no
  OpenAI/AWS/GitHub/Slack key or PEM private-key marker. This is a bounded
  content screen, not a guarantee about arbitrary binary documents.
- The two CPU tools match their bound SHA-256 values. The candidate CPU shape
  uses clean `env -i`, exact science suffix, LNC2/four ranks, seed 42, 1,800-s
  science default, and only G2's CPU/no-compile/three-step/isolated-output
  additions. `preflight_dryrun.py` uses synthetic data by default and puts
  its work under `TMPDIR`, which the command sets to the distinct `/dev/shm`
  G2 root. The mode-0700 step-log parent is created before launch and the
  step file must be absent; `prepare.record_step_boundary` then writes there.
  The appended `--out-dir` overrides the science-output environment default.
  The containment command binds the distinct G2 label and its own process
  directory. The prior c5c CPU anchor is a PASS with matching source,
  prepare, argv, four ranks, three steps, and 2x256 micro-batch.

## Conditions that remain HOLD

The Operator must freshly prove hostname, disk >=10 GiB, `/dev/shm` capacity,
no process/lock/Neuron holder, closed applicable breakers, unique absent stage
and G2 paths/labels, exact archive/tool hashes, and unchanged protected
checkpoint/cache identities **before** any host write. Stage into an absent
path without overwrite, then archive source identity and obtain a different
Verifier's stage review before CPU G2. This package itself grants no host
action. Before the CPU launch, prove the separate tmpfs step-log parent is
owned and writable, not merely present, while its step file remains absent.
Even a CPU G2 PASS does not authorize the native A0 label.

For later G2 numerical review, do not rely on the tool's `verdict` or
`--compare identical` alone: `aggregate()` can report PASS from four PASS
rank reports without checking a nonzero `torchrun_rc`, while `compare()` does
not compare argv or exit status. Independently require `torchrun_rc == 0`,
`timed_out == false`, exact SHA/argv/environment, four finite three-step rank
reports, step-log lines `0,1,2`, checkpoint/load check, clean quiescence, and
then lockstep comparison. Any mismatch is HOLD/FAIL, not a control or win.
