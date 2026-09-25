# Independent Verifier — trn21 c5c A0 stage-r2 checkout implementation

**Verdict: PASS for the frozen, non-dispatchable implementation review only.**
The code and local tests support a one-shot, absent-path, owner-private source
checkout from the pinned persistent capsule, with fail-closed archive/Git/asset
checks. This is **not** permission to invoke the function on trn21. Checkout,
CPU G2, native training, evaluation, adoption, AWS changes and submission all
remain **HOLD** until their separate role gates and exact requests pass.

Reviewed the package introduced at canonical commit
`04bc824ddfc0c37565c9087c5ac912c95c0155f7`; its five package files
were unchanged at `09ba9e1d3d394f148fbd2f7a2787819bfcd1927a`. The
implementation SHA-256 is
`e66afb7a1e3368ed7854eaa679e2bad9a6b8529e3b793e0f8fff2c69c496785d`,
the test SHA-256 is
`b0407b233bbd1c6b09075765c66e25de7ea912a22ef1f8d5dcf353ac40e7d373`,
and the gate contract SHA-256 is
`cb034ffea544c560104a4d558a8a18119f50a4db5088045e0c5d711635bfbd65`.
All equal the frozen contract's pins. The contract still says
`extraction_authorized=false` and has no exact invocation or one-shot
assignment. Its pre-transfer-result fields are a historical HOLD snapshot,
not a newly granted authority.

## Checks and evidence

- The previously independent-reviewed transfer result is source-stage evidence
  only: canonical `d8c16692e668e801a76134857ce743c25992c291`, review
  receipt SHA-256
  `bfafcb56ff449708dc31a1ebbe9023448018ea505e37a15847ae78bc3a3a2916`.
  I separately assembled the nine archived writer receipts into the exact
  `write/verify/final_verify` input shape. Local
  `validate_persistence_bundle` accepted all three artifacts; the canonical
  sorted bundle SHA-256 is
  `13db20bb2887eb6ff1c9feda31ea8453c9252ace18ff571021f866780e7cfe86`.
  Mutating the boot ID, mount source, capsule inode, an exact
  `(PID,start_ticks)` session pair, or removing a final verification caused
  rejection. The validator binds recorded host `ip-172-31-34-85`, UID 1000,
  boot/mount namespace, ext4 root mount, parent identity, target inodes,
  modes, bytes, SHA and chronological cross-session passes. It does **not**
  authenticate raw SSH transport; the prior raw-record review did that, and a
  new request must bind both pieces of evidence.
- The existing nine local tests passed (`python3 -B .../test_stage_r2_checkout.py
  -v`). They include the real pinned capsule's 6,338-member scan: 6,283
  regular files, 55 directories, four GNU long-name headers, 188,636,254
  unpacked bytes, and manifest SHA-256
  `a9eff72e760f314717e712a5e3d23d300232db4feeb361896c57b16f7bdd1bf4`.
  Independently added local synthetic negative checks for character devices,
  block devices and GNU long-name traversal; each was rejected before
  extraction. The supplied tests also reject absolute/traversal/duplicate
  paths, missing parents, symlinks, hardlinks, FIFO, PAX metadata, unsafe Git
  config, setuid, credential/checkpoint paths and symlink collisions.
- Source-stage order is fail-closed: the implementation checks the complete
  archived receipt bundle and current host/boot/mount/parent identity; requires
  an absent checkout/output path and a live 10-GiB root free floor; snapshots
  the two pinned checkpoints and seven cache-directory identities; verifies
  the persistent capsule's final-session inode, `0600` mode, 104,005,787
  bytes and full SHA-256; and rehashes both persistent CPU tools. It completes
  two archive-parser passes and a pinned manifest **before** creating the
  checkout. On success it rechecks the capsule, tools, host identity, disk
  floor, protected assets and output absence.
- Creation is restricted to the one previously absent `/home/ubuntu` checkout.
  `mkdir` is exclusive; directory walks use `O_DIRECTORY|O_NOFOLLOW` against
  owner-private directories on the same device; member files use
  `O_EXCL|O_NOFOLLOW`, are written in bounded chunks, read back, checked for
  inode/mode/length/digest, and fsynced. Archive ownership and group/world
  modes are not applied. A synthetic pre-existing file remained unchanged
  while earlier new members remained as partial output. No deletion, cleanup,
  rename, replacement, same-path retry, SSH or AWS call is present. Direct
  `python3 -B stage_r2_checkout.py` exited nonzero with `DISPATCH HOLD`.
- The success verifier requires the pinned detached e65 HEAD, exact shallow
  and config files, one reachable commit, no remote or alternates, a clean
  status, an independent all-untracked check, `git fsck`, and exact root
  `train.py`/`prepare.py` hashes. Git runs with a minimal configuration and
  no optional index locks. A local synthetic Git check confirmed that
  `git ls-files --others --full-name` detects a `.gitignore`-ignored file
  even when porcelain status omits it. The final receipt is explicitly
  `SOURCE_CHECKOUT_VERIFIED_FOR_INDEPENDENT_REVIEW_ONLY` and returns all
  downstream authorizations as false.

## Limits and required next gate

This was a **local code/fixture review**. I did not SSH to trn21, run this
stage on a host, independently repeat a full unpack of the real capsule,
perform live preflight/postflight, or run CPU G2, training or evaluation. The
bounded text secret-marker scan excludes opaque Git packs; this review does
not certify a general absence of secrets. The implementation returns no
success receipt on an exception and preserves a partial checkout, but its
`finally` only closes descriptors: a future one-shot Operator wrapper must
archive the exact failure and perform a separate **read-only full postflight
even on failure**, without cleanup or retry. That postflight and the fresh
full preflight are not supplied by this package.

Before any checkout attempt, the Supervisor must obtain a separately reviewed
exact request and argv binding this code SHA-256, the complete archived
receipt-bundle SHA-256, the prior independent transfer review, the fixed
capsule/checkout paths and a fresh full trn21 preflight. The Operator must
perform only that one checkout-stage action, retrieve the complete result and
postflight, and leave any partial checkout untouched on failure. A distinct
Verifier must then review the host result. This PASS is not an extraction
authorization or a substitute for those gates.
