# Author implementation receipt; not independent acceptance

Base20576a9d, isolated sparse worktree, local development only. No SSH/host
operation, native package import, full model, launch, cleanup or guardian action.
The initial full /tmp worktree checkout failed with write errors; its partial
files were preserved. Continued in a fresh no-checkout sparse worktree without
editing the canonical checkout or another agent's worktree.

21 author tests pass in each of normal and optimized Python; raw logs are
author-normal.log and author-optimized.log. Seven new cache tests and14 applicable
existing transport regressions cover real links to outside files, broken targets,
directories and cycles; a guarded os.open detects any attempt to open those
targets. Descriptor scans count only regular sizes and link lstat sizes. Tests
also exercise noncache/root/special refusal, nested scan errors, link replacement
during readlink, directory substitution before open, final retrieval cache change,
lossless allowed artifact transfer and absence of copied caches.

Inherited transport regressions exercise before-open/path/header/in-place
checkpoint substitution, immutable leaf/chunk authentication, late changes,
receiver failures, complete hidden/partial retrieval, caps without truncation,
real local flock staging and failure preservation. All host observations in
these tests are mocked. No test results establish native feasibility.

The r2 probe inverse label substitution reproduces the exact donor937e2f23 hash.
INPUTS, foundation, guardian, idle and dispatch source bytes remain unchanged.
The separate r1 recovery pins the unchanged consumed packet and rejects staging.
K2 delivers only a monitor/SOURCE_PINS overlay; its current host binding and all
consumer/worker/lifetime/resource/scientific logic stay donor-identical.
No r1 packet, r1 receipt, selected-science file or controller state was edited.

Independent review must use the frozen input-only bundle, never this receipt,
author test source or author logs as its own reference. Fresh review, runtime/
host gates and separate Operator releases remain pending.
