# F0/F2 mechanics repair candidate

This is a repair of the independently refused packet0ef8a56a, not approval of it.
The full original refusal and its probes remain immutable at
`/home/poojith-devan/k2-f0f2-independent-gmVZVW/`. Its verdict remains
REFUSE_IMPLEMENTATION_MECHANICS_REVIEW. A new independent review is required.
The selected partial scope, original six-fixture gate, all resource/time caps,
consumer9425390c, integer oracle and `BINDING=None` are unchanged.

## Four concrete repairs

1. Finalization has a release-safe flag initially false. Every stream/monitor
   close is exception-contained; owned-process termination proceeds despite close
   failures. Postflight, final census/reaping, terminal creation/encoding/write/
   close, caps and deadline must finish before that flag can permit release.
   Terminal ENOSPC/exclusive-create/cap or any unexpected finalization exception
   cannot bypass the outer protection hold. An additional quiet containment
   attempt precedes that hold after unexpected exceptions. The hold does not
   require any filesystem/logging operation and catches sleep/signal exceptions.
   TERM/INT/HUP become Python exceptions after lock acquisition, so they cannot
   turn a failed hold into an ordinary successful lock release. SIGKILL, process
   corruption and host loss remain outside userspace guarantees.
2. The driver enables Linux subreaper and creates its persistent PID/pidfd
   tracker immediately after authenticating the inherited lock, before the first
   git/ps/neuron-ls helper. Each bounded helper also requires prior kernel ECHILD,
   takes live descendant censuses, rejects a zero-exit helper with surviving
   descendants, and runs TERM/KILL/reaping in its finally block. Cleanup is
   reserved INSIDE its existing timeout (not15s plus extra cleanup). Existing
   ≤2MiB helper output cap and common deadline remain. Helper evidence is kept
   in memory until terminal recording; cleanup is not conditional on a log write.
3. After the final postflight helper, even when its check raises, the driver
   performs a new descendant census, bounded stop and reap. Release requires
   no live pidfds followed by kernel `waitpid(-1,WNOHANG)` reporting ECHILD.
   Final cleanup ends by1198 and terminal completion before1200. An attempted
   late helper survivor cannot be hidden by the earlier termination receipt or
   the selected ps-name filters. Failed cleanup/postflight/evidence retains the
   original lock, with no successor.
4. Coordinator admission rejects the torchrun launcher as a worker and refuses
   reuse of a previously admitted PID or PID/start pair. It reads the actual
   peer process's four rank variables from bounded `/proc/PID/environ`, retaining
   only those variables, and matches the claimed rank/world to those bytes. It
   checks actual executable inode, native-venv interpreter argv (allowing sibling
   interpreter symlinks to the same executable), exact worker/root arguments,
   ancestry to the frozen live launcher identity, and unchanged peer starttime.
   All four distinct workers must still be live with their admitted identities
   before scheduling. Rejected peers are closed. Source/process association is
   not a hardware execution counter or arbitrary same-UID adversary guarantee.

`IMPLEMENTATION_SPEC.md` describes unchanged numerical/native behavior; these
mechanics supersede its earlier helper sequencing and peer-admission description.
There is no SDK API change, model/default/threshold change or host-specific binding.

## Tiny regression receipts

`test_repairs.py` executes actual driver main with mocked host-only preconditions
and a real local test-owned flock. Terminal ENOSPC, monitor-log close failure and
log-cap exhaustion each reach the real protection hold, retain the lock, and stay
held after TERM. The test then uses SIGKILL only on its own fixture and reaps it.
These are local control-flow/FD-lifetime tests, not a native util-linux host run.

Actual `bounded_run` is exercised with the review's timeout-plus-fork and
zero-exit-detached-child patterns. Both fail, contain descendants and demonstrate
kernel ECHILD. Another actual-main probe creates a new-session double-fork child
inside the postflight check AFTER the earlier stop; the final recensus discovers,
signals and reaps it. All fixture processes are test-owned; no preexisting process
is signaled.

An actual coordinator-main admission regression uses the prior one-PID/four-rank
claim with generation/launch/route mocks. It refuses before environment receipts
or native routes are scheduled. Four separate local children then exercise real
Unix SO_PEERCRED, actual rank environment/argv/executable/ancestry checks without
SDK imports. Duplicate process, false rank environment and launcher-as-worker
counterchecks reject. Local success is not four native ranks or a native result.

Normal and `-O` repair receipts, existing packet receipts and20,006-pair oracle
receipts are frozen separately. No full-size tensor, host access, stage/launch,
training/evaluation or independent self-approval occurred. The original refusal
is preserved; only a fresh reviewer can decide whether these fixes resolve it.
