# Independent Verifier — trn21 stage-r2 CPU G2 final request

**Verdict: PASS for exact CPU-only final-request static enablement; HOLD for
host dispatch until an actual separate Supervisor one-attempt assignment and
fresh live preflight.** I reviewed Implementer commit
`b6b9e88bbafa0efb307ab2d07f31338c499eec4b` and its canonical
cherry-pick `00a70aa30596bafde84771fdb77f740e923ccbc3`. Their six
new file blobs are identical. No existing source, scoring or evaluator file
changed. I performed no SSH, CPU probe, native training, evaluation, AWS
action, cleanup or submission.

| Final package file | SHA-256 |
| --- | --- |
| `CPU_G2_REQUEST.FINAL_PENDING_REVIEW.json` | `7cd7b7aaf34c58da76f1072dbe3e1609c9b6838ca6a975de6fa5c5a7ab85be15` |
| `INVOCATION_BINDING.FINAL_PENDING_REVIEW.json` | `2463f37e7ae97f5a37632fa3d20fa2fa5b7b5f5df2e07749eda6c9a7af72dcbf` |
| `RUNBOOK.md` | `c2370abc2e865493bbfdb3f327e7755bcadfc4b138ab53dc907d788695766f5c` |
| `cpu_g2_invocation_final.py` | `8518e47d259a7ca9ecd592328093b4332a4a79f1ca96548ef8fe74bf22cacf6a` |
| `remote_cpu_g2_once.py` | `966e1c67d909e8be0a63cdb5f18e3801e73cd199c3fda63890916131ca48355a` |
| `test_cpu_g2_final.py` | `27c88f921594b138477ca9ca92bf4e0b0022c35fbffc3c264af4a88b3b4cce92` |

The final request pins the disabled predecessor request `20bb8c73…`, its
independent static PASS `32c9a11c…`, the accepted checkout-result PASS
`b3c82bd5…`, and the e65 source checkout. I recomputed those ancestor
hashes; the local e65 Git object and archived checkout agree on `train.py`
`c5c410af…` and `prepare.py` `5807c46e…`. The two persistent ext4 CPU
helpers remain SHA-pinned to `6a545891…` and `0b5aae14…`, with reviewed
inodes, owner, single links and mode 0600. The frozen three-step/four-rank
synthetic argv, strict-host-key SSH arrays and stdin payload recompute to
the full digests in the binding. The launch has no `--real-data` and includes
`--no-eval-public`; the request, review and raw-result status deny native,
public-evaluation and submission authority.

The wrapper requires **actual fixed-path** Verifier and Supervisor files,
reads both with `O_NOFOLLOW`, and compares their complete structured objects
and one-way review-byte hash. It then checks local STOPs and breakers, creates
the fixed `O_EXCL|O_NOFOLLOW` attempt marker, fsyncs its file and parent
before any SSH, and rejects replay. The first SSH is a full read-only audit.
The second repeats the full audit, proves the exact new stage-r2 scratch
root/label and science paths absent, checks source/tool/asset identity, and
bridges boot, mount, source, tool and host STOP state immediately before the
contained launch, including in the launch child's `preexec_fn`. After any
attempted host contact, a separate full read-only postflight is attempted
even on transport or run failure. A successful CPU run yields raw evidence
pending a different Verifier's numeric comparison, not a science result.

`git diff 00a70aa3^ 00a70aa3 --check` passed. The supplied offline suite
passed **10/10 unittest methods**, including its internal mutation subtests.
My additional offline negatives rejected six fresh-preflight drifts (prepare
hash, helper mode, cache identity, Neuron holder, scratch step log and dirty
Git); confirmed marker file and parent fsync plus replay refusal; forced a
STOP in the child preexec gate and observed HOLD before launch; and forced a
run-SSH exception and observed a separate postflight and terminal HOLD.

The fixed-path structured Verifier receipt
`INDEPENDENT_FINAL_CPU_G2_REVIEW.PASS.json` exactly equals
`expected_review(build_plan())`; its SHA-256 is
`c3e2be88d6a813c9aae95387ecb98f8f0b07771b82dfbaa7c4de618e43f76e55`.
It is owner-owned, single-link and mode 0644 in this review worktree.
The **Supervisor assignment file was not created**. I verified that with
only the review present, `operator_once()` rejects before marker creation or
transport. Git checkouts here may give new files mode 0664; the fixed review
file must be checked and set to 0644 in the eventual Operator worktree or
the wrapper correctly refuses it. The e65 source still lacks the later
paired disk/STOP interlock and cannot be used for native launch.
