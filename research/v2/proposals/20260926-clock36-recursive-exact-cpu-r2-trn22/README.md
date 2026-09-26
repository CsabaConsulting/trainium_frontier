# CPU r2: retained trained-checkpoint causality/state qualification

Proposal6d5a2fac precedes this separate successor. Original03f20e7f/8f93ead3
packet is untouched. Frozen source44926a6f/SHA4bfaec50 remains unchanged.
Fresh source review84448aa2 is scoped source/schedule PASS only; complete CPU
qualification was explicitly refused until trained-checkpoint causal/state
checks execute. This packet prepares those checks; it does not claim they ran.

Four full-size CPU G2 arms/reference/default/OFF/ON retain exact r1 science:
startup clock ON throughout, Recursive omitted/OFF/ON only. Same113,256,724
parameters/65 tensors,4 ranks,3 synthetic2x256 updates. Unchanged standard G2,
receipt validator,guard,dispatcher and full driver flow. Only unique r2 root/
labels/request/helper hashes and checkpoint verification helper differ.

After clean G2 gates, the existing600s+60s checkpoint child loads each of all
four retained trained checkpoints using weights_only. It authenticates exact
request/source/checkpoint hashes,full8x1024 config,65 state tensors/113256724
elements,then calls ordinary source load_for_eval on CPU without any global,
dtype,forward or loader override. Loaded checkpoint bytes must match. Its
deterministic synthetic input is B1T256,seed7,in-vocabulary; no data or tokenizer
loader is called. All named parameters and ALL registered buffers,including
nonpersistent buffers,are hashed before and after every forward. Input bytes,
finite/shape outputs,eval mode and strict byte repeat equality are checked.

Synthetic causality mirrors pinned prepare.py663–684 ONLY: suffix reversal from
cut=max(2,T//2),FP32 prefix[:cut-1] maxabs<=1e-2. Exactness/error are reported.
This is NOT organizer causality_check/data/scoring or native evidence. Output
snapshots are cloned so aliased scratch storage cannot erase earlier observations.
No new tolerance is used for state/repeat/loaded-model bytes. The original
three-arm BF16 CPU logit comparison remains separately disclosed and unchanged;
that old dtype shim is never applied to the new ordinary-loader tests. ON's
state/purity/causality checks are mandatory but ON need not match OFF weights.

Boundaries unchanged: inherited original trn22 lifetime flock832589/device66305,
in-lock fresh ps+neuron-ls every child,64GiB RAM,Operator disk>=10.5GiB plus
footprint (old inner6.5GiB guard retained),STOP/hash/host gates. No checkpoint
child budget extension,timeout retry,cache/data/evaluator/production source change.
Three G2 updates do not cross prepare's5-step startup boundary. CPU-only timings
are not an1800-second quality result or hardware microbatch proof.

Own tiny helper tests use <=32 tensor elements each,one CPU thread,no train.py
import: causal positive,noncausal/aliased-output leakage,param/nonpersistent-buffer
mutation,input mutation,mode change,nondeterminism,nonfinite/malformed outputs,
signed-zero repeat difference. Nonfinite tensor records also refuse. These are
helper tests only; no full-size qualification or review is self-approved.

New root/labels prevent overwrite of8f93. Source/CPU outcome/independent r2 binding
review and explicit sole-Operator release remain separate. Every outcome needs
actual PID/starttime/PPID/PGID/SID descendant/compiler absence,device-holder,
fresh ps+neuron-ls and original lock identity postflight. No automatic cold/full/
eval: child-specific cold awaits distinct accepted CPU outcome and root release.
