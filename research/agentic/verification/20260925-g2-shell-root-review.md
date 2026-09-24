# Independent CPU/G2 operator-shell review

Reviewer: root/controller, not the wrapper implementer. Date: 25 September IST.
Scope: exact commit `445cc9d0e911b7fd4a22c6540713a60cc69686ad`,
`research/v2/run_g2_baseline_s0_trn22_r1.sh` SHA-256
`fe65a6e8b38236329557ca8942d54a200b0d8ad0bae8b8b2a0bb6e32fcd330bd`.

Verdict: **PASS for the four synthetic CPU/G2 probes only**, subject to the Operator's fresh host,
disk/RAM/tmpfs/idle/device checks and working user systemd scope. No native or scored run is authorized.

I read the complete wrapper, static tests, called containment implementation, harness launcher,
aggregate and comparison fields. I reran the supplied test file independently: 7/7 PASS; bash syntax
PASS. These tests are static, not proof that G2 itself passes.

- Exact host, absent unique stage/run/label paths, seven source/tool/prepare SHA identities and
  four argv vectors are bound. Training sources and prepare.py are not edited.
- The parent shell retains the existing shared host flock across launch/wait/stop/quiescence.
  The second nonblocking flock must fail. No parallel sleep-based lock holder is introduced.
- Every probe is externally tagged/contained. Timeout/error/interrupt traps perform scoped stop
  and 10-second quiescence. A surviving tagged child fails the batch even after cleanup succeeds.
- Each receipt must be PASS, exact source/argv/prepare identity, four PASS ranks, checkpoint load,
  no cross-rank mismatch, zero torchrun return code, and no timeout.
- Two required comparisons and the additional 35c9-versus-clean comparison require exact losses,
  plan, key set and parameter digests. No BPB result is generated or inferred.
- All TMPDIRs are unique under the private tmpfs stage; `--keep` retains artifacts. No SSH,
  deletion, data manipulation, AWS action or Neuron training occurs inside the script.

Approved staging allowlist at that commit: unchanged prepare.py; the exact 35c9 and 116188 base
files; research-v2 and S0 train.py; contained.py; preflight_dryrun.py; the wrapper and its test.
The Operator may build a clean thin staging Git snapshot, recording original commit plus exact
hashes separately. Its local staging commit must not be represented as the upstream commit.
Operator owns transfer/launch/retrieval; root did not launch a host job during this review.

If the host lacks a functioning user systemd scope, stop before launch and report it. Do not edit
the reviewed wrapper ad hoc. Any failure is a gate failure, with new labels required for retries.
