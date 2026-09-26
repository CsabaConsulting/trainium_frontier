# Exact414c prospective table bindings — not a launch release

Supervisor requested a mechanical rebind of the existing Researcher-authored
timing/full proposals. Implementer performed that rebind without changing the
hypothesis, prediction, base, precision, launch argv, selection thresholds,
resource floors or scientific design. No proposer/verifier/Operator authority
is inferred; Root independently reads the exact delta and receipts.

Source414cdd76df85450659375c0928b7f0c882b963088529ca049c19be32814fec34
at commit2a51d40c53da8b5707c499abc765f2d3d47ba500 replaces failed4079
for prospective jobs only. Failed4079 parser evidence remains1899d6ea0.
Base remains K2 source662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00,
batch131072, matrixLR.015, pack32/context1024, four LNC2 ranks, seed42.
No batch/LR/schedule stack is silently inserted.

## Actual evidence available for Root's independent read

- Source: `research/v2/lanes/tables-runtime-select/train.py`, SHA414cdd76.
- Spec SHAa472cb85d1ce9fbcd02e3830549c2209c2eb288d837b04d224ffaccf5d74789b.
- Complete diff SHAe95f4725114bdb087898934497337547e6ea908ebba8c28c40c1baed17cf4788.
- CPU int32 receipt SHA4e0874a36c6881fac54482b57cbcc57566098bc4ba4979f9c338f3f9bf7220b1.
- CPU int64 receipt SHA3ec2fac569e226a748d7affadcdae7b3967aedbebb5544948d4152de6df8c43b.
- Both suites: FP32 reference, OFF3 bit identity, ON2 registered-vs-ordinary
  bit identity, actual table RMS, hashing/clipping/strict checkpoint PASS.
  Mocked collectives, reduced geometry and K2-disabled OFF3 limits remain.
- Exact production real table parser and authenticated neuronx-cc backend
  PASS, NEFF61386bytes. Both logical-core instruction streams are nonempty.
- Two full-production official simulator fixtures PASS with maxerror0;
  second traverses all4096 active groups and516096 untouched rows, including
  final-row addressing. No native execution or full-model startup claim.
- Existing cross-shape prefix discrepancy7.62939453125e-6 is not erased.

The only source delta is runtime table HBM group selection. Address claim:
counts/starts `g*128+p`, maxima `g`, output `(g*128+p)*128+o*64+f`;
g0..4095,p0..127,o0..1,f0..63. Root can independently verify disjoint owner
halves and complete row coverage. FP32 encounter-order operations are unchanged.
All three existing K2 helper ASTs equal exact662c in both CPU receipts.

## Still contingent, not invented

Current dependency is actual `fs-20260926-k2-localconstants-native201-trn21-r1`.
Its precommitted35*131072/(t44-t9) rate and terminal correctness have not been
invented or relabelled here. Native qualification flags remain false. Table
timing45 is conditional on that actual pass and fresh sole-Operator checks;
it adds no3% table-speed requirement or separate local-forward/review layer.

Full removes only --num-steps45. Its unchanged real same-host K2 full-control
binding remains null, not an e1e2-only score or a timing surrogate. There is
no actual full-control row or retrospective relabel by this rebind. Follow
the Supervisor's current scheduling authority; these disabled proposals are
not queue edits or launch assignments.

Native/full host alternatives use fresh runtime-select labels, unselected.
Original lifetime lock, exact source/argv/environment, measured resources,
official1800-second accounting and fresh20M/context1024 remain unchanged.
Source414c local checks completed sequentially; no local-heavy process was
present at10:11:25UTC. No further model/compiler test was started for this
JSON rebind. No SSH, host launch, shared state/queue/ledger, data or upload.
