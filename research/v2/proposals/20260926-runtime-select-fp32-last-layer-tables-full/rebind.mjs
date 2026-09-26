// Mechanical source/receipt binding of the existing Researcher-authored plans.
// Does not change the scientific design, argv, precision, thresholds or gates.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const root = '/home/poojith-devan/Desktop/Trainium';
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const read = p => fs.readFileSync(path.join(root, p));
const oldSha = '4079e8e6bff1f8438e0430a8b8b2099085da5d6f908149268ac17e8b318fdbff';
const newSha = '414cdd76df85450659375c0928b7f0c882b963088529ca049c19be32814fec34';
const oldCommit = 'fe9bb842b2e899c6e280eb6832dddcf5c6de8d20';
const newCommit = '2a51d40c53da8b5707c499abc765f2d3d47ba500';
const oldPrefix = 'research/v2/proposals/20260926-compilerlocals-fp32-last-layer-tables-';
const newPrefix = 'research/v2/proposals/20260926-runtime-select-fp32-last-layer-tables-';
const lane = 'research/v2/lanes/tables-runtime-select/';
if (sha(read(lane + 'train.py')) !== newSha) throw Error('SOURCE_HASH_MISMATCH');
const receipts = {};
for (const [key, file] of Object.entries({
  cpu_int32: 'cpu-result-int32.json',
  cpu_int64: 'cpu-result-int64.json',
  table_parser: 'parser-result-table.json',
  backward_backend: 'compile-production-result-1790416485199864469.json',
  official_simulator_clustered: 'simulate-result-1790416504816097348.json',
  official_simulator_all_groups: 'simulate-result-1790416657288176259.json',
})) {
  const raw = read(lane + file), result = JSON.parse(raw);
  const bound = result.source_sha256 ?? result.sha256?.[path.join(root, lane, 'train.py')];
  if (bound !== newSha) throw Error('RECEIPT_SOURCE_MISMATCH:' + key);
  receipts[key] = {path: lane + file, sha256: sha(raw),
    status: result.status ?? 'AUTHOR_CPU_PASS', native_executed: false};
}
const replacements = [
  ['20260926-compilerlocals-fp32-last-layer-tables-', '20260926-runtime-select-fp32-last-layer-tables-'],
  ['fs-20260926-compilerlocals-fp32-tables-', 'fs-20260926-runtime-select-fp32-tables-'],
  ['K2-compilerlocals-P2-FP32-last-layer-tables', 'K2-runtime-select-P2-FP32-last-layer-tables'],
  [oldSha, newSha], [oldCommit, newCommit],
  ['research/v2/lanes/tables-compiler-local-constants/TEST_COMMANDS.md', lane + 'REPRODUCE.md'],
  ['research/v2/lanes/tables-compiler-local-constants/', lane],
  ['4079', '414c'], ['fe9bb842', '2a51d40c5'],
];
function bind(value) {
  if (typeof value === 'string') {
    for (const [from, to] of replacements) value = value.split(from).join(to);
    return value;
  }
  if (Array.isArray(value)) return value.map(bind);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k,v]) => [k,bind(v)]));
  return value;
}
const results = {};
for (const kind of ['timing', 'full']) {
  const oldPath = oldPrefix + kind + '/EXPERIMENT_PROPOSAL.json';
  const newPath = newPrefix + kind + '/EXPERIMENT_PROPOSAL.json';
  if (fs.existsSync(path.join(root, newPath))) throw Error('TARGET_ALREADY_EXISTS:' + newPath);
  const oldBytes = read(oldPath), original = JSON.parse(oldBytes), proposal = bind(original);
  // Preserve original historical evidence as evidence, not source-ready flags.
  proposal.historical_repair_binding = original.historical_repair_binding;
  proposal.status = 'complete_exact_source_cpu_off_local_backend_simulator_pass_native_and_actual_control_pending';
  proposal.prepared_utc = new Date().toISOString();
  proposal.author_session = original.author_session ?? '/root/p2_full_proposal_researcher';
  proposal.mechanical_binding_by = '/root/tables_scope_packaging, Implementer under direct Supervisor assignment; no new proposal/scientific-selection authority or self-review';
  proposal.superseded_proposal = oldPath;
  proposal.superseded_proposal_sha256 = sha(oldBytes);
  proposal.supersession = 'Mechanical rebind of unchanged Researcher-authored table timing/full design to repaired414c/base662 and actual source-bound CPU/compiler/simulator receipts. Failed4079 runtime-index parser receipt remains in parent lane1899d6ea0. No old source or launch metadata may be executed; no scientific change, extra gate or native qualification is implied.';
  proposal.candidate.source_commit_status = 'Exact414c at2a51d40c5; author CPU/OFF, real table backward NEFF and both official production-shape simulators passed. K2 native201 result, table native execution and source/argv/environment preflight remain pending.';
  if (sha(Buffer.from(proposal.candidate.launch_command, 'utf8')) !== original.candidate.launch_command_sha256) throw Error('ARGV_CHANGED');
  proposal.gates.cpu_exactness_author = true;
  proposal.gates.flag_off_three_step_identity_author = true;
  proposal.gates.independent_review = 'Existing P2/guard/constants reviews retained at exact original scopes; Root independently reads this minimal runtime-address delta and current receipts. Implementer does not self-review or claim Root acceptance. No extra review loop requested.';
  const limitations = 'Fresh414c author int32/int64 suites pass. OFF3 uses K2=false in both parent/P2 with SGD, not production-optimizer K2-ON/table-OFF; ON2 uses actual table/RMS and interpreted backward at reduced geometry. Real production table parser/backend and official simulator separately pass. Cross-shape prefix discrepancy7.62939453125e-6 remains disclosed. No native/distributed/BPB evidence yet.';
  proposal.gates.independent_review_limitations = limitations;
  if ('review_limitations' in proposal.gates) proposal.gates.review_limitations = limitations;
  proposal.gates.exact_source_composition_bound = false;
  proposal.gates.forward_local_compile_extra_gate = false;
  proposal.control.native_qualified = false;
  if (proposal.base) proposal.base.native_qualified = false;
  proposal.k2_live_dependency = {
    source_sha256: '662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00',
    host: 'trn21', run_label: 'fs-20260926-k2-localconstants-native201-trn21-r1',
    assignment: 'research/v2/assignments/20260926-k2-native201-trn21.md',
    qualification: 'PENDING_ACTUAL_NATIVE201_TERMINAL_AND_TIMING_RECEIPT',
    result: null,
    primary_statistic: '35*131072/(t44-t9), unchanged preselected window; rate>=1.03 versus actual same-host e1e2 control',
  };
  proposal.scope_binding = {
    source_sha256: newSha, source_commit: newCommit,
    parent_source_sha256: oldSha,
    spec: lane+'SPEC.md', spec_sha256: sha(read(lane+'SPEC.md')),
    diff: lane+'SOURCE.diff', diff_sha256: sha(read(lane+'SOURCE.diff')),
    test_commands: lane+'REPRODUCE.md', test_commands_sha256: sha(read(lane+'REPRODUCE.md')),
    status_report: lane+'STATUS.md', status_report_sha256: sha(read(lane+'STATUS.md')),
    scope: 'Only four table HBM group access expressions become reshape/select dynamic views, removing invalid VirtualRegister multiplication. Runtime4096 loop, FP32 order, metadata, forward, optimizer and all model/eval/budget code unchanged. K2 helpers AST-equal to662c.',
    independent_composition_verified: false,
    active_k2_source: 'research/v2/lanes/k2-compiler-local-constants/train.py',
    active_k2_sha256: '662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00',
    active_k2_commit: 'da34c88e90c0f7f281188315a3e63734a8d55521',
    execution_evidence: receipts,
    native: 'PENDING',
    gate_policy: 'Existing native forward accepted by human. CPU exactness/OFF identity plus one native timing is the bar; no extra local-framework-forward prerequisite, OFF run or review cycle. Current receipts are real evidence, not native timing.',
    local_process_policy: 'All author tests completed sequentially; no heavy process remains and local slot was released.'
  };
  proposal.evidence = [...new Set([...proposal.evidence, lane+'STATUS.md', ...Object.values(receipts).map(x => x.path)])];
  proposal.local_scope = 'Original hypothesis/design/argv/precision/thresholds authored by Researcher remain unchanged. Implementer mechanically rebound exact bytes and actual own test receipts only under Supervisor instruction; no self-verification, host access, queue/state/ledger/selector edit, launch assignment or launch.';
  if ('role_separation' in proposal) proposal.role_separation = proposal.local_scope;
  if (proposal.candidate_shortlist) proposal.candidate_shortlist[0].why_ranked = 'Human-selected fixed FP32 tables. Repaired414c has actual author CPU/OFF, production backward NEFF and official simulator PASS; native K2/table and full-control requirements remain pending.';
  proposal.dispatch.sequence = 'Preserve live K2 native201 and batch confirmation. Tables are contingent on actual qualified trn21 K2 native201, source-bound table CPU/OFF evidence (complete), Root source-composition acceptance and fresh native admission. Full additionally retains actual same-host K2 full-control binding. One selected host only; no launch release here.';
  proposal.dispatch.binding = 'Sole Operator binds exact414c/2a51d40c5, this proposal/hash, unused selected label, effective argv/environment/cache/resources under the original lifetime guardian. Never execute old4079/0127/b472/a318/639 prepared bindings.';
  if (proposal.evidence_hashes) {
    proposal.evidence_hashes.historical_binding_note = 'Legacy original spec/diff/review/CPU/K2-full hashes are historical only; current scope_binding carries414c evidence. No fabricated current native/full control.';
    proposal.evidence_hashes.p2_timing_proposal_sha256 = results.timing.sha256;
  }
  // Fresh actual controls remain unbound; do not manufacture scores or adoptions.
  if (proposal.control.public_val_bpb !== undefined && proposal.control.public_val_bpb !== null) throw Error('UNEXPECTED_CONTROL_SCORE');
  fs.mkdirSync(path.dirname(path.join(root,newPath)), {recursive:true});
  const output = Buffer.from(JSON.stringify(proposal, null, 2)+'\n');
  fs.writeFileSync(path.join(root,newPath), output, {flag:'wx'});
  results[kind] = {path:newPath, sha256:sha(output), launch_command_sha256:proposal.candidate.launch_command_sha256};
}
console.log(JSON.stringify(results,null,2));
