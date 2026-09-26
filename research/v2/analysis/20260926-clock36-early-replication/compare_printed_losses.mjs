// Read-only comparison of archived printed losses, not a correctness or BPB gate.
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
const source = '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb';
const inputs = [
  ['research/v2/operator/20260926-trn21-startup-clock-replication/progress/0127-readonly.json',
   '656ba5c30bd4ba686443683552150fe1f3b4e6f97ffb92ec714f0b5139ecfbc4'],
  ['research/v2/operator/20260926-trn22-startup-clock-full/completed/results-bundle.json',
   '2557fbc01b642425201eb077fe84a1967ac4b518b52a9f9bf6ac1c0f8be2cb99'],
];
const hash = x => createHash('sha256').update(x).digest('hex');
function require(ok, why) { if (!ok) throw Error(why); }
const runs = inputs.map(([relative, pin]) => {
  const raw = fs.readFileSync(path.join(repo, relative));
  require(hash(raw) === pin, 'bundle pin: ' + relative);
  const bundle = JSON.parse(raw);
  function extract(name) {
    const f = bundle.files[name];
    require(hash(f.text) === f.sha256 && Buffer.byteLength(f.text) === f.bytes, name);
    return f.text;
  }
  require(JSON.parse(extract('NATIVE_BINDING.disabled.json')).source_sha256 === source, 'source identity');
  const log = extract('training.log');
  const rows = [...log.matchAll(/^step (\d+) \| loss (\d+\.\d{4}) \| lrm (\d+\.\d{3}) \| dt ([\d.]+)s \| tok\/s ([\d,]+)/gm)]
    .map(m => ({ step: Number(m[1]), loss: m[2], loss_units: Math.round(Number(m[2]) * 10000), lrm: m[3] }));
  require(rows.length > 0 && new Set(rows.map(r => r.step)).size === rows.length, 'nonempty unique printed steps');
  return { path: relative, bundle_sha256: pin, training_log_sha256: hash(log), rows };
});
const right = new Map(runs[1].rows.map(r => [r.step, r]));
const pairs = runs[0].rows.filter(r => right.has(r.step)).map(a => {
  const b = right.get(a.step);
  return { step: a.step, trn21_loss: a.loss, trn22_loss: b.loss,
    printed_loss_delta: (a.loss_units - b.loss_units) / 10000, trn21_lrm: a.lrm, trn22_lrm: b.lrm };
});
require(pairs.length > 0, 'shared printed steps');
console.log(JSON.stringify({
  scope: 'Printed precision only; trn21 ongoing. No weight, update, complete-loss or final-BPB equivalence claim.',
  source_sha256: source,
  sources: runs.map(({ rows, ...s }) => s),
  common_printed_steps: pairs.length, through_step: pairs.at(-1).step,
  exact_printed_losses: pairs.filter(x => x.printed_loss_delta === 0).length,
  first_loss_difference: pairs.find(x => x.printed_loss_delta !== 0) ?? null,
  first_printed_lrm_difference: pairs.find(x => x.trn21_lrm !== x.trn22_lrm) ?? null,
  max_abs_printed_loss_delta: Math.max(...pairs.map(x => Math.abs(x.printed_loss_delta))),
  first_five: pairs.slice(0, 5), milestones: pairs.filter(x => [50, 100, 500].includes(x.step)),
  changes_any_gate: false, prediction_of_final_bpb: null,
}, null, 2));
