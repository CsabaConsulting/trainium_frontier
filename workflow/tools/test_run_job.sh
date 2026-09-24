#!/usr/bin/env bash
# End-to-end test of run_job.py on any Linux box (no Neuron needed): fake train/eval commands, a scratch
# git repo, two controls, a nominated flag arm, a job with an escaping descendant, a bad-loss job and a
# wrong-bytes job. Expected: C1/C2 control, F6 NOMINATE, E1 recorded WITH an anomaly, B1 rc=5, X1 rc=4,
# no tagged processes left.
set -u
W="$(cd "$(dirname "$0")" && pwd)"; T="$(mktemp -d)"; mkdir -p "$T/repo" "$T/archive"; cd "$T/repo"
git init -q && echo "print('train')" > train.py && git add train.py && git -c user.email=t@t -c user.name=t commit -qm init
SHA=$(sha256sum train.py | cut -d' ' -f1); HEAD=$(git rev-parse HEAD)
cat > "$T/host.json" <<J
{"host": "trn21", "python": "python3", "out_root": "out", "archive_dir": "$T/archive", "disk_floor_gib": 0.5, "use_cgroup": false,
 "env": {"NEURON_LOGICAL_NC_CONFIG": "2"}, "eval_args": [], "eval_args_verified": true,
 "train_cmd_override": ["python3", "$W/fake/fake_train.py"], "eval_cmd_override": ["python3", "$W/fake/fake_eval.py"]}
J
mkjob(){ l=$1; a=$2; k=$3; s=${4:-$SHA}; mkdir -p research/v2/experiments/$l
  echo "{\"verdict\":\"PASS\",\"train_py_sha256\":\"$SHA\",\"argv\":[\"--foo\"]}" > research/v2/experiments/$l/g2.json
  echo "{\"label\":\"$l\",\"arm_id\":\"$a\",\"kind\":\"$k\",\"train_py\":\"train.py\",\"train_sha256\":\"$s\",\"base_sha256\":\"$SHA\",\"git_commit\":\"$HEAD\",\"argv\":[\"--foo\"],\"g2_receipt\":\"research/v2/experiments/$l/g2.json\",\"footprint_gib\":0.1,\"timeout_s\":120,\"eval_timeout_s\":60}" > "$T/$l.json"; }
runj(){ l=$1; shift; env "$@" python3 "$W/run_job.py" --host-config "$T/host.json" --job "$T/$l.json" --repo "$T/repo" > "$T/$l.out" 2>"$T/$l.err"; rc=$?
  python3 -c "import json;d=json.load(open('$T/$l.out'));print('%-3s rc=%d %s'%('$l',$rc,{k:d.get(k) for k in ('public_20m','verdict','anomalies','failed_stage') if d.get(k) not in (None,[])}))"; }
mkjob C1 A0 control;  runj C1 FAKE_BPB=0.9920
mkjob C2 A0 control;  runj C2 FAKE_BPB=0.9918
mkjob F6 F6 flag_arm; runj F6 FAKE_BPB=0.9905
mkjob E1 F7 flag_arm; runj E1 FAKE_BPB=0.9930 FAKE_MODE=escape
mkjob B1 F8 flag_arm; runj B1 FAKE_MODE=badloss
mkjob X1 F9 flag_arm 0000000000000000000000000000000000000000000000000000000000000000; runj X1
python3 -c "
import sys; sys.path.insert(0, '$W'); import contained
left = {l: contained.tagged_pids(l) for l in ('C1','C2','F6','E1','E1-eval','B1','X1')}
print('tagged processes left:', {k: v for k, v in left.items() if v} or 'none')"
echo "ledger rows: $(wc -l < research/v2/ledger.jsonl) (expected 4); scratch dir: $T"
