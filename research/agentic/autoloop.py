#!/usr/bin/env python3
"""CLI for the unattended flag-only experiment loop. See research/agentic/AUTOLOOP.md.

Subcommands:
  status      local readiness gates (no remote call)
  preflight   read-only remote inventory of every configured box
  once        one controller cycle
  daemon      repeat cycles until STOP/breaker/caps
  dry-run     build and print the manifest the loop would launch next, without any remote call
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autoloop import config as C  # noqa: E402
from autoloop.archive import GitArchiver  # noqa: E402
from autoloop.controller import Controller  # noqa: E402
from autoloop.manifest import build_manifest  # noqa: E402
from autoloop.models import BedrockClient, ModelError  # noqa: E402
from autoloop.remote import LocalGit, RemoteError, RemoteOperator, SSHTransport, check_key_path  # noqa: E402
from autoloop.queue_plan import FixedQueuePlan  # noqa: E402
from autoloop.state import Store  # noqa: E402


def _build(cfg: C.LoopConfig, *, need_model: bool):
    store = Store()
    transport = SSHTransport()
    local_git = LocalGit()
    operator = RemoteOperator(transport, local_git, cfg, store.controller_id)
    model = None
    if need_model:
        try:
            model = BedrockClient(cfg)
        except ModelError as exc:
            print(f"model client unavailable: {exc}", file=sys.stderr)
    archiver = GitArchiver(push=cfg.git_push, remote=cfg.git_remote)
    return Controller(cfg=cfg, store=store, operator=operator, model=model, archiver=archiver, local_git=local_git)


def status(_a) -> int:
    cfg = C.load_config()
    rows = []
    errs = cfg.validate()
    rows.append(("config", not errs, "; ".join(errs) or f"shape lnc={cfg.lnc} nproc={cfg.nproc_per_node}"))
    try:
        check_key_path()
        rows.append(("ssh_key", True, f"{C.SSH_KEY_PATH} mode 600 (contents never read)"))
    except RemoteError as exc:
        rows.append(("ssh_key", False, str(exc)))
    rows.append(("stop_absent", not C.STOP_FILE.exists(), str(C.STOP_FILE)))
    lg = LocalGit()
    rows.append(("authoritative_commit_local", lg.commit_exists(C.AUTHORITATIVE_COMMIT), C.AUTHORITATIVE_COMMIT))
    try:
        store = Store()
        b = store.breaker()
        rows.append(("breaker_closed", b is None, json.dumps(b) if b else "closed"))
        rows.append(("runs_total", True, str(store.state["runs_total"])))
        for pid, rec in store.proposals().items():
            rows.append((f"proposal:{pid}", True, f"{rec['stage']} verdict={rec.get('verdict')} box={rec.get('box')}"))
    except RuntimeError as exc:
        rows.append(("state", False, str(exc)))
    ok_all = True
    for name, ok, detail in rows:
        ok_all &= ok
        print(f"{'OK' if ok else 'BLOCKED':8} {name}: {detail}")
    return 0 if ok_all else 2


def preflight(_a) -> int:
    cfg = C.load_config()
    ctl = _build(cfg, need_model=False)
    rc = 0
    for box in cfg.boxes:
        try:
            inv = ctl.op.inventory(box)
            print(json.dumps(inv.to_dict(), indent=2))
            rc |= 0 if inv.clean_and_idle else 2
        except RemoteError as exc:
            print(f"{box}: {exc}", file=sys.stderr)
            rc |= 1
    return rc


def once(_a) -> int:
    cfg = C.load_config()
    ctl = _build(cfg, need_model=True)
    rep = ctl.cycle()
    print(json.dumps(rep.to_dict(), indent=2))
    return 3 if rep.stopped_reason else 0


def daemon(a) -> int:
    cfg = C.load_config()
    ctl = _build(cfg, need_model=True)
    return ctl.daemon(max_cycles=a.max_cycles)


def dry_run(a) -> int:
    cfg = C.load_config()
    errs = cfg.validate()
    if errs:
        print("\n".join(errs), file=sys.stderr)
        return 2
    if cfg.plan_mode == "fixed_full_only":
        if a.tier != "full" or a.flags:
            raise ValueError("fixed_full_only dry-run requires --tier full and no flags")
        entry = FixedQueuePlan.load().entries[0]
        m = build_manifest(
            proposal_id=entry.proposal_id, tier="full", box=entry.box,
            remote_run_dir=f"{C.REMOTE_RUNS_DIR}/{entry.commit[:8]}--{entry.proposal_id}--al-20260923-fixed-dry-run-full-r1",
            label="al-20260923-fixed-dry-run-full-r1", train_flags=[], cfg=cfg,
            frozen_at="dry-run", control_label=None, abort_conditions=[],
            description_note="fixed queue exact control dry run", candidate_commit=entry.commit,
            execution_mode="exact-dense1024-cold-eval", comparison_role="control",
            expected_train_sha256=entry.train_sha256,
        )
    else:
        flags = a.flags or ["--weight-ema"]
        m = build_manifest(
            proposal_id="20260101-dry-run", tier=a.tier, box=cfg.boxes[0],
            remote_run_dir=f"{C.REMOTE_RUNS_DIR}/{C.AUTHORITATIVE_COMMIT[:8]}--20260101-dry-run--al-20260101-dry-run-{a.tier}-r1",
            label=f"al-20260101-dry-run-{a.tier}-r1", train_flags=flags, cfg=cfg, frozen_at="dry-run",
            control_label=None, abort_conditions=[], description_note="dry run",
        )
    print(json.dumps(m.to_dict(), indent=2))
    return 0


def clear_fixed_breaker(a) -> int:
    """Operator-only local state transition; this command performs no remote action."""
    cfg = C.load_config()
    if cfg.plan_mode != "fixed_full_only":
        raise ValueError("breaker receipt transition is only valid in fixed_full_only mode")
    store = Store()
    store.clear_breaker_for_fixed_plan(
        receipt_commit=a.receipt_commit, receipt_path=a.receipt_path,
        receipt_sha256=a.receipt_sha256, cleared_by_role=a.cleared_by_role,
    )
    print("fixed-plan breaker cleared with pinned Operator receipt; fresh preflight still required")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(func=status)
    sub.add_parser("preflight").set_defaults(func=preflight)
    sub.add_parser("once").set_defaults(func=once)
    d = sub.add_parser("daemon")
    d.add_argument("--max-cycles", type=int, default=0)
    d.set_defaults(func=daemon)
    dr = sub.add_parser("dry-run")
    dr.add_argument("--tier", choices=["smoke", "full"], default="smoke")
    dr.add_argument("flags", nargs="*", default=[])
    dr.set_defaults(func=dry_run)
    cb = sub.add_parser("clear-fixed-breaker")
    cb.add_argument("--receipt-commit", required=True)
    cb.add_argument("--receipt-path", required=True)
    cb.add_argument("--receipt-sha256", required=True)
    cb.add_argument("--cleared-by-role", required=True)
    cb.set_defaults(func=clear_fixed_breaker)
    args = p.parse_args()
    try:
        raise SystemExit(args.func(args))
    except (RemoteError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
