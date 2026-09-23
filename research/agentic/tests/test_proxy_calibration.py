from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxy_calibration import collect, markdown, summarize  # noqa: E402


class ProxyCalibrationTests(unittest.TestCase):
    def _experiment(self, root: Path, name: str, smoke: float, full: float, *, cls: str | None = None):
        directory = root / name
        directory.mkdir()
        proposal = {
            "hypothesis": "BF16 kernel throughput boundary improves useful fixed-budget steps.",
            "single_causal_change": "Cast one kernel input to BF16.",
            "flags": [],
        }
        if cls:
            proposal["mechanism_class"] = cls
        (directory / "proposal.json").write_text(json.dumps(proposal))
        (directory / "smoke-verdict.json").write_text(json.dumps({
            "decision": "KEEP", "metrics": {"bpb_delta": smoke},
        }))
        (directory / "full-verdict.json").write_text(json.dumps({
            "decision": "KEEP" if full < 0 else "DISCARD", "metrics": {"bpb_delta": full},
        }))

    def test_collect_joins_only_complete_pairs_and_marks_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._experiment(root, "bf16-a", -0.1, -0.2)
            incomplete = root / "b"
            incomplete.mkdir()
            (incomplete / "proposal.json").write_text("{}")
            rows = collect(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["mechanism_class"], "throughput")
            self.assertTrue(rows[0]["mechanism_class_inferred"])

    def test_summary_reports_sign_agreement_and_spearman(self):
        rows = [
            {"proposal_id": "a", "mechanism_class": "schedule", "mechanism_class_inferred": False,
             "smoke_bpb_delta": -0.3, "full_bpb_delta": -0.2, "full_comparable": True,
             "smoke_decision": "KEEP", "full_decision": "KEEP"},
            {"proposal_id": "b", "mechanism_class": "schedule", "mechanism_class_inferred": False,
             "smoke_bpb_delta": -0.1, "full_bpb_delta": 0.1, "full_comparable": True,
             "smoke_decision": "KEEP", "full_decision": "DISCARD"},
        ]
        report = summarize(rows)
        self.assertEqual(report["summary"]["schedule"]["sign_agreement"], 0.5)
        self.assertAlmostEqual(report["summary"]["schedule"]["spearman_rho"], 1.0)
        self.assertEqual(report["summary"]["schedule"]["routing_evidence"], "insufficient_evidence")
        self.assertIn("50.0%", markdown(report))


if __name__ == "__main__":
    unittest.main()
