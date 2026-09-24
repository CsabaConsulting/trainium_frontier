import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class DeterministicWorkflowV2Tests(unittest.TestCase):
    def test_root_contract_is_additive_and_points_to_workflow(self):
        contract = (ROOT / "AGENTS.md").read_text()
        self.assertIn("# Trainium Frontier agent contract", contract)
        self.assertIn("## Non-negotiable rules", contract)
        self.assertIn("## Deterministic workflow v2", contract)
        self.assertIn("research/agentic/DETERMINISTIC_WORKFLOW_V2.md", contract)

    def test_workflow_has_one_state_and_selection_authority(self):
        workflow = (ROOT / "research/agentic/DETERMINISTIC_WORKFLOW_V2.md").read_text()
        self.assertIn("Do not create a parallel ledger, queue, state file", workflow)
        self.assertIn("autoloop/selection.py` is the sole advisory deterministic engine", workflow)
        self.assertIn("fresh 2M screen can only nominate", workflow)
        self.assertIn("full20M", workflow)
        self.assertIn("“Porter” is a duty, not a seventh authority", workflow)

    def test_audit_pins_source_and_rejects_untrusted_runtime_imports(self):
        audit = (
            ROOT / "research/agentic/imports/claude-workflow-d08ea46/AUDIT.md"
        ).read_text()
        self.assertIn("d08ea46dd8135a6ceafa1b0856e5a2f0fc1fe474", audit)
        self.assertIn("d5823b32a299109c18b5fbe4bf46e96879628caee8e02138e56005b8a98dc40f", audit)
        self.assertIn("### ADOPT", audit)
        self.assertIn("### PATCH", audit)
        self.assertIn("### REJECT", audit)
        self.assertIn("Importing `workflow/tools/decide.py`, `preflight_dryrun.py`, or `contained.py`", audit)

    def test_proposal_template_exposes_research_metadata(self):
        template = json.loads((ROOT / "research/agentic/EXPERIMENT_PROPOSAL.json").read_text())
        for field in (
            "p_improve", "estimated_host_hours", "trainium_feasibility",
            "cheapest_falsifier", "prior_result_refs",
        ):
            self.assertIn(field, template)


if __name__ == "__main__":
    unittest.main()
