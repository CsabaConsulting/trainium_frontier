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
        self.assertIn("752ea5e00d020e1f2e601ee633448bf5ea52f522", audit)
        self.assertIn("c30422e2a9f0ed7203c85597eef1cb78c9480b62", audit)
        self.assertIn("dd4c05002ef269a0fae292ae90bf60cda5182104", audit)
        self.assertIn("d5823b32a299109c18b5fbe4bf46e96879628caee8e02138e56005b8a98dc40f", audit)
        self.assertIn("ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb", audit)
        self.assertIn("c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586", audit)
        self.assertIn("3c225d9acb7a835939eeda36588851c6ff062df8adb4ab8e6dacc529caf642f1", audit)
        self.assertIn("`workflow/candidates/S0/train.py`", audit)
        self.assertIn("`workflow/proposals/research-v2/train.py`", audit)
        self.assertIn("`workflow/tools/test_run_job.sh`", audit)
        self.assertIn("### ADOPT", audit)
        self.assertIn("### PATCH", audit)
        self.assertIn("### REJECT", audit)
        self.assertIn("`116188cf` source remains current", audit)
        self.assertIn("Neuron lockstep, two cold comparable public20M runs", audit)
        self.assertIn("fresh-tail is the only new hypothesis", audit)
        self.assertIn("table-freeze/epoch ablation frozen", audit)
        self.assertIn("private IMPL-1/research-v2 implementation", audit)
        self.assertIn("A2-before-A1 ordering", audit)
        self.assertIn("an externally authorized, digest-bound request and exact schema", audit)
        self.assertIn("a durable host lock", audit)
        self.assertIn("no-follow safe-root path confinement", audit)
        self.assertIn("causality gate (`None` can pass)", audit)
        self.assertIn("escaped-descendant handling", audit)
        self.assertIn("safe archive/delete ordering", audit)
        self.assertIn("`run_job.py`", audit)

        workflow = (ROOT / "research/agentic/DETERMINISTIC_WORKFLOW_V2.md").read_text()
        self.assertIn("752ea5e00d020e1f2e601ee633448bf5ea52f522", workflow)
        self.assertIn("grants it no canonical authority", workflow)

    def test_proposal_template_exposes_research_metadata(self):
        template = json.loads((ROOT / "research/agentic/EXPERIMENT_PROPOSAL.json").read_text())
        for field in (
            "p_improve", "estimated_host_hours", "trainium_feasibility",
            "cheapest_falsifier", "prior_result_refs",
        ):
            self.assertIn(field, template)


if __name__ == "__main__":
    unittest.main()
