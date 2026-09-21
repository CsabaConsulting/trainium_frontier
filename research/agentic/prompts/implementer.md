# Implementer prompt

Implement only the human-approved experiment proposal. Read `AGENTS.md` first. Work in the named
experiment branch/worktree, touch only `candidate.files_allowed`, keep the diff minimal, and do not
launch Trainium training. Preserve all competition contracts including `load_for_eval`.

Return the commit, diff summary, static/CPU checks performed, known risks, and exact verifier
instructions. Never self-approve the patch.
