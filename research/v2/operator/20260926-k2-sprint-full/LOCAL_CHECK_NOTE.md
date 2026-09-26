The initial local mechanical checker stopped with `FULL_FOLLOWUP_MECHANICAL_CHECK_FAILED`
at function-AST equality. The exact diff was only the two mandatory embedded hashes in
`preflight`: launch-command.txt and provenance.json changed to the authenticated new
package pins. No remote call occurred and no packet/scientific/driver logic was changed.

The checker now normalizes precisely those two old→new authenticated hash literals
before comparing function ASTs; each old pin must appear exactly once. This is a local
checker-scope correction, not an ignored native or scientific test failure.
