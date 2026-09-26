# Disabled clean cold-five r2: diagnosed root-only repair

Prepared from Supervisor e80692e2. Scientific source, argv, harness, guard, dispatcher and driver control flow are unchanged. The entire prior driver is transformed only by exact root and label literals. r1 failure 56b5220b remains immutable.

The static regression compiles the actual unchanged harness containment assertion AST: old sandbox fails; corrected sandbox under the actual harness project/out passes. Future transport also resolves and enforces that predicate on the host before creating the root. This is not a containment waiver.

77 local mechanical checks pass under normal Python and -O; no source execution, tensors, host operations or launch authority. One diagnosed retry only after separate scoped release. Explicit postflight is mandatory on every outcome; no automatic full or eval.
