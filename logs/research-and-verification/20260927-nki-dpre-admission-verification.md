# Independent prospective NKI admission review

**PASS_LOCAL_PROSPECTIVE_ADMISSION_ONLY** — no blocker found in exact diff `6df7fdb5c1834a8167233ec9eb76b829b0a3e433f45e11aad5c79b70ea7b95ab`. Reviewed 2026-09-27 09:50:46 UTC. This is a local review, not application, release, launch, full-run or submission authority. The sole Operator retains normal admission checks; no extra source/native gate is introduced.

The patch admits only the disabled trn22 native60 r5/r6 alternatives, one singleton role after the actually selected own MuonEq r3 or split-cooldown r5 predecessor. The native timing itself remains the first native test; no new compiler/simulator prerequisite is requested.

## Independent evidence

- Reran all four author test groups: PASS, 104 reconstructed packets and 312 guard comparisons. `git apply --check`: PASS.
- Own stdlib reference separately reconstructs all 104 packets (102 existing), authenticates 1,456 encoded members, and confirms all 312 original lifetime/idle/dispatch guard bytes unchanged. Both new clean committed snapshots and source-bundle hashes authenticate.
- Thirty-eight binding/rehash mutations refuse altered source, parent, review, predecessor, scope or guards; direct source-poison and review-parent-poison tests also refuse. Fifteen actual prospective stage-predicate cases prove the narrow two-key behavior and unchanged ordinary-arm behavior.
- All unchanged manual functions remain AST-identical. The common stage changes only the base predicate, preserving authentic own predecessor terminal/archive/postflight, controller, one-shot, under-lock idle/device/disk and transport/lifetime checks. Release adds reporting metadata only. Existing active bytes remained exactly pinned before/after.

## Administrative ancestry does not replace science

The two exact keys require queue `research_base.source_sha256=c239...` as the administrative adoption token. Independently, their immutable packet base/scientific base is `500f...`, source is `0bc97383...`, and source-review receipt is `7a750095...`. Source, review, snapshots and packet contents remain hash-bound. Queued500f or any other newly selected base fails the actual stage predicate, rather than silently retaining stale science; ordinary arms still compare queue base to packet base.

This protection relies on the existing controller contract updating the authoritative queue base when adopting a scientific parent. An unrecorded adoption or stale authoritative state is not detectable from prose and is not authorized here. No additional operational gate is proposed.

The new packets remain disabled, 60 steps, no evaluation, no full-run/BPB-adoption authorization. Same-host fresh500f comparison remains pending/null until authentic results exist. Native execution, speed, cold-accounted gain and quality remain UNKNOWN. The original source/CPU receipt is reused explicitly; this review is not new native evidence.

## Exact integration bindings

| File | Current SHA256 | Prospective SHA256 |
| --- | --- | --- |
| ramp-family/build.py | `9509b53eb9a1b95dd6414d920681fa87b89feac41f658c9c916d2d3763a94601` | `bd3c539b55f262e89a0581c1bd30dafae0a4430d8ccd89700824aa49a7b3e4fe` |
| operator-recovery/manual_job.py | `01717a02f94336167901772e0ce12d52bb69ad84aebf4c611be982fc9009d716` | `3def737a1ab36df6bc986eca3f0ac0288e9c59c84f4ec4de7c71b8aa750d984f` |
| ramp-family/release_root.py | `4dbf2f6fb593b2053fc0496e9f10b162c65d742f40b521e54193578e2c96cb52` | `a0352237267b5c24096e5373500c2fabbe615af20452c8e05988b8fc418aa1bf` |

Gate SHA256: `eeba7ce10c68be3c3d6ce971b38a613055816b7896d0c9d15245174972489676`.

Own reference `20260927-nki-dpre-admission-reference-verification.py`, SHA256 `aca4b6e1652b2e385d9aa54be3e22a49c7d8afbd73e47dfd240cb05a837f11b1`, final PASS in 1.672 seconds. Two initial verifier-fixture issues (newline restoration and missing wrapper name) were corrected without changing implementation files. No host, active role/queue/release, native/compiler/simulator or training-source action occurred. Full structured evidence is in the adjacent JSON.
