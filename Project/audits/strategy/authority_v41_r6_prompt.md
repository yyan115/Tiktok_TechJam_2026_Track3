FINAL round (6) of an adversarial review of my benchmark project's control
system. Fresh eyes each round. Trajectory: R1 REVISE(8) -> R2 REVISE(7) ->
R3 REVISE(5) -> R4 REVISE(6) -> R5 REVISE(5, at TOCTOU/doc-wording
caliber; confirmed all R4 changes). R5 fixes landed at commit d7616ca.
GATE_DESIGN.md Amendments v4.2-v4.6 record every adoption;
OWNER_PATCH_card_gate.md "Deliberate decisions" records ratified
non-adoptions and middles, each with reasoning.

Adopted from round 5: candidate bytes hashed once per issuance (shared by
eligibility checks and the permit binding); the ever-crowned audit backlog
is prospective (every champion persisted before launch) and the file is
write-protected; the guard matches audit directory NODES, not only
descendants (moving the whole packets/strategy/auto directory bounces);
JUDGE_ERROR/TIMEOUT verdict rows no longer suppress refiring — failed
judgments retry, capped at 3 attempts then park for the owner; the parser
documentation was corrected to match the implementation (exactly one
verdict-bearing document; non-verdict JSON documents are tolerated because
audit responses quote JSON evidence); re-auditing the six pre-binding
champion receipts is queued as runtime work.

This is the closing round. Verify the round-5 fixes, then render a
judgment on the WHOLE system's readiness for installation. The standard:
are there remaining holes through which the automated pipeline can grant
itself authority (run without a permit, judge its own results, erase or
suppress adverse findings, or launder evidence) ABOVE the documented,
ratified residual level of this single-user machine? Polish, hypothetical
sub-second races requiring deliberate self-attack, and re-arguments of
ratified decisions do not count.

Files: Project/tools/run_gate.py, Project/tools/audit_champion.py,
Project/tools/champion_watch.py, Project/tools/tests/,
Project/loop/OWNER_PATCH_card_gate.md, Project/loop/GATE_DESIGN.md.
Run the committed suites if your sandbox allows.

If the standard above is met, say APPROVE — installation proof-tests are
the final live check. If you find a genuine authority hole above the
residual line, list it as a numbered REQUIRED change. Finish with exactly
one final line: VERDICT: APPROVE or VERDICT: REVISE
