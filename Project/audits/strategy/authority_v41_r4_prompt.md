Round 4 of an adversarial review of my benchmark project's control system.
Fresh eyes each round. Trajectory: round 1 REVISE (8 items), round 2
REVISE (7 — confirmed round-1 fixes present and the suite green), round 3
REVISE (5 — confirmed round-2 fixes on nominal paths; all items
implementation-layer). Round-3 fixes landed at commit 48c1380.
Project/loop/GATE_DESIGN.md Amendments v4.2-v4.4 record what was adopted;
OWNER_PATCH_card_gate.md "Deliberate decisions" records what I ratified as
deliberate design with reasoning.

Adopted from round 3: audit verdict recording now happens only under the
shared gate lock with no attempt in flight (no post-verdict run window);
the watcher's launch-time cache is replaced by handled-ness DERIVED from
durable verdict rows plus live process markers with stale cleanup and
exclusive-create claims (the 24-entry suppressed-audit backlog this
exposed is queued to refire); response artifacts are unique per attempt;
the verdict is parsed from one complete JSON document with a token
cross-check; sed and less are out of the guard's read allowance; other
command segments may not reference the bound run-record by name or by a
--ledger token; guard and parser regression tests are committed, with the
guard tested from the owner patch's own extracted bytes (30/30) alongside
the CLI suite (37/37).

Deliberately not adopted across rounds (do not re-argue unless UNSOUND,
not merely improvable — reasoning is written down): critic-authorized
reopening of closed research directions; agent-writable cards; local
timestamp ordering with tamper brakes; the frozen benchmark runner staying
unmodified (cooperative hook enforcement, documented honestly); text
provenance on one machine being detectable rather than preventable;
screening on scratch run-records (segment guard + falsifiable ranges +
strike mechanics; screening can never crown a champion).

Your job: verify the round-3 fixes and hunt for remaining AUTHORITY holes
(not polish). Run the two committed suites if your sandbox allows (they
write only to temp dirs). Files: Project/tools/run_gate.py,
Project/tools/audit_champion.py, Project/tools/champion_watch.py,
Project/tools/tests/, Project/loop/OWNER_PATCH_card_gate.md,
Project/loop/GATE_DESIGN.md.

Judge maturity honestly: three rounds have converged 8 -> 7 -> 5 with each
round confirming the previous fixes. If what remains is polish or accepted
residuals rather than authority holes, say APPROVE — the live proof-tests
at installation are the final check, and endless revision now costs more
than it protects. Finish with numbered REQUIRED changes (if any), then
exactly one final line: VERDICT: APPROVE or VERDICT: REVISE
