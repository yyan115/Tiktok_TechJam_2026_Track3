#!/usr/bin/env python3
"""Owner batch: retire the pre-citation-gate RULE_VIOLATIONs that freeze permits.

WHY THIS EXISTS
---------------
16 RULE_VIOLATION verdicts were recorded on 30 Aug 07:48-08:05 by the auto-audit
sweeping the pre-gate board.  All of them say the same thing: the run predates the
citation gate, so it has no plan, no quoted source and no reasoning chain to verify.
The auditors said the GPU numbers themselves looked credible and that the fix is to
rerun under a genuine pre-run cited plan -- never to attach citations retroactively.

That is already the plan (HANDOVER 3.1: the whole board is re-measured).  But the
gate treats ANY unresolved hard verdict as a brake on every new permit, so until
these are resolved the first optimization card cannot issue.  The findings are not
in dispute.  What is being removed is their grip on new work.

WHAT DOES **NOT** WORK, AND WHY THIS SCRIPT EXISTS AT ALL
--------------------------------------------------------
``run_gate.py verdict-clear --kind violation`` looks like the unlock.  Its own
docstring says so.  It consumes a real signed owner capability, prints
"RULE_VIOLATION <key> resolved by controller-verified owner authority", and exits 0.

It does not lift the brake.

Rehearsed on a full copy of this repo with a real LOCK: 16 cleared, 16 still
outstanding afterwards.  ``verdict-clear`` appends to ``gate_state.cleared_verdicts``,
which ``unacked_hard_verdicts`` explicitly treats as "legacy display state only ...
cannot clear the prospective audit authority's first-write-wins hard event".  The
real brake is a ``resolution`` event in the AUDIT AUTHORITY journal.  An owner who
follows the documented path spends 16 signatures and stays frozen.

THE PATH THAT WORKS
-------------------
Per verdict: ``authorize`` an ``audit.resolve`` capability against the AUDIT EVENT
hash, then ``audit_authority.record_resolution``.  One capability covers all of them
-- ``verify_capability`` accepts an ``audit:*`` wildcard while the journal still
records the specific ``audit:<entry_id>`` each use was spent on.

So it is ONE owner signature, not 16.

Resolution kind is ``FINDING_ACCEPTED_ROW_RETIRED``: the finding stands, the row is
withdrawn from contention.  Deliberately not ``FINDING_OVERTURNED`` -- that asserts
the auditor was wrong, and the auditor was right.

HOW TO RUN IT
-------------
After LOCK is activated (this needs a live controller), from the repo root:

  1. python3 Project/tools/owner_lock_ceremony.py mint-capability \
         --action audit.resolve --target 'audit:*' --campaign <campaign-id> \
         --reason "retire pre-citation-gate rows so the post-LOCK board can be measured" \
         --max-uses 25 --expires-minutes 60 \
         --private-key <your owner key outside the repo> \
         --out /tmp/cap_audit.json

  2. python3 Project/tools/clear_pregate_verdicts.py \
         --capability /tmp/cap_audit.json --campaign <campaign-id>      # dry run

  3. same command with --yes                                            # does it

  4. shred /tmp/cap_audit.json

The dry run lists every verdict it would retire and touches nothing.  The real run
stops at the first failure rather than half-finishing.  Re-running is safe: an
already-resolved verdict is skipped, and the audit authority refuses a second
resolution against the same target anyway.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Project" / "tools"))
sys.path.insert(0, str(ROOT / "Project" / "harness"))

RESOLUTION_KIND = "FINDING_ACCEPTED_ROW_RETIRED"
RATIONALE = (
    "Finding accepted, not overturned. The 30 Aug audit correctly found this row has "
    "no plan, quoted source or reasoning chain, because the run predates the citation "
    "gate that requires them. The row is retired as promotion-ineligible and is not "
    "promoted on any board; its shape is re-measured from scratch under the post-LOCK "
    "campaign (HANDOVER 3.1). This resolution removes the verdict's brake on NEW "
    "permits. It does not rehabilitate the row and makes no claim about its numbers."
)


def outstanding() -> list[dict]:
    """Hard RULE_VIOLATIONs still braking new permits, newest state each call."""
    import run_gate as g
    state = g.load_json(g.STATE, {})
    state.setdefault("cleared_verdicts", [])
    return [h for h in g.unacked_hard_verdicts(state)
            if h.get("verdict") == "RULE_VIOLATION" and h.get("_audit_event_sha256")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capability", required=True,
                    help="signed audit.resolve capability from mint-capability")
    ap.add_argument("--campaign", required=True,
                    help="campaign id the capability was minted against")
    ap.add_argument("--yes", action="store_true",
                    help="actually write resolutions (without this it is a dry run)")
    args = ap.parse_args()

    import audit_authority as aa

    capability = Path(args.capability).resolve()
    if not capability.is_file():
        print(f"REFUSED: capability file not found: {capability}")
        return 1

    rows = outstanding()
    if not rows:
        print("Nothing to do: no RULE_VIOLATION is braking new permits.")
        return 0

    print(f"{len(rows)} verdict(s) currently brake every new permit:\n")
    for row in rows:
        print(f"  {row['entry_id']}  recorded {row.get('recorded')}"
              f"  audit event {row['_audit_event_sha256'][:16]}")
    print(f"\nresolution kind : {RESOLUTION_KIND}")
    print(f"rationale       : {RATIONALE[:80]}...")

    if not args.yes:
        print("\nDRY RUN — nothing was written. Re-run with --yes to do it.")
        return 0

    print()
    done = 0
    for row in rows:
        entry_id = row["entry_id"]
        target_sha = row["_audit_event_sha256"]
        proc = subprocess.run(
            [sys.executable, str(ROOT / "Project/harness/trusted_controller.py"),
             "authorize", "--capability", str(capability),
             "--action", "audit.resolve", "--target", f"audit:{entry_id}",
             "--subject-sha256", target_sha, "--campaign", args.campaign],
            cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            detail = (proc.stdout + proc.stderr).strip()[:400]
            print(f"STOPPED at {entry_id}: controller refused the authorization.\n"
                  f"  {detail}\n"
                  f"  {done} of {len(rows)} were resolved before this. Re-running "
                  f"after you fix the cause will pick up the rest.")
            return 1
        receipt = json.loads(proc.stdout)
        try:
            aa.record_resolution(
                entry_id=entry_id, target_event_sha256=target_sha,
                resolution_kind=RESOLUTION_KIND, rationale=RATIONALE,
                authority_event_id=receipt["authority_event_id"],
                capability_nonce=receipt["capability_nonce"])
        except Exception as exc:
            print(f"STOPPED at {entry_id}: the audit authority refused the "
                  f"resolution.\n  {type(exc).__name__}: {str(exc)[:400]}\n"
                  f"  A capability use was spent on this one. {done} of {len(rows)} "
                  f"were resolved before it.")
            return 1
        done += 1
        print(f"  retired {entry_id}")

    left = outstanding()
    print(f"\n{done} retired. RULE_VIOLATIONs still braking permits: {len(left)}")
    if left:
        print("  Not clean yet — these remain:")
        for row in left:
            print(f"    {row['entry_id']}")
        return 1
    print("  The permit brake is off. Shred the capability file now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
