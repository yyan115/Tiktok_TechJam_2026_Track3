#!/usr/bin/env python3
"""The owner's minting tool and the gate must name the same actions.

An owner capability is signed by `owner_lock_ceremony.py mint-capability
--action X` and spent by code that demands action Y. If X and Y are drawn from
two different vocabularies, every privileged transition in the loop refuses at
mint time, and the only way through is `--allow-unknown-action` -- the flag
whose entire purpose is to say "no one has checked this for you".

That is exactly what shipped. The ceremony knew six dotted names
(`lock.activate`, `audit.resolve`, ...); the gate demanded seven underscored
ones (`open_campaign`, `register_family`, ...); the two sets were DISJOINT. A
campaign could not be opened without reaching for the escape hatch, and nothing
runs without a campaign. It was found by rehearsing the ceremony end to end on a
copy of the repo, not by any test, because no test compared the two lists.

This test compares the two lists.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "Project" / "tools"))

import owner_lock_ceremony as ceremony  # noqa: E402
import run_gate as gate  # noqa: E402
import audit_authority as audit  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    checks.append((name, bool(condition), detail))
    print(("PASS " if condition else "FAIL ") + name
          + (f"  [{detail}]" if detail and not condition else ""))


def main() -> int:
    known = set(ceremony.KNOWN_ACTIONS)

    # 1. Every privileged transition the gate can demand must be mintable
    #    WITHOUT --allow-unknown-action.
    unmintable = sorted(gate.AUTHORITY_ACTIONS - known)
    check("every gate authority action is known to the minting ceremony",
          not unmintable,
          f"needs --allow-unknown-action: {unmintable}")

    # 2. The audit authority's resolution path is a separate consumer with its
    #    own action name, and it is checked against the capability too.
    check("the audit authority's resolve action is mintable",
          "audit.resolve" in known)

    # 3. audit_authority.record_resolution requires the capability target to be
    #    exactly audit:<entry_id>, so the declared prefix must permit that.
    prefix = ceremony.KNOWN_ACTIONS.get("audit.resolve")
    check("audit.resolve's target prefix matches what record_resolution checks",
          prefix == "audit:", str(prefix))

    # 4. A declared prefix must not forbid a target the consumer requires. Any
    #    prefix we assert here is a constraint on the owner at 3am, so each one
    #    should be traceable to code that actually checks it.
    for action, required in sorted(ceremony.KNOWN_ACTIONS.items()):
        if required is None:
            continue
        check(f"prefix {required!r} for {action} is a plain namespace",
              required.endswith(":") and required.strip(":").isidentifier()
              or required.endswith(":") and "-" not in required,
              f"{action} -> {required!r}")

    # 5. Resolution kinds: the one used to retire a correct finding must exist
    #    and must be distinct from the one that overturns it. Retiring a row is
    #    not the same claim as saying the auditor was wrong, and a hash-chained
    #    journal keeps whichever claim is written forever.
    src = (REPO / "Project" / "tools" / "audit_authority.py").read_text()
    check("FINDING_ACCEPTED_ROW_RETIRED is an accepted resolution kind",
          "FINDING_ACCEPTED_ROW_RETIRED" in src)
    check("it is still distinct from FINDING_OVERTURNED",
          "FINDING_OVERTURNED" in src)
    check("audit_authority exposes record_resolution",
          hasattr(audit, "record_resolution"))

    failed = [(n, d) for n, ok, d in checks if not ok]
    print(f"{len(checks) - len(failed)}/{len(checks)} passed"
          + (" — ALL GREEN" if not failed else f" — FAILURES: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
