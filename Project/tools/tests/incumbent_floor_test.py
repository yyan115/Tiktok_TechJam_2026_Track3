#!/usr/bin/env python3
"""The incumbent channel: where a win prediction's floor comes from.

Before this existed, ``_validate_prediction`` read ``incumbent_speedup`` out of
the profile artifact -- a file written by a worker inside the jail -- and the
controller's diagnostic request never carried one, so the metric was always
absent and every ``win`` card was refused.  The grind could not take its first
optimization step.

The fix was not to start supplying the number through the jail.  It was to read
it from the one place that already means "this is the champion":
``gate_state.groups[family|shape]["best_speedup"]``, written only by
``audit-finalize`` and only for a run that was controller-measured, correct,
clean and audited promotion-eligible.

So there are two properties to hold down, and the second matters more than the
first: the floor must exist, and it must not be nameable by the thing being
judged.  These tests are unit-level on purpose -- driving a real champion onto
the board through the CLI takes a bound audit, and the logic under test is
worth pinning directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "Project" / "tools"))

import run_gate  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    checks.append((name, bool(condition), detail))
    print(("PASS " if condition else "FAIL ") + name
          + (f"  [{detail}]" if detail and not condition else ""))


def refuses(fn, fragment: str) -> tuple[bool, str]:
    """Call fn and report whether it refused with `fragment` in the message."""
    try:
        fn()
    except run_gate.GateRefusal as exc:
        return fragment in str(exc), str(exc)
    except Exception as exc:  # a crash is not a refusal
        return False, f"{type(exc).__name__}: {exc}"
    return False, "no refusal raised"


def accepts(fn) -> tuple[float, str, str]:
    """Call fn expecting acceptance; a refusal reports as a failed check.

    A mutant that refuses where it should accept must show up as one FAIL
    line, not as a traceback that hides every check after it.
    """
    try:
        _, _, incumbent, source = fn()
    except Exception as exc:
        return float("nan"), "", f"{type(exc).__name__}: {exc}"
    return incumbent, source, ""


def groups(best: dict[str, float | None]) -> dict:
    """Build a gate-state ``groups`` map from gkey -> best_speedup."""
    return {"groups": {key: {"best_speedup": value, "strikes": 0,
                             "closed": False}
                       for key, value in best.items()}}


def profile(shape: int, reported=run_gate._MISSING) -> dict:
    """A counter-evidence profile record, optionally claiming an incumbent."""
    metrics = {"kernel_launches": 41}
    if reported is not run_gate._MISSING:
        metrics["incumbent_speedup"] = reported
    return {"shape": shape, "metrics": metrics}


NOISE = 0.005
CAMPAIGN = {"calibrations": {"3": {"noise": NOISE}, "5": {"noise": NOISE}}}

policy = run_gate.load_catalog()["prediction_policy"]
EFFECT = max(run_gate.IMPROVE_MARGIN - 1.0,
             policy["minimum_effect_noise_multiples"] * NOISE)


def main() -> int:
    # ---- where the number comes from ------------------------------------
    incumbent, source = run_gate._incumbent_speedup({}, 3)
    check("an empty board makes the baseline itself the incumbent",
          incumbent == 1.0 and "baseline" in source, f"{incumbent} {source}")

    incumbent, source = run_gate._incumbent_speedup(groups({"F-A|3": 1.09}), 3)
    check("a champion-eligible row becomes the incumbent",
          incumbent == 1.09 and "F-A|3" in source, f"{incumbent} {source}")

    incumbent, _ = run_gate._incumbent_speedup(
        groups({"F-A|3": 1.09, "F-B|3": 1.30, "F-C|3": None}), 3)
    check("the incumbent is the best on the shape, across every family",
          incumbent == 1.30, str(incumbent))

    incumbent, _ = run_gate._incumbent_speedup(groups({"F-A|5": 6.00}), 3)
    check("a champion on another shape does not raise this shape's floor",
          incumbent == 1.0, str(incumbent))

    incumbent, _ = run_gate._incumbent_speedup(groups({"F-A|3": None}), 3)
    check("a family with no champion-eligible row yet is skipped, not fatal",
          incumbent == 1.0, str(incumbent))

    # ---- malformed state refuses rather than guessing low ----------------
    # Every one of these, if it "recovered", would silently lower the floor.
    ok, detail = refuses(
        lambda: run_gate._incumbent_speedup({"groups": {"F-A-3": {}}}, 3),
        "malformed")
    check("a group key with no shape separator refuses", ok, detail)

    ok, detail = refuses(
        lambda: run_gate._incumbent_speedup({"groups": {"F-A|three": {}}}, 3),
        "no readable shape")
    check("a group key with an unreadable shape refuses", ok, detail)

    ok, detail = refuses(
        lambda: run_gate._incumbent_speedup({"groups": {"F-A|3": "1.09"}}, 3),
        "is malformed")
    check("a group that is not a record refuses", ok, detail)

    for bad in ("1.09", float("inf"), float("nan"), 0.0, -2.0, True):
        ok, detail = refuses(
            lambda bad=bad: run_gate._incumbent_speedup(
                {"groups": {"F-A|3": {"best_speedup": bad}}}, 3),
            "malformed best_speedup")
        check(f"best_speedup={bad!r} refuses instead of lowering the floor",
              ok, detail)

    ok, detail = refuses(
        lambda: run_gate._incumbent_speedup({"groups": ["F-A|3"]}, 3),
        "groups are malformed")
    check("a groups map that is not a map refuses", ok, detail)

    # ---- the floor actually binds ---------------------------------------
    def validate(st, kind, pmin, pmax, reported=run_gate._MISSING, shape=3):
        return run_gate._validate_prediction(
            CAMPAIGN, st, profile(shape, reported), kind, pmin, pmax)

    incumbent, source, err = accepts(lambda: validate({}, "win", 1.08, 1.10))
    check("on an empty board a win must beat the baseline by the effect floor",
          not err and incumbent == 1.0 and 1.08 > 1.0 * (1.0 + EFFECT),
          err or f"{incumbent} {source}")

    ok, detail = refuses(lambda: validate({}, "win", 1.0 + EFFECT, 1.05),
                         "must exclude the incumbent")
    check("a win exactly at the floor is refused (the bound is strict)",
          ok, detail)

    champion = groups({"F-A|3": 1.09})
    ok, detail = refuses(lambda: validate(champion, "win", 1.08, 1.10),
                         "must exclude the incumbent")
    check("a win that would not beat the standing champion is refused",
          ok, detail)
    check("the refusal shows the arithmetic, not just a verdict",
          "1.0900" in detail and "1.1227" in detail
          and f"{EFFECT:.4f}" in detail, detail)

    incumbent, source, err = accepts(
        lambda: validate(champion, "win", 1.13, 1.15))
    check("a win that clears the champion by the effect floor is accepted",
          not err and incumbent == 1.09 and "F-A|3" in source,
          err or f"{incumbent} {source}")

    incumbent, source, err = accepts(
        lambda: validate(champion, "characterization", 1.00, 1.04))
    check("characterization is not held to the win floor, but still reports it",
          not err and incumbent == 1.09, err or str(incumbent))

    # ---- the judged thing cannot name its own floor ----------------------
    # This is the whole point of the change. The profile artifact is written by
    # a worker in the jail; if the gate read the incumbent from it, a candidate
    # could pick the number it has to beat.
    ok, detail = refuses(
        lambda: validate(champion, "win", 1.08, 1.10, reported=0.001),
        "must exclude the incumbent")
    check("a profile claiming a tiny incumbent cannot lower the real floor",
          ok and "1.0900" in detail, detail)

    incumbent, _, err = accepts(
        lambda: validate({}, "win", 1.08, 1.10, reported=99.0))
    check("a profile claiming a huge incumbent cannot raise the real floor",
          not err and incumbent == 1.0, err or str(incumbent))

    incumbent, _, err = accepts(
        lambda: validate(champion, "win", 1.13, 1.15, reported=None))
    check("a profile with no incumbent metric at all no longer blocks a win",
          not err and incumbent == 1.09, err or str(incumbent))

    # ---- the band checks still run before the floor ----------------------
    ok, detail = refuses(lambda: validate({}, "win", 0.9, 1.8), "uninformative")
    check("an uninformative band is still refused first", ok, detail)

    ok, detail = refuses(
        lambda: run_gate._validate_prediction(
            {"calibrations": {}}, {}, profile(3), "win", 1.08, 1.10),
        "campaign-bound calibration is required")
    check("a shape with no campaign calibration still refuses", ok, detail)

    failed = [(n, d) for n, ok, d in checks if not ok]
    print(f"{len(checks) - len(failed)}/{len(checks)} passed"
          + (" — ALL GREEN" if not failed else f" — FAILURES: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
