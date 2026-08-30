#!/usr/bin/env python3
"""The gate's CLI surface contract -- and the anti-shim rule that this file broke.

WHAT THIS FILE USED TO BE
  Three lines: ``from competence_gate_test import main``.  Running it re-ran
  another suite under a second name, so "37/37 + 37/37" in this project's commit
  history was one suite of 37 counted twice.  A pass-through test file is worse
  than no test file: it manufactures confidence out of nothing.  It is replaced
  here rather than deleted because the post-LOCK allowlist
  (Project/lock_staging/guard_bash.py) names this path as a runnable entrypoint,
  and that file is owner-pasted, not agent-editable.

WHAT IT IS NOW
  A real suite, in the lane its name claims: the run_gate.py command-line
  surface, which nothing else covers.  competence_gate_test.py exercises gate
  POLICY through the CLI; this exercises the CLI ITSELF -- that every advertised
  subcommand is reachable, that a bad invocation refuses instead of crashing,
  and that a stack trace never substitutes for a refusal.  Plus one rule aimed
  squarely at the defect above: no suite in this directory may be a pass-through
  for another suite's main.

  The deepest gate/controller integration lives in integration_authority_test.py.
  This file deliberately does not repeat it.

Run: python3 Project/tools/tests/gate_v4_cli_test.py   (no GPU, no network,
writes only under a temporary directory).
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TESTS = Path(__file__).resolve().parent
GATE_SRC = REPO / "Project" / "tools" / "run_gate.py"
CATALOG = REPO / "Project" / "loop" / "mechanism_catalog.json"
CATALOG_SCHEMA = REPO / "Project" / "loop" / "mechanism_catalog.schema.json"
AUDIT_AUTHORITY = REPO / "Project" / "tools" / "audit_authority.py"
AUTHORITY = REPO / "Project" / "harness" / "authority.py"
CONTROLLER = REPO / "Project" / "harness" / "trusted_controller.py"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    print(("PASS " if condition else "FAIL ") + name
          + (f"  [{detail}]" if detail and not condition else ""))


def build() -> Path:
    """A tree the gate can start in, with no state of its own yet."""
    root = Path(tempfile.mkdtemp(prefix="gate_cli_"))
    for relative in ("Project/tools", "Project/harness", "Project/loop",
                     "Project/research", "Project/results", "Project/audits",
                     "Project/authority/blobs", "Project/submission"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(GATE_SRC, root / "Project/tools/run_gate.py")
    shutil.copyfile(CATALOG, root / "Project/loop/mechanism_catalog.json")
    shutil.copyfile(CATALOG_SCHEMA,
                    root / "Project/loop/mechanism_catalog.schema.json")
    shutil.copyfile(AUDIT_AUTHORITY, root / "Project/tools/audit_authority.py")
    shutil.copyfile(AUTHORITY, root / "Project/harness/authority.py")
    shutil.copyfile(CONTROLLER, root / "Project/harness/trusted_controller.py")
    (root / "Project/results/JOURNAL.jsonl").write_text("")
    (root / "Project/audits/verdicts.jsonl").write_text("")
    # The research index is not optional furniture: the gate reads it on every
    # invocation, so a tree without it tests file-not-found, not the CLI.
    (root / "Project/research/INDEX.md").write_text("cli test index\n")
    (root / "Project/research/note-a.md").write_text("note a\n")
    return root


def gate_runner(root: Path):
    def run(*args: str) -> tuple[int, str, str]:
        process = subprocess.run(
            [sys.executable, str(root / "Project/tools/run_gate.py"), *args],
            cwd=str(root), text=True, capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        return process.returncode, process.stdout, process.stderr

    return run


def gate_subcommands(path: Path) -> tuple[set[str], set[str]]:
    """Every subcommand the parser advertises, and every one main() dispatches.

    Two lists in one file that must be the same list: a subcommand added to the
    parser but not to the dispatch table parses fine and then raises KeyError at
    the moment an operator uses it.
    """
    tree = ast.parse(path.read_text())
    advertised: set[str] = set()
    dispatched: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_parser"
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            advertised.add(node.args[0].value)
        if isinstance(node, ast.Dict):
            keys = [key.value for key in node.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)]
            if {"reconcile", "status"} <= set(keys):
                dispatched |= set(keys)
    return advertised, dispatched


def passthrough_suites(directory: Path) -> list[str]:
    """Test files whose real content is another suite's main().

    ``from <other>_test import main`` at module level is the exact shape of the
    defect this file used to be.  It is banned outright: a suite that re-runs a
    sibling adds no coverage and inflates every count that quotes it.
    """
    offenders: list[str] = []
    for path in sorted(directory.glob("*_test.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.endswith("_test")
                    and node.module != path.stem
                    and any(alias.name == "main" for alias in node.names)):
                offenders.append(f"{path.name} -> {node.module}")
    return offenders


def main() -> int:
    trees = [build(), build()]
    run = gate_runner(trees[0])

    print("gate_v4_cli_test — run_gate.py command-line surface\n")

    # -- the parser and the dispatch table are one list --------------------
    advertised, dispatched = gate_subcommands(GATE_SRC)
    check("the gate advertises subcommands", bool(advertised),
          repr(sorted(advertised)))
    check("every advertised subcommand is dispatched",
          not advertised - dispatched,
          f"advertised only: {sorted(advertised - dispatched)}")
    check("every dispatched subcommand is advertised",
          not dispatched - advertised,
          f"dispatched only: {sorted(dispatched - advertised)}")

    # -- the surface is reachable ------------------------------------------
    code, out, err = run("--help")
    check("gate --help succeeds", code == 0 and "usage" in (out + err).lower(),
          (out + err)[:300])
    missing = sorted(name for name in advertised if name not in out)
    check("gate --help lists every subcommand", not missing, repr(missing))

    for name in sorted(advertised):
        code, out, err = run(name, "--help")
        check(f"{name} --help is reachable",
              code == 0 and "Traceback" not in (out + err), (out + err)[-300:])

    # -- a bad invocation refuses; it never crashes ------------------------
    code, out, err = run("not-a-subcommand")
    check("an unknown subcommand is refused, not crashed",
          code != 0 and "Traceback" not in (out + err), (out + err)[-300:])

    code, out, err = run()
    check("no subcommand at all is refused, not crashed",
          code != 0 and "Traceback" not in (out + err), (out + err)[-300:])

    # -- an uninitialised tree: readable, but not actionable ---------------
    # status is read-only and stays readable before init (it reports empty
    # state).  reconcile changes things, so it must refuse in words rather
    # than act on state that does not exist.
    code, out, err = run("status")
    try:
        empty = json.loads(out)
    except json.JSONDecodeError:
        empty = None
    check("status is readable before init and reports nothing active",
          code == 0 and isinstance(empty, dict)
          and not empty.get("active_campaign") and not empty.get("campaigns"),
          (out + err).strip()[:200])
    code, out, err = run("reconcile")
    check("reconcile on an uninitialised tree refuses in words",
          code != 0 and "REFUSED" in (out + err), (out + err).strip()[-200:])

    code, out, err = run("init")
    check("init creates gate state", code == 0
          and (trees[0] / "Project/loop/gate_state.json").is_file(),
          (out + err)[:300])
    code, out, err = run("status")
    parsed = None
    if code == 0:
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            parsed = None
    check("status on an initialised tree prints JSON state",
          isinstance(parsed, dict), (out + err)[:300])
    check("a fresh gate has no active campaign",
          isinstance(parsed, dict) and not parsed.get("active_campaign"),
          json.dumps(parsed.get("active_campaign")) if parsed else "no state")

    # Every subcommand invoked bare, in its own untouched tree: those with
    # required flags must exit on argparse's usage error, those without must
    # refuse on state.  Neither may ever print a stack trace -- a traceback is
    # an unhandled defect wearing a refusal's clothes, and an operator cannot
    # tell the two apart.
    bare = gate_runner(trees[1])
    crashed: list[str] = []
    for name in sorted(advertised):
        code, out, err = bare(name)
        if "Traceback" in (out + err):
            crashed.append(f"{name}: {(out + err).strip().splitlines()[-1][:120]}")
    check("no subcommand crashes when invoked bare", not crashed,
          "; ".join(crashed))

    # -- no suite in this directory may re-run another ---------------------
    offenders = passthrough_suites(TESTS)
    check("no test suite is a pass-through for another suite's main",
          not offenders, "; ".join(offenders))

    for tree in trees:
        shutil.rmtree(tree, ignore_errors=True)
    failed = [(name, detail) for name, ok, detail in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} passed"
          + (" — ALL GREEN" if not failed else ""))
    for name, detail in failed:
        print(f"  FAILED: {name}" + (f"\n          {detail}" if detail else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
