#!/usr/bin/env python3
"""Adversarial tests for the exact guard bytes staged for owner LOCK.

Every case below runs ``Project/lock_staging/guard_bash.py`` as a real
subprocess over real stdin.  Nothing here re-implements the guard's logic:
an earlier suite in this repo tested extracted snippets instead of installed
bytes, which is exactly how a sed-script command-execution hole and an
``rg --pre=CMD`` hole survived review.

The settings block at the end pins the Edit/Write half of the boundary, so
the hook and Project/lock_staging/settings.json cannot drift apart silently.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GUARD = ROOT / "Project" / "lock_staging" / "guard_bash.py"
SETTINGS = ROOT / "Project" / "lock_staging" / "settings.json"

FAILURES: list[str] = []


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def run_guard(stdin_text: str, project_dir: Path | None = None) -> str:
    environment = dict(os.environ)
    environment["CLAUDE_PROJECT_DIR"] = str(project_dir or ROOT)
    result = subprocess.run(
        [sys.executable, str(GUARD)],
        input=stdin_text,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
        cwd=str(project_dir or ROOT),
        env=environment,
    )
    if result.returncode != 0:
        return f"crash(rc={result.returncode})"
    output = result.stdout.strip()
    if not output:
        return "allow"
    try:
        value = json.loads(output)
        return value["hookSpecificOutput"]["permissionDecision"]
    except Exception as exc:  # noqa: BLE001
        return f"unparseable({exc})"


def decision(payload: object, project_dir: Path | None = None) -> str:
    return run_guard(json.dumps(payload), project_dir)


def bash(command: object) -> dict:
    return {"tool_input": {"command": command}}


def check(name: str, actual: str, expected: str) -> None:
    passed = actual == expected
    print(("PASS " if passed else "FAIL ") + name)
    if not passed:
        FAILURES.append(f"{name}: expected {expected}, got {actual}")


def run_cases(cases: list[tuple[str, object, str]],
              project_dir: Path | None = None) -> None:
    for name, payload, expected in cases:
        check(name, decision(payload, project_dir), expected)


# --------------------------------------------------------------------------
# 1. malformed and hostile payloads: none may crash open
# --------------------------------------------------------------------------
def malformed_payloads() -> None:
    raw_cases = [
        ("empty stdin", ""),
        ("whitespace stdin", "   \n"),
        ("invalid json", "{not json"),
        ("truncated json", '{"tool_input": {"command":'),
        ("json null", "null"),
        ("json number", "7"),
        ("json string", '"git status"'),
        ("json array", "[]"),
        ("two json documents", '{"tool_input":{"command":"pwd"}}{"x":1}'),
    ]
    for name, text in raw_cases:
        check(name, run_guard(text), "deny")

    run_cases([
        ("empty object", {}, "deny"),
        ("tool_input null", {"tool_input": None}, "deny"),
        ("tool_input list", {"tool_input": []}, "deny"),
        ("tool_input string", {"tool_input": "pwd"}, "deny"),
        ("command int", bash(7), "deny"),
        ("command null", bash(None), "deny"),
        ("command list", bash(["pwd"]), "deny"),
        ("command dict", bash({"a": 1}), "deny"),
        ("command empty", bash(""), "deny"),
        ("command whitespace", bash("   "), "deny"),
        ("command bool", bash(True), "deny"),
        ("over length", bash("x" * 5000), "deny"),
        ("over length valid prefix", bash("pwd " + "a" * 5000), "deny"),
        ("non-object payload", [], "deny"),
        ("nul in command", bash("pwd\x00git"), "deny"),
        ("unbalanced quote", bash("head -- 'Project/PLAN.md"), "deny"),
    ])


# --------------------------------------------------------------------------
# 2. the allowed GRIND surface must actually work
# --------------------------------------------------------------------------
def allowed_surface() -> None:
    run_cases([
        ("controller status",
         bash("python3 Project/harness/trusted_controller.py status"), "allow"),
        ("controller issue-permit",
         bash("python3 Project/harness/trusted_controller.py issue-permit "
              "--request Project/loop/requests/a.json --capability cap.json"),
         "allow"),
        ("controller side lane",
         bash("python3 Project/harness/trusted_controller.py side --permit p1"),
         "allow"),
        ("runner shim onto the controller",
         bash("python3 Project/harness/runner.py status"), "allow"),
        ("gate status", bash("python3 Project/tools/run_gate.py status"), "allow"),
        ("gate diagnostic request",
         bash("python3 Project/tools/run_gate.py diagnostic --campaign c1 "
              "--shape 3 --target-sha256 abc --tool nsys --supports "
              "kernel-launch-overhead --question q --route r"), "allow"),
        ("gate side-evaluate",
         bash("python3 Project/tools/run_gate.py side-evaluate --campaign c1 "
              "--shape 6 --submission Project/submission/x.py"), "allow"),
        ("build submission", bash("python3 Project/tools/build_submission.py"),
         "allow"),
        ("build submission check-only",
         bash("python3 Project/tools/build_submission.py --check-only"), "allow"),
        ("session bootstrap",
         bash("python3 Project/tools/session_bootstrap.py"), "allow"),
        ("sensitivity board",
         bash("python3 Project/tools/sensitivity_board.py"), "allow"),
        ("champion watch", bash("python3 Project/tools/champion_watch.py"),
         "allow"),
        ("champion watch dry run",
         bash("python3 Project/tools/champion_watch.py --dry-run"), "allow"),
        ("locked test", bash("python3 Project/tools/tests/trusted_controller_test.py"),
         "allow"),
        ("gate cli test", bash("python3 Project/tools/tests/gate_v4_cli_test.py"),
         "allow"),
        ("champion watch test",
         bash("python3 Project/tools/tests/champion_watch_test.py"), "allow"),
        ("staged guard test",
         bash("python3 Project/tools/tests/staged_guard_test.py"), "allow"),
        ("git status", bash("git status --short --branch"), "allow"),
        ("git status bare", bash("git status"), "allow"),
        ("git diff stat", bash("git diff --stat"), "allow"),
        ("git diff path", bash("git diff -- Project/kernels/k100.py"), "allow"),
        ("git log", bash("git log --oneline -20"), "allow"),
        ("git show", bash("git show --stat a1b2c3d4"), "allow"),
        ("git rev-parse", bash("git rev-parse HEAD"), "allow"),
        ("git branch", bash("git branch --show-current"), "allow"),
        ("git mutable add", bash("git add -- Project/kernels/k100.py"), "allow"),
        ("git commit", bash("git commit -m measured-candidate"), "allow"),
        ("read sed", bash("sed -n 1,40p -- Project/GRIND_ENTRYPOINT.md"), "allow"),
        ("read sed single line", bash("sed -n 12p -- Project/PLAN.md"), "allow"),
        ("read rg", bash("rg -n permit -- Project/harness"), "allow"),
        ("read rg files", bash("rg --files -- Project/kernels"), "allow"),
        ("read rg context", bash("rg -n -C3 permit -- Project/harness"), "allow"),
        ("read head", bash("head -n 40 -- Project/PLAN.md"), "allow"),
        ("read tail", bash("tail -n 40 -- Project/PLAN.md"), "allow"),
        ("read wc", bash("wc -l -- Project/PLAN.md"), "allow"),
        ("read ls", bash("ls -la -- Project/kernels"), "allow"),
        ("read sha256sum", bash("sha256sum -- Project/PLAN.md"), "allow"),
        ("read stat", bash("stat -- Project/PLAN.md"), "allow"),
        ("read protected file is fine",
         bash("head -n 5 -- Project/harness/runner.py"), "allow"),
        ("pwd", bash("pwd"), "allow"),
        ("uname", bash("uname -a"), "allow"),
        ("date", bash("date -u"), "allow"),
        ("gpu identity", bash("nvidia-smi -L"), "allow"),
        ("gpu machine state",
         bash("nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,"
              "utilization.gpu,power.draw --format=csv,noheader"), "allow"),
        ("ship manifest",
         bash("python3 Project/tools/ship_manifest.py --evidence-map "
              "Project/results_side/final_evidence_map.json"), "allow"),
        ("ship manifest report",
         bash("python3 Project/tools/ship_manifest.py --evidence-map "
              "Project/results_side/final_evidence_map.json --report"), "allow"),
        # Constrained entrypoints must not be order-sensitive.  An allowlist
        # that admits one spelling of a flag pair and refuses the other is a
        # block whose cause the agent cannot see from the denial, which is the
        # same failure class as the old guard's "clean after git".
        ("ship manifest report before the map",
         bash("python3 Project/tools/ship_manifest.py --report --evidence-map "
              "Project/results_side/final_evidence_map.json"), "allow"),
        ("ship manifest diagnose is read-only and needs no map",
         bash("python3 Project/tools/ship_manifest.py --diagnose"), "allow"),
        ("ship manifest diagnose with report",
         bash("python3 Project/tools/ship_manifest.py --diagnose --report"),
         "allow"),
        ("champion watch both flags",
         bash("python3 Project/tools/champion_watch.py --dry-run --no-reconcile"),
         "allow"),
        ("champion watch both flags reversed",
         bash("python3 Project/tools/champion_watch.py --no-reconcile --dry-run"),
         "allow"),
        # The controller's diagnostic lane needs no allowlist entry of its own:
        # trusted_controller.py maps to None (any arguments) because it does
        # its own permit, capability and schema checks.
        ("controller diagnostic lane",
         bash("python3 Project/harness/trusted_controller.py diagnostic "
              "--permit p1 --target "
              "Project/submission/torch_transformer_benchmark_submission.py"),
         "allow"),
        ("controller diagnostic with a timeout",
         bash("python3 Project/harness/trusted_controller.py diagnostic "
              "--permit p1 --timeout 10800"), "allow"),
        ("controller verify-lock replaces the owner ceremony's verify",
         bash("python3 Project/harness/trusted_controller.py verify-lock"),
         "allow"),
        # Reading committed bytes: the "commit the candidate before first
        # controller contact" step is unverifiable without these.
        ("git show a revision", bash("git show HEAD --stat"), "allow"),
        ("git show an ancestor", bash("git show HEAD~2 --stat"), "allow"),
        ("git show committed file bytes",
         bash("git show HEAD:Project/kernels/k100.py"), "allow"),
        ("git diff against a revision",
         bash("git diff HEAD~1 HEAD -- Project/kernels"), "allow"),
        ("git log for one path",
         bash("git log --oneline -20 -- Project/memory/LESSONS.md"), "allow"),
        ("git rev-parse short", bash("git rev-parse --short HEAD"), "allow"),
        # Searching for a literal that starts with a dash.  The live guard's
        # ancestor refused a grep because the PATTERN contained "rm -rf"; a
        # pattern is data here, never a rule input.
        ("rg for a pattern containing rm -rf",
         bash("""rg -n "rm -rf" -- Project/tools"""), "allow"),
        ("rg -e for a pattern starting with a dash",
         bash("rg -n -e -rf -- Project/tools"), "allow"),
        ("rg --regexp long form",
         bash("rg --regexp --pre=id -- Project/tools"), "allow"),
    ])


# --------------------------------------------------------------------------
# 3. profilers: version probe only; every launcher form denied
# --------------------------------------------------------------------------
def profiler_policy() -> None:
    """Version probes only.

    The decision this pins: no profiler launcher is admitted, and that is
    stated rather than quietly left out.  nsys/ncu/compute-sanitizer exist to
    run an arbitrary command line and each takes an argument naming a file to
    write, so no argument grammar keeps them from re-admitting `python3 -c`
    and arbitrary writes.  nsys is fully replaced: `run_gate.py diagnostic` ->
    one-use permit -> the controller's sandboxed diagnostic lane, which is the
    only route producing evidence `plan --counter-evidence` can cite, so a
    shell-run profile would be unusable even if it were allowed.  Two gaps are
    real and named: ncu needs root here, and compute-sanitizer is not in the
    catalog's evidence_tools, so neither is reachable by any agent route --
    both are owner work, pre-LOCK or through break-glass.
    """
    run_cases([
        ("nsys version probe", bash("nsys --version"), "allow"),
        ("ncu version probe", bash("ncu --version"), "allow"),
        ("sanitizer version probe", bash("compute-sanitizer --version"), "allow"),
        ("nsys launches a process",
         bash("nsys profile -o out python3 Project/tools/run_gate.py status"),
         "deny"),
        ("nsys profile allowlisted script still denied",
         bash("nsys profile python3 Project/tools/run_gate.py status"), "deny"),
        ("nsys stats runs recipes", bash("nsys stats -- Project/x.nsys-rep"),
         "deny"),
        ("nsys launch", bash("nsys launch python3 x.py"), "deny"),
        ("ncu without sudo", bash("ncu python3 x.py"), "deny"),
        ("sudo ncu", bash("sudo ncu python3 x.py"), "deny"),
        ("sudo anything", bash("sudo -n true"), "deny"),
        ("compute-sanitizer run",
         bash("compute-sanitizer --tool racecheck python3 x.py"), "deny"),
        ("nsys bare", bash("nsys"), "deny"),
        ("ncu version with extra arg", bash("ncu --version foo"), "deny"),
        ("profiler on python entrypoint via python3 prefix",
         bash("python3 nsys"), "deny"),
        ("nsys with an output file", bash("nsys profile -o Project/x pwd"),
         "deny"),
        ("ncu log-file is a write primitive",
         bash("ncu --log-file Project/x --version"), "deny"),
        ("sanitizer racecheck on a kernel",
         bash("compute-sanitizer --tool racecheck python3 "
              "Project/tools/run_gate.py status"), "deny"),
        ("sanitizer memcheck", bash("compute-sanitizer --tool memcheck pwd"),
         "deny"),
        ("torch profiler has no shell form either",
         bash("python3 -m torch.profiler"), "deny"),
        # The sanctioned profiling route is the gate + controller pair, and
        # both halves are reachable.
        ("gate diagnostic opens the sanctioned route",
         bash("python3 Project/tools/run_gate.py diagnostic --campaign c1 "
              "--shape 3 --target-sha256 abc --tool nsys --supports "
              "kernel-launch-overhead --question q --route r"), "allow"),
        ("controller runs the profiler inside the sandbox",
         bash("python3 Project/harness/trusted_controller.py diagnostic "
              "--permit p1"), "allow"),
    ])


# --------------------------------------------------------------------------
# 4. execution and write primitives that the previous staged bytes allowed
# --------------------------------------------------------------------------
def command_execution_holes() -> None:
    run_cases([
        # GNU sed's `e` command runs a shell; `w`/`W` write any path.  None of
        # these contains a character the metacharacter screen rejects.
        ("sed e executes a command",
         bash("""sed -n "1e id" -- Project/PLAN.md"""), "deny"),
        ("sed e with an argument",
         bash("""sed -n '1e touch /tmp/pwned' -- Project/PLAN.md"""), "deny"),
        ("sed w writes the live hook",
         bash("""sed -n "1s/a/b/w .claude/hooks/guard_bash.py" -- """
              "Project/PLAN.md"), "deny"),
        ("sed w writes the frozen referee",
         bash("""sed -n "1s/a/b/w Project/harness/runner.py" -- """
              "Project/PLAN.md"), "deny"),
        ("sed W writes", bash("""sed -n "1,2W out.txt" -- Project/PLAN.md"""),
         "deny"),
        ("sed r reads outside the repo",
         bash("""sed -n "1r /etc/shadow" -- Project/PLAN.md"""), "deny"),
        ("sed script with a substitution",
         bash("""sed -n "s/a/b/p" -- Project/PLAN.md"""), "deny"),
        ("sed without -n", bash("sed 1,5p -- Project/PLAN.md"), "deny"),
        ("sed with -e", bash("sed -n -e 1,5p -- Project/PLAN.md"), "deny"),
        ("sed with -i", bash("sed -n -i 1,5p -- Project/PLAN.md"), "deny"),
        ("sed with no path", bash("sed -n 1,5p --"), "deny"),
        # ripgrep's executing flags all accept a --flag=value spelling, which
        # an exact-token blocklist never sees.
        ("rg --pre= executes", bash("rg --pre=id -n x -- Project/PLAN.md"),
         "deny"),
        ("rg --pre separated", bash("rg --pre id -n x -- Project/PLAN.md"),
         "deny"),
        ("rg --pre-glob=", bash("rg --pre-glob=y --pre=id -n x -- Project/PLAN.md"),
         "deny"),
        ("rg --hostname-bin=", bash("rg --hostname-bin=id -n x -- Project/PLAN.md"),
         "deny"),
        ("rg -z spawns decompressors", bash("rg -z -n x -- Project/PLAN.md"),
         "deny"),
        ("rg --search-zip", bash("rg --search-zip -n x -- Project/PLAN.md"),
         "deny"),
        ("rg unknown flag", bash("rg --frobnicate -n x -- Project/PLAN.md"),
         "deny"),
        ("rg without separator", bash("rg -n x Project/PLAN.md"), "deny"),
        ("rg without a path", bash("rg -n x --"), "deny"),
        ("rg two patterns", bash("rg -n a b -- Project/PLAN.md"), "deny"),
        ("rg --files with a pattern", bash("rg --files x -- Project/kernels"),
         "deny"),
        ("rg -e with no pattern after it", bash("rg -e -- Project/PLAN.md"),
         "deny"),
        ("rg -e twice is two patterns",
         bash("rg -n -e a -e b -- Project/PLAN.md"), "deny"),
        ("rg -e plus a bare pattern", bash("rg -n -e a b -- Project/PLAN.md"),
         "deny"),
        ("rg --regexp= value form is still an unknown flag",
         bash("rg --regexp=x -- Project/PLAN.md"), "deny"),
        ("rg --files cannot take -e either",
         bash("rg --files -e x -- Project/kernels"), "deny"),
        # read tools must not become executors, oracles or hangs
        ("head with no path", bash("head --"), "deny"),
        ("ls with no path", bash("ls --"), "deny"),
        ("wc with no path", bash("wc -l --"), "deny"),
        ("tail -f never terminates", bash("tail -f -- Project/PLAN.md"), "deny"),
        ("tail --follow", bash("tail --follow -- Project/PLAN.md"), "deny"),
        ("sha256sum -c is a path oracle",
         bash("sha256sum -c -- Project/PLAN.md"), "deny"),
        ("stat -L follows out of the repo",
         bash("stat -L -- Project/PLAN.md"), "deny"),
        ("head unknown flag", bash("head -z -- Project/PLAN.md"), "deny"),
        ("nvidia-smi power limit", bash("nvidia-smi -pl 100"), "deny"),
        ("nvidia-smi lock clocks", bash("nvidia-smi -lgc 1000"), "deny"),
        ("nvidia-smi gpu reset", bash("nvidia-smi --gpu-reset"), "deny"),
    ])


# --------------------------------------------------------------------------
# 5. metacharacters, tokenization and shell/shlex divergence
# --------------------------------------------------------------------------
def tokenization() -> None:
    run_cases([
        ("chained command", bash("git status --short && python3 evil.py"), "deny"),
        ("semicolon chain", bash("pwd ; id"), "deny"),
        ("pipe", bash("git status --short | tee out"), "deny"),
        ("redirect", bash("git status --short > Project/drafts/out"), "deny"),
        ("append redirect", bash("pwd >> Project/drafts/out"), "deny"),
        ("here-string", bash("python3 - <<< print(1)"), "deny"),
        ("command substitution", bash("sha256sum -- $(pwd)"), "deny"),
        ("backticks", bash("sha256sum -- `pwd`"), "deny"),
        ("brace expansion", bash("ls -- Project/{harness,tools}"), "deny"),
        ("glob expansion", bash("ls -- Project/*"), "deny"),
        ("question glob", bash("ls -- Project/kernel?"), "deny"),
        ("bracket glob", bash("ls -- Project/kernels/k[0-9].py"), "deny"),
        ("backslash escape", bash("ls -- Project/kernels\\k1.py"), "deny"),
        ("newline in command", bash("pwd\nid"), "deny"),
        ("carriage return", bash("pwd\rid"), "deny"),
        ("env assignment prefix", bash("PYTHONPATH=/tmp python3 "
                                       "Project/tools/run_gate.py status"), "deny"),
        ("leading env var only", bash("PATH=/tmp pwd"), "deny"),
        ("tab separated tokens still parse",
         bash("python3\tProject/tools/run_gate.py\tstatus"), "allow"),
        ("repeated spaces still parse", bash("git      status --short"), "allow"),
        ("comment character is not a comment",
         bash("git status --short#comment"), "deny"),
        ("quoted metacharacter is still refused",
         bash("""git log --oneline "-5;id" """), "deny"),
        # shlex's whitespace set and bash's IFS agree on space/tab/newline and
        # nothing else, so every other control character is refused outright
        # rather than left to two parsers to disagree about.
        ("vertical tab in a command", bash("pwd\x0bid"), "deny"),
        ("vertical tab glued to an allowed argument",
         bash("python3 Project/tools/run_gate.py status\x0b"), "deny"),
        ("form feed", bash("pwd\x0cid"), "deny"),
        ("escape character", bash("pwd\x1b[2J"), "deny"),
        ("delete character", bash("head -- Project/PLAN.md\x7f"), "deny"),
        ("bell in a commit message",
         bash("git commit -m 'ring\x07the bell'"), "deny"),
        ("non-breaking space is not a separator", bash("git\xa0status"), "deny"),
        ("tab inside a commit message is fine",
         bash("git commit -m 'tabbed\tmessage'"), "allow"),
    ])


# --------------------------------------------------------------------------
# 6. path policy: prefixes, traversal, symlinks, protected components
# --------------------------------------------------------------------------
def path_policy() -> None:
    run_cases([
        ("dotdot traversal", bash("head -- Project/../Project/PLAN.md"), "deny"),
        ("dotdot escape", bash("head -- Project/../../etc/passwd"), "deny"),
        ("absolute path", bash("head -- /etc/passwd"), "deny"),
        ("double slash absolute", bash("head -- //etc/passwd"), "deny"),
        ("dot git component", bash("head -- Project/.git/config"), "deny"),
        ("git dir at root", bash("head -- .git/config"), "deny"),
        ("claude dir", bash("head -- .claude/settings.json"), "deny"),
        ("claude hook", bash("head -- .claude/hooks/guard_bash.py"), "deny"),
        ("break glass file is unreadable",
         bash("head -- .claude/BREAK_GLASS"), "deny"),
        ("tilde home", bash("head -- ~/secret"), "deny"),
        ("bare dot", bash("ls -- ."), "deny"),
        ("outside the readable roots", bash("head -- etc/passwd"), "deny"),
        ("readable root prefix extension is not a match",
         bash("head -- CLAUDE.md.evil"), "deny"),
        ("readme prefix extension", bash("head -- README.md.bak"), "deny"),
        ("normalized double slash", bash("head -- Project//PLAN.md"), "allow"),
        ("normalized leading dot slash", bash("head -- ./Project/PLAN.md"),
         "allow"),
        ("normalized inner dot", bash("head -- Project/./PLAN.md"), "allow"),
        ("quoted path with a space is still checked",
         bash("""head -- "Project/a b.md" """), "allow"),
        ("path that leaves the roots when quoted",
         bash("""head -- "../etc/passwd" """), "deny"),
        # .git and .claude are refused in EVERY position, not just the first.
        ("dot git in a deeper position",
         bash("head -- Project/kernels/.git/config"), "deny"),
        ("dot claude in a deeper position",
         bash("head -- Project/kernels/.claude/settings.json"), "deny"),
        ("dot git as a trailing component",
         bash("head -- Project/kernels/.git"), "deny"),
        ("staging a path with a .git component",
         bash("git add -- Project/kernels/.git/config"), "deny"),
        ("dot gitignore is not the git directory",
         bash("head -- Project/.gitignore"), "allow"),
        # Unicode: the roots are byte-compared, so no homoglyph, width variant
        # or invisible character can spell its way into one.
        ("unicode homoglyph root", bash("head -- Ｐroject/PLAN.md"), "deny"),
        ("combining-mark root", bash("head -- Projéct/PLAN.md"),
         "deny"),
        ("zero-width space inside the root",
         bash("head -- Project​/PLAN.md"), "deny"),
        ("non-breaking space inside the root",
         bash("head -- Project\xa0/PLAN.md"), "deny"),
        ("cyrillic lookalike root", bash("head -- Рroject/PLAN.md"), "deny"),
        ("unicode below a real root is still inside it",
         bash("head -- Project/kernels/kérnel.py"), "allow"),
        # Trailing dots and slashes normalize inside the root, never out of it.
        ("trailing dot component", bash("head -- Project/."), "allow"),
        ("trailing dotdot component", bash("head -- Project/.."), "deny"),
        ("dotdot rejoining the same root",
         bash("head -- Project/../Project/PLAN.md"), "deny"),
        ("dotdot inside a staged path",
         bash("git add -- Project/kernels/../../etc/passwd"), "deny"),
        ("trailing dot on a filename stays in the root",
         bash("head -- Project/PLAN.md."), "allow"),
        ("trailing slash on the root", bash("head -- Project/"), "allow"),
    ])


def _load_guard_module():
    """Import the STAGED FILE ITSELF so `_under` is tested, not a copy of it."""
    spec = importlib.util.spec_from_file_location("staged_guard_bytes", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def path_boundary_semantics() -> None:
    """Prefix boundaries, asserted directly against the staged bytes.

    An earlier revision of this hook matched roots with a bare
    ``normalized.startswith(root)`` over a list that mixed FILE roots with
    DIRECTORY roots.  A file root spelled
    ``Project/submission/dispatcher_region.py`` therefore also matched
    ``Project/submission/dispatcher_region.py.evil`` and every other sibling
    sharing that prefix, and a directory root spelled ``Project/res`` would
    have swallowed ``Project/results_side``.  ``_under`` is the fix: file
    roots match exactly, directory roots match on a separator.

    These go through the module rather than the subprocess because the
    property belongs to ``_under`` itself and must hold for every caller,
    including roots that are not reachable from any single command today.
    The module is loaded from the same path the subprocess cases execute, so
    it is still the staged bytes under test -- nothing here re-implements it.
    """
    guard = _load_guard_module()
    under = guard._under
    cases = [
        # file root: exact, and nothing that merely starts with its name
        ("Project/submission/dispatcher_region.py",
         "Project/submission/dispatcher_region.py", True),
        ("Project/submission/dispatcher_region.py.evil",
         "Project/submission/dispatcher_region.py", False),
        ("Project/submission/dispatcher_region.pyx",
         "Project/submission/dispatcher_region.py", False),
        ("Project/submission/dispatcher_region.py/child",
         "Project/submission/dispatcher_region.py", False),
        ("CLAUDE.md", "CLAUDE.md", True),
        ("CLAUDE.md.evil", "CLAUDE.md", False),
        # directory root: the directory itself, and anything beneath it
        ("Project", "Project/", True),
        ("Project/kernels/k1.py", "Project/", True),
        ("Projectile.md", "Project/", False),
        ("Project/results_side/x.json", "Project/res", False),
        ("Project/harness", "Project/harness/", True),
        ("Project/harness/runner.py", "Project/harness/", True),
        ("Project/harnessX/f.py", "Project/harness/", False),
        ("Project/harness.py", "Project/harness/", False),
    ]
    for value, root, expected in cases:
        check(f"_under({value!r}, {root!r})",
              str(under(value, root)), str(expected))

    # Every root the hook actually ships must be spelled unambiguously: a
    # directory root ends in "/", a file root names an existing file.
    roots = (list(guard.READABLE_ROOTS) + [guard.STAGEABLE_ROOT]
             + list(guard.STAGEABLE_EXCLUDED))
    for root in roots:
        well_formed = root.endswith("/") or (ROOT / root).is_file()
        check(f"root is unambiguously a file or a directory: {root!r}",
              "ok" if well_formed else "ambiguous", "ok")

    # And the same property observed through the real subprocess surface.
    run_cases([
        ("file root extension is not readable", bash("head -- CLAUDE.md.evil"),
         "deny"),
        ("file root itself is readable", bash("head -- CLAUDE.md"), "allow"),
        ("directory root boundary holds for the reader",
         bash("head -- Projectile.md"), "deny"),
    ])


def symlink_policy() -> None:
    """A planted symlink inside an allowed root must not launder a read."""
    with tempfile.TemporaryDirectory(prefix="staged-guard-symlink-") as temp:
        sandbox = Path(temp)
        (sandbox / "Project" / "kernels").mkdir(parents=True)
        (sandbox / "Project" / "kernels" / "real.py").write_text("x\n")
        (sandbox / "outside.txt").write_text("secret\n")
        (sandbox / "Project" / "kernels" / "escape").symlink_to(
            sandbox / "outside.txt")
        (sandbox / "Project" / "linkdir").symlink_to(sandbox)
        run_cases([
            ("symlinked file inside a root",
             bash("head -- Project/kernels/escape"), "deny"),
            ("path through a symlinked directory",
             bash("head -- Project/linkdir/outside.txt"), "deny"),
            ("git add of a symlink", bash("git add -- Project/kernels/escape"),
             "deny"),
            ("real file beside the symlink still reads",
             bash("head -- Project/kernels/real.py"), "allow"),
        ], project_dir=sandbox)


# --------------------------------------------------------------------------
# 7. git surface: workability and safety
# --------------------------------------------------------------------------
def git_surface() -> None:
    run_cases([
        # destructive git is unreachable
        ("stash", bash("git stash"), "deny"),
        ("hard reset", bash("git reset --hard"), "deny"),
        ("checkout", bash("git checkout -- Project/harness/runner.py"), "deny"),
        ("restore", bash("git restore Project/harness/runner.py"), "deny"),
        ("clean", bash("git clean -fd"), "deny"),
        ("rebase", bash("git rebase main"), "deny"),
        ("push", bash("git push origin grind-day1"), "deny"),
        ("config", bash("git config core.pager cat"), "deny"),
        ("git -c prefix", bash("git -c core.pager=id status"), "deny"),
        ("git exec-path prefix", bash("git --exec-path=/tmp status"), "deny"),
        ("commit -a", bash("git commit -a -m msg"), "deny"),
        ("commit amend", bash("git commit --amend -m msg"), "deny"),
        ("commit empty message", bash("git commit -m ''"), "deny"),
        ("commit over-long message", bash("git commit -m " + "x" * 201), "deny"),
        ("diff output file", bash("git diff --output=x"), "deny"),
        ("diff output-indicator", bash("git diff --output-indicator-new=x"),
         "deny"),
        ("diff output file even with a revision",
         bash("git diff HEAD --output=x"), "deny"),
        ("log patch could reach an external differ", bash("git log -p"), "deny"),
        ("log ext-diff", bash("git log --ext-diff"), "deny"),
        ("rev-parse with no argument", bash("git rev-parse"), "deny"),
        ("rev-parse parseopt", bash("git rev-parse --parseopt"), "deny"),
        ("rev-parse sq-quote", bash("git rev-parse --sq-quote id"), "deny"),
        ("show a revision path outside the repo",
         bash("git show HEAD:/etc/passwd"), "deny"),
        ("show a revision path that traverses out",
         bash("git show HEAD:../etc/passwd"), "deny"),
        ("show a revision path into the live hook",
         bash("git show HEAD:.claude/settings.json"), "deny"),
        ("a revision-looking token is not a free path",
         bash("git show HEAD:etc/passwd"), "deny"),
        ("add without separator", bash("git add Project/kernels/k1.py"), "deny"),
        ("add all", bash("git add -A"), "deny"),
        ("add with no path", bash("git add --"), "deny"),
        # the enforcer and referee bytes are not stageable by the agent
        ("add the frozen referee",
         bash("git add -- Project/harness/runner.py"), "deny"),
        ("add the controller",
         bash("git add -- Project/harness/trusted_controller.py"), "deny"),
        ("add the harness tree", bash("git add -- Project/harness"), "deny"),
        ("add the gate", bash("git add -- Project/tools/run_gate.py"), "deny"),
        ("add the tools tree", bash("git add -- Project/tools"), "deny"),
        ("add all of Project would contain the enforcers",
         bash("git add -- Project"), "deny"),
        ("add outside Project", bash("git add -- CLAUDE.md"), "deny"),
        ("add the official benchmark",
         bash("git add -- torch_transformer_benchmark.py"), "deny"),
        ("add the live hook", bash("git add -- .claude/hooks/guard_bash.py"),
         "deny"),
        ("add the git dir", bash("git add -- .git/config"), "deny"),
        ("add the staged guard bytes",
         bash("git add -- Project/lock_staging/guard_bash.py"), "deny"),
        ("add the staging tree", bash("git add -- Project/lock_staging"), "deny"),
        ("add the gate tests tree",
         bash("git add -- Project/tools/tests"), "deny"),
        ("an excluded root is not escaped by a dot component",
         bash("git add -- Project/./harness/runner.py"), "deny"),
        ("an excluded root is not escaped by traversal",
         bash("git add -- Project/kernels/../harness/runner.py"), "deny"),
        ("one bad path poisons the whole add",
         bash("git add -- Project/kernels/k1.py Project/harness/runner.py"),
         "deny"),
        # A sibling that merely shares an excluded root's name prefix is a
        # different tree and stays stageable; the exclusion must match on a
        # path boundary, not on a string prefix, in BOTH directions.
        ("a name-prefix sibling of an excluded root is not excluded",
         bash("git add -- Project/harnessX/notes.md"), "allow"),
        # the work record and the machine-written ledgers must be committable
        ("add kernels", bash("git add -- Project/kernels"), "allow"),
        ("add the dispatcher region",
         bash("git add -- Project/submission/dispatcher_region.py"), "allow"),
        ("add the generated submission",
         bash("git add -- Project/submission/"
              "torch_transformer_benchmark_submission.py"), "allow"),
        ("add research", bash("git add -- Project/research"), "allow"),
        ("add drafts", bash("git add -- Project/drafts"), "allow"),
        ("add DECISIONS.md", bash("git add -- Project/memory/DECISIONS.md"),
         "allow"),
        ("add LESSONS.md", bash("git add -- Project/memory/LESSONS.md"), "allow"),
        ("add STATE.md", bash("git add -- Project/memory/STATE.md"), "allow"),
        ("add HANDOVER.md", bash("git add -- Project/HANDOVER.md"), "allow"),
        ("add GRIND_ENTRYPOINT.md",
         bash("git add -- Project/GRIND_ENTRYPOINT.md"), "allow"),
        ("add cards ledger", bash("git add -- Project/loop/cards.jsonl"), "allow"),
        ("add lineage ledger", bash("git add -- Project/loop/lineage.jsonl"),
         "allow"),
        ("add gate state", bash("git add -- Project/loop/gate_state.json"),
         "allow"),
        ("add the audit ledger", bash("git add -- Project/audits/verdicts.jsonl"),
         "allow"),
        ("add runner results", bash("git add -- Project/results"), "allow"),
        ("add side results", bash("git add -- Project/results_side"), "allow"),
        ("add authority events", bash("git add -- Project/authority"), "allow"),
        ("add several paths at once",
         bash("git add -- Project/kernels/k1.py Project/memory/LESSONS.md"),
         "allow"),
    ])


def commit_messages() -> None:
    """A legitimate commit message must never be blocked for its wording.

    The old blacklist guard refused any commit whose message contained the
    word "clean" after "git"; two real commits were lost to it.  The staged
    metacharacter screen reproduced the same class for punctuation, so the
    single-quoted verbatim form exists to take that class off the table.
    """
    run_cases([
        ("plain single-token message", bash("git commit -m measured-candidate"),
         "allow"),
        ("message containing the word clean",
         bash("""git commit -m 'clean up the guard allowlist'"""), "allow"),
        ("message containing the word reset",
         bash("""git commit -m 'reset the calibration baseline'"""), "allow"),
        ("message with an arrow", bash("""git commit -m 'warmup 3->1'"""),
         "allow"),
        ("message with a dollar sign",
         bash("""git commit -m 'refuse $HOME in auditor paths'"""), "allow"),
        ("message with a pipe and parens",
         bash("""git commit -m 'gate: plan|delta (bottleneck required)'"""),
         "allow"),
        ("message with brackets and a star",
         bash("""git commit -m 'k015: fix mask [B,1,S,S] * scale'"""), "allow"),
        ("message with a semicolon",
         bash("""git commit -m 'shape 6; shape 14 side lanes'"""), "allow"),
        ("message with backticks",
         bash("""git commit -m 'document `nsys` route'"""), "allow"),
        ("verbatim form rejects an embedded single quote",
         bash("""git commit -m 'don't'"""), "deny"),
        ("verbatim form rejects a trailing command",
         bash("""git commit -m 'ok'; id"""), "deny"),
        ("verbatim form rejects a leading command",
         bash("""id; git commit -m 'ok'"""), "deny"),
        ("verbatim form rejects extra flags",
         bash("""git commit -m 'ok' --amend"""), "deny"),
        ("verbatim form rejects a different subcommand",
         bash("""git push -m 'ok'"""), "deny"),
        ("verbatim form rejects an empty message",
         bash("""git commit -m ''"""), "deny"),
        ("verbatim form rejects an over-long message",
         bash("git commit -m '" + "x" * 201 + "'"), "deny"),
        ("double quotes do not get the verbatim carve-out",
         bash('''git commit -m "warmup 3->1"'''), "deny"),
    ])


# --------------------------------------------------------------------------
# 8. entrypoint policy: what is reachable directly and what must be routed
# --------------------------------------------------------------------------
def entrypoint_policy() -> None:
    run_cases([
        ("python -c", bash("python3 -c print(1)"), "deny"),
        ("python -m", bash("python3 -m pathlib"), "deny"),
        ("python -X flag", bash("python3 -X importtime "
                                "Project/tools/run_gate.py status"), "deny"),
        ("python stdin", bash("python3 -"), "deny"),
        ("python unknown script", bash("python3 evil.py"), "deny"),
        ("python relative-prefixed allowlisted script",
         bash("python3 ./Project/tools/run_gate.py status"), "deny"),
        ("python absolute allowlisted script",
         bash("python3 /home/admin/Project/tools/run_gate.py"), "deny"),
        ("other interpreter", bash("python Project/tools/run_gate.py"), "deny"),
        ("bash interpreter", bash("bash Project/tools/run_gate.py"), "deny"),
        # side evaluators: gate + controller only (external-audit finding)
        ("direct shape 6 evaluator",
         bash("python3 Project/tools/shape6_local_eval.py"), "deny"),
        ("direct shape 14 evaluator",
         bash("python3 Project/tools/shape14_eval.py"), "deny"),
        # auditor: champion_watch mints the durable attempt row first
        ("direct auditor", bash("python3 Project/tools/audit_champion.py e1"),
         "deny"),
        ("direct codex", bash("codex exec review"), "deny"),
        ("codex as an argument token", bash("python3 codex exec"), "deny"),
        # sandbox payloads are launched by the controller, never by a shell
        ("direct candidate worker",
         bash("python3 Project/harness/candidate_worker.py"), "deny"),
        ("direct profile worker",
         bash("python3 Project/harness/profile_worker.py"), "deny"),
        # owner-only tooling
        ("owner lock ceremony",
         bash("python3 Project/tools/owner_lock_ceremony.py verify"), "deny"),
        ("streamlit dashboard", bash("python3 Project/tools/dashboard.py"),
         "deny"),
        ("streamlit runner", bash("streamlit run Project/tools/dashboard.py"),
         "deny"),
        ("raw-dial smoke", bash("python3 Project/tools/smokes/k007_smoke.py"),
         "deny"),
        # constrained-argument entrypoints
        ("ship manifest without an evidence map",
         bash("python3 Project/tools/ship_manifest.py"), "deny"),
        ("ship manifest naming an output file",
         bash("python3 Project/tools/ship_manifest.py --evidence-map "
              "Project/results_side/m.json --output Project/results_side/x.json"),
         "deny"),
        ("ship manifest with an escaping evidence map",
         bash("python3 Project/tools/ship_manifest.py --evidence-map /etc/passwd"),
         "deny"),
        ("build submission provenance-out is a write primitive",
         bash("python3 Project/tools/build_submission.py --provenance-out x.json"),
         "deny"),
        ("champion watch max-launches is not tunable",
         bash("python3 Project/tools/champion_watch.py --max-launches 50"),
         "deny"),
        ("session bootstrap takes no arguments",
         bash("python3 Project/tools/session_bootstrap.py --x"), "deny"),
        ("test file with arguments",
         bash("python3 Project/tools/tests/lock_manifest_test.py --x"), "deny"),
        ("sensitivity board takes no arguments",
         bash("python3 Project/tools/sensitivity_board.py --out x"), "deny"),
        ("champion watch flags may not repeat",
         bash("python3 Project/tools/champion_watch.py --dry-run --dry-run"),
         "deny"),
        ("ship manifest report alone has no evidence map",
         bash("python3 Project/tools/ship_manifest.py --report"), "deny"),
        ("ship manifest evidence map with no value",
         bash("python3 Project/tools/ship_manifest.py --evidence-map"), "deny"),
        ("ship manifest evidence map swallowing a flag",
         bash("python3 Project/tools/ship_manifest.py --evidence-map --report"),
         "deny"),
        ("ship manifest two evidence maps",
         bash("python3 Project/tools/ship_manifest.py --evidence-map "
              "Project/drafts/a.json --evidence-map Project/drafts/b.json"),
         "deny"),
        # owner_lock_ceremony.py is the only code that touches a private key.
        # Even `verify` is refused: it shares an argv surface with keygen and
        # mint-capability, and trusted_controller.py verify-lock/status already
        # reports everything verify would tell the agent.
        ("owner ceremony keygen",
         bash("python3 Project/tools/owner_lock_ceremony.py keygen"), "deny"),
        ("owner ceremony mint-capability",
         bash("python3 Project/tools/owner_lock_ceremony.py mint-capability "
              "--action permit.issue --campaign c1 --reason r --private-key k"),
         "deny"),
        ("owner ceremony sign-lock",
         bash("python3 Project/tools/owner_lock_ceremony.py sign-lock "
              "--private-key k"), "deny"),
    ])


# --------------------------------------------------------------------------
# 9. general shell surface that must stay gone
# --------------------------------------------------------------------------
def general_surface() -> None:
    run_cases([
        ("relative cd", bash("cd Project"), "deny"),
        ("recursive delete", bash("rm -rf Project"), "deny"),
        ("single delete", bash("rm Project/kernels/k1.py"), "deny"),
        ("find delete", bash("find Project -delete"), "deny"),
        ("find exec", bash("find Project -exec id"), "deny"),
        ("copy referee", bash("cp candidate.py Project/harness/runner.py"),
         "deny"),
        ("move referee", bash("mv candidate.py Project/harness/runner.py"),
         "deny"),
        ("install", bash("install candidate.py Project/harness/runner.py"),
         "deny"),
        ("hardlink", bash("ln candidate.py Project/harness/runner.py"), "deny"),
        ("symlink", bash("ln -s /etc Project/kernels/etc"), "deny"),
        ("dd", bash("dd if=x of=Project/manifest.json"), "deny"),
        ("rsync", bash("rsync x Project/harness/"), "deny"),
        ("shred", bash("shred Project/audits/verdicts.jsonl"), "deny"),
        ("truncate", bash("truncate -s 0 Project/audits/verdicts.jsonl"), "deny"),
        ("chmod", bash("chmod 777 Project/harness/runner.py"), "deny"),
        ("touch", bash("touch Project/kernels/new.py"), "deny"),
        ("mkdir", bash("mkdir Project/kernels/new"), "deny"),
        ("echo", bash("echo hello"), "deny"),
        ("cat", bash("cat Project/PLAN.md"), "deny"),
        ("grep", bash("grep -rn x Project"), "deny"),
        ("awk", bash("awk 1 Project/PLAN.md"), "deny"),
        ("perl", bash("perl -e print"), "deny"),
        ("curl", bash("curl https://example.com"), "deny"),
        ("wget", bash("wget https://example.com"), "deny"),
        ("pip install", bash("pip install torch"), "deny"),
        ("modprobe", bash("modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0"),
         "deny"),
        ("env", bash("env"), "deny"),
        ("export", bash("export PATH=/tmp"), "deny"),
        ("kill", bash("kill -9 1"), "deny"),
        ("nohup", bash("nohup python3 Project/tools/run_gate.py status"), "deny"),
        ("xargs", bash("xargs id"), "deny"),
        ("timeout wrapper",
         bash("timeout 10 python3 Project/tools/run_gate.py status"), "deny"),
        ("nice wrapper",
         bash("nice python3 Project/tools/run_gate.py status"), "deny"),
        ("strace wrapper",
         bash("strace python3 Project/tools/run_gate.py status"), "deny"),
        ("uname without -a", bash("uname"), "deny"),
        ("pwd with an argument", bash("pwd -P"), "deny"),
    ])


# --------------------------------------------------------------------------
# 10. owner break-glass: usable by the owner, never mintable by the agent
# --------------------------------------------------------------------------
def break_glass() -> None:
    with tempfile.TemporaryDirectory(prefix="staged-guard-breakglass-") as temp:
        sandbox = Path(temp)
        claude = sandbox / ".claude"
        claude.mkdir()
        marker = claude / "BREAK_GLASS"

        run_cases([
            ("no break-glass file: exotic command denied",
             bash("nsys profile -o t python3 x.py"), "deny"),
        ], project_dir=sandbox)

        marker.write_text(
            "# owner break-glass; delete this file when the incident is over\n"
            "\n"
            "nsys profile -o Project/loop/profile_evidence/t python3 x.py\n"
            "sudo ncu --set full python3 y.py\n"
        )
        run_cases([
            ("break-glass grants the exact line",
             bash("nsys profile -o Project/loop/profile_evidence/t python3 x.py"),
             "allow"),
            ("break-glass grants a sudo line the owner wrote",
             bash("sudo ncu --set full python3 y.py"), "allow"),
            ("break-glass tolerates surrounding whitespace",
             bash("  sudo ncu --set full python3 y.py  "), "allow"),
            ("break-glass is exact, not prefix",
             bash("sudo ncu --set full python3 y.py ; id"), "deny"),
            # Python strips \x0b as whitespace, bash keeps it inside the final
            # argument, so an entry must not authorize its control-character
            # neighbour.  The control screen runs before the lookup.
            ("break-glass does not stretch over a control character",
             bash("sudo ncu --set full python3 y.py\x0b"), "deny"),
            ("break-glass does not grant a longer line",
             bash("sudo ncu --set full python3 y.py --set roofline"), "deny"),
            ("break-glass does not grant a neighbouring command",
             bash("sudo ncu --set full python3 z.py"), "deny"),
            ("break-glass comment lines grant nothing",
             bash("# owner break-glass; delete this file when the incident "
                  "is over"), "deny"),
            ("break-glass blank lines grant nothing", bash(" "), "deny"),
            ("unrelated command is still denied", bash("rm -rf /"), "deny"),
            ("malformed payload still denies with break-glass present",
             {"tool_input": None}, "deny"),
        ], project_dir=sandbox)

        # An oversized or non-regular marker grants nothing.
        marker.write_text("pwd -P\n" + ("#" + "x" * 200 + "\n") * 200)
        check("oversized break-glass grants nothing",
              decision(bash("pwd -P"), sandbox), "deny")
        marker.unlink()
        (sandbox / "elsewhere").write_text("pwd -P\n")
        marker.symlink_to(sandbox / "elsewhere")
        check("symlinked break-glass grants nothing",
              decision(bash("pwd -P"), sandbox), "deny")
        marker.unlink()
        marker.mkdir()
        check("directory break-glass grants nothing",
              decision(bash("pwd -P"), sandbox), "deny")
        shutil.rmtree(marker)

        # Without CLAUDE_PROJECT_DIR the lookup is simply unavailable.
        environment = dict(os.environ)
        environment.pop("CLAUDE_PROJECT_DIR", None)
        result = subprocess.run(
            [sys.executable, str(GUARD)],
            input=json.dumps(bash("nsys profile -o t python3 x.py")),
            text=True, capture_output=True, timeout=20, check=False,
            cwd=str(sandbox), env=environment,
        )
        check("break-glass is unavailable without CLAUDE_PROJECT_DIR",
              "allow" if not result.stdout.strip() else "deny", "deny")

    # The agent has no shell primitive that could create the marker.
    run_cases([
        ("agent cannot write the marker with a redirect",
         bash("echo x > .claude/BREAK_GLASS"), "deny"),
        ("agent cannot write the marker with tee",
         bash("tee .claude/BREAK_GLASS"), "deny"),
        ("agent cannot stage the marker",
         bash("git add -- .claude/BREAK_GLASS"), "deny"),
        ("agent cannot write the marker with sed",
         bash("""sed -n "1w .claude/BREAK_GLASS" -- Project/PLAN.md"""), "deny"),
    ])


# --------------------------------------------------------------------------
# 11. settings.json must agree with the hook
# --------------------------------------------------------------------------
def _rule_matches(pattern: str, path: str) -> bool:
    regex = ""
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            regex += "(?:.*/)?"
            index += 3
        elif pattern.startswith("**", index):
            regex += ".*"
            index += 2
        elif pattern[index] == "*":
            regex += "[^/]*"
            index += 1
        else:
            regex += re.escape(pattern[index])
            index += 1
    return re.fullmatch(regex, path) is not None


def settings_consistency() -> None:
    try:
        document = json.loads(SETTINGS.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        check("settings.json parses as JSON", f"error({exc})", "ok")
        return
    check("settings.json parses as JSON", "ok", "ok")

    deny = document.get("permissions", {}).get("deny", [])
    check("settings deny list is a list of strings",
          "ok" if isinstance(deny, list) and all(isinstance(x, str) for x in deny)
          else "bad", "ok")

    def denied(path: str) -> bool:
        return any(
            _rule_matches(rule[len(tool) + 1:-1], path)
            for rule in deny
            for tool in ("Edit", "Write")
            if rule.startswith(tool + "(") and rule.endswith(")")
        )

    must_be_denied = [
        ".claude/settings.json",
        ".claude/hooks/guard_bash.py",
        ".git/config",
        # A writable .gitignore/.gitattributes hides work from the owner's own
        # `git status`/`git diff` review, which is the only human check on
        # everything the agent commits.
        ".gitignore",
        ".gitattributes",
        "Project/.gitignore",
        "Project/kernels/.gitattributes",
        "CLAUDE.md",
        "README.md",
        "torch_transformer_benchmark.py",
        "tensorflow_transformer_benchmark.py",
        "Project/PLAN.md",
        "Project/RUNBOOK.md",
        "Project/GRIND_ENTRYPOINT.md",
        "Project/shapes.json",
        "Project/manifest.json",
        "Project/device_peaks.json",
        "Project/harness/runner.py",
        "Project/harness/trusted_controller.py",
        "Project/tools/run_gate.py",
        "Project/tools/champion_watch.py",
        "Project/tools/tests/staged_guard_test.py",
        "Project/authority/blobs/a.json",
        "Project/audits/verdicts.jsonl",
        "Project/results/JOURNAL.jsonl",
        "Project/results_side/SHIP_MANIFEST.json",
        "Project/lock_staging/guard_bash.py",
        "Project/loop/gate_state.json",
        "Project/loop/gate_log.jsonl",
        "Project/loop/mechanism_catalog.json",
        "Project/loop/mechanism_catalog.schema.json",
        "Project/loop/permits_used/p1.json",
        "Project/loop/requests/r1.json",
        "Project/loop/profile_evidence/profile-1.json",
        "Project/submission/torch_transformer_benchmark_submission.py",
    ]
    for path in must_be_denied:
        check(f"settings denies Edit/Write on {path}",
              "denied" if denied(path) else "writable", "denied")

    # Deliberately writable.  Two different reasons, both deliberate:
    #
    # 1. The agent's own work record (memory/*, HANDOVER.md, research, drafts).
    #    The standing orders mandate these -- "log decisions in DECISIONS.md,
    #    lessons in LESSONS.md as they happen" -- and owner_lock_ceremony.py's
    #    own DELIBERATE_EXCLUSIONS says Project/memory/** is "written every
    #    session by design".  A write lock on a mandated file does not make the
    #    prose true; it just stops the record, and an agent that cannot
    #    discharge a standing order improvises an unpoliced shadow record
    #    instead.  LESSONS #24 (unsourced numbers typed into STATE.md and later
    #    quoted as fact) is a claim-provenance failure, and the countermeasure
    #    that actually works is above: every number that can be checked lives
    #    in a machine-written artifact the agent cannot Edit at all.
    #
    # 2. Project/loop/cards.jsonl is the direction card, and run_gate.py
    #    REFUSES `plan` until an open card exists for the family ("Open the
    #    card first"), so locking it would freeze the loop shut -- which is
    #    exactly what the ceremony's exclusion list says.  Cards carry no
    #    authority: identity comes from trusted_family()/the catalog, never
    #    from card text.
    must_be_writable = [
        "Project/memory/DECISIONS.md",
        "Project/memory/LESSONS.md",
        "Project/memory/STATE.md",
        "Project/HANDOVER.md",
        "Project/loop/cards.jsonl",
        "Project/loop/lineage.jsonl",
        "Project/kernels/k100.py",
        "Project/submission/dispatcher_region.py",
        "Project/research/new-note.md",
        "Project/drafts/tech_report_draft.md",
    ]
    for path in must_be_writable:
        check(f"settings leaves {path} writable",
              "writable" if not denied(path) else "denied", "writable")

    hooks = document.get("hooks", {})
    pre = json.dumps(hooks.get("PreToolUse", []))
    check("PreToolUse installs the guard on Bash",
          "ok" if ".claude/hooks/guard_bash.py" in pre and '"Bash"' in pre
          else "missing", "ok")
    session = json.dumps(hooks.get("SessionStart", []))
    check("SessionStart still injects STATE.md",
          "ok" if "STATE.md" in session else "missing", "ok")
    check("SessionStart runs the bootstrap health check",
          "ok" if "session_bootstrap.py" in session else "missing", "ok")

    # Every python entrypoint the hook allows must be a file that exists and
    # must itself be Edit/Write protected, or the allowlist is self-defeating.
    source = GUARD.read_text(encoding="utf-8")
    scripts = sorted(set(re.findall(r'"(Project/[A-Za-z0-9_/]+\.py)":', source)))
    check("hook lists python entrypoints", "ok" if scripts else "none", "ok")
    for script in scripts:
        check(f"allowlisted entrypoint exists: {script}",
              "present" if (ROOT / script).is_file() else "missing", "present")
        check(f"allowlisted entrypoint is write-protected: {script}",
              "denied" if denied(script) else "writable", "denied")


def main() -> int:
    malformed_payloads()
    allowed_surface()
    profiler_policy()
    command_execution_holes()
    tokenization()
    path_policy()
    path_boundary_semantics()
    symlink_policy()
    git_surface()
    commit_messages()
    entrypoint_policy()
    general_surface()
    break_glass()
    settings_consistency()
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nALL GREEN")
    for failure in FAILURES:
        print(failure)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
