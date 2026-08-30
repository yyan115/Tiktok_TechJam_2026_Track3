#!/usr/bin/env python3
"""The auditor backend: which binary reviews the work, and how it is trusted.

The audit lane's whole value rests on the auditor being something the agent
under review cannot become. Two properties carry that, and both are cheap to
lose in a refactor:

  1. The auditor binary is pinned by absolute path AND content hash, and lives
     somewhere the agent cannot write. `resolve_auditor_identity` refuses
     anything under HOME for exactly this reason -- the stock Claude Code
     install is `~/.local/share/claude/versions/<v>`, which the agent can
     overwrite with a script that prints PASS.
  2. The verdict is parsed and schema-validated from the backend's output, and
     a transport failure is reported as a transport failure rather than as an
     empty or malformed verdict that some later branch might read as benign.

There is a third property that no test can enforce, so it is asserted as a
recorded fact instead: whether the auditor is a different vendor from the work
under review. Codex is; Claude is not. The backend table says so, every audit
artifact carries it, and the report has to say it out loud.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "Project" / "tools"))

import audit_champion as ac  # noqa: E402
from audit_authority import AuditAuthorityError, resolve_auditor_identity  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    checks.append((name, bool(condition), detail))
    print(("PASS " if condition else "FAIL ") + name
          + (f"  [{detail}]" if detail and not condition else ""))


def raises(fn, fragment: str) -> tuple[bool, str]:
    try:
        fn()
    except Exception as exc:
        return fragment in str(exc), f"{type(exc).__name__}: {exc}"
    return False, "no exception raised"


def main() -> int:
    home = Path(os.environ.get("HOME") or Path.home()).resolve()

    # ---- 1. no backend may live where the agent can rewrite it -----------
    for name, backend in sorted(ac.AUDITOR_BACKENDS.items()):
        exe = Path(backend["executable"])
        check(f"{name}: auditor path is absolute", exe.is_absolute(), str(exe))
        under_home = True
        try:
            exe.resolve().relative_to(home)
        except ValueError:
            under_home = False
        check(f"{name}: auditor path is outside HOME", not under_home, str(exe))
        check(f"{name}: pinned hash is 64 hex",
              len(backend["sha256"]) == 64
              and all(c in "0123456789abcdef" for c in backend["sha256"]),
              backend["sha256"])

    # The resolver must refuse a HOME binary even when the hash matches, since
    # a hash the agent can recompute after editing is not a control.
    stock = Path.home() / ".local" / "bin" / "claude"
    if stock.exists():
        ok, detail = raises(
            lambda: resolve_auditor_identity(stock.resolve(), "0" * 64),
            "under HOME is refused")
        check("the stock ~/.local Claude install is refused as an auditor",
              ok, detail)

    # ---- 2. backend selection --------------------------------------------
    name, _ = ac.selected_backend()
    check("a default backend is selected", name in ac.AUDITOR_BACKENDS, name)
    name, _ = ac.selected_backend("codex")
    check("codex can still be selected explicitly", name == "codex", name)
    ok, detail = raises(lambda: ac.selected_backend("gpt5-turbo"),
                        "unknown auditor backend")
    check("an unknown backend refuses instead of falling back", ok, detail)

    # ---- 3. the envelope must not launder a failure into a verdict -------
    good = '{"is_error": false, "result": "{\\"schema_version\\": 2}"}'
    check("a healthy envelope yields the verdict text",
          ac._claude_extract(good) == '{"schema_version": 2}',
          ac._claude_extract(good))

    for label, payload, fragment in (
        ("non-JSON output", "Error: something broke", "not JSON"),
        ("a JSON array", "[1, 2, 3]", "not a JSON object"),
        ("an error envelope",
         '{"is_error": true, "result": "rate limit"}', "reported an error"),
        ("an empty result",
         '{"is_error": false, "result": ""}', "no result text"),
        ("a missing result", '{"is_error": false}', "no result text"),
    ):
        ok, detail = raises(lambda p=payload: ac._claude_extract(p), fragment)
        check(f"{label} is reported, not treated as a verdict", ok, detail)

    # ---- 4. the schema reaches a backend that cannot enforce it ----------
    fields = dict(entry_id="run-" + "0" * 32, packet_path="p",
                  candidate_source_path="c", attempt_nonce="a" * 64,
                  packet_sha256="b" * 64, candidate_sha256="c" * 64,
                  measurement_event_sha256="d" * 64, lane="primary")
    claude_prompt = ac.build_prompt(ac.AUDITOR_BACKENDS["claude"], **fields)
    codex_prompt = ac.build_prompt(ac.AUDITOR_BACKENDS["codex"], **fields)
    check("the claude prompt carries the verdict schema inline",
          "BEGIN SCHEMA" in claude_prompt
          and '"attempt_nonce"' in claude_prompt)
    check("it warns that empty required fields are still required",
          "retest_request" in claude_prompt and "findings" in claude_prompt)
    check("the codex prompt does not, because --output-schema enforces it",
          "BEGIN SCHEMA" not in codex_prompt)
    check("both prompts still carry the untrusted-data warning",
          "UNTRUSTED QUOTED DATA" in claude_prompt
          and "UNTRUSTED QUOTED DATA" in codex_prompt)

    # ---- 5. the launch is read-only on both backends ---------------------
    cargv = ac._claude_argv(Path("/usr/local/bin/claude-auditor"), "prompt")
    check("claude runs in non-interactive print mode", "-p" in cargv)
    check("claude ignores this repository's settings and hooks",
          "--restricted" in cargv and "--strict-mcp-config" in cargv)
    denied = next((cargv[i + 1] for i, a in enumerate(cargv)
                   if a == "--disallowedTools"), "")
    for tool in ("Bash", "Edit", "Write", "WebFetch", "Task"):
        check(f"claude auditor cannot use {tool}", tool in denied, denied)
    check("the prompt is NOT passed as argv (variadic flags swallow it)",
          "prompt" not in cargv, str(cargv[-2:]))
    check("claude backend therefore sends the prompt on stdin",
          ac.AUDITOR_BACKENDS["claude"]["prompt_on_stdin"] is True)

    zargv = ac._codex_argv(Path("/usr/local/bin/codex"), "prompt")
    check("codex still runs read-only sandboxed",
          "-s" in zargv and "read-only" in zargv)
    check("codex still gets the machine-checked output schema",
          "--output-schema" in zargv)

    # ---- 6. independence is recorded, not assumed ------------------------
    check("codex is marked as an independent vendor",
          ac.AUDITOR_BACKENDS["codex"]["independent_vendor"] is True)
    check("claude is marked as NOT an independent vendor",
          ac.AUDITOR_BACKENDS["claude"]["independent_vendor"] is False)

    failed = [(n, d) for n, ok, d in checks if not ok]
    print(f"{len(checks) - len(failed)}/{len(checks)} passed"
          + (" — ALL GREEN" if not failed else f" — FAILURES: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
