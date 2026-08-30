#!/usr/bin/env python3
"""Fail-closed Bash allowlist installed by the owner during LOCK.

This hook is deliberately a command-shape parser, not a shell regex trying to
recognize every dangerous spelling.  Anything outside the small GRIND surface
is denied.  Candidate editing uses Claude's Edit/Write tools in the mutable
implementation roots; authority-changing execution goes through locked Python
entrypoints below.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

MAX_COMMAND_LENGTH = 4096
META = re.compile(r"[\n\r|;&><`$\\{}*?\[\]]")
PYTHON = {"python3", "/usr/bin/python3"}
SAFE_PYTHON_SCRIPTS = {
    "Project/tools/run_gate.py": None,
    "Project/harness/trusted_controller.py": None,
    "Project/tools/champion_watch.py": {
        "", "--dry-run", "--no-reconcile", "--dry-run --no-reconcile"
    },
    "Project/tools/build_submission.py": {""},
    "Project/tools/session_bootstrap.py": {""},
    "Project/tools/tests/authority_store_test.py": {""},
    "Project/tools/tests/audit_authority_test.py": {""},
    "Project/tools/tests/competence_gate_test.py": {""},
    "Project/tools/tests/evidence_submission_test.py": {""},
    "Project/tools/tests/guard_and_auditor_test.py": {""},
    "Project/tools/tests/lock_manifest_test.py": {""},
    "Project/tools/tests/sandbox_boundary_test.py": {""},
    "Project/tools/tests/staged_guard_test.py": {""},
    "Project/tools/tests/trusted_controller_test.py": {""},
}
MUTABLE_GIT_ROOTS = (
    "Project/kernels/",
    "Project/submission/dispatcher_region.py",
    "Project/submission/torch_transformer_benchmark_submission.py",
    "Project/research/",
    "Project/drafts/",
)
READABLE_ROOTS = (
    "Project/", "README.md", "CLAUDE.md", "PLAN.md", "RUNBOOK.md",
    "torch_transformer_benchmark.py", "tensorflow_transformer_benchmark.py",
)


class Denied(RuntimeError):
    pass


def deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"LOCK allowlist: {reason}",
        }
    }, sort_keys=True))


def safe_repo_path(value: str, *, mutable: bool = False) -> bool:
    if not value or value.startswith("-") or os.path.isabs(value):
        return False
    pure = Path(value)
    if ".." in pure.parts or ".git" in pure.parts or ".claude" in pure.parts:
        return False
    normalized = pure.as_posix()
    roots = MUTABLE_GIT_ROOTS if mutable else READABLE_ROOTS
    return any(
        normalized == root.rstrip("/") or normalized.startswith(root)
        for root in roots
    )


def allow_python(tokens: list[str]) -> bool:
    if len(tokens) < 2 or tokens[0] not in PYTHON:
        return False
    script = tokens[1]
    policy = SAFE_PYTHON_SCRIPTS.get(script, "absent")
    if policy == "absent":
        return False
    if policy is None:
        # These locked programs perform their own exact schema/path/authority
        # checks.  Python flags, -c/-m, alternate scripts, and stdin execution
        # never reach this branch.
        return True
    return " ".join(tokens[2:]) in policy


def allow_git(tokens: list[str]) -> bool:
    if not tokens or tokens[0] != "git" or len(tokens) < 2:
        return False
    subcommand = tokens[1]
    args = tokens[2:]
    if any(arg.startswith("--output") for arg in args):
        return False
    if subcommand == "status":
        return all(arg in {"--short", "--porcelain", "--branch"} for arg in args)
    if subcommand == "diff":
        allowed = {"--check", "--cached", "--stat", "--name-only", "--name-status", "--"}
        return all(arg in allowed or safe_repo_path(arg) for arg in args)
    if subcommand == "log":
        return all(
            arg in {"--oneline", "--decorate", "--stat", "--all"}
            or re.fullmatch(r"-[0-9]{1,3}", arg)
            for arg in args
        )
    if subcommand == "show":
        return all(
            arg in {"--stat", "--name-only", "--name-status", "--oneline"}
            or re.fullmatch(r"[0-9a-f]{4,64}", arg)
            for arg in args
        )
    if subcommand == "rev-parse":
        return args in (["HEAD"], ["--show-toplevel"], ["--abbrev-ref", "HEAD"])
    if subcommand == "branch":
        return args == ["--show-current"]
    if subcommand == "add":
        return len(args) >= 2 and args[0] == "--" and all(
            safe_repo_path(arg, mutable=True) for arg in args[1:]
        )
    if subcommand == "commit":
        return len(args) == 2 and args[0] == "-m" and 1 <= len(args[1]) <= 200
    return False


def allow_read(tokens: list[str]) -> bool:
    if not tokens:
        return False
    command = tokens[0]
    if command == "pwd":
        return len(tokens) == 1
    if command in {"uname", "date"}:
        return tokens in (["uname", "-a"], ["date", "-u"])
    if command == "nvidia-smi":
        return all(
            token in {"-L", "-q", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"}
            for token in tokens[1:]
        )
    if command == "rg":
        if any(token in {"--pre", "--pre-glob", "--hostname-bin"} for token in tokens):
            return False
        if "--files" in tokens:
            paths = [token for token in tokens[1:] if not token.startswith("-")]
            return not paths or all(safe_repo_path(path) for path in paths)
        if "--" not in tokens:
            return False
        separator = tokens.index("--")
        return separator >= 2 and all(safe_repo_path(path) for path in tokens[separator + 1:])
    if command == "sed":
        return (
            len(tokens) >= 5
            and tokens[1] == "-n"
            and tokens[3] == "--"
            and all(safe_repo_path(path) for path in tokens[4:])
        )
    if command in {"head", "tail", "wc", "sha256sum", "stat", "ls"}:
        if "--" not in tokens:
            return False
        separator = tokens.index("--")
        return all(safe_repo_path(path) for path in tokens[separator + 1:])
    return False


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise Denied("payload is not an object")
        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            raise Denied("tool_input is absent or malformed")
        command = tool_input.get("command")
        if not isinstance(command, str) or not command.strip():
            raise Denied("command is absent or empty")
        if len(command) > MAX_COMMAND_LENGTH:
            raise Denied("command exceeds the length cap")
        if META.search(command):
            raise Denied("shell metacharacters, expansion, chaining, and redirection are forbidden")
        tokens = shlex.split(command, posix=True)
        if not tokens or any("\x00" in token for token in tokens):
            raise Denied("command tokenization failed")
        if allow_python(tokens) or allow_git(tokens) or allow_read(tokens):
            return
        raise Denied("command shape is not on the post-LOCK allowlist")
    except Denied as exc:
        deny(str(exc))
    except Exception as exc:
        deny(f"malformed payload or internal guard failure ({type(exc).__name__})")


if __name__ == "__main__":
    main()
