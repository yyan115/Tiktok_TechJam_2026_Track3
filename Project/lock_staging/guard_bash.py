#!/usr/bin/env python3
"""Fail-closed Bash allowlist installed by the owner during LOCK.

This hook is deliberately a command-shape parser, not a shell regex trying to
recognize every dangerous spelling.  Anything outside the small GRIND surface
is denied.  Candidate editing uses Claude's Edit/Write tools in the mutable
implementation roots; authority-changing execution goes through locked Python
entrypoints below.

Design rules this file obeys, in order of importance:

1.  DENY IS THE DEFAULT AND THE ONLY SAFE FAILURE.  Every exception path,
    including BaseException, prints a deny decision.  A hook that stays
    silent is read as "no opinion" and the command runs.

2.  NO ALLOWED PROGRAM MAY BE A GENERAL EXECUTOR OR A GENERAL WRITER.
    A tool is on this list only if the argument grammar accepted here makes
    it incapable of launching another program or naming an arbitrary output
    file.  This is why ``sed`` accepts only a numeric print range (GNU sed's
    ``e`` command executes shells and ``w``/``W`` write arbitrary paths), why
    ``rg`` flags are an allowlist rather than a blocklist (``--pre=CMD``
    executes, and an exact-token blocklist never sees the ``=`` form), and
    why no profiler is here.

    On profilers specifically, since this is the project's own thesis:
    ``nsys``, ``ncu`` and ``compute-sanitizer`` are process launchers whose
    whole purpose is to run an arbitrary command line, and each also takes an
    ``-o``/``--log-file`` argument naming a file to write.  There is no
    argument grammar that keeps them from re-admitting ``python3 -c``, so
    admitting one admits everything.  The replacement is not "no profiling":
    ``run_gate.py diagnostic`` -> a one-use permit -> the controller's
    sandboxed ``diagnostic`` lane, which is additionally the *only* route that
    produces gate-admissible counter evidence.  A shell-run profile could not
    be cited by ``plan --counter-evidence`` even if it were allowed, so
    admitting the launcher would buy capability the gate cannot consume while
    costing the whole allowlist.  Two honest gaps remain and are named rather
    than papered over: ``ncu`` needs root on this box, and
    ``compute-sanitizer`` is not in the catalog's ``evidence_tools``, so
    neither is reachable by any agent route.  Both are owner work -- pre-LOCK,
    or post-LOCK through the break-glass file below.  The inert ``--version``
    probe is allowed so the agent can tell "tool missing" from "gate refused".

3.  THE ALLOWLIST MUST NOT STRANGLE THE WORK IT POLICES.  A block whose
    cause the agent cannot see is a block it will retry forever.  Denials
    name the rule.  Legitimate commit messages are not rejected for their
    punctuation (see ``allow_verbatim_commit``), and every workflow the
    standing orders require -- opening a direction card, logging lessons and
    decisions, committing the machine-written ledgers -- is reachable.

4.  ONE OWNER BREAK-GLASS PATH, NEVER MINTABLE BY THE AGENT.  See
    ``break_glass_allows``.  The agent has no write primitive of any kind to
    ``.claude/``: no shell command here writes files, and the companion
    settings.json denies Edit/Write on ``.claude/**``.

Silence means allow, so this hook is a veto.  It presumes the owner runs the
grind with Bash pre-approved (bypass/accept mode); deny decisions and the
settings.json Edit/Write deny rules are still enforced in those modes.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

MAX_COMMAND_LENGTH = 4096
MAX_COMMIT_MESSAGE = 200
MAX_BREAK_GLASS_BYTES = 16384
MAX_BREAK_GLASS_LINES = 64

# Shell metacharacters.  Chaining, redirection, expansion, substitution,
# globbing and backslash escapes are all removed before tokenization, which
# is what keeps shlex's parse and bash's parse from being able to disagree.
META = re.compile(r"[\n\r|;&><`$\\{}*?\[\]]")
# Control characters never legitimately appear inside a path token.
CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# ... nor anywhere in a command.  Tab, newline and carriage return are left to
# META and to shlex (which treats them as separators, as bash does); every
# other control character is a place where shlex's idea of a token and bash's
# could be argued about, so none is admitted.
COMMAND_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

PYTHON = {"python3", "/usr/bin/python3"}

# ---------------------------------------------------------------------------
# Python entrypoints
#
# value None      -> any arguments; the program performs its own exact
#                    schema/path/authority checks and is itself protected
#                    from edits by settings.json.
# value frozenset -> the space-joined argument tail must be an exact member.
# value callable  -> validator over the argument tail.
#
# DELIBERATELY ABSENT, with the route that replaces each one:
#   Project/harness/candidate_worker.py  - sandbox payload; only the
#       controller may launch it, and only after consuming a one-use permit.
#   Project/harness/profile_worker.py    - same, for the diagnostic lane.
#   Project/tools/audit_champion.py      - champion_watch.py mints the
#       attempt id and nonce and writes the durable attempt_started row
#       first; record_audit_result() refuses any attempt without exactly one
#       matching start (audit_authority.py), so a hand-run auditor cannot
#       record a verdict -- it can only burn codex runs, strand markers and
#       re-fire outside the watch's retry cap.
#   Project/tools/shape6_local_eval.py   - reachable only as
#   Project/tools/shape14_eval.py          `run_gate.py side-evaluate` ->
#       `trusted_controller.py side --permit`.  Shapes 6 and 14 bypassing
#       all authority was a top external-audit finding; this keeps it shut.
#   Project/tools/owner_lock_ceremony.py - the owner half of the control
#       plane; it is the only code that touches private key material.  Not
#       even its read-only `verify` subcommand is admitted: `verify` shares an
#       argv surface with `keygen` and `mint-capability --private-key`, and an
#       agent that can reach the signing program at all is one allowlist edit
#       away from signing its own authorization.  Everything `verify` reports
#       is already available through `trusted_controller.py verify-lock` and
#       `status`, which are on this list and touch no private material.
#   Project/tools/dashboard.py           - a streamlit server, not a python3
#       entrypoint, and long-running.
#   Project/tools/smokes/*.py            - raw-dial GPU benchmarking, which
#       standing order 3 forbids outright.
# ---------------------------------------------------------------------------


def _only_flags(*permitted: str):
    """Accept any subset of ``permitted``, in any order, with no duplicates.

    Deliberately NOT an exact match on the joined argument tail.  That older
    form made the allowlist order-sensitive: ``--dry-run --no-reconcile`` was
    admitted and ``--no-reconcile --dry-run`` was refused, for no reason the
    agent could see from the denial.  Rule 3 calls that out as its own kind of
    failure -- a bounce whose cause is invisible is a bounce the agent retries
    forever.  None of the flags spelled here takes a value, so no argument
    accepted by this validator can name a file.
    """
    allowed = frozenset(permitted)
    if any(not flag.startswith("--") for flag in allowed):  # pragma: no cover
        raise ValueError("only long value-less flags may be listed here")

    def check(args: list[str]) -> bool:
        return len(set(args)) == len(args) and set(args) <= allowed

    return check


def _ship_manifest_args(args: list[str]) -> bool:
    """`--evidence-map <repo path>`, plus any of `--report` / `--diagnose`.

    `--diagnose` is read-only by construction (it writes nothing and explains
    per shape what evidence is missing), and it is the natural first call when
    the manifest refuses -- so it is admitted on its own, without an evidence
    map, which is exactly how ship_manifest.py's own argument parser treats it.

    `--output` is refused here even though ship_manifest.py confines it to
    Project/results_side: pinning the destination keeps this allowlist free of
    any argument that names a file to be written.  Order does not matter.
    """
    flags: list[str] = []
    evidence_map: str | None = None
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"--report", "--diagnose"}:
            if token in flags:
                return False
            flags.append(token)
            index += 1
            continue
        if token == "--evidence-map":
            if evidence_map is not None or index + 1 >= len(args):
                return False
            evidence_map = args[index + 1]
            index += 2
            continue
        return False
    if evidence_map is None:
        return "--diagnose" in flags
    return readable_path(evidence_map)


SAFE_PYTHON_SCRIPTS: dict[str, object] = {
    "Project/tools/run_gate.py": None,
    "Project/harness/trusted_controller.py": None,
    # Post-LOCK runner.py is a bare os.execv shim onto trusted_controller.py,
    # so this is the controller under the name standing order 3 uses.
    "Project/harness/runner.py": None,
    "Project/tools/champion_watch.py": _only_flags("--dry-run", "--no-reconcile"),
    "Project/tools/build_submission.py": _only_flags("--check-only"),
    "Project/tools/session_bootstrap.py": _only_flags(),
    "Project/tools/sensitivity_board.py": _only_flags(),
    "Project/tools/ship_manifest.py": _ship_manifest_args,
    "Project/tools/tests/audit_authority_test.py": _only_flags(),
    "Project/tools/tests/authority_store_test.py": _only_flags(),
    "Project/tools/tests/champion_watch_test.py": _only_flags(),
    "Project/tools/tests/competence_gate_test.py": _only_flags(),
    "Project/tools/tests/evidence_submission_test.py": _only_flags(),
    "Project/tools/tests/gate_v4_cli_test.py": _only_flags(),
    "Project/tools/tests/integration_authority_test.py": _only_flags(),
    "Project/tools/tests/guard_and_auditor_test.py": _only_flags(),
    "Project/tools/tests/lock_manifest_test.py": _only_flags(),
    "Project/tools/tests/sandbox_boundary_test.py": _only_flags(),
    "Project/tools/tests/staged_guard_test.py": _only_flags(),
    "Project/tools/tests/trusted_controller_test.py": _only_flags(),
}

# ---------------------------------------------------------------------------
# Path policy
#
# STAGEABLE (`git add`) is broad on purpose.  Staging cannot alter a byte; it
# only makes what is already on disk durable.  Integrity comes from the write
# side -- no shell command here writes files, and settings.json denies
# Edit/Write on every machine-written and canonical path.  Refusing to stage
# the ledgers would leave the audit trail alive only in an uncommitted
# working tree, which is how this repo nearly lost it once already.
#
# The exclusions are the referee and enforcer bytes themselves -- the control
# plane, the gate and its tests, and the pre-image of this hook: even if a
# write primitive to them were ever found, it could not be made durable
# through this allowlist.  Owner commits those.
#
# Prefix boundaries matter on both lists; see ``_under``.  A root naming a
# FILE must match that file and nothing that merely starts with its name, or
# "Project/submission/dispatcher_region.py" would also admit
# "...dispatcher_region.py.evil"; a root naming a DIRECTORY must match on a
# path separator, or "Project/harness/" would fail to exclude nothing while
# "Project/res" would wrongly capture "Project/results_side".
# ---------------------------------------------------------------------------
STAGEABLE_ROOT = "Project/"
STAGEABLE_EXCLUDED = (
    "Project/harness/",
    "Project/tools/",
    # The exact bytes the owner installs at LOCK.  Post-LOCK they are a
    # historical artifact; letting the agent commit a changed copy would let a
    # future reader quote a file that is not what is actually enforced.
    "Project/lock_staging/",
)
READABLE_ROOTS = (
    "Project/",
    "README.md",
    "CLAUDE.md",
    "torch_transformer_benchmark.py",
    "tensorflow_transformer_benchmark.py",
)

READ_COMMANDS = ("head", "tail", "wc", "sha256sum", "stat", "ls")
NUMERIC_READ_FLAG = re.compile(r"-[nc][0-9]{1,9}")
LS_FLAG = re.compile(r"-[1aAdFhlrRSt]{1,8}")
WC_FLAG = re.compile(r"-[lwmc]{1,4}")
SED_SCRIPT = re.compile(r"[0-9]{1,9}(,[0-9]{1,9})?p")
RG_CONTEXT_FLAG = re.compile(r"-[ABC][0-9]{1,4}")
# Allowlist, not blocklist.  rg's executing flags (--pre, --pre-glob,
# --hostname-bin, -z/--search-zip) all accept a `--flag=value` spelling that
# an exact-token blocklist silently misses.
# -e/--regexp takes the next token as a pattern, whatever it looks like.  It
# is the only way to search for a literal that begins with "-" (``rg -n "-rf"``
# is refused because the pattern parses as a flag), and rg never reinterprets
# that token as an option, so nothing executable can hide in it.
RG_PATTERN_FLAGS = frozenset({"-e", "--regexp"})
RG_FLAGS = frozenset({
    "-n", "--line-number", "-N", "--no-line-number",
    "-i", "--ignore-case", "-S", "--smart-case", "-s", "--case-sensitive",
    "-w", "--word-regexp", "-F", "--fixed-strings", "-x", "--line-regexp",
    "-v", "--invert-match", "-o", "--only-matching",
    "-l", "--files-with-matches", "--files-without-match",
    "-c", "--count", "--count-matches",
    "-H", "--with-filename", "-I", "--no-filename",
    "--no-heading", "--heading", "--hidden", "--no-ignore", "--no-ignore-vcs",
    "--stats", "--trim", "--vimgrep", "--column", "--byte-offset",
    "-U", "--multiline", "--multiline-dotall",
    "--files", "--no-messages", "--json",
    "--color=never", "--sort=path", "--no-config",
})
# Read-only nvidia-smi surface.  No control token (-pl, -lgc, -rgc, -ac, -pm,
# -e, --gpu-reset) is present, and none may ever be added.
NVIDIA_SMI_TOKENS = frozenset({
    "-L", "-q", "--query-gpu=name,driver_version,memory.total",
    "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,power.draw",
    "--query-compute-apps=pid,used_memory",
    "--format=csv", "--format=csv,noheader",
})
# Profilers are launchers, so only their inert version probe is admitted.
# It lets the agent tell "tool missing / driver blocked" apart from "gate
# refused" and lets it verify the tool_version a profile artifact claims.
PROFILER_PROBES = (
    ["nsys", "--version"],
    ["ncu", "--version"],
    ["compute-sanitizer", "--version"],
)
# Anchored on the ORIGINAL command text.  A single-quoted bash word has
# exactly one terminator and no expansion, escape or history behaviour of any
# kind, so a message that contains no single quote, CR, LF or NUL provably
# parses as one literal word.  This is what keeps punctuation ("3->1",
# "fix $HOME path") out of the false-positive class that the old blacklist
# guard fell into when it rejected the word "clean" after "git".
VERBATIM_COMMIT = re.compile(
    r"git commit -m '([^'\r\n\x00]{1," + str(MAX_COMMIT_MESSAGE) + r"})'"
)
# Read-only revision names for `git diff`/`show`/`log`/`rev-parse`.  None of
# these is executable and none can name an output file; git resolves them and
# prints.  Without them the agent cannot say `git show HEAD --stat` after the
# "commit candidate bytes before first controller contact" step, which is a
# workflow the standing orders require.
REVISION = re.compile(r"HEAD(~[0-9]{1,3}|\^{1,3})?|[0-9a-f]{4,64}")
# `git show <rev>:<repo path>` prints the committed bytes of one file.  That is
# how the agent verifies that what it measured is what it committed, so the
# path half is checked with exactly the same reader policy as everything else.
REVISION_PATH = re.compile(r"(HEAD(~[0-9]{1,3}|\^{1,3})?|[0-9a-f]{4,64}):(.+)",
                           re.DOTALL)


def git_revision(arg: str) -> bool:
    if REVISION.fullmatch(arg):
        return True
    match = REVISION_PATH.fullmatch(arg)
    return bool(match) and readable_path(match.group(3))


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


def project_root() -> Path:
    value = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if value and os.path.isabs(value):
        return Path(value)
    return Path.cwd()


def traverses_symlink(value: str) -> bool:
    """True if any existing component of a repo-relative path is a symlink.

    safe_repo_path is otherwise purely lexical, so a symlink planted inside a
    permitted root would launder a read of anything on the filesystem.  Only
    the relative components are inspected: the project root itself is allowed
    to sit under a symlinked mount (/home -> /var/home on Fedora).
    """
    try:
        current = project_root()
        for part in Path(value).parts:
            current = current / part
            if current.is_symlink():
                return True
            if not current.exists():
                return False
    except OSError:
        return True
    return False


def _plain_repo_path(value: str) -> str | None:
    """Reject anything that is not an ordinary relative in-repo path."""
    if not value or value.startswith("-") or os.path.isabs(value):
        return None
    if CONTROL.search(value):
        return None
    pure = Path(value)
    parts = pure.parts
    if ".." in parts or ".git" in parts or ".claude" in parts:
        return None
    if any(part in {"", "."} for part in parts):
        return None
    normalized = pure.as_posix()
    if normalized.startswith("/") or ".." in normalized.split("/"):
        return None
    if traverses_symlink(normalized):
        return None
    return normalized


def _under(normalized: str, root: str) -> bool:
    """Directory roots match with a separator boundary; file roots exactly.

    Without the boundary, root "Project/submission/dispatcher_region.py"
    also matches "...dispatcher_region.py.evil", and a root spelled
    "Project/res" would match "Project/results_side".
    """
    if root.endswith("/"):
        return normalized == root[:-1] or normalized.startswith(root)
    return normalized == root


def readable_path(value: str) -> bool:
    normalized = _plain_repo_path(value)
    if normalized is None:
        return False
    return any(_under(normalized, root) for root in READABLE_ROOTS)


def stageable_path(value: str) -> bool:
    normalized = _plain_repo_path(value)
    if normalized is None:
        return False
    if not _under(normalized, STAGEABLE_ROOT):
        return False
    for blocked in STAGEABLE_EXCLUDED:
        if _under(normalized, blocked):
            return False
        # Refuse a path that is an ancestor of an excluded root, so that
        # `git add -- Project` cannot stage Project/harness by containment.
        if (blocked + "/").startswith(normalized.rstrip("/") + "/"):
            return False
    return True


def safe_repo_path(value: str, *, mutable: bool = False) -> bool:
    return stageable_path(value) if mutable else readable_path(value)


def allow_python(tokens: list[str]) -> bool:
    if len(tokens) < 2 or tokens[0] not in PYTHON:
        return False
    script = tokens[1]
    if script not in SAFE_PYTHON_SCRIPTS:
        return False
    policy = SAFE_PYTHON_SCRIPTS[script]
    args = tokens[2:]
    if policy is None:
        # These locked programs perform their own exact schema/path/authority
        # checks.  Python flags, -c/-m, alternate scripts, and stdin execution
        # never reach this branch: tokens[1] must equal a key exactly.
        return True
    if callable(policy):
        return bool(policy(args))
    return " ".join(args) in policy


def allow_git(tokens: list[str]) -> bool:
    if not tokens or tokens[0] != "git" or len(tokens) < 2:
        return False
    subcommand = tokens[1]
    args = tokens[2:]
    # `git -c ...` / `git --exec-path=...` land here as the subcommand and
    # fall through to the final return False.
    if any(arg.startswith("--output") for arg in args):
        return False
    if subcommand == "status":
        return all(arg in {"--short", "--porcelain", "--branch"} for arg in args)
    if subcommand == "diff":
        allowed = {"--check", "--cached", "--staged", "--stat", "--shortstat",
                   "--numstat", "--name-only", "--name-status", "--"}
        return all(
            arg in allowed or git_revision(arg) or readable_path(arg)
            for arg in args
        )
    if subcommand == "log":
        # `--patch`/`-p` is absent: with an (unreachable) diff.external config
        # it would be an execution path, and the diff is available separately.
        allowed = {"--oneline", "--decorate", "--stat", "--all", "--graph",
                   "--name-only", "--name-status", "--"}
        return all(
            arg in allowed
            or re.fullmatch(r"-[0-9]{1,3}", arg)
            or git_revision(arg)
            or readable_path(arg)
            for arg in args
        )
    if subcommand == "show":
        allowed = {"--stat", "--name-only", "--name-status", "--oneline", "--"}
        return all(
            arg in allowed or git_revision(arg) or readable_path(arg)
            for arg in args
        )
    if subcommand == "rev-parse":
        allowed = {"--show-toplevel", "--abbrev-ref", "--short", "--verify"}
        return bool(args) and all(
            arg in allowed or git_revision(arg) for arg in args
        )
    if subcommand == "branch":
        return args == ["--show-current"]
    if subcommand == "add":
        return len(args) >= 2 and args[0] == "--" and all(
            stageable_path(arg) for arg in args[1:]
        )
    if subcommand == "commit":
        return (
            len(args) == 2
            and args[0] == "-m"
            and 1 <= len(args[1]) <= MAX_COMMIT_MESSAGE
        )
    return False


def allow_verbatim_commit(command: str) -> bool:
    """Exact `git commit -m '<message>'` over the raw, untokenized text."""
    return VERBATIM_COMMIT.fullmatch(command.strip()) is not None


def _read_flags_ok(command: str, flags: list[str]) -> bool:
    index = 0
    while index < len(flags):
        token = flags[index]
        if command in {"head", "tail"}:
            # -f/--follow is absent on purpose: it never terminates.
            if token in {"-q", "-v"} or NUMERIC_READ_FLAG.fullmatch(token):
                index += 1
                continue
            if (token in {"-n", "-c"} and index + 1 < len(flags)
                    and flags[index + 1].isdigit()):
                index += 2
                continue
            return False
        if command == "wc" and WC_FLAG.fullmatch(token):
            index += 1
            continue
        if command == "ls" and LS_FLAG.fullmatch(token):
            index += 1
            continue
        # sha256sum and stat take no flags at all: sha256sum -c turns a read
        # tool into an arbitrary-path existence oracle, and stat -L follows
        # symlinks back out of the repo.
        return False
    return True


def allow_read(tokens: list[str]) -> bool:
    if not tokens:
        return False
    command = tokens[0]
    rest = tokens[1:]
    if command == "pwd":
        return not rest
    if command in {"uname", "date"}:
        return tokens in (["uname", "-a"], ["date", "-u"])
    if tokens in PROFILER_PROBES:
        return True
    if command == "nvidia-smi":
        return all(token in NVIDIA_SMI_TOKENS for token in rest)
    if command == "rg":
        if "--" not in rest:
            return False
        separator = rest.index("--")
        head, paths = rest[:separator], rest[separator + 1:]
        if not paths or not all(readable_path(path) for path in paths):
            return False
        pattern_seen = False
        index = 0
        while index < len(head):
            token = head[index]
            if token in RG_PATTERN_FLAGS:
                if pattern_seen or index + 1 >= len(head):
                    return False
                pattern_seen = True
                index += 2
                continue
            if token.startswith("-"):
                if token not in RG_FLAGS and not RG_CONTEXT_FLAG.fullmatch(token):
                    return False
            elif pattern_seen:
                return False
            else:
                pattern_seen = True
            index += 1
        return not pattern_seen if "--files" in head else pattern_seen
    if command == "sed":
        # Only a numeric print range.  A free-form script would admit GNU
        # sed's `e` (execute) and `w`/`W` (write anywhere) commands, neither
        # of which contains a metacharacter this guard screens for.
        return (
            len(tokens) >= 5
            and tokens[1] == "-n"
            and SED_SCRIPT.fullmatch(tokens[2]) is not None
            and tokens[3] == "--"
            and all(readable_path(path) for path in tokens[4:])
        )
    if command in READ_COMMANDS:
        if "--" not in rest:
            return False
        separator = rest.index("--")
        if not _read_flags_ok(command, rest[:separator]):
            return False
        paths = rest[separator + 1:]
        return bool(paths) and all(readable_path(path) for path in paths)
    return False


def break_glass_allows(command: str) -> bool:
    """The single owner escape hatch: <project>/.claude/BREAK_GLASS.

    Each non-blank, non-``#`` line is one command the owner has authorized
    verbatim; matching is exact, never prefix, so an entry grants that
    command and nothing adjacent to it.  The owner deletes the file when the
    incident is over.

    The agent cannot mint one.  Nothing on this allowlist writes a file, and
    the companion settings.json denies Edit/Write on ``.claude/**``; the same
    ``.claude`` component is also refused by _plain_repo_path, so the file
    cannot even be read back.  Its integrity therefore rests on exactly the
    same anchor as this hook's own bytes, which is the correct level: an
    agent that could write ``.claude/`` would simply rewrite the hook.
    """
    root = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if not root or not os.path.isabs(root):
        return False
    path = Path(root) / ".claude" / "BREAK_GLASS"
    try:
        if path.is_symlink() or not path.is_file():
            return False
        if path.stat().st_size > MAX_BREAK_GLASS_BYTES:
            return False
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    wanted = command.strip()
    if not wanted:
        return False
    for line in text.splitlines()[:MAX_BREAK_GLASS_LINES]:
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry == wanted:
            return True
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
        # Ahead of break-glass on purpose.  break_glass_allows compares the
        # STRIPPED command against its entries, and Python strips \x0b/\x0c as
        # whitespace while bash keeps them inside the final argument -- so an
        # entry for "... python3 k.py" would otherwise also authorize
        # "... python3 k.py\x0b", which runs a different file.  Screening
        # first keeps break-glass matching exact, as documented.
        if COMMAND_CONTROL.search(command):
            raise Denied("command contains a control character")
        if break_glass_allows(command):
            return
        if allow_verbatim_commit(command):
            return
        if META.search(command):
            raise Denied(
                "shell metacharacters, expansion, chaining, and redirection "
                "are forbidden; a commit message may keep its punctuation if "
                "the whole command is written as: git commit -m '<message>'"
            )
        tokens = shlex.split(command, posix=True)
        if not tokens or any("\x00" in token for token in tokens):
            raise Denied("command tokenization failed")
        if allow_python(tokens) or allow_git(tokens) or allow_read(tokens):
            return
        raise Denied("command shape is not on the post-LOCK allowlist")
    except Denied as exc:
        deny(str(exc))
    except BaseException as exc:  # noqa: BLE001 - silence would mean allow
        deny(f"malformed payload or internal guard failure ({type(exc).__name__})")


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 - last resort; never exit silent
        try:
            deny("guard failed before reaching a decision")
        except BaseException:  # noqa: BLE001
            sys.stdout.write(
                '{"hookSpecificOutput":{"hookEventName":"PreToolUse",'
                '"permissionDecision":"deny",'
                '"permissionDecisionReason":"LOCK allowlist: guard failure"}}'
            )
    raise SystemExit(0)
