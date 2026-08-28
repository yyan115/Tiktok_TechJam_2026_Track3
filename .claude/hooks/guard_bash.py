#!/usr/bin/env python3
"""PreToolUse guard for Bash commands.

Blocks shell commands that would WRITE to protected files (official benchmark
scripts, README.md, shapes.json, manifest.json, results ledger). Reads are fine.
This guards against accidents, not malice — see Project/PLAN.md trust model.
"""
import json
import re
import sys

PROTECTED = [
    r"torch_transformer_benchmark\.py",
    r"tensorflow_transformer_benchmark\.py",
    r"README\.md",
    r"shapes\.json",
    r"manifest\.json",
    r"Project/results/",
    r"JOURNAL\.jsonl",
    r"LEADERBOARD\.md",
    r"\.claude/",
    r"Project/harness/",
    r"runner\.py",
]
PROT = "(" + "|".join(PROTECTED) + ")"

# Write-style operations aimed at a protected path. Deliberately simple:
# catches redirects, in-place edits, deletes, moves, copies-onto, tee, truncate.
WRITE_PATTERNS = [
    r">>?\s*\S*" + PROT,                      # > file, >> file
    r"\btee\b(\s+-\S+)*\s+\S*" + PROT,        # tee [-a] file
    r"\bsed\b[^|;&]*-i[^|;&]*" + PROT,        # sed -i ... file
    r"\brm\b[^|;&]*" + PROT,                  # rm ... file
    r"\bmv\b[^|;&]*" + PROT,                  # mv ... file (as src or dst)
    r"\bcp\b[^|;&]*\s\S*" + PROT + r"\s*($|[;&|])",  # cp src protected_dst
    r"\btruncate\b[^|;&]*" + PROT,
    r"\bchmod\b[^|;&]*" + PROT,
    r"\bln\b[^|;&]*" + PROT,
    # Destructive git, tolerant of flags/options anywhere (git -C . -q reset --hard …)
    r"\bgit\b[^|;&]*\bclean\b",
    r"\bgit\b[^|;&]*\breset\b[^|;&]*--hard",
    r"\bgit\b[^|;&]*\brestore\b",
    r"\bgit\b[^|;&]*\bcheckout\b[^|;&]*(\s--(\s|$)|\sHEAD\b|\s\.(\s|$))",
    r"\bgit\b[^|;&]*\b(checkout|reset)\b[^|;&]*" + PROT,
]

# Abbreviated GNU long options are valid (git reset --har, rm --recur) — match prefixes.
WRITE_PATTERNS = [pat.replace("--hard", "--ha\\S*").replace("--recursive", "--recu\\S*")
                  for pat in WRITE_PATTERNS]


def recursive_rm_outside_tmp(command: str) -> bool:
    """True when an rm with a recursive flag targets ANY operand outside /tmp
    (codex round 4: a /tmp operand must not exempt the rest of the command)."""
    for segment in re.split(r"[|;&]+", command):
        tokens = segment.strip().split()
        if not tokens or tokens[0] != "rm":
            continue
        flags = [t for t in tokens[1:] if t.startswith("-")]
        operands = [t for t in tokens[1:] if not t.startswith("-")]
        recursive = any(
            ("recursive".startswith(t[2:]) and len(t) > 2) if t.startswith("--")
            else any(c in "rR" for c in t[1:])
            for t in flags
        )
        if recursive and any(not op.startswith("/tmp/") for op in operands):
            return True
    return False


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return  # unparseable input: do not block
    command = payload.get("tool_input", {}).get("command", "") or ""
    if recursive_rm_outside_tmp(command):
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
              "permissionDecision": "deny",
              "permissionDecisionReason": "Blocked: recursive delete with a target outside /tmp."}}))
        return
    for pattern in WRITE_PATTERNS:
        if re.search(pattern, command):
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                "Blocked: this command writes to a protected file "
                                "(official benchmark / README / manifests / results "
                                "ledger). See Project/PLAN.md. Results files are "
                                "written only by the trusted runner."
                            ),
                        }
                    }
                )
            )
            return


if __name__ == "__main__":
    main()
