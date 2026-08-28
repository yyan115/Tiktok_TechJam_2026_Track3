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
    # Recursive deletes (-r/-R/--recursive, any flag spelling) except under /tmp
    r"\brm\b(?![^|;&]*\s/tmp/)[^|;&]*(\s-[a-zA-Z]*[rR][a-zA-Z]*(\s|$)|--recursive)",
]


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return  # unparseable input: do not block
    command = payload.get("tool_input", {}).get("command", "") or ""
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
