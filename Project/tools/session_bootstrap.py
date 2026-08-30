#!/usr/bin/env python3
"""Cold-session, read-only GRIND briefing and fail-closed health check."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "Project" / "GRIND_ENTRYPOINT.md"
CONTROLLER = ROOT / "Project" / "harness" / "trusted_controller.py"
GATE = ROOT / "Project" / "tools" / "run_gate.py"


def checked(label: str, command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return f"{label}: CLOSED (exit {result.returncode})\n{output}"
    return f"{label}: OK\n{output}"


def main() -> int:
    if ENTRYPOINT.is_symlink() or not ENTRYPOINT.is_file():
        print("GRIND SESSION CLOSED: canonical entrypoint missing", file=sys.stderr)
        return 2
    print("=== CANONICAL GRIND ENTRYPOINT ===")
    print(ENTRYPOINT.read_text(encoding="utf-8"))
    print("=== LIVE LOCK/CONTROLLER STATUS ===")
    print(checked("controller", [sys.executable, str(CONTROLLER), "status"]))
    print("=== LIVE COMPETENCE-GATE STATUS ===")
    print(checked("gate", [sys.executable, str(GATE), "status"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
