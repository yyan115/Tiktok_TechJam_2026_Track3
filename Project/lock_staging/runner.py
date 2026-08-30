#!/usr/bin/env python3
"""LOCK-time replacement for the legacy in-process benchmark runner.

This compatibility entrypoint contains no referee or candidate logic.  Every
operation is delegated to the protected trusted controller, whose one-use
permit and sandbox boundary are therefore unavoidable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    controller = Path(__file__).resolve().with_name("trusted_controller.py")
    if controller.is_symlink() or not controller.is_file():
        raise SystemExit("REFUSED: trusted_controller.py is absent or linked")
    os.execv(
        sys.executable,
        [sys.executable, str(controller), *sys.argv[1:]],
    )


if __name__ == "__main__":
    main()
