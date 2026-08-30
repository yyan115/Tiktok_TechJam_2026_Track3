#!/usr/bin/env python3
"""Adversarial mount/namespace tests for the candidate worker boundary."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "Project" / "harness"))

from sandbox import (  # noqa: E402
    IsolatedMount,
    SandboxFiles,
    run_isolated_command,
    run_sandbox,
)


WORKER = r'''
import argparse, json, os, socket
from pathlib import Path
p=argparse.ArgumentParser()
for n in ("request","candidate","official","shapes","output"):
    p.add_argument("--"+n, required=True)
a=p.parse_args()
checks={}
checks["home_absent"] = not Path("/home/admin").exists()
checks["authority_absent"] = not Path("/work/authority").exists()
checks["candidate_present"] = Path(a.candidate).is_file()
try:
    Path(a.candidate).write_text("forged")
    checks["candidate_read_only"] = False
except OSError:
    checks["candidate_read_only"] = True
try:
    Path("/etc/worker-forgery").write_text("x")
    checks["system_read_only"] = False
except OSError:
    checks["system_read_only"] = True
try:
    s=socket.socket()
    s.connect(("1.1.1.1", 53))
    checks["network_isolated"] = False
except OSError:
    checks["network_isolated"] = True
Path(a.output,"response.json").write_text(json.dumps(checks,sort_keys=True))
'''

SLEEP_WORKER = r'''
import argparse, time
p=argparse.ArgumentParser()
for n in ("request","candidate","official","shapes","output"):
    p.add_argument("--"+n, required=True)
p.parse_args()
time.sleep(60)
'''


def main() -> int:
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(("PASS " if condition else "FAIL ") + name + (f" [{detail}]" if detail and not condition else ""))
        if not condition:
            failures.append(name)

    with tempfile.TemporaryDirectory(prefix="sandbox-boundary-") as temp:
        base = Path(temp)
        worker = base / "worker.py"
        worker.write_text(WORKER)
        candidate = base / "candidate.py"
        candidate.write_text("VALUE = 1\n")
        official = base / "official.py"
        official.write_text("VALUE = 2\n")
        shapes = base / "shapes.json"
        shapes.write_text('{"shapes":[]}\n')
        request = base / "request.json"
        request.write_text('{"operation":"probe"}\n')
        output = base / "output"
        output.mkdir()
        files = SandboxFiles(worker, candidate, official, shapes, request, output)

        result = run_sandbox(files, timeout_seconds=10)
        check("sandbox worker exits cleanly", result.returncode == 0, result.stderr.decode(errors="replace"))
        response_path = output / "response.json"
        response = json.loads(response_path.read_text()) if response_path.exists() else {}
        for name in (
            "home_absent",
            "authority_absent",
            "candidate_present",
            "candidate_read_only",
            "system_read_only",
            "network_isolated",
        ):
            check(name.replace("_", " "), response.get(name) is True, str(response))
        check("host candidate bytes unchanged", candidate.read_text() == "VALUE = 1\n")

        sleeper = base / "sleeper.py"
        sleeper.write_text(SLEEP_WORKER)
        timeout_output = base / "timeout-output"
        timeout_output.mkdir()
        timeout_result = run_sandbox(
            SandboxFiles(sleeper, candidate, official, shapes, request, timeout_output),
            timeout_seconds=1,
        )
        check("timeout kills sandbox process group", timeout_result.timed_out and timeout_result.returncode != 0)

        layout_worker = base / "layout_worker.py"
        layout_worker.write_text(
            "from pathlib import Path\n"
            "source=Path('/sandbox/source.py')\n"
            "try:\n source.write_text('changed')\n readonly=False\n"
            "except OSError:\n readonly=True\n"
            "Path('/sandbox/output/result.txt').write_text(str(readonly))\n"
        )
        layout_output = base / "layout-output"
        layout_output.mkdir()
        layout_result = run_isolated_command(
            mounts=[
                IsolatedMount(layout_worker, "/sandbox/tool.py"),
                IsolatedMount(candidate, "/sandbox/source.py"),
                IsolatedMount(layout_output, "/sandbox/output", writable=True),
            ],
            argv=["/usr/bin/python3", "/sandbox/tool.py"],
            cwd="/sandbox",
            timeout_seconds=10,
        )
        check("generic isolated evaluator exits cleanly", layout_result.returncode == 0)
        check(
            "generic isolated evaluator has one writable output mount",
            (layout_output / "result.txt").read_text() == "True",
        )
        check("generic isolated source remains unchanged", candidate.read_text() == "VALUE = 1\n")

    print(f"\n{len(failures)} failure(s)" if failures else "\nALL GREEN")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
