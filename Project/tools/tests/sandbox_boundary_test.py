#!/usr/bin/env python3
"""Adversarial mount/namespace tests for the candidate worker boundary.

Two classes of assertion live here and both must stay:

1. Isolation -- the jail hides the home directory, the authority state and the
   repository, the candidate bytes are read-only, the system tree is read-only,
   the network is dead, and a hung worker is killed with its process group.
2. Capability -- the jail can actually run the work it exists to run.  A
   sandbox that refuses to start, or that starts but cannot compile a GPU
   kernel, is not "safe"; it is broken, and every result taken through it is
   vacuous.  The Triton case below is the regression test for two real bugs:
   a machine-wide RLIMIT_NPROC cap that killed bwrap's clone(), and a missing
   /sbin (so Triton's driver bootstrap could not exec ldconfig).
"""

from __future__ import annotations

import glob
import importlib.util
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
    s.settimeout(5)
    s.connect(("1.1.1.1", 53))
    checks["network_isolated"] = False
except OSError:
    checks["network_isolated"] = True
try:
    u=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    u.settimeout(5)
    u.sendto(b"x", ("1.1.1.1", 53))
    checks["network_isolated_udp"] = False
except OSError:
    checks["network_isolated_udp"] = True
# Loader tooling: Triton's CUDA bootstrap execs ldconfig to find libcuda.
checks["ldconfig_present"] = os.path.isfile("/sbin/ldconfig") and os.access("/sbin/ldconfig", os.X_OK)
checks["bin_present"] = os.path.isdir("/bin")
checks["sbin_on_path"] = "/usr/sbin" in os.environ.get("PATH","").split(":")
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

# Real work, inside the jail: import torch, take the GPU, compile and launch an
# authored Triton kernel, verify the numbers.  Any regression in the mount set
# or the resource limits shows up here as a hard failure with a traceback.
GPU_TRITON_WORKER = r'''
import argparse, json, os, traceback
from pathlib import Path
p=argparse.ArgumentParser()
for n in ("request","candidate","official","shapes","output"):
    p.add_argument("--"+n, required=True)
a=p.parse_args()
checks={}
detail={}
for key in ("torch_imported","cuda_available","triton_compiled_and_ran","triton_numerics_exact"):
    checks[key]=False
checks["home_absent"] = not Path("/home/admin").exists()
checks["ldconfig_present"] = os.path.isfile("/sbin/ldconfig")
try:
    import torch
    checks["torch_imported"]=True
    detail["torch"]=torch.__version__
    checks["cuda_available"]=bool(torch.cuda.is_available())
    if checks["cuda_available"]:
        detail["device"]=torch.cuda.get_device_name(0)
        import triton
        import triton.language as tl
        detail["triton"]=triton.__version__

        @triton.jit
        def fused_scale_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
            tl.store(out_ptr + offsets, x * 2.0 + y, mask=mask)

        n = 4096
        x = torch.arange(n, device="cuda", dtype=torch.float32)
        y = torch.full((n,), 3.0, device="cuda", dtype=torch.float32)
        out = torch.zeros(n, device="cuda", dtype=torch.float32)
        fused_scale_add[(triton.cdiv(n, 128),)](x, y, out, n, BLOCK=128)
        torch.cuda.synchronize()
        checks["triton_compiled_and_ran"]=True
        expected = x * 2.0 + y
        max_err = (out - expected).abs().max().item()
        detail["max_abs_error"]=max_err
        detail["out_head"]=[float(v) for v in out[:4].tolist()]
        checks["triton_numerics_exact"]= max_err == 0.0
except Exception:
    detail["error"]=traceback.format_exc()[-4000:]
Path(a.output,"response.json").write_text(json.dumps({"checks":checks,"detail":detail},sort_keys=True))
'''

LAYOUT_WORKER = r'''
import json, os
from pathlib import Path
result={}
source=Path('/sandbox/source.py')
try:
    source.write_text('changed')
    result["source_read_only"]=False
except OSError:
    result["source_read_only"]=True
result["ldconfig_present"]=os.path.isfile('/sbin/ldconfig') and os.access('/sbin/ldconfig', os.X_OK)
result["bin_present"]=os.path.isdir('/bin')
result["sbin_on_path"]="/usr/sbin" in os.environ.get("PATH","").split(":")
result["home_absent"]= not Path('/home/admin').exists()
Path('/sandbox/output/result.json').write_text(json.dumps(result,sort_keys=True))
'''


def _host_gpu_stack_present() -> str:
    """Return "" when a real GPU run is possible here, else the reason it is not."""
    if not glob.glob("/dev/nvidia*"):
        return "host has no /dev/nvidia* device nodes"
    for module in ("torch", "triton"):
        try:
            if importlib.util.find_spec(module) is None:
                return f"host has no {module} installed"
        except (ImportError, ValueError):
            return f"host has no {module} installed"
    return ""


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


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

        result = run_sandbox(files, timeout_seconds=60)
        check("sandbox worker exits cleanly", result.returncode == 0, result.stderr.decode(errors="replace"))
        response = _read_json(output / "response.json")
        for name in (
            "home_absent",
            "authority_absent",
            "candidate_present",
            "candidate_read_only",
            "system_read_only",
            "network_isolated",
            "network_isolated_udp",
            "ldconfig_present",
            "bin_present",
            "sbin_on_path",
        ):
            check(name.replace("_", " "), response.get(name) is True, str(response))
        check("host candidate bytes unchanged", candidate.read_text() == "VALUE = 1\n")

        # Real GPU work through the same builder: torch + CUDA + a Triton kernel.
        gpu_reason = _host_gpu_stack_present()
        if gpu_reason:
            print(f"SKIP triton kernel runs on GPU inside jail [{gpu_reason}]")
        else:
            gpu_worker = base / "gpu_worker.py"
            gpu_worker.write_text(GPU_TRITON_WORKER)
            gpu_output = base / "gpu-output"
            gpu_output.mkdir()
            gpu_result = run_sandbox(
                SandboxFiles(gpu_worker, candidate, official, shapes, request, gpu_output),
                timeout_seconds=600,
            )
            gpu_response = _read_json(gpu_output / "response.json")
            gpu_checks = gpu_response.get("checks", {})
            gpu_detail = gpu_response.get("detail", {})
            gpu_stderr = gpu_result.stderr.decode(errors="replace")[-4000:]
            check("gpu worker exits cleanly", gpu_result.returncode == 0, gpu_stderr)
            check("torch imports inside jail", gpu_checks.get("torch_imported") is True, str(gpu_detail))
            check("cuda is available inside jail", gpu_checks.get("cuda_available") is True, str(gpu_detail))
            check(
                "triton kernel compiles and launches inside jail",
                gpu_checks.get("triton_compiled_and_ran") is True,
                str(gpu_detail),
            )
            check(
                "triton kernel result is numerically exact",
                gpu_checks.get("triton_numerics_exact") is True,
                str(gpu_detail),
            )
            check("gpu worker still sees no home directory", gpu_checks.get("home_absent") is True, str(gpu_checks))
            if gpu_detail.get("device"):
                print(
                    f"      device={gpu_detail.get('device')} torch={gpu_detail.get('torch')} "
                    f"triton={gpu_detail.get('triton')} max_abs_error={gpu_detail.get('max_abs_error')} "
                    f"out[:4]={gpu_detail.get('out_head')}"
                )

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
        layout_worker.write_text(LAYOUT_WORKER)
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
            timeout_seconds=60,
        )
        check(
            "generic isolated evaluator exits cleanly",
            layout_result.returncode == 0,
            layout_result.stderr.decode(errors="replace"),
        )
        layout = _read_json(layout_output / "result.json")
        check(
            "generic isolated evaluator has one writable output mount",
            layout.get("source_read_only") is True,
            str(layout),
        )
        check("generic isolated source remains unchanged", candidate.read_text() == "VALUE = 1\n")
        for name in ("ldconfig_present", "bin_present", "sbin_on_path", "home_absent"):
            check("generic isolated " + name.replace("_", " "), layout.get(name) is True, str(layout))

    print(f"\n{len(failures)} failure(s)" if failures else "\nALL GREEN")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
