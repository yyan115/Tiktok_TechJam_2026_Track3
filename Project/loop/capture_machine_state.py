#!/usr/bin/env python3
"""Capture a machine-state attestation for a calibration request.

The gate binds a machine_state_sha256 to every calibration so that a noise
floor can never be quoted without a record of the box it was measured on.
This reads the device directly; nothing here is hand-written.
"""
import json
import subprocess
import sys
import time

FIELDS = "utilization.gpu,memory.used,clocks.sm"


def main() -> int:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={FIELDS}",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip()
    util, mem, clock = [x.strip() for x in out.split(",")]

    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip()
    competing = [line.strip() for line in apps.splitlines() if line.strip()]

    state = {
        "captured_epoch": time.time(),
        "gpu_utilization": int(util),
        "gpu_memory_used_mib": int(mem),
        "sm_clock_mhz": int(clock),
        "competing_processes": competing,
    }
    path = sys.argv[1] if len(sys.argv) > 1 else "machine_state.json"
    with open(path, "w") as fh:
        json.dump(state, fh, indent=1)
    print(json.dumps(state, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
