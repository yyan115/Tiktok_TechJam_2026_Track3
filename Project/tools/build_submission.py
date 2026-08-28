#!/usr/bin/env python3
"""Build the judge-facing submission script and PROVE its provenance.

The submission is the untouched official torch_transformer_benchmark.py with
exactly ONE region replaced: the UserOptimizedTransformer class (the region
the official template explicitly designates for the optimized
implementation). This tool:

  1. splices Project/submission/dispatcher_region.py into that region,
  2. re-verifies from the OUTPUT file that every byte outside the region is
     identical to the official script (the acceptance property),
  3. byte-compiles the result.

Output: Project/submission/torch_transformer_benchmark_submission.py
Run:    python3 Project/tools/build_submission.py
"""
import hashlib
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "torch_transformer_benchmark.py"
REGION = ROOT / "Project" / "submission" / "dispatcher_region.py"
OUT = ROOT / "Project" / "submission" / "torch_transformer_benchmark_submission.py"

START_MARK = "class UserOptimizedTransformer(BaselineTransformer):"
END_MARK = "def copy_model_weights("


def split_official(text: str):
    lines = text.splitlines(keepends=True)
    start = end = None
    for i, line in enumerate(lines):
        if start is None and line.startswith(START_MARK):
            start = i
        elif start is not None and line.startswith(END_MARK):
            end = i
            break
    if start is None or end is None:
        raise SystemExit("markers not found in official script")
    return "".join(lines[:start]), "".join(lines[start:end]), "".join(lines[end:])


def main() -> int:
    official_text = OFFICIAL.read_text()
    prefix, _original_region, suffix = split_official(official_text)
    region_text = REGION.read_text()
    if not region_text.endswith("\n"):
        region_text += "\n"

    OUT.write_text(prefix + region_text + "\n\n" + suffix)

    # Acceptance property, verified from the OUTPUT file's bytes: everything
    # before and after the (single, contiguous) replaced region is byte-
    # identical to the official script.
    out_bytes = OUT.read_bytes()
    pb, sb = prefix.encode(), suffix.encode()
    if not (out_bytes.startswith(pb) and out_bytes.endswith(sb)
            and len(out_bytes) == len(pb) + len(region_text.encode()) + 2 + len(sb)):
        raise SystemExit("PROVENANCE FAILURE: bytes outside the designated "
                         "region differ from the official script")

    py_compile.compile(str(OUT), doraise=True)

    print(f"submission: {OUT}")
    print(f"official sha256:   {hashlib.sha256(official_text.encode()).hexdigest()[:16]}…")
    print(f"submission sha256: {hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]}…")
    print("outside-region bytes identical to official: VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
