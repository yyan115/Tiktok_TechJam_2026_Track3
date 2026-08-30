#!/usr/bin/env python3
"""Build and independently verify the judge-facing submission artifact.

Only the official ``UserOptimizedTransformer`` region may differ. Verification
re-parses the completed output and compares its raw prefix and suffix bytes to
the pinned official script; it does not trust the splice operation itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "torch_transformer_benchmark.py"
OFFICIAL_MANIFEST = ROOT / "Project" / "manifest.json"
REGION = ROOT / "Project" / "submission" / "dispatcher_region.py"
OUT = ROOT / "Project" / "submission" / "torch_transformer_benchmark_submission.py"

START_MARK = b"class UserOptimizedTransformer(BaselineTransformer):"
END_MARK = b"def copy_model_weights("


class ProvenanceError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pinned_official_proof() -> dict[str, str]:
    """Verify the official source against the separately pinned manifest."""
    if OFFICIAL.is_symlink() or OFFICIAL_MANIFEST.is_symlink():
        raise ProvenanceError("official source and manifest must be regular files")
    try:
        manifest = json.loads(OFFICIAL_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot read official integrity manifest: {exc}") from exc
    expected = manifest.get("files", {}).get(OFFICIAL.name)
    actual = sha256(OFFICIAL.read_bytes())
    if not isinstance(expected, str) or actual != expected:
        raise ProvenanceError(
            f"official source is not pinned: expected {expected!r}, got {actual}"
        )
    commit = manifest.get("official_commit")
    if not isinstance(commit, str) or not commit:
        raise ProvenanceError("official integrity manifest lacks official_commit")
    return {
        "official_commit": commit,
        "official_manifest_sha256": sha256(OFFICIAL_MANIFEST.read_bytes()),
        "official_sha256": actual,
    }


def line_marker(data: bytes, marker: bytes, *, start: int = 0) -> int:
    matches = []
    cursor = start
    while True:
        index = data.find(marker, cursor)
        if index < 0:
            break
        if index == 0 or data[index - 1:index] == b"\n":
            matches.append(index)
        cursor = index + 1
    if len(matches) != 1:
        raise ProvenanceError(
            f"expected one line-start marker {marker!r}, found {len(matches)}"
        )
    return matches[0]


def split_designated_region(data: bytes) -> tuple[bytes, bytes, bytes]:
    start = line_marker(data, START_MARK)
    end = line_marker(data, END_MARK, start=start + len(START_MARK))
    if end <= start:
        raise ProvenanceError("official replacement markers are out of order")
    return data[:start], data[start:end], data[end:]


def normalized_region() -> bytes:
    if REGION.is_symlink() or not REGION.is_file():
        raise ProvenanceError("dispatcher region must be a regular file")
    region = REGION.read_bytes()
    if not region.endswith(b"\n"):
        region += b"\n"
    line_marker(region, START_MARK)
    if END_MARK in region:
        raise ProvenanceError("dispatcher region contains the protected suffix marker")
    return region + b"\n"


def expected_submission() -> bytes:
    pinned_official_proof()
    official = OFFICIAL.read_bytes()
    prefix, _original_region, suffix = split_designated_region(official)
    return prefix + normalized_region() + suffix


def verify_output(output: bytes) -> dict[str, object]:
    """Independently prove both protected byte ranges from completed output."""
    pin = pinned_official_proof()
    official = OFFICIAL.read_bytes()
    official_prefix, _official_region, official_suffix = split_designated_region(
        official
    )
    if not output.startswith(official_prefix):
        raise ProvenanceError("output prefix differs from pinned official bytes")
    if not output.endswith(official_suffix):
        raise ProvenanceError("output suffix differs from pinned official bytes")
    region_end = len(output) - len(official_suffix)
    output_region = output[len(official_prefix):region_end]
    if output_region != normalized_region():
        raise ProvenanceError("output optimized region differs from dispatcher bytes")
    compile(output.decode("utf-8"), str(OUT), "exec")
    return {
        "schema_version": "submission-provenance-v2",
        "verified": True,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "official_path": str(OFFICIAL.relative_to(ROOT)),
        "official_sha256": pin["official_sha256"],
        "official_commit": pin["official_commit"],
        "official_manifest_path": str(OFFICIAL_MANIFEST.relative_to(ROOT)),
        "official_manifest_sha256": pin["official_manifest_sha256"],
        "dispatcher_path": str(REGION.relative_to(ROOT)),
        "dispatcher_sha256": sha256(REGION.read_bytes()),
        "submission_path": str(OUT.relative_to(ROOT)),
        "submission_sha256": sha256(output),
        "protected_prefix_bytes": len(official_prefix),
        "protected_prefix_sha256": sha256(official_prefix),
        "protected_suffix_bytes": len(official_suffix),
        "protected_suffix_sha256": sha256(official_suffix),
        "replacement_region_bytes": len(output_region),
        "replacement_region_sha256": sha256(output_region),
    }


def write_provenance(path: Path, proof: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(proof, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_submission(data: bytes) -> None:
    temporary = OUT.with_name(f".{OUT.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, OUT)
        directory_fd = os.open(str(OUT.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/verify generated submission")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify existing output without modifying it",
    )
    parser.add_argument(
        "--provenance-out",
        help="optionally create an immutable JSON proof artifact",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = expected_submission()
    proof = verify_output(expected)
    if args.check_only:
        if not OUT.is_file():
            raise SystemExit(f"submission does not exist: {OUT}")
        output = OUT.read_bytes()
        if output != expected:
            raise SystemExit("submission is stale relative to dispatcher/official bytes")
    else:
        atomic_write_submission(expected)
        output = OUT.read_bytes()
    if output != expected:
        raise SystemExit("submission bytes changed during atomic generation")
    proof = verify_output(output)
    if args.provenance_out:
        proof_path = Path(args.provenance_out).resolve()
        try:
            proof_path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise SystemExit("provenance output must remain inside repository") from exc
        proof_path.parent.mkdir(parents=True, exist_ok=True)
        write_provenance(proof_path, proof)
    print(json.dumps(proof, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
