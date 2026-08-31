"""Compute the authorization subject hash for registering shape 12's family.

The gate binds a privileged transition to the SHA-256 of a canonical JSON
subject, so the owner signs the exact object rather than a description of it.
This script builds that object the same way `run_gate.cmd_family_register`
does and prints both the hash and the commands that consume it.

Run:  python3 Project/loop/shape12_subject.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "Project" / "loop" / "families" / "F-shape12-fusion.json"

# Must match, byte for byte, the --novelty-basis passed to family-register.
NOVELTY_BASIS = (
    "Not a new mechanism and not claimed as one. The parent family spent all "
    "twelve attempts characterising builds that predate three correctness "
    "defects fixed on 31 August: the mask predicate host synchronisation, the "
    "storage offset replay invariant, and the output buffer aliasing. This "
    "registers a fresh budget so shape 12 can be measured on the corrected "
    "artifact, which is required because every row of the reported board must "
    "carry one artifact hash. The mechanism under test is identical to the "
    "parent."
)


def canonical_json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def main():
    family = json.loads(SPEC.read_bytes())
    subject = {
        "campaign_id": "CAMP-FINAL",
        "family": family,
        "parent_family_id": family["parent_family_id"],
        "novelty_basis": NOVELTY_BASIS.strip(),
    }
    digest = hashlib.sha256(canonical_json(subject)).hexdigest()

    print("subject sha256:")
    print(f"  {digest}\n")
    print("Step A - mint the capability:\n")
    print("python3 Project/tools/owner_lock_ceremony.py mint-capability \\")
    print("    --action resolve_family_novelty \\")
    print("    --campaign CAMP-FINAL \\")
    print('    --reason "fresh budget for shape 12 on the corrected artifact" \\')
    print("    --private-key ~/techjam-keys/owner_private_key.pem \\")
    print("    --key-dir ~/techjam-keys\n")
    print("Step B - spend it (substitute the capability path Step A prints):\n")
    print("python3 Project/harness/trusted_controller.py authorize \\")
    print("    --capability <path from step A> \\")
    print("    --action resolve_family_novelty \\")
    print("    --target F-shape12-fusion \\")
    print(f"    --subject-sha256 {digest} \\")
    print("    --campaign CAMP-FINAL\n")
    print("Then paste me the receipt path it prints.")


if __name__ == "__main__":
    raise SystemExit(main())
