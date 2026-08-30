#!/usr/bin/env python3
"""Owner-run LOCK ceremony for the AI-control harness.

This tool is the OWNER half of the control plane.  It is deliberately outside
the working agent's reach: it is the only place an Ed25519 private key is ever
touched, and it refuses to put private material anywhere the agent can read.

An agent that can sign its own authorization has no authority system at all.
Everything here exists to keep that from happening.

Subcommands
-----------
  keygen           make the owner and critic keypairs (private keys leave the repo)
  build-lock       assemble the LOCK document over the protected file set
  sign-lock        detached Ed25519 signature over canonical_json(LOCK document)
  mint-capability  issue one signed, scoped, expiring owner/critic capability
  verify           re-run the whole chain and print PASS/FAIL in plain language

Everything fails closed.  No subcommand ever prints private key material.

Exit codes: 0 = all good, 2 = hard failure, 3 = advisories only (see `verify`).
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import secrets
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent
REPO_DEFAULT = HERE.parents[1]

# The harness modules are the authority.  Import the real ones so that the
# owner tool and the controller can never drift apart in their serialization.
if str(REPO_DEFAULT / "Project" / "harness") not in sys.path:
    sys.path.insert(0, str(REPO_DEFAULT / "Project" / "harness"))

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from authority import (  # noqa: E402
    AUTHORITY_SCHEMA_VERSION,
    AuthorityError,
    AuthorityPaths,
    AuthorityStore,
    atomic_write,
    canonical_json,
    iso_utc,
    private_key_path_is_external,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from lock_manifest import (  # noqa: E402
    LOCK_KEYS,
    create_lock_document,
    safe_relative_path,
    verify_lock,
)


# --------------------------------------------------------------------------
# The protected file set, with the reason each entry is in it.
#
# Rule of thumb used here: a file belongs in the LOCK if changing it silently
# changes what the control plane ENFORCES or what the evidence MEANS.  A file
# stays out if it is working state that legitimately changes during GRIND —
# those are bound by other mechanisms (the hash-chained authority journal, and
# the per-run candidate_sha256 inside each one-use permit).
# --------------------------------------------------------------------------

GROUPS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "control-plane": (
        "the code that decides what is allowed to run at all",
        (
            ("Project/harness/trusted_controller.py",
             "sole post-LOCK entrypoint; refuses every command until the lock validates"),
            ("Project/harness/authority.py",
             "journal, capability verification, one-use permits, canonical_json"),
            ("Project/harness/lock_manifest.py",
             "the verifier itself — if this is editable the lock means nothing"),
            ("Project/harness/sandbox.py",
             "the Bubblewrap boundary that keeps candidate bytes off the repo"),
            ("Project/harness/candidate_worker.py",
             "the isolated worker that actually touches candidate code"),
            ("Project/harness/runner.py",
             "post-LOCK shim: every benchmark must go through the controller"),
        ),
    ),
    "gate-and-audit": (
        "the code and schemas that decide what counts as evidence",
        (
            ("Project/tools/run_gate.py",
             "competence gate: bottleneck + counter evidence + prediction bands"),
            ("Project/tools/audit_authority.py",
             "verdict ledger, resolution rules, owner receipt binding"),
            ("Project/tools/audit_champion.py",
             "blind audit packet construction"),
            ("Project/tools/champion_watch.py",
             "the hook that fires audits on every new champion"),
            ("Project/tools/session_bootstrap.py",
             "SessionStart health check — the first thing a cold agent sees"),
            ("Project/audits/verdict_schema.json",
             "the only accepted verdict shape; a loosened schema launders PASSes"),
            ("Project/loop/mechanism_catalog.json",
             "trusted family ids; free-text families are how three-strikes never closed"),
            ("Project/loop/mechanism_catalog.schema.json",
             "the catalog's own shape"),
            ("Project/tools/final_evidence_map.schema.json",
             "what may be selected into the shipped evidence map"),
        ),
    ),
    "benchmark-truth": (
        "the definition of the problem — the numbers are meaningless if these move",
        (
            ("torch_transformer_benchmark.py",
             "official scoring script; byte-identity claims are measured against it"),
            ("tensorflow_transformer_benchmark.py",
             "official second reference"),
            ("Project/shapes.json",
             "the 14 test shapes; every permit is bound to a shape id"),
            ("Project/manifest.json",
             "the pinned project manifest"),
        ),
    ),
    "evidence-tools": (
        "the tools that assemble what actually ships",
        (
            ("Project/tools/build_submission.py",
             "builds the submitted file; a change here changes the artifact"),
            ("Project/tools/ship_manifest.py",
             "writes the shipped claim text and the verdict filter"),
            ("Project/tools/shape6_local_eval.py",
             "side evidence for shape 6 — a shape that can score zero"),
            ("Project/tools/shape14_eval.py",
             "side evidence for shape 14 — decomposed streamed execution"),
        ),
    ),
    "staging": (
        "the reviewed bytes the owner installs during this ceremony",
        (
            ("Project/lock_staging/guard_bash.py",
             "the allowlist guard the owner pastes into .claude/hooks/"),
            ("Project/lock_staging/settings.json",
             "the matching deny rules and hooks"),
            ("Project/lock_staging/runner.py",
             "the controller shim that replaces the legacy in-process runner"),
        ),
    ),
    "enforcement": (
        "the live hook files — protected so a silent edit breaks the lock loudly",
        (
            (".claude/hooks/guard_bash.py",
             "the running guard; must equal the staged bytes"),
            (".claude/settings.json",
             "the running deny rules and hook wiring"),
        ),
    ),
}

DEFAULT_GROUPS: tuple[str, ...] = (
    "control-plane",
    "gate-and-audit",
    "benchmark-truth",
    "evidence-tools",
    "staging",
    "enforcement",
)

# Printed at build time so the owner sees what is deliberately NOT locked.
DELIBERATE_EXCLUSIONS: tuple[tuple[str, str], ...] = (
    ("Project/results/**, Project/results_side/**",
     "runner-written and changes with every measurement; integrity comes from "
     "the hash-chained authority journal, not from the lock"),
    ("Project/submission/**, Project/kernels/**",
     "this is the work product — GRIND changes it by design; each measurement "
     "is bound to exact bytes by candidate_sha256 inside its one-use permit"),
    ("Project/loop/cards.jsonl, gate_log.jsonl, gate_state.json, lineage.jsonl",
     "append-only working state; locking it would freeze the loop shut"),
    ("Project/authority/**",
     "the journal protects itself by hash chain, and LOCK.json cannot hash itself"),
    ("CLAUDE.md, README.md, Project/PLAN.md, Project/RUNBOOK.md, Project/HANDOVER.md",
     "living rule documents — recorded instead as rules_snapshot_sha256, so "
     "drift is REPORTED by `verify` without bricking the controller"),
    ("Project/memory/**",
     "STATE/DECISIONS/LESSONS are written every session by design"),
    ("Project/tools/dashboard.py, Project/tools/sensitivity_board.py",
     "read-only reporting; they cannot grant authority"),
)

# What the owner copies into place during the ceremony: staged -> live.
STAGING_INSTALL: tuple[tuple[str, str], ...] = (
    ("Project/lock_staging/guard_bash.py", ".claude/hooks/guard_bash.py"),
    ("Project/lock_staging/settings.json", ".claude/settings.json"),
    ("Project/lock_staging/runner.py", "Project/harness/runner.py"),
)

# Hashed into rules_snapshot_sha256.  Stable rule documents only.
RULES_DOCUMENTS: tuple[str, ...] = (
    "CLAUDE.md",
    "README.md",
    "Project/PLAN.md",
    "Project/RUNBOOK.md",
    "Project/HANDOVER.md",
)

# action -> required target prefix (None = free-form target)
KNOWN_ACTIONS: dict[str, str | None] = {
    "lock.activate": "lock:",
    "lock.rotate": "lock:",
    "permit.issue": "shape:",
    "audit.resolve": "audit:",
    "verdict.resolve": None,
    "technical.review": None,
}

CRITIC_ONLY_ACTION = "technical.review"


# --------------------------------------------------------------------------
# plain-language output helpers
# --------------------------------------------------------------------------

_ADVISORIES: list[str] = []


def say(text: str = "") -> None:
    print(text)


def rule(title: str) -> None:
    print()
    print(title)
    print("-" * max(8, len(title)))


def warn(text: str) -> None:
    print(f"  WARNING: {text}")


def advise(text: str) -> None:
    _ADVISORIES.append(text)
    print(f"  ADVISORY: {text}")


def fail(text: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"\nFAILED: {text}", file=sys.stderr)
    raise SystemExit(2)


def banner(lines: Sequence[str]) -> None:
    width = max(len(line) for line in lines) + 4
    print()
    print("!" * width)
    for line in lines:
        print("! " + line.ljust(width - 4) + " !")
    print("!" * width)
    print()


# --------------------------------------------------------------------------
# small primitives
# --------------------------------------------------------------------------


def repo_root(args: argparse.Namespace) -> Path:
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        fail(f"repository root is not a directory: {root}")
    return root


def public_fingerprint(key: Ed25519PublicKey) -> str:
    return sha256_bytes(
        key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )


def load_private_key(path: Path, *, label: str) -> Ed25519PrivateKey:
    """Read a private key.  Its bytes never leave this function."""
    if path.is_symlink():
        fail(f"{label} private key must not be a symlink: {path}")
    if not path.is_file():
        fail(
            f"{label} private key not found at {path}\n"
            "        Run `keygen` first, or pass --private-key with the right path."
        )
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        fail(
            f"{label} private key at {path} is readable by other accounts "
            f"(mode {mode:o}).\n"
            "        Restrict it to owner-only read/write before using it."
        )
    data = path.read_bytes()
    key: Any
    try:
        key = serialization.load_pem_private_key(data, password=None)
    except TypeError:
        secret = getpass.getpass(f"passphrase for the {label} private key: ")
        if not secret:
            fail(f"{label} private key is encrypted and no passphrase was given")
        try:
            key = serialization.load_pem_private_key(data, password=secret.encode())
        except Exception:
            fail(f"cannot decrypt the {label} private key — wrong passphrase?")
        finally:
            secret = ""
    except Exception as exc:
        fail(f"cannot load the {label} private key: {exc}")
    finally:
        data = b""
    if not isinstance(key, Ed25519PrivateKey):
        fail(f"{label} private key must be Ed25519")
    return key


def rules_snapshot(root: Path) -> tuple[str, dict[str, str]]:
    """Hash-of-hashes over the stable rule documents."""
    components: dict[str, str] = {}
    missing: list[str] = []
    for relative in RULES_DOCUMENTS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            missing.append(relative)
            continue
        components[relative] = sha256_file(path)
    if missing:
        fail("rule documents are missing or not regular files: " + ", ".join(missing))
    digest = sha256_bytes(canonical_json(components))
    return digest, components


def resolve_protected(
    root: Path,
    *,
    groups: Sequence[str],
    extra: Sequence[str],
    exclude: Sequence[str],
) -> list[tuple[str, str, str]]:
    """Return [(relative_path, group, reason)] sorted, fully validated."""
    unknown = [name for name in groups if name not in GROUPS]
    if unknown:
        fail("unknown --group value(s): " + ", ".join(unknown))
    dropped = {safe_relative_path(value) for value in exclude}
    chosen: dict[str, tuple[str, str]] = {}
    for name in groups:
        _, entries = GROUPS[name]
        for relative, reason in entries:
            chosen[safe_relative_path(relative)] = (name, reason)
    for value in extra:
        chosen[safe_relative_path(value)] = ("extra", "added by the owner on the command line")
    for value in dropped:
        chosen.pop(value, None)
    resolved: list[tuple[str, str, str]] = []
    missing: list[str] = []
    for relative in sorted(chosen):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            missing.append(relative)
            continue
        group, reason = chosen[relative]
        resolved.append((relative, group, reason))
    if missing:
        fail(
            "these protected files are missing or are not regular files:\n        "
            + "\n        ".join(missing)
            + "\n        Fix the tree, or drop them with --exclude, then run again."
        )
    if not resolved:
        fail("the protected file set is empty — refusing to build a meaningless lock")
    return resolved


def staging_report(root: Path) -> list[tuple[str, str, bool, bool]]:
    """[(staged, live, live_exists, matches)] for the copy-into-place step."""
    report: list[tuple[str, str, bool, bool]] = []
    for staged, live in STAGING_INSTALL:
        staged_path = root / staged
        live_path = root / live
        if staged_path.is_symlink() or not staged_path.is_file():
            report.append((staged, live, False, False))
            continue
        exists = live_path.is_file() and not live_path.is_symlink()
        matches = exists and sha256_file(staged_path) == sha256_file(live_path)
        report.append((staged, live, exists, matches))
    return report


def read_lock_document(root: Path) -> dict[str, Any]:
    paths = AuthorityPaths(root)
    path = paths.lock_manifest
    if path.is_symlink() or not path.is_file():
        fail(f"LOCK.json is absent or is not a regular file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"LOCK.json is not readable JSON: {exc}")
    if not isinstance(document, dict):
        fail("LOCK.json must be a JSON object")
    if set(document) != LOCK_KEYS:
        fail("LOCK.json has unexpected or missing fields — rebuild it with build-lock")
    return document


def activation_events(root: Path) -> list[dict[str, Any]]:
    store = AuthorityStore(root)
    try:
        events = store.read_events()
    except AuthorityError as exc:
        fail(f"the authority journal will not verify: {exc}")
    return [
        event for event in events
        if event.get("kind") in {"lock_activated", "lock_rotated"}
    ]


def journal_length(root: Path) -> int:
    store = AuthorityStore(root)
    try:
        return len(store.read_events())
    except AuthorityError as exc:
        fail(f"the authority journal will not verify: {exc}")


def scan_for_private_keys(root: Path) -> list[str]:
    """Any PEM private-key header anywhere in the repository is a hard failure."""
    hits: list[str] = []
    for base, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = [
            name for name in directories
            if name not in {".git", "__pycache__", "node_modules", ".venv"}
        ]
        for name in filenames:
            path = Path(base) / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                if path.stat().st_size > 10 * 1024 * 1024:
                    continue
                with path.open("rb") as stream:
                    head = stream.read(8192)
            except OSError:
                continue
            if b"PRIVATE KEY-----" in head:
                hits.append(str(path.relative_to(root)))
    return sorted(hits)


# --------------------------------------------------------------------------
# keygen
# --------------------------------------------------------------------------


def resolve_key_dir(args: argparse.Namespace, root: Path) -> Path:
    key_dir = Path(args.key_dir).expanduser().resolve()
    if key_dir.is_relative_to(root):
        fail(
            f"refusing to keep private keys inside the repository: {key_dir}\n"
            "        The working agent can read every file under the repository.\n"
            "        A key the agent can read is a key the agent can sign with,\n"
            "        which means there is no authority system at all.\n"
            "        Pick a directory outside the repo, e.g. a mounted USB stick."
        )
    if not private_key_path_is_external(key_dir, root):
        # authority.private_key_path_is_external also rejects $HOME, because
        # the agent runs as this same user and can read the whole home tree.
        if not args.allow_home_key:
            fail(
                f"{key_dir} is inside your home directory.\n"
                "        The agent runs as the SAME operating-system user, so it can\n"
                "        read anything under your home directory. Project/harness/\n"
                "        authority.py:private_key_path_is_external() rejects this on\n"
                "        purpose.\n"
                "        Best: put the keys on removable media or a separate volume,\n"
                "              e.g. --key-dir /run/media/<you>/<stick>/techjam-owner-keys\n"
                "        If you accept the weaker protection (guard rules only, not\n"
                "        the operating system), re-run with --allow-home-key."
            )
        banner([
            "PRIVATE KEYS ARE GOING INSIDE YOUR HOME DIRECTORY.",
            "The working agent runs as the same user and can read them.",
            "Only the guard hook stands between the agent and your key.",
            "Move them to removable media as soon as you can.",
        ])
    return key_dir


def cmd_keygen(args: argparse.Namespace) -> int:
    root = repo_root(args)
    key_dir = resolve_key_dir(args, root)
    paths = AuthorityPaths(root)

    if (paths.lock_manifest.exists() or paths.lock_signature.exists()) and not args.force:
        fail(
            "a LOCK already exists in this repository.\n"
            "        Generating new keys would silently invalidate it.\n"
            "        To replace the keys deliberately: keygen --force, then\n"
            "        build-lock, sign-lock, and activate with a lock.rotate capability."
        )

    rule("1. where the private keys will live")
    say(f"  directory : {key_dir}")
    say(f"  repository: {root}")
    say("  The repository never sees a private key. Only the two PUBLIC keys go in.")

    key_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(key_dir, 0o700)

    use_passphrase = False
    secret = ""
    if not args.no_passphrase:
        secret = getpass.getpass(
            "  passphrase for both private keys (press Enter for none): "
        )
        if secret:
            confirm = getpass.getpass("  repeat the passphrase: ")
            if confirm != secret:
                fail("the two passphrases do not match")
            use_passphrase = True

    encryption: Any
    if use_passphrase:
        encryption = serialization.BestAvailableEncryption(secret.encode())
    else:
        encryption = serialization.NoEncryption()

    store = AuthorityStore(root)
    store.ensure_layout()

    rule("2. generating keypairs")
    results: list[tuple[str, Path, str]] = []
    for role, public_target in (
        ("owner", paths.public_key),
        ("critic", paths.critic_public_key),
    ):
        private_target = key_dir / f"{role}_private_key.pem"
        if private_target.exists() or private_target.is_symlink():
            fail(
                f"{private_target} already exists.\n"
                "        This tool never overwrites private key material.\n"
                "        Move the old key aside yourself, or choose another --key-dir."
            )
        if public_target.exists() and not args.force:
            fail(
                f"{public_target} already exists — refusing to overwrite it.\n"
                "        Use --force only if you intend to replace this identity."
            )

        private_key = Ed25519PrivateKey.generate()
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption,
        )
        descriptor = os.open(
            private_target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        try:
            offset = 0
            while offset < len(private_pem):
                offset += os.write(descriptor, private_pem[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        private_pem = b""

        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        atomic_write(public_target, public_pem, mode=0o444)
        fingerprint = public_fingerprint(private_key.public_key())
        results.append((role, private_target, fingerprint))
        say(f"  {role:<6} private key -> {private_target}  (owner-only, never printed)")
        say(f"  {role:<6} public  key -> {public_target.relative_to(root)}")
        say(f"  {role:<6} fingerprint  = {fingerprint}")
    secret = ""

    # Prove the pair actually round-trips before the owner walks away.
    rule("3. self-check")
    for role, private_target, fingerprint in results:
        reloaded = load_private_key(private_target, label=role)
        if public_fingerprint(reloaded.public_key()) != fingerprint:
            fail(f"{role} key does not round-trip — do not proceed")
        installed = serialization.load_pem_public_key(
            (paths.public_key if role == "owner" else paths.critic_public_key).read_bytes()
        )
        if not isinstance(installed, Ed25519PublicKey) or \
                public_fingerprint(installed) != fingerprint:
            fail(f"{role} public key in the repository does not match the private key")
        say(f"  {role}: private key loads and matches the installed public key  OK")

    banner([
        "WRITE THESE TWO FINGERPRINTS DOWN SOMEWHERE OFF THIS MACHINE.",
        f"owner : {results[0][2]}",
        f"critic: {results[1][2]}",
        "",
        "Then back up the private keys to somewhere offline.",
        "If you lose them you cannot ever re-sign or rotate this lock.",
        "If someone else gets them they own the whole control plane.",
        "",
        "Later you can prove nobody swapped your identity with:",
        "  verify --expect-owner-key-sha256 <the owner fingerprint above>",
    ])
    return 0


# --------------------------------------------------------------------------
# build-lock
# --------------------------------------------------------------------------


def cmd_build_lock(args: argparse.Namespace) -> int:
    root = repo_root(args)
    paths = AuthorityPaths(root)
    groups = tuple(args.group) if args.group else DEFAULT_GROUPS
    protected = resolve_protected(
        root, groups=groups, extra=args.extra, exclude=args.exclude
    )

    rule("1. protected file set (review this before anything is written)")
    width = max(len(relative) for relative, _, _ in protected)
    current_group = ""
    for relative, group, reason in sorted(protected, key=lambda row: (row[1], row[0])):
        if group != current_group:
            current_group = group
            title, _ = GROUPS.get(group, ("owner-supplied additions", ()))
            say(f"\n  [{group}] {title}")
        digest = sha256_file(root / relative)
        size = (root / relative).stat().st_size
        say(f"    {relative.ljust(width)}  {digest[:16]}  {size:>8} bytes")
        say(f"    {' ' * width}  why: {reason}")
    say(f"\n  TOTAL: {len(protected)} files will be hash-pinned by this lock.")

    rule("2. deliberately NOT locked (and what protects them instead)")
    for pattern, reason in DELIBERATE_EXCLUSIONS:
        say(f"  - {pattern}")
        say(f"      {reason}")

    rule("3. staged bytes: are they installed yet?")
    installed_all = True
    for staged, live, exists, matches in staging_report(root):
        if matches:
            say(f"  installed  {staged}  ==  {live}")
            continue
        installed_all = False
        state = "differs from staged" if exists else "absent"
        say(f"  NOT YET    {staged}  ->  {live}   ({state})")
    if not installed_all:
        say()
        say("  The lock records the hashes of the files AS THEY ARE RIGHT NOW.")
        say("  If you install the staged bytes AFTER building the lock, the lock")
        say("  will immediately fail and the controller will refuse everything.")
        say("  Copy the staged files into place FIRST, then run build-lock again.")
        if not args.allow_uninstalled_staging:
            fail(
                "staged bytes are not installed yet — install them first, or pass "
                "--allow-uninstalled-staging if you truly mean to lock the current bytes"
            )
        warn("continuing anyway because --allow-uninstalled-staging was given")

    rule("4. rules snapshot (recorded in the lock, not enforced by it)")
    digest, components = rules_snapshot(root)
    for relative, component in components.items():
        say(f"  {component[:16]}  {relative}")
    say(f"  rules_snapshot_sha256 = {digest}")
    say("  If these documents change later, `verify` reports drift. The controller")
    say("  keeps working — that is on purpose, they are living documents.")

    rule("5. current authority state")
    activations = activation_events(root)
    events_total = journal_length(root)
    say(f"  authority journal events: {events_total}")
    if activations:
        latest = activations[-1]["payload"]
        say(f"  currently activated lock_id: {latest.get('lock_id')}")
        say("  A new LOCK document therefore needs a `lock.rotate` capability,")
        say("  not `lock.activate`.")
    else:
        say("  no lock has ever been activated in this journal")
        if events_total:
            warn(
                "the journal is not empty but has no activation — the controller "
                "requires an EMPTY journal for the very first activation"
            )

    if paths.lock_manifest.exists():
        warn(f"{paths.lock_manifest.relative_to(root)} already exists and will be replaced")

    if not args.yes:
        rule("DRY RUN — nothing was written")
        say("  Nothing on disk changed. If the list above is what you want, re-run")
        say("  the exact same command with --yes on the end.")
        return 0

    rule("6. writing the LOCK document")
    if not paths.public_key.is_file():
        fail(
            f"owner public key missing: {paths.public_key}\n"
            "        Run `keygen` first."
        )
    if not paths.critic_public_key.is_file():
        fail(
            f"critic public key missing: {paths.critic_public_key}\n"
            "        Run `keygen` first."
        )
    lock_id = args.lock_id or f"lock-{secrets.token_hex(10)}"
    epoch = args.epoch or f"post-fix-{utc_now().strftime('%Y%m%dT%H%M%SZ')}"
    try:
        document = create_lock_document(
            root=root,
            protected_files=[relative for relative, _, _ in protected],
            owner_public_key=paths.public_key,
            critic_public_key=paths.critic_public_key,
            rules_snapshot_sha256=digest,
            epoch=epoch,
            lock_id=lock_id,
        )
    except AuthorityError as exc:
        fail(f"the lock document would not build: {exc}")

    body = canonical_json(document) + b"\n"
    atomic_write(paths.lock_manifest, body, mode=0o444)
    # A document change invalidates any signature already on disk.  Removing it
    # is the fail-closed move: a stale signature would look like tampering.
    if paths.lock_signature.exists():
        paths.lock_signature.unlink()
        warn("removed the previous LOCK.sig — it no longer matches this document")

    say(f"  wrote {paths.lock_manifest.relative_to(root)}")
    say(f"  lock_id             = {document['lock_id']}")
    say(f"  epoch               = {document['epoch']}")
    say(f"  created_at          = {document['created_at']}")
    say(f"  owner_key_sha256    = {document['owner_key_sha256']}")
    say(f"  critic_key_sha256   = {document['critic_key_sha256']}")
    say(f"  protected files     = {len(document['protected_files'])}")
    say(f"  document sha256     = {sha256_bytes(canonical_json(document))}")
    say()
    say("  NEXT: sign-lock. Until then the lock is unsigned and worth nothing.")
    return 0


# --------------------------------------------------------------------------
# sign-lock
# --------------------------------------------------------------------------


def cmd_sign_lock(args: argparse.Namespace) -> int:
    root = repo_root(args)
    paths = AuthorityPaths(root)
    document = read_lock_document(root)

    if document["schema_version"] != AUTHORITY_SCHEMA_VERSION:
        fail("LOCK.json was built for a different authority schema")

    rule("1. loading the owner private key")
    private_path = Path(args.private_key).expanduser()
    if private_path.is_relative_to(root.resolve()):
        fail(f"refusing to read a private key from inside the repository: {private_path}")
    owner_key = load_private_key(private_path, label="owner")
    fingerprint = public_fingerprint(owner_key.public_key())
    say(f"  key fingerprint: {fingerprint}")

    rule("2. does this key actually own this lock?")
    if document["owner_key_sha256"] != fingerprint:
        fail(
            "this private key is NOT the owner key recorded in LOCK.json.\n"
            f"        LOCK.json expects: {document['owner_key_sha256']}\n"
            f"        this key is:       {fingerprint}\n"
            "        Signing anyway would produce a lock that fails verification."
        )
    installed = serialization.load_pem_public_key(paths.public_key.read_bytes())
    if not isinstance(installed, Ed25519PublicKey) or \
            public_fingerprint(installed) != fingerprint:
        fail("Project/authority/owner_public_key.pem does not match this private key")
    say("  yes — LOCK.json, the installed public key, and this private key all agree")

    rule("3. signing")
    payload = canonical_json(document)
    signature = owner_key.sign(payload)
    atomic_write(paths.lock_signature, base64.b64encode(signature) + b"\n", mode=0o444)
    say(f"  signed {len(payload)} bytes of canonical JSON")
    say(f"  wrote {paths.lock_signature.relative_to(root)}")

    rule("4. immediate re-verification")
    try:
        result = verify_lock(root)
    except AuthorityError as exc:
        fail(f"the freshly signed lock does not verify: {exc}")
    say(f"  verify_lock() says valid={result['valid']}")
    say(f"  lock_id            = {result['lock_id']}")
    say(f"  epoch              = {result['epoch']}")
    say(f"  manifest_sha256    = {result['manifest_sha256']}")
    say(f"  protected files    = {result['protected_file_count']} (all hashes match)")
    say()
    say("  The lock is now signed but NOT yet activated. The controller will still")
    say("  refuse: 'signed LOCK bytes are valid but the epoch is not owner-activated'.")
    say("  NEXT: mint-capability --action lock.activate, then activate-lock.")
    return 0


# --------------------------------------------------------------------------
# mint-capability
# --------------------------------------------------------------------------


def cmd_mint_capability(args: argparse.Namespace) -> int:
    root = repo_root(args)
    paths = AuthorityPaths(root)
    action = args.action
    role = args.role

    rule("1. checking the request makes sense")
    if action not in KNOWN_ACTIONS and not args.allow_unknown_action:
        fail(
            f"unknown action {action!r}.\n"
            "        Known actions: " + ", ".join(sorted(KNOWN_ACTIONS)) + "\n"
            "        Pass --allow-unknown-action only if you know the controller "
            "handler that consumes it."
        )
    if role == "critic" and action != CRITIC_ONLY_ACTION:
        fail(
            "a critic capability may only carry 'technical.review'.\n"
            "        authority.py refuses anything else from the critic key."
        )
    if role == "owner" and action == CRITIC_ONLY_ACTION:
        fail(
            "the owner key may NOT sign 'technical.review'.\n"
            "        authority.py keeps the owner and the independent critic apart\n"
            "        on purpose. Sign this with the critic key: --role critic."
        )

    target = args.target
    if action in {"lock.activate", "lock.rotate"} and not target:
        document = read_lock_document(root)
        target = f"lock:{document['lock_id']}"
        say(f"  derived target from LOCK.json: {target}")
    if not target:
        fail("--target is required for this action")
    required_prefix = KNOWN_ACTIONS.get(action)
    if required_prefix and not (
        target.startswith(required_prefix) or target == "*"
    ):
        fail(
            f"action {action} expects a target starting with {required_prefix!r}, "
            f"got {target!r}"
        )
    if target == "*" or target.endswith(":*"):
        warn(
            f"target {target!r} is a wildcard — this capability will authorize every "
            "matching subject, not one"
        )
    if not args.reason.strip():
        fail("--reason must not be empty; it is your own record of why this was issued")

    rule("2. checking the current authority state")
    activations = activation_events(root)
    events_total = journal_length(root)
    say(f"  authority journal events: {events_total}")
    if action == "lock.activate":
        if activations:
            fail(
                "this journal already contains an activation.\n"
                "        The controller will treat the next activation as a ROTATION.\n"
                "        Re-run with --action lock.rotate."
            )
        if events_total:
            fail(
                "the controller requires an EMPTY authority journal for the very "
                "first activation, and this journal has "
                f"{events_total} event(s).\n"
                "        Investigate before proceeding — something wrote authority "
                "state before the owner did."
            )
        say("  empty journal, no prior activation: lock.activate is correct")
    elif action == "lock.rotate":
        if not activations:
            fail(
                "nothing has ever been activated in this journal, so there is "
                "nothing to rotate.\n        Use --action lock.activate."
            )
        previous = activations[-1]["payload"].get("lock_id")
        document = read_lock_document(root)
        if previous == document["lock_id"]:
            fail(
                f"lock_id {previous} is already the activated epoch.\n"
                "        Build and sign a NEW lock document first, then rotate to it."
            )
        say(f"  rotating from {previous} to {document['lock_id']}")

    rule("3. minting")
    if args.expires_minutes < 1:
        fail("--expires-minutes must be at least 1")
    if args.expires_minutes > 24 * 60:
        fail("--expires-minutes above 24 hours is refused; mint a fresh one instead")
    if args.max_uses < 1:
        fail("--max-uses must be at least 1")
    if args.max_uses > 1:
        warn(
            f"max_uses={args.max_uses}: this capability can be spent "
            f"{args.max_uses} times before it expires"
        )

    now = utc_now()
    capability_id = f"capability-{secrets.token_hex(12)}"
    capability: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "capability_id": capability_id,
        "role": role,
        "campaign_id": args.campaign,
        "actions": [action],
        "targets": [target],
        "issued_at": iso_utc(now),
        "expires_at": iso_utc(now + timedelta(minutes=args.expires_minutes)),
        "max_uses": args.max_uses,
        "nonce": secrets.token_hex(16),
    }

    private_path = Path(args.private_key).expanduser()
    if private_path.is_relative_to(root.resolve()):
        fail(f"refusing to read a private key from inside the repository: {private_path}")
    signing_key = load_private_key(private_path, label=role)
    fingerprint = public_fingerprint(signing_key.public_key())
    expected_public = paths.public_key if role == "owner" else paths.critic_public_key
    if not expected_public.is_file():
        fail(f"{expected_public} is missing — run keygen first")
    installed = serialization.load_pem_public_key(expected_public.read_bytes())
    if not isinstance(installed, Ed25519PublicKey) or \
            public_fingerprint(installed) != fingerprint:
        fail(
            f"this private key is not the {role} key the repository trusts.\n"
            f"        installed {expected_public.name}: "
            f"{public_fingerprint(installed) if isinstance(installed, Ed25519PublicKey) else 'not Ed25519'}\n"
            f"        this key:  {fingerprint}"
        )
    capability["signature"] = base64.b64encode(
        signing_key.sign(canonical_json(capability))
    ).decode("ascii")

    rule("4. proving the controller will accept it")
    store = AuthorityStore(root)
    try:
        store.verify_capability(
            capability,
            action=action,
            target=target,
            campaign_id=args.campaign,
        )
    except AuthorityError as exc:
        fail(f"the capability we just minted does not verify: {exc}")
    say("  authority.verify_capability() accepts this capability  OK")

    rule("5. writing it out")
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_dir = Path(args.key_dir).expanduser().resolve() / "capabilities"
        out_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(out_dir, 0o700)
        out_path = out_dir / f"{capability_id}.json"
    if out_path.is_relative_to(root.resolve()):
        warn(
            "this capability file is inside the repository, where the agent can "
            "read it. Until it expires or is spent, anyone who can read it can "
            "use it. Prefer a path outside the repo."
        )
    body = canonical_json(capability) + b"\n"
    descriptor = os.open(
        out_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    # The capability schema is closed, so the human reason cannot live inside
    # the signed document.  Keep it beside the capability for the owner's log.
    provenance = {
        "capability_id": capability_id,
        "capability_sha256": sha256_bytes(body),
        "reason": args.reason,
        "role": role,
        "action": action,
        "target": target,
        "campaign_id": args.campaign,
        "issued_at": capability["issued_at"],
        "expires_at": capability["expires_at"],
        "max_uses": args.max_uses,
        "signed_by_key_sha256": fingerprint,
        "repository": str(root),
    }
    provenance_path = out_path.with_suffix(".provenance.json")
    atomic_write(provenance_path, canonical_json(provenance) + b"\n", mode=0o600)

    say(f"  capability : {out_path}")
    say(f"  reason log : {provenance_path}")
    say(f"  expires at : {capability['expires_at']}  "
        f"(in {args.expires_minutes} minute(s))")
    say()
    say("  Run this before it expires:")
    controller = "python3 Project/harness/trusted_controller.py"
    if action in {"lock.activate", "lock.rotate"}:
        say(f"    {controller} activate-lock --capability {out_path}")
    elif action == "permit.issue":
        say(f"    {controller} issue-permit --request <request.json> "
            f"--capability {out_path}")
    else:
        say(f"    {controller} authorize --capability {out_path} \\")
        say(f"        --action {action} --target {target} \\")
        say(f"        --subject-sha256 <64 hex> --campaign {args.campaign}")
    say()
    say("  Delete the capability file once it is spent. It is single-purpose.")
    return 0


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    root = repo_root(args)
    paths = AuthorityPaths(root)
    hard: list[str] = []
    _ADVISORIES.clear()

    say(f"OWNER LOCK VERIFICATION")
    say(f"repository: {root}")

    rule("1. key material")
    fingerprints: dict[str, str] = {}
    for label, path in (("owner", paths.public_key), ("critic", paths.critic_public_key)):
        if path.is_symlink() or not path.is_file():
            say(f"  {label:<6} public key : MISSING ({path})")
            hard.append(f"{label} public key missing")
            continue
        try:
            key = serialization.load_pem_public_key(path.read_bytes())
        except Exception as exc:
            say(f"  {label:<6} public key : UNREADABLE ({exc})")
            hard.append(f"{label} public key unreadable")
            continue
        if not isinstance(key, Ed25519PublicKey):
            say(f"  {label:<6} public key : WRONG TYPE (not Ed25519)")
            hard.append(f"{label} public key is not Ed25519")
            continue
        fingerprints[label] = public_fingerprint(key)
        say(f"  {label:<6} public key : present  sha256 {fingerprints[label]}")

    leaks = scan_for_private_keys(root)
    if leaks:
        say("  private key material inside the repository: FOUND")
        for leak in leaks:
            say(f"      {leak}")
        hard.append("private key material is inside the repository")
    else:
        say("  private key material inside the repository: none  OK")

    if args.expect_owner_key_sha256:
        expected = args.expect_owner_key_sha256.strip().lower()
        actual = fingerprints.get("owner")
        if actual == expected:
            say("  owner fingerprint matches the value you recorded  OK")
        else:
            say(f"  owner fingerprint MISMATCH")
            say(f"      you expected: {expected}")
            say(f"      on disk:      {actual}")
            hard.append("owner public key was substituted")

    rule("2. LOCK document")
    if paths.lock_manifest.is_symlink() or not paths.lock_manifest.is_file():
        say(f"  LOCK.json : MISSING ({paths.lock_manifest})")
        hard.append("LOCK.json missing")
        document = None
    else:
        try:
            document = json.loads(paths.lock_manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            say(f"  LOCK.json : UNREADABLE ({exc})")
            hard.append("LOCK.json unreadable")
            document = None
        if isinstance(document, dict) and set(document) == LOCK_KEYS:
            say(f"  lock_id   : {document['lock_id']}")
            say(f"  epoch     : {document['epoch']}")
            say(f"  created   : {document['created_at']}")
            say(f"  protected : {len(document['protected_files'])} files")
            say(f"  doc sha256: {sha256_bytes(canonical_json(document))}")
        elif document is not None:
            say("  LOCK.json : WRONG SHAPE (unexpected or missing fields)")
            hard.append("LOCK.json schema mismatch")
            document = None

    rule("3. protected files, one by one")
    if isinstance(document, dict) and isinstance(document.get("protected_files"), dict):
        protected = document["protected_files"]
        width = max(len(name) for name in protected)
        matched = 0
        for relative in sorted(protected):
            expected_hash = protected[relative]
            path = root / relative
            if path.is_symlink() or not path.is_file():
                say(f"  MISSING  {relative.ljust(width)}")
                hard.append(f"protected file missing: {relative}")
                continue
            actual = sha256_file(path)
            if actual == expected_hash:
                matched += 1
                say(f"  match    {relative.ljust(width)}  {actual[:16]}")
            else:
                say(f"  CHANGED  {relative.ljust(width)}  {actual[:16]} "
                    f"(lock says {str(expected_hash)[:16]})")
                hard.append(f"protected file changed: {relative}")
        say(f"\n  {matched} of {len(protected)} protected files hash-match the lock.")
    else:
        say("  cannot check: no usable LOCK document")

    rule("4. detached signature (the authoritative check)")
    try:
        result = verify_lock(root)
        say(f"  verify_lock(): PASS")
        say(f"      manifest_sha256  = {result['manifest_sha256']}")
        say(f"      owner_key_sha256 = {result['owner_key_sha256']}")
        say(f"      critic_key_sha256= {result['critic_key_sha256']}")
    except AuthorityError as exc:
        say(f"  verify_lock(): FAIL — {exc}")
        hard.append(f"verify_lock failed: {exc}")
        result = None

    rule("5. what the controller itself says")
    try:
        import trusted_controller  # noqa: PLC0415

        controller = trusted_controller.TrustedController(root)
        try:
            view = controller.verify_lock_only()
            say(f"  signed lock valid : {view['valid']}")
            say(f"  epoch activated   : {view['active']}")
            if not view["active"]:
                say("  The controller will refuse every command until an owner")
                say("  capability activates this epoch. That is correct behaviour.")
        except (AuthorityError, trusted_controller.ControllerRefusal) as exc:
            say(f"  controller refuses: {exc}")
            if result is not None:
                hard.append(f"controller refuses a valid lock: {exc}")
    except Exception as exc:  # import problems must not hide the lock result
        advise(f"could not ask the controller directly: {exc}")

    rule("6. advisories (reported, not enforced by the controller)")
    if isinstance(document, dict):
        digest, components = rules_snapshot(root)
        if digest == document.get("rules_snapshot_sha256"):
            say("  rules snapshot: unchanged since the lock was signed  OK")
        else:
            say("  rules snapshot: DRIFT since the lock was signed")
            say(f"      lock says: {document.get('rules_snapshot_sha256')}")
            say(f"      now:       {digest}")
            for relative, component in components.items():
                say(f"      {component[:16]}  {relative}")
            advise(
                "one or more rule documents changed; re-sign the lock to re-baseline"
            )
    for staged, live, exists, matches in staging_report(root):
        if matches:
            say(f"  staged bytes installed: {live}  OK")
        else:
            state = "differs" if exists else "absent"
            say(f"  staged bytes NOT installed: {live} ({state})")
            advise(f"{staged} has not been copied to {live}")

    rule("RESULT")
    if hard:
        say("  FAIL")
        for item in hard:
            say(f"    - {item}")
        say()
        say("  Nothing is authorized while this is failing. That is the design.")
        return 2
    if _ADVISORIES:
        say("  PASS with advisories")
        for item in _ADVISORIES:
            say(f"    - {item}")
        return 3
    say("  PASS — the lock is signed, intact, and every protected byte matches.")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    default_key_dir = "~/.techjam-owner-keys"
    parser = argparse.ArgumentParser(
        prog="owner_lock_ceremony.py",
        description="Owner-run LOCK ceremony. Run this yourself, not from an agent session.",
    )
    parser.add_argument(
        "--root",
        default=str(REPO_DEFAULT),
        help="repository root (default: this checkout)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    keygen = sub.add_parser(
        "keygen", help="generate the owner and critic Ed25519 keypairs"
    )
    keygen.add_argument("--key-dir", default=default_key_dir,
                        help=f"where the PRIVATE keys go (default {default_key_dir})")
    keygen.add_argument("--no-passphrase", action="store_true",
                        help="do not prompt for a passphrase")
    keygen.add_argument("--allow-home-key", action="store_true",
                        help="accept a key directory inside your home directory")
    keygen.add_argument("--force", action="store_true",
                        help="replace existing PUBLIC keys (never private ones)")
    keygen.set_defaults(handler=cmd_keygen)

    build = sub.add_parser("build-lock", help="assemble the LOCK document")
    build.add_argument("--group", action="append", choices=sorted(GROUPS),
                       help="protected group (repeatable; default: all of them)")
    build.add_argument("--extra", action="append", default=[],
                       help="extra repo-relative file to protect (repeatable)")
    build.add_argument("--exclude", action="append", default=[],
                       help="drop one repo-relative file (repeatable)")
    build.add_argument("--epoch", help="epoch label (default: post-fix-<utc timestamp>)")
    build.add_argument("--lock-id", help="lock id (default: random, 25 chars)")
    build.add_argument("--allow-uninstalled-staging", action="store_true",
                       help="build even though the staged bytes are not installed")
    build.add_argument("--yes", action="store_true",
                       help="actually write LOCK.json (without this it is a dry run)")
    build.set_defaults(handler=cmd_build_lock)

    sign = sub.add_parser("sign-lock", help="detached Ed25519 signature over the lock")
    sign.add_argument("--private-key", required=True,
                      help="path to the owner private key (outside the repository)")
    sign.set_defaults(handler=cmd_sign_lock)

    mint = sub.add_parser("mint-capability", help="issue one signed, scoped capability")
    mint.add_argument("--action", required=True,
                      help="e.g. " + ", ".join(sorted(KNOWN_ACTIONS)))
    mint.add_argument("--target",
                      help="e.g. lock:<lock_id>, shape:7, audit:<entry_id>; "
                           "derived from LOCK.json for lock.* actions")
    mint.add_argument("--campaign", required=True, help="campaign id this belongs to")
    mint.add_argument("--reason", required=True,
                      help="why you are issuing it (kept in a sidecar log)")
    mint.add_argument("--role", choices=("owner", "critic"), default="owner")
    mint.add_argument("--expires-minutes", type=int, default=20)
    mint.add_argument("--max-uses", type=int, default=1)
    mint.add_argument("--private-key", required=True,
                      help="path to the signing private key (outside the repository)")
    mint.add_argument("--key-dir", default=default_key_dir,
                      help="where capability files are written when --out is absent")
    mint.add_argument("--out", help="explicit output path for the capability JSON")
    mint.add_argument("--allow-unknown-action", action="store_true")
    mint.set_defaults(handler=cmd_mint_capability)

    verify = sub.add_parser("verify", help="re-run the whole chain, print PASS/FAIL")
    verify.add_argument("--expect-owner-key-sha256",
                        help="the owner fingerprint you wrote down at keygen time")
    verify.set_defaults(handler=cmd_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except AuthorityError as exc:
        fail(str(exc))
    except KeyboardInterrupt:
        print("\nFAILED: interrupted — nothing was completed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
