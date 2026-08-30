#!/usr/bin/env python3
"""Run gate v5 — competence policy and permit-request architecture.

This module validates the scientific preconditions for an experiment and
emits an immutable, content-addressed request.  The trusted controller owns
the authoritative permit, its one-use consumption, the candidate worker,
and the authority event log.  ``permit.json`` is retained only as a legacy
transport projection for the owner hook; it is NEVER sufficient authority.

Thinking tiers:
  research + plan   full two steps — required to OPEN a new direction card
                    (research: current INDEX hash, >=2 existing notes,
                    summary; plan: hypothesis, numeric prediction, kill
                    criteria, file:line citations quoted into the log).
  delta             concise structured step for the NEXT attempt within an
                    open direction: what changed since the last attempt +
                    a numeric prediction. No filler packets for tuning
                    iterations; budget still enforced.
Both tiers emit exactly one request. A separate DIAGNOSTIC request can only
profile immutable current-champion bytes and can never authorize candidate
bytes, promotion, a primary-ledger write, or a scientific success/strike.

Families and strikes:
  family ids come only from the trusted catalog/migration registry or from
  a controller-verified admission event. Candidate prose cannot create or
  rename a family. Variants inherit the same family unless an externally
  authorized novelty resolution admits a child family.
  optimization mode: improvement = clean, correct, promoted-comparable row
    whose speedup exceeds the group's best by >=3% (the promotion margin
    floor — epsilon noise never resets strikes).
  screening mode: the attempt's declared prediction range decides
    hit/miss; a miss is a strike.
  failed scratch falsifiers and equivalent/renamed variants count against
    the family; diagnostic work and infrastructure failures do not.
  confirmation and calibration never strike or improve.
  3 strikes without improvement => direction CLOSED, postmortem debt
  registered. Closed directions reopen ONLY via a one-use critic receipt
  bound to the closure nonce. There is no unlock command.

State: Project/loop/gate_state.json · Requests: Project/loop/requests/
Transport projection: Project/loop/permit.json · Log: gate_log.jsonl

Owner/reviewer actions are accepted only as controller receipts bound to an
exact action and subject hash. Workspace quote strings and critic-authored
files have no authority.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as _datetime
import hashlib
import json
import math
import os
import re
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOOP = ROOT / "Project" / "loop"
STATE = LOOP / "gate_state.json"
PERMIT = LOOP / "permit.json"
USED = LOOP / "permits_used"
INFLIGHT = LOOP / "in_flight.json"
LOG = LOOP / "gate_log.jsonl"
VERDICTS = ROOT / "Project" / "audits" / "verdicts.jsonl"
INDEX = ROOT / "Project" / "research" / "INDEX.md"
CARDS = LOOP / "cards.jsonl"
DEFAULT_JOURNAL = ROOT / "Project" / "results" / "JOURNAL.jsonl"
CATALOG = LOOP / "mechanism_catalog.json"
REQUESTS = LOOP / "requests"
PROFILE_EVIDENCE = LOOP / "profile_evidence"
AUTHORITY_RECEIPTS = LOOP / "authority_receipts"
AUTHORITY_BLOBS = ROOT / "Project" / "authority" / "blobs"
CONTROLLER = ROOT / "Project" / "harness" / "trusted_controller.py"
SCRATCH_LEDGERS = LOOP / "scratch_ledgers"
MAX_STRIKES = 3
MAX_EXEC_FAILURES = 3
PERMIT_TTL_S = 45 * 60
IMPROVE_MARGIN = 1.03
SCIENTIFIC_MODES = ("optimization", "screening", "confirmation", "correctness")
SIDE_MODES = ("shape6", "shape14")
MODES = ("diagnostic", *SCIENTIFIC_MODES, "calibration", *SIDE_MODES)
PRIMARY_MODES = ("optimization", "confirmation")
SCRATCH_MODES = ("screening", "correctness")
HEX64 = re.compile(r"[0-9a-f]{64}")
AUTHORITY_ACTIONS = {
    "open_campaign", "register_family", "resolve_family_novelty",
    "resume_stalled_campaign", "reopen_family", "resolve_integrity_verdict",
    "quarantine_request",
}
_MISSING = object()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


class GateRefusal(RuntimeError):
    """A fail-closed validation failure suitable for a concise CLI error."""


def canonical_json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha_json(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _strict_json_file(path: Path, what: str):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw, parse_constant=reject_constant)
    except Exception as exc:
        raise GateRefusal(f"{what} missing, unreadable, or malformed: {exc}")
    if not isinstance(value, dict):
        raise GateRefusal(f"{what} must be one JSON object")
    return value


def _parse_epoch(value, what: str) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        raise GateRefusal(f"{what} is not a valid timestamp")
    try:
        return _datetime.datetime.fromisoformat(
            value.replace("Z", "+00:00")).timestamp()
    except Exception as exc:
        raise GateRefusal(f"{what} is not a valid timestamp: {exc}")


def load_catalog() -> dict:
    """Load and semantically validate the owner-frozen mechanism catalog.

    The JSON Schema is shipped for external validation. These checks are kept
    in-process so a missing optional ``jsonschema`` dependency cannot turn a
    malformed catalog into an allow.
    """
    c = _strict_json_file(CATALOG, "mechanism catalog")
    required = {"schema_version", "catalog_version", "prediction_policy",
                "campaign_policy", "bottlenecks", "mechanisms",
                "legacy_families"}
    if set(c) != required or c.get("schema_version") != 1:
        raise GateRefusal("mechanism catalog has an unknown/missing field or schema version")
    if not all(isinstance(c.get(k), dict) for k in
               ("prediction_policy", "campaign_policy", "bottlenecks",
                "mechanisms", "legacy_families")):
        raise GateRefusal("mechanism catalog sections must be JSON objects")
    if not c["bottlenecks"] or not c["mechanisms"]:
        raise GateRefusal("mechanism catalog cannot have empty taxonomies")
    for bid, b in c["bottlenecks"].items():
        if (not isinstance(bid, str) or not isinstance(b, dict)
                or set(b) != {"description", "evidence_tools", "required_metrics"}
                or not isinstance(b["evidence_tools"], list)
                or not b["evidence_tools"]
                or not isinstance(b["required_metrics"], list)
                or not b["required_metrics"]):
            raise GateRefusal(f"malformed bottleneck catalog entry: {bid!r}")
    for mid, m in c["mechanisms"].items():
        if (not isinstance(mid, str) or not isinstance(m, dict)
                or set(m) != {"description", "addresses", "changed_resource"}
                or not isinstance(m["addresses"], list) or not m["addresses"]
                or any(b not in c["bottlenecks"] for b in m["addresses"])):
            raise GateRefusal(f"malformed mechanism catalog entry: {mid!r}")
    for fid, family in c["legacy_families"].items():
        _validate_family(family, c, expected_id=fid, trusted_legacy=True)
    pp = c["prediction_policy"]
    cp = c["campaign_policy"]
    if set(pp) != {"minimum_effect_noise_multiples",
                   "maximum_win_band_noise_multiples",
                   "maximum_characterization_band_noise_multiples",
                   "minimum_relative_band", "maximum_relative_band"}:
        raise GateRefusal("prediction policy has unknown/missing fields")
    if set(cp) != {"maximum_total_attempts", "maximum_calibrations_per_shape",
                   "maximum_total_calibrations", "stall_window",
                   "maximum_family_attempts"}:
        raise GateRefusal("campaign policy has unknown/missing fields")
    return c


def _validate_family(family: dict, catalog: dict, expected_id=None,
                     trusted_legacy=False) -> dict:
    required = {"family_id", "shape", "mechanism", "bottleneck",
                "changed_resource", "expected_counter_change",
                "parent_family_id", "budget_attempts", "budget_minutes",
                "admission", "allow_new_attempts"}
    if not isinstance(family, dict) or set(family) != required:
        raise GateRefusal("family record has unknown/missing fields")
    fid = family.get("family_id")
    if (not isinstance(fid, str) or not re.fullmatch(r"[A-Za-z0-9._-]{3,96}", fid)
            or (expected_id is not None and fid != expected_id)):
        raise GateRefusal("family_id is invalid or does not match its registry key")
    if (type(family.get("shape")) is not int
            or not 1 <= family["shape"] <= 14):
        raise GateRefusal(f"family {fid} has invalid shape")
    mechanism = family.get("mechanism")
    bottleneck = family.get("bottleneck")
    if mechanism not in catalog["mechanisms"] or bottleneck not in catalog["bottlenecks"]:
        raise GateRefusal(f"family {fid} names an unknown mechanism/bottleneck")
    if bottleneck not in catalog["mechanisms"][mechanism]["addresses"]:
        raise GateRefusal(f"family {fid}'s mechanism does not address its bottleneck")
    if (family.get("changed_resource")
            != catalog["mechanisms"][mechanism]["changed_resource"]):
        raise GateRefusal(f"family {fid} changed_resource disagrees with the catalog")
    if (not isinstance(family.get("expected_counter_change"), dict)
            or not family["expected_counter_change"]):
        raise GateRefusal(f"family {fid} needs expected_counter_change")
    if (type(family.get("budget_attempts")) is not int
            or type(family.get("budget_minutes")) is not int
            or not (1 <= family["budget_attempts"]
                    <= catalog["campaign_policy"]["maximum_family_attempts"])
            or family["budget_minutes"] <= 0):
        raise GateRefusal(f"family {fid} has an invalid budget")
    if type(family.get("allow_new_attempts")) is not bool:
        raise GateRefusal(f"family {fid} allow_new_attempts must be boolean")
    if trusted_legacy and family.get("admission") != "legacy-history-only":
        raise GateRefusal(f"legacy family {fid} must be history-only")
    return family


_LOCK_REF = []


def load_state_strict():
    """Fail CLOSED: issuance and reconciliation refuse to run on missing or
    corrupt state — a wiped state file must not erase closures/debts. Also
    acquires the shared gate lock for the life of this process, serializing
    every state transition against the watcher; and checks the state's
    sequence number against the log so a stale git-restored file (missing
    later transitions) is refused."""
    if not _LOCK_REF:
        _LOCK_REF.append(gate_lock())
    if not STATE.exists():
        raise SystemExit("REFUSED: gate state missing. Run `run_gate.py init` "
                         "once (records the event) if this is genuinely new.")
    try:
        st = json.loads(STATE.read_text())
    except Exception:
        raise SystemExit("REFUSED: gate state unreadable/corrupt — fail "
                         "closed. Bring Project/loop/gate_state.json back "
                         "from version control before any further attempts.")
    logged_seq = 0
    if LOG.exists():
        for i, line in enumerate(LOG.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                s = json.loads(line).get("state_seq")
            except Exception:
                raise SystemExit(f"REFUSED: gate_log.jsonl line {i} is "
                                 "malformed — fail closed. Repair or remove "
                                 "that line (with a note) before proceeding.")
            if isinstance(s, int):
                logged_seq = max(logged_seq, s)
    if not isinstance(st, dict):
        raise SystemExit("REFUSED: gate state must be one JSON object")
    if type(st.get("seq", 0)) is not int or st.get("seq", 0) < 0:
        raise SystemExit("REFUSED: gate state seq must be a nonnegative integer")
    if st.get("seq", 0) < logged_seq:
        raise SystemExit(f"REFUSED: state seq {st.get('seq', 0)} is BEHIND the "
                         f"log's {logged_seq} — a stale restore is missing "
                         "transitions. Reconstruct state to match the log "
                         "before proceeding.")
    _ensure_state_schema(st)
    return st


def _ensure_state_schema(st: dict) -> None:
    """Prospective v5 fields. Old history remains readable but cannot grant
    a campaign, family, profile, capability, or attempted-byte credential."""
    defaults = {
        "family_registry": {}, "family_admissions": {},
        "campaigns": {}, "active_campaign": None,
        "profiles": {}, "consumed_capability_nonces": [],
        "request_shas": [], "pending_screen_judgment": None,
        "pending_audit_decisions": {},
        "reconciled_authority_event_shas": [],
        "settled_request_shas": [],
        "groups": {}, "pending_postmortem": [], "cleared_verdicts": [],
        "quarantined_requests": [],
    }
    for key, default in defaults.items():
        if key not in st:
            st[key] = default.copy() if isinstance(default, (dict, list)) else default
    typed = {
        "family_registry": dict, "family_admissions": dict,
        "campaigns": dict, "profiles": dict,
        "pending_audit_decisions": dict,
        "groups": dict, "consumed_capability_nonces": list,
        "request_shas": list, "pending_postmortem": list,
        "reconciled_authority_event_shas": list,
        "settled_request_shas": list,
        "cleared_verdicts": list,
        "quarantined_requests": list,
    }
    for key, typ in typed.items():
        if not isinstance(st.get(key), typ):
            raise SystemExit(f"REFUSED: gate state field {key} is malformed")
    if st.get("active_campaign") is not None and not isinstance(
            st.get("active_campaign"), str):
        raise SystemExit("REFUSED: active_campaign must be a string or null")


def save_state(st: dict) -> None:
    LOOP.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    tmp.replace(STATE)
    dir_fd = os.open(LOOP, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def commit(st: dict, entry: dict) -> None:
    """Crash-safe transition: the log row carrying the NEW seq is durably
    appended BEFORE state is saved. A crash in between leaves state BEHIND
    the log, which the strict loader refuses — fail closed, never laundered.

    ``_dry_run`` marks a throwaway copy of state used by
    ``_reconcile_probe_reason`` to reproduce a reconciler's refusal text. A
    probe must never write the log or the state file, so it stops here."""
    if st.get("_dry_run"):
        return
    seq = st.get("seq", 0) + 1
    st.pop("_verdict_lines_snapshot", None)  # transient, never persisted
    entry["state_seq"] = seq
    log(entry)
    st["seq"] = seq
    save_state(st)


def log(entry: dict) -> None:
    import os
    LOOP.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def index_hash() -> str:
    return hashlib.sha256(INDEX.read_bytes()).hexdigest()[:16]


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def controller_timing_protocol() -> tuple[dict | None, str]:
    """Read the trusted controller's timing protocol from the controller.

    ONE source of truth for the benchmark timing protocol.  The controller
    stamps its module-level ``TIMING`` constant into both the worker request
    and the ``measurement_recorded`` payload, and
    ``_reconcile_authority_calibration`` demands
    ``payload["timing_args"] == request["timing_config"]``.  A campaign whose
    ``timing_config`` differs from that constant therefore can NEVER
    reconcile its calibration; and since a campaign-bound calibration is a
    hard prerequisite for ``plan``, the entire gate wedges permanently with
    no recovery.  The cheap, recoverable place to catch that is campaign
    open, before any GPU time is spent.

    The constant is READ, never copied.  A second literal in this file would
    drift out of step with the controller and recreate exactly the class of
    bug it is meant to close.  The controller is parsed with ``ast`` instead
    of imported because importing it drags in the whole sandbox/worker stack
    (and its module-level side effects) for the sake of one dict.

    Returns ``(protocol, source)``.  ``protocol`` is ``None`` only in the one
    case where the controller source parses cleanly but publishes no
    module-level ``TIMING`` at all -- a controller that stamps no protocol
    cannot wedge a campaign on one, and the caller records that the binding
    was unverified.  Every other failure raises: those are cases where a
    protocol exists and the gate cannot read it, and guessing is what caused
    the defect.
    """
    keys = {"warmup", "repeats", "rounds"}
    try:
        source = CONTROLLER.read_text()
    except OSError as exc:
        raise GateRefusal(
            "trusted controller source is unreadable, so its timing protocol "
            f"cannot be bound; authority fails closed ({exc})")
    try:
        tree = ast.parse(source, filename=str(CONTROLLER))
    except SyntaxError as exc:
        raise GateRefusal(
            f"trusted controller source does not parse ({exc}); its timing "
            "protocol cannot be bound")
    found = _MISSING
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        if "TIMING" not in names or getattr(node, "value", None) is None:
            continue
        try:
            found = ast.literal_eval(node.value)
        except Exception as exc:
            raise GateRefusal(
                "trusted controller TIMING is not a literal constant, so the "
                f"gate cannot bind the timing protocol ({exc})")
    if found is _MISSING:
        return None, f"{CONTROLLER.name}: no module-level TIMING constant"
    if (not isinstance(found, dict) or set(found) != keys
            or any(type(found[k]) is not int or found[k] <= 0 for k in keys)):
        raise GateRefusal(
            "trusted controller TIMING is malformed (expected positive int "
            f"warmup/repeats/rounds, found {found!r})")
    return dict(found), f"{CONTROLLER.name}::TIMING"


def open_cards() -> dict:
    """Return latest non-closed card rows, rejecting malformed history.

    Cards describe a proposal; they do not create family identity. Identity
    is resolved separately through ``trusted_family``.
    """
    fams = {}
    try:
        lines = CARDS.read_text().splitlines()
    except Exception as exc:
        raise GateRefusal(f"cards ledger missing or unreadable: {exc}")
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            c = json.loads(line)
        except Exception as exc:
            raise GateRefusal(f"cards.jsonl line {lineno} malformed: {exc}")
        if (not isinstance(c, dict)
                or not isinstance(c.get("direction_family_id"), str)
                or not c["direction_family_id"]
                or not isinstance(c.get("status"), str)):
            raise GateRefusal(f"cards.jsonl line {lineno} has invalid schema")
        fams[c["direction_family_id"]] = c
    return {k: v for k, v in fams.items()
            if "killed" not in str(v.get("status", "")).lower()
            and "closed" not in str(v.get("status", "")).lower()}


def trusted_family(st: dict, family_id: str, *, require_active=True) -> dict:
    catalog = load_catalog()
    if family_id in st.get("family_registry", {}):
        family = _validate_family(st["family_registry"][family_id], catalog,
                                  expected_id=family_id)
    elif family_id in catalog["legacy_families"]:
        family = _validate_family(catalog["legacy_families"][family_id], catalog,
                                  expected_id=family_id, trusted_legacy=True)
    else:
        raise GateRefusal(
            f"unknown family {family_id!r}; candidate text cannot create identity. "
            "Obtain a controller-verified family admission first.")
    if require_active and not family.get("allow_new_attempts"):
        raise GateRefusal(f"family {family_id} is history-only/closed to new attempts")
    return family


def active_campaign(st: dict, campaign_id=None) -> dict:
    current = st.get("active_campaign")
    if not current or current not in st.get("campaigns", {}):
        raise GateRefusal("no controller-authorized active campaign")
    if campaign_id is not None and campaign_id != current:
        raise GateRefusal(f"campaign mismatch: active campaign is {current}")
    campaign = st["campaigns"][current]
    if not isinstance(campaign, dict) or campaign.get("status") != "active":
        raise GateRefusal("active campaign record is malformed or not active")
    campaign.setdefault("side_evaluation_requests", 0)
    campaign.setdefault("side_evaluations", [])
    if (type(campaign["side_evaluation_requests"]) is not int
            or campaign["side_evaluation_requests"] < 0
            or not isinstance(campaign["side_evaluations"], list)):
        raise GateRefusal("active campaign side-evaluation state is malformed")
    return campaign


def _request_artifact(payload: dict) -> tuple[str, str]:
    """Persist canonical request bytes under their SHA before transport.

    The trusted controller must load this exact object and bind its own
    authority event to ``request_sha256``. The path itself carries no power.
    """
    REQUESTS.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(payload)
    digest = hashlib.sha256(raw).hexdigest()
    path = REQUESTS / f"{digest}.json"
    if path.exists():
        if path.read_bytes() != raw:
            raise GateRefusal("content-addressed request path contains different bytes")
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o444)
        try:
            with os.fdopen(fd, "wb", closefd=False) as fh:
                fh.write(raw)
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            os.close(fd)
        dir_fd = os.open(REQUESTS, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    return str(path.relative_to(ROOT)), digest


def _authority_api():
    """Load the hash-chain reader without trusting an ad-hoc JSON tail."""
    harness = ROOT / "Project" / "harness"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    try:
        from authority import AuthorityError, AuthorityStore
    except Exception as exc:
        raise GateRefusal(f"controller authority API unavailable: {exc}")
    return AuthorityError, AuthorityStore


def _authority_events_strict() -> list[dict]:
    AuthorityError, AuthorityStore = _authority_api()
    try:
        return AuthorityStore(ROOT).read_events()
    except (AuthorityError, OSError, ValueError) as exc:
        raise GateRefusal(f"controller authority journal failed closed: {exc}")


def _gate_request_from_sha(request_sha256: str) -> dict:
    if not HEX64.fullmatch(str(request_sha256 or "")):
        raise GateRefusal("gate state contains a malformed request SHA-256")
    path = REQUESTS / f"{request_sha256}.json"
    if path.is_symlink():
        raise GateRefusal("gate request artifact cannot be a symlink")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GateRefusal(f"gate request artifact is missing: {request_sha256}: {exc}")
    if hashlib.sha256(raw).hexdigest() != request_sha256:
        raise GateRefusal("gate request filename/content hash binding is broken")
    try:
        request = json.loads(raw, parse_constant=lambda v: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {v}")))
    except Exception as exc:
        raise GateRefusal(f"gate request artifact is malformed: {exc}")
    if (not isinstance(request, dict) or request.get("schema_version") != 1
            or not isinstance(request.get("request_id"), str)
            or not request["request_id"] or request.get("mode") not in MODES
            or type(request.get("shape")) is not int
            or not 1 <= request["shape"] <= 14):
        raise GateRefusal("gate request artifact has an unsupported core schema")
    return request


def _request_candidate_and_family(request: dict) -> tuple[str | None, str | None]:
    kind = request.get("request_kind")
    if kind == "diagnostic":
        candidate = request.get("target_sha256")
        family_id = None
    elif kind == "calibration":
        candidate = None
        family_id = None
    elif kind == "scientific_attempt":
        candidate = request.get("impl_sha256")
        family = request.get("family")
        family_id = family.get("family_id") if isinstance(family, dict) else None
    elif kind == "side_evaluation":
        candidate = request.get("impl_sha256")
        family_id = None
    else:
        raise GateRefusal(f"unknown gate request kind {kind!r}")
    if candidate is not None and not HEX64.fullmatch(str(candidate)):
        raise GateRefusal("gate request candidate binding is malformed")
    if family_id is not None and (not isinstance(family_id, str) or not family_id):
        raise GateRefusal("gate request family binding is malformed")
    return candidate, family_id


def _expected_lane(mode: str) -> str:
    if mode in SCRATCH_MODES:
        return "scratch"
    if mode in ("shape6", "shape14"):
        return mode
    return "primary"


def _validate_issued_event(event: dict, request_sha: str,
                           request: dict) -> dict:
    if event.get("kind") != "permit_issued" or event.get("actor") != "trusted-controller":
        raise GateRefusal("gate request was not issued by the trusted controller")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise GateRefusal("permit_issued payload is malformed")
    candidate, family_id = _request_candidate_and_family(request)
    expected = {
        "request_sha256": request_sha,
        "request_blob_sha256": request_sha,
        "campaign_id": request.get("campaign_id"),
        "mode": request.get("mode"),
        "shape_id": request.get("shape"),
        "candidate_sha256": candidate,
        "family_id": family_id,
        "capability_consumed": True,
        "capability_action": "permit.issue",
        "capability_target": f"shape:{request.get('shape')}",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise GateRefusal("permit_issued payload disagrees with its immutable gate request")
    if (not isinstance(payload.get("permit_id"), str) or not payload["permit_id"]
            or not isinstance(payload.get("expires_at"), str)
            or not isinstance(payload.get("capability_id"), str)
            or not isinstance(payload.get("capability_nonce"), str)
            or payload.get("capability_role") != "owner"
            or not HEX64.fullmatch(str(payload.get("owner_key_sha256", "")))):
        raise GateRefusal("permit_issued authority/capability binding is malformed")
    expected_modify = request.get("candidate_authorized") is True
    if payload.get("may_modify_candidate") is not expected_modify:
        raise GateRefusal("permit mutation privilege disagrees with the gate request")
    expected_may_promote = request.get("mode") == "optimization"
    if payload.get("may_promote") is not expected_may_promote:
        raise GateRefusal("permit promotion privilege disagrees with controller policy")
    expires_epoch = _parse_epoch(payload["expires_at"], "permit expiry")
    if abs(expires_epoch - float(request.get("expires_epoch", -1))) > 1.0:
        raise GateRefusal("permit expiry disagrees with its immutable gate request")
    return payload


def _validate_measurement_packet(chain: dict) -> dict:
    measurement = chain["measurement"]
    binding = chain["binding"]
    payload = measurement["payload"]
    bound = binding.get("payload")
    if binding.get("actor") != "trusted-controller" or not isinstance(bound, dict):
        raise GateRefusal("measurement packet binding is malformed")
    side_lane = payload.get("mode") in SIDE_MODES
    expected_binding = {
        "entry_id": payload.get("entry_id") if side_lane else payload["run_id"],
        "measurement_event_id": measurement["event_id"],
        "measurement_event_sha256": measurement["event_sha256"],
        "candidate_sha256": payload["candidate_sha256"],
        "lane": payload["lane"],
    }
    if side_lane:
        expected_binding["side_evidence_sha256"] = payload.get("side_evidence_sha256")
    if any(bound.get(key) != value for key, value in expected_binding.items()):
        raise GateRefusal("measurement packet binding disagrees with the measurement")
    packet_sha = bound.get("packet_sha256")
    if not HEX64.fullmatch(str(packet_sha or "")):
        raise GateRefusal("measurement packet SHA-256 is malformed")
    packet_path = AUTHORITY_BLOBS / f"{packet_sha}.json"
    if packet_path.is_symlink():
        raise GateRefusal("measurement packet blob cannot be a symlink")
    try:
        raw = packet_path.read_bytes()
    except OSError as exc:
        raise GateRefusal(f"measurement packet blob is absent: {exc}")
    if hashlib.sha256(raw).hexdigest() != packet_sha:
        raise GateRefusal("measurement packet filename/content binding is broken")
    try:
        packet = json.loads(raw)
    except Exception as exc:
        raise GateRefusal(f"measurement packet JSON is malformed: {exc}")
    if side_lane:
        required = {
            "schema_version", "entry_id", "lane", "measurement_event_id",
            "measurement_event_sha256", "candidate_sha256", "permit_id",
            "gate_request_sha256", "campaign_id", "mode", "shape_id",
            "side_evidence_sha256", "side_stage_artifacts",
            "controller_validation", "side_evidence_packets",
            "lock_manifest_sha256",
        }
        if (not isinstance(packet, dict) or set(packet) != required
                or packet.get("schema_version") != 1):
            raise GateRefusal("side measurement packet has an unknown/missing field")
        expected = {
            "entry_id": payload.get("entry_id"), "lane": payload["lane"],
            "measurement_event_id": measurement["event_id"],
            "measurement_event_sha256": measurement["event_sha256"],
            "candidate_sha256": payload["candidate_sha256"],
            "permit_id": payload["permit_id"],
            "gate_request_sha256": chain["request_sha256"],
            "campaign_id": payload["campaign_id"], "mode": payload["mode"],
            "shape_id": payload["shape_id"],
            "side_evidence_sha256": payload["side_evidence_sha256"],
            "side_stage_artifacts": payload["side_stage_artifacts"],
            "controller_validation": payload["controller_validation"],
            "lock_manifest_sha256": payload["lock_manifest_sha256"],
        }
        if any(packet.get(key) != value for key, value in expected.items()):
            raise GateRefusal("side measurement packet disagrees with authority")
        stages = packet.get("side_stage_artifacts")
        packets = packet.get("side_evidence_packets")
        if (not isinstance(stages, list) or not stages
                or not isinstance(packets, list) or len(packets) != len(stages)):
            raise GateRefusal("side measurement stage evidence is malformed")
        stage_shas = set()
        for stage in stages:
            if (not isinstance(stage, dict) or set(stage) != {"stage", "sha256"}
                    or not isinstance(stage.get("stage"), str)
                    or not HEX64.fullmatch(str(stage.get("sha256", "")))):
                raise GateRefusal("side measurement stage reference is malformed")
            stage_shas.add(stage["sha256"])
            stage_path = AUTHORITY_BLOBS / f"{stage['sha256']}.json"
            if (stage_path.is_symlink() or not stage_path.is_file()
                    or sha_file(stage_path) != stage["sha256"]):
                raise GateRefusal("side stage artifact is missing or changed")
        if payload["side_evidence_sha256"] not in stage_shas:
            raise GateRefusal("side evidence SHA is not one of its bound stages")
        return {"sha256": packet_sha, "path": packet_path, "payload": packet}
    required = {
        "schema_version", "entry_id", "lane", "measurement_event_id",
        "measurement_event_sha256", "candidate_sha256", "permit_id",
        "gate_request_sha256", "campaign_id", "mode", "shape_id",
        "family_id", "worker_request_sha256", "worker_response_sha256",
        "controller_correctness", "supporting_timing", "lock_manifest_sha256",
    }
    optional = {"diagnostic_profile_sha256"}
    if (not isinstance(packet, dict) or not required.issubset(packet)
            or set(packet) - required - optional or packet.get("schema_version") != 1):
        raise GateRefusal("measurement packet has an unknown/missing field")
    packet_expected = {
        "entry_id": payload["run_id"],
        "lane": payload["lane"],
        "measurement_event_id": measurement["event_id"],
        "measurement_event_sha256": measurement["event_sha256"],
        "candidate_sha256": payload["candidate_sha256"],
        "permit_id": payload["permit_id"],
        "gate_request_sha256": chain["request_sha256"],
        "campaign_id": payload["campaign_id"],
        "mode": payload["mode"],
        "shape_id": payload["shape_id"],
        "family_id": payload["family_id"],
        "worker_request_sha256": payload["worker_request_sha256"],
        "worker_response_sha256": payload["worker_response_sha256"],
        "controller_correctness": payload["controller_correctness"],
        "supporting_timing": payload["supporting_timing"],
        "lock_manifest_sha256": payload["lock_manifest_sha256"],
    }
    if any(packet.get(key) != value for key, value in packet_expected.items()):
        raise GateRefusal("measurement packet disagrees with the authority event")
    if payload.get("mode") == "diagnostic":
        profile_sha = payload.get("diagnostic_profile_sha256")
        if (not HEX64.fullmatch(str(profile_sha or ""))
                or packet.get("diagnostic_profile_sha256") != profile_sha):
            raise GateRefusal("diagnostic measurement lacks an authority-bound profile")
    return {"sha256": packet_sha, "path": packet_path, "payload": packet}


def _authority_chains(st: dict, events=None) -> list[dict]:
    """Resolve gate requests solely through the validated authority chain.

    Workspace ledgers, process lists and mutable in-flight projections have no
    evidentiary role. An incomplete chain remains outstanding after a crash.
    """
    events = _authority_events_strict() if events is None else events
    request_shas = st.get("request_shas", [])
    if (not isinstance(request_shas, list) or len(set(request_shas)) != len(request_shas)
            or any(not HEX64.fullmatch(str(v or "")) for v in request_shas)):
        raise GateRefusal("gate request registry is malformed or duplicated")
    request_set = set(request_shas)
    campaign_id = st.get("active_campaign")
    issued_events = [event for event in events if event.get("kind") == "permit_issued"]
    for event in issued_events:
        payload = event.get("payload", {})
        if (payload.get("campaign_id") == campaign_id
                and payload.get("request_sha256") not in request_set):
            raise GateRefusal("active campaign contains an unrecognized permit request")
    by_request = {}
    permit_ids = set()
    for event in issued_events:
        request_sha = event.get("payload", {}).get("request_sha256")
        if request_sha not in request_set:
            continue
        if request_sha in by_request:
            raise GateRefusal("one gate request was issued more than once")
        request = _gate_request_from_sha(request_sha)
        permit = _validate_issued_event(event, request_sha, request)
        if permit["permit_id"] in permit_ids:
            raise GateRefusal("controller reused a permit_id")
        permit_ids.add(permit["permit_id"])
        by_request[request_sha] = {
            "request_sha256": request_sha, "request": request,
            "issued": event, "permit": permit,
        }
    for chain in by_request.values():
        permit_id = chain["permit"]["permit_id"]
        consumed = [event for event in events
                    if event.get("kind") == "permit_consumed"
                    and event.get("payload", {}).get("permit_id") == permit_id]
        if len(consumed) > 1:
            raise GateRefusal("permit was consumed more than once")
        chain["consumed"] = consumed[0] if consumed else None
        if consumed:
            cp = consumed[0].get("payload", {})
            expected = {
                "issued_event_id": chain["issued"]["event_id"],
                "mode": chain["permit"]["mode"],
                "shape_id": chain["permit"]["shape_id"],
                "candidate_sha256": chain["permit"]["candidate_sha256"],
            }
            if (consumed[0].get("actor") != "trusted-controller"
                    or any(cp.get(k) != v for k, v in expected.items())):
                raise GateRefusal("permit_consumed event has broken bindings")
        started = [event for event in events
                   if event.get("kind") == "run_started"
                   and event.get("payload", {}).get("permit_id") == permit_id]
        if len(started) > 1:
            raise GateRefusal("permit has multiple run_started events")
        chain["started"] = started[0] if started else None
        if started:
            if not consumed:
                raise GateRefusal("run_started exists without permit consumption")
            sp = started[0].get("payload", {})
            expected = {
                "consumed_event_id": consumed[0]["event_id"],
                "campaign_id": chain["permit"]["campaign_id"],
                "mode": chain["permit"]["mode"],
                "shape_id": chain["permit"]["shape_id"],
                "candidate_sha256": chain["permit"]["candidate_sha256"],
            }
            if (started[0].get("actor") != "trusted-controller"
                    or not isinstance(sp.get("run_id"), str)
                    or any(sp.get(k) != v for k, v in expected.items())):
                raise GateRefusal("run_started event has broken bindings")
        run_id = started[0]["payload"]["run_id"] if started else None
        measured = [event for event in events
                    if event.get("kind") == "measurement_recorded"
                    and event.get("payload", {}).get("permit_id") == permit_id]
        failed = [event for event in events
                  if event.get("kind") == "run_failed"
                  and run_id is not None
                  and event.get("payload", {}).get("run_id") == run_id]
        if len(measured) > 1 or len(failed) > 1 or (measured and failed):
            raise GateRefusal("controller run has duplicate/conflicting terminal events")
        chain["measurement"] = measured[0] if measured else None
        chain["failed"] = failed[0] if failed else None
        chain["binding"] = None
        chain["packet"] = None
        if measured:
            if not started:
                raise GateRefusal("measurement exists without a bound run_started event")
            mp = measured[0].get("payload", {})
            request = chain["request"]
            candidate, family_id = _request_candidate_and_family(request)
            expected = {
                "run_id": run_id,
                "started_event_id": started[0]["event_id"],
                "campaign_id": request.get("campaign_id"),
                "mode": request.get("mode"),
                "lane": _expected_lane(request.get("mode")),
                "shape_id": request.get("shape"),
                "candidate_sha256": candidate,
                "family_id": family_id,
            }
            if (measured[0].get("actor") != "trusted-controller"
                    or any(mp.get(k) != v for k, v in expected.items())):
                raise GateRefusal("measurement event disagrees with request/run bindings")
            if request.get("request_kind") == "side_evaluation":
                validation = mp.get("controller_validation")
                stages = mp.get("side_stage_artifacts")
                if (not isinstance(mp.get("entry_id"), str)
                        or not re.fullmatch(
                            r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", mp["entry_id"])
                        or not HEX64.fullmatch(str(mp.get("gate_request_sha256", "")))
                        or mp["gate_request_sha256"] != chain["request_sha256"]
                        or not HEX64.fullmatch(str(mp.get("side_evidence_sha256", "")))
                        or not isinstance(stages, list) or not stages
                        or not isinstance(validation, dict)
                        or type(validation.get("passed")) is not bool
                        or mp.get("evidence_eligible_pre_audit")
                            is not validation["passed"]
                        or mp.get("promotion_eligible") is not False
                        or not HEX64.fullmatch(str(mp.get("lock_manifest_sha256", "")))):
                    raise GateRefusal("side measurement event has malformed trusted fields")
            else:
                correctness = mp.get("controller_correctness")
                timing = mp.get("supporting_timing")
                speedup = timing.get("event_speedup") if isinstance(timing, dict) else None
                if (not isinstance(correctness, dict)
                        or type(correctness.get("passed")) is not bool
                        or not isinstance(timing, dict)
                        or type(timing.get("suspicious")) is not bool
                        or not isinstance(speedup, (int, float))
                        or isinstance(speedup, bool) or not math.isfinite(float(speedup))
                        or mp.get("promotion_eligible") is not False
                        or type(mp.get("performance_eligible")) is not bool
                        or not HEX64.fullmatch(str(mp.get("worker_request_sha256", "")))
                        or not HEX64.fullmatch(str(mp.get("worker_response_sha256", "")))
                        or not HEX64.fullmatch(str(mp.get("lock_manifest_sha256", "")))):
                    raise GateRefusal("measurement event has malformed trusted result fields")
            bindings = [event for event in events
                        if event.get("kind") == "measurement_packet_bound"
                        and event.get("payload", {}).get("measurement_event_id")
                            == measured[0]["event_id"]]
            if len(bindings) > 1:
                raise GateRefusal("measurement has multiple packet bindings")
            if bindings:
                chain["binding"] = bindings[0]
                chain["packet"] = _validate_measurement_packet(chain)
        if failed:
            if not started:
                raise GateRefusal("run_failed exists without run_started")
            fp = failed[0].get("payload", {})
            if (failed[0].get("actor") != "trusted-controller"
                    or fp.get("started_event_id") != started[0]["event_id"]
                    or not isinstance(fp.get("reason"), str) or not fp["reason"]):
                raise GateRefusal("run_failed event has broken bindings")
        chain["terminal"] = (measured[0] if measured and chain["binding"]
                             else failed[0] if failed else None)
    return [by_request[key] for key in request_shas if key in by_request]


def _pending_authority_reason(st: dict) -> str | None:
    chains = _authority_chains(st)
    settled = set(st.get("settled_request_shas", []))
    reconciled = set(st.get("reconciled_authority_event_shas", []))
    issued_by_sha = {chain["request_sha256"]: chain for chain in chains}
    for request_sha in st.get("request_shas", []):
        request = _gate_request_from_sha(request_sha)
        chain = issued_by_sha.get(request_sha)
        if chain is None:
            if request_sha not in settled and float(request.get("expires_epoch", 0)) > time.time():
                return f"gate request {request['request_id']} is emitted but not yet issued/expired"
            continue
        terminal = chain.get("terminal")
        terminal_sha = terminal.get("event_sha256") if terminal else None
        if request_sha in settled:
            if terminal_sha not in reconciled:
                raise GateRefusal("settled request lacks its reconciled authority event")
            continue
        return (f"controller request {request['request_id']} is unreconciled "
                f"({chain['permit']['permit_id']})")
    return None


def verify_controller_receipt(receipt_path: str, action: str,
                              subject: dict, st: dict) -> dict:
    """Re-query the trusted controller for every privileged transition.

    A receipt is a pointer into the controller's hash-chained authority log,
    not a bearer token whose workspace text is trusted. The private Ed25519
    key and verification authority stay outside this module.
    """
    import subprocess
    if action not in AUTHORITY_ACTIONS:
        raise GateRefusal(f"unknown privileged action: {action}")
    if not CONTROLLER.exists():
        raise GateRefusal("trusted controller is unavailable; authority fails closed")
    receipt = Path(receipt_path).resolve() if receipt_path else None
    if receipt is None or not receipt.exists():
        raise GateRefusal("controller receipt is required")
    subject_sha = sha_json(subject)
    proc = subprocess.run(
        [sys.executable, str(CONTROLLER), "verify-receipt",
         "--receipt", str(receipt), "--action", action,
         "--subject-sha256", subject_sha],
        cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stdout + proc.stderr).strip()[:500]
        raise GateRefusal(f"controller rejected authority receipt: {detail}")
    try:
        verified = json.loads(proc.stdout)
    except Exception as exc:
        raise GateRefusal(f"controller returned malformed verification JSON: {exc}")
    required = {"valid", "authority_event_id", "authority_event_sha256",
                "capability_nonce", "role", "action", "subject_sha256"}
    if (not isinstance(verified, dict) or set(verified) != required
            or verified.get("valid") is not True
            or verified.get("action") != action
            or verified.get("subject_sha256") != subject_sha
            or not isinstance(verified.get("capability_nonce"), str)
            or not verified["capability_nonce"]
            or not HEX64.fullmatch(str(verified.get("authority_event_sha256", ""))
                                   )):
        raise GateRefusal("controller verification response failed closed schema/binding checks")
    nonce = verified["capability_nonce"]
    if nonce in st.get("consumed_capability_nonces", []):
        raise GateRefusal("controller capability nonce has already been consumed")
    return verified


def consume_controller_receipt(st: dict, verified: dict) -> None:
    st.setdefault("consumed_capability_nonces", []).append(
        verified["capability_nonce"])


def _strict_log_rows() -> list[dict]:
    rows = []
    try:
        lines = LOG.read_text().splitlines()
    except Exception as exc:
        raise GateRefusal(f"gate log unreadable: {exc}")
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise GateRefusal(f"gate_log.jsonl line {lineno} malformed: {exc}")
        if not isinstance(row, dict):
            raise GateRefusal(f"gate_log.jsonl line {lineno} is not an object")
        rows.append(row)
    return rows


def prior_family_verdict(family_id: str, shape: int) -> str:
    """Stable caller-visible reference to the latest scientific outcome."""
    latest = None
    group = f"{family_id}|{int(shape)}"
    for row in _strict_log_rows():
        direction_match = (row.get("direction") == family_id
                           and int(row.get("shape", -1)) == int(shape))
        group_match = row.get("group") == group
        if ((direction_match or group_match)
                and row.get("step") in
                    ("reconcile", "screen_judge", "audit_finalize")):
            latest = row
    if latest is None:
        return "NONE"
    seq = latest.get("state_seq")
    if type(seq) is not int:
        raise GateRefusal("latest family verdict lacks a valid state_seq")
    result = latest.get("result")
    if not isinstance(result, str) or not result:
        if "improved" in latest:
            result = "improved" if latest.get("improved") else "miss"
        elif latest.get("correct") is not None:
            result = "pass" if latest.get("correct") else "failed-correctness"
        else:
            result = "recorded"
    return f"seq:{seq}:{result}"


def _profile_record(st: dict, ref: str, family: dict, target_sha: str) -> dict:
    record = st.get("profiles", {}).get(ref)
    if not isinstance(record, dict):
        raise GateRefusal(f"counter-evidence profile {ref!r} is unknown")
    required = {"profile_record_id", "artifact_path", "artifact_sha256",
                "shape", "target_sha256", "tool", "supported_bottlenecks",
                "metrics", "created_epoch", "reconciled_state_seq"}
    if set(record) != required:
        raise GateRefusal("counter-evidence profile state row is malformed")
    if (record["profile_record_id"] != ref
            or type(record["shape"]) is not int
            or record["shape"] != family["shape"]
            or record["target_sha256"] != target_sha
            or family["bottleneck"] not in record["supported_bottlenecks"]):
        raise GateRefusal("counter-evidence is not bound to this family/shape/target")
    path = (ROOT / record["artifact_path"]).resolve()
    try:
        path.relative_to(PROFILE_EVIDENCE.resolve())
    except ValueError:
        raise GateRefusal("counter-evidence path escaped the trusted profile namespace")
    if not path.exists() or sha_file(path) != record["artifact_sha256"]:
        raise GateRefusal("counter-evidence artifact is missing or changed")
    catalog = load_catalog()
    b = catalog["bottlenecks"][family["bottleneck"]]
    if record["tool"] not in b["evidence_tools"]:
        raise GateRefusal("counter-evidence tool is inappropriate for the bottleneck")
    missing = [m for m in b["required_metrics"] if m not in record["metrics"]]
    if missing:
        raise GateRefusal(f"counter-evidence lacks required metric(s): {missing}")
    campaign = active_campaign(st)
    created = float(record["created_epoch"])
    if created < float(campaign.get("opened_epoch", 0)):
        raise GateRefusal("counter-evidence predates the active campaign")
    if created > time.time() + 60 or time.time() - created > 24 * 60 * 60:
        raise GateRefusal("counter-evidence is not fresh (must be within 24 hours)")
    return record


def _validate_prediction(campaign: dict, group: dict, profile: dict,
                         kind: str, pmin, pmax) -> tuple[float, float]:
    if kind not in ("win", "characterization"):
        raise GateRefusal("prediction-kind must be win or characterization")
    try:
        pmin, pmax = float(pmin), float(pmax)
    except (TypeError, ValueError):
        raise GateRefusal("predict-min and predict-max must be numbers")
    if not (math.isfinite(pmin) and math.isfinite(pmax)
            and 0.0 < pmin < pmax):
        raise GateRefusal("prediction bounds must be finite and 0 < min < max")
    shape_key = str(profile["shape"])
    calibration = campaign.get("calibrations", {}).get(shape_key)
    if not isinstance(calibration, dict):
        raise GateRefusal("a campaign-bound calibration is required before predicting")
    noise = calibration.get("noise")
    if not isinstance(noise, (int, float)) or not math.isfinite(noise) or noise <= 0:
        raise GateRefusal("campaign calibration has invalid noise")
    policy = load_catalog()["prediction_policy"]
    center = (pmin + pmax) / 2.0
    relative_width = (pmax - pmin) / max(abs(center), 1.0)
    multiple = (policy["maximum_win_band_noise_multiples"] if kind == "win"
                else policy["maximum_characterization_band_noise_multiples"])
    max_width = min(policy["maximum_relative_band"],
                    max(policy["minimum_relative_band"], multiple * noise))
    if relative_width > max_width:
        raise GateRefusal(
            f"prediction band is uninformative: relative width {relative_width:.4f} "
            f"> noise-calibrated maximum {max_width:.4f}")
    incumbent = profile["metrics"].get("incumbent_speedup")
    if kind == "win":
        if not isinstance(incumbent, (int, float)) or incumbent <= 0:
            raise GateRefusal("win prediction evidence needs numeric incumbent_speedup")
        effect = max(IMPROVE_MARGIN - 1.0,
                     policy["minimum_effect_noise_multiples"] * noise)
        if pmin <= incumbent * (1.0 + effect):
            raise GateRefusal(
                "win prediction must exclude the incumbent by the calibrated "
                f"effect floor ({effect:.4f})")
    return pmin, pmax


def parse_citations(spec: str):
    out = []
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        m = re.fullmatch(r"(.+?):(\d+)(?:-(\d+))?", item)
        if not m:
            return None, f"bad citation format: '{item}'"
        rel, a, b = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
        path = ROOT / rel
        if not path.exists():
            path = ROOT / "Project" / rel
        if not path.exists():
            return None, f"cited file does not exist: '{rel}'"
        lines = path.read_text().splitlines()
        if not (1 <= a <= b <= len(lines)):
            return None, f"cited lines {a}-{b} out of range for {rel}"
        out.append({"file": rel, "lines": f"{a}-{b}",
                    "quoted": "\n".join(lines[a - 1:b])[:2000]})
    return out, None


def _audit_api():
    tools_dir = ROOT / "Project" / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    try:
        from audit_authority import (AuditAuthorityError, audit_decision,
                                     read_events)
    except Exception as exc:
        raise GateRefusal(f"audit authority unavailable: {exc}")
    return AuditAuthorityError, audit_decision, read_events


def audit_decision_strict(entry_id: str, candidate_sha256=None,
                          packet_sha256=None):
    AuditAuthorityError, audit_decision, _ = _audit_api()
    try:
        return audit_decision(entry_id, candidate_sha256, packet_sha256)
    except AuditAuthorityError as exc:
        raise GateRefusal(f"audit authority failed closed for {entry_id}: {exc}")


def unacked_hard_verdicts(st):
    """RULE_VIOLATION / RETEST audit verdicts recorded after the gate went
    live that have no journaled acknowledgment. Any such verdict brakes ALL
    new permits (Track-2 lesson: an audit that only displays is telemetry,
    not a governor). Malformed verdict lines fail closed as blockers.
    Timestamps compare lexically — every writer on this box uses +0800."""
    cutoff = st.get("created")
    if not cutoff:
        try:
            first = next(l for l in LOG.read_text().splitlines() if l.strip())
            cutoff = json.loads(first).get("ts") or "0000"
        except Exception:
            cutoff = "0000"
        st["created"] = cutoff  # persisted by the caller's next commit()
    # ``cleared_verdicts`` is legacy display state only. It cannot clear the
    # prospective audit authority's first-write-wins hard event.
    out = []
    try:
        # ONE read: the same snapshot feeds evaluation AND the line count
        # bound into permits (no evaluate-then-recount race window).
        raw = VERDICTS.read_text()
        lines = raw.splitlines()
        st["_verdict_lines_snapshot"] = len([l for l in lines if l.strip()])
    except Exception:
        # Missing OR unreadable verdict record = someone/something removed
        # the audit trail. That is never a green light. Fail closed.
        st["_verdict_lines_snapshot"] = -1
        return [{"_clear_key": "VERDICT-LEDGER-MISSING-OR-UNREADABLE|?",
                 "verdict": "RULE_VIOLATION"}]
    seen = {}
    prospective_entry_ids = set()
    for line in lines:
        if not line.strip():
            continue
        try:
            v = json.loads(line)
        except Exception:
            out.append({"_clear_key": "MALFORMED-VERDICT-LINE|?",
                        "verdict": "RULE_VIOLATION"})
            continue
        if not (v.get("entry_id") and v.get("recorded") and v.get("verdict")):
            # Structurally hollow rows (e.g. {}) are tampering, not noise.
            out.append({"_clear_key": "STRUCTURALLY-INVALID-VERDICT-ROW|?",
                        "verdict": "RULE_VIOLATION"})
            continue
        if v.get("verdict") not in (
                "PASS", "RULE_VIOLATION", "RETEST", "NEEDS_CONTEXT"):
            out.append({"_clear_key":
                        f"UNKNOWN-VERDICT|{v.get('entry_id')}|{v.get('recorded')}",
                        "verdict": "RULE_VIOLATION"})
            continue
        key = f"{v.get('entry_id')}|{v.get('recorded')}"
        if key in seen and seen[key] != v.get("verdict"):
            # Two rows claiming the same identity with different verdicts
            # means the record was tampered with or double-written: brake.
            out.append({"_clear_key": f"CONFLICTING-DUPLICATE|{key}",
                        "verdict": "RULE_VIOLATION"})
            continue
        seen[key] = v.get("verdict")
        if v.get("verdict") not in ("RULE_VIOLATION", "RETEST"):
            continue
        if str(v.get("recorded", "")) <= cutoff:
            continue
        prospective_entry_ids.add(v["entry_id"])
        try:
            decision = audit_decision_strict(v["entry_id"])
        except GateRefusal:
            out.append({"_clear_key": f"AUDIT-AUTHORITY-ERROR|{key}",
                        "verdict": "RULE_VIOLATION"})
            continue
        if decision.active_hard_event_sha256:
            v["_clear_key"] = key
            v["_audit_event_sha256"] = decision.active_hard_event_sha256
            out.append(v)
    # Prospective audit events may not have a legacy verdict row. They are
    # still global hard brakes until the audit authority resolves them.
    try:
        _err, _decision, read_events = _audit_api()
        for event in read_events():
            eid = event.get("entry_id")
            integrity = (event.get("result", {}).get("integrity", {}).get("verdict")
                         if event.get("event_type") == "audit_result" else None)
            if (isinstance(eid, str) and integrity in ("RULE_VIOLATION", "RETEST")
                    and str(event.get("recorded", "")) > cutoff):
                prospective_entry_ids.add(eid)
        already = {x.get("entry_id") for x in out}
        for eid in sorted(prospective_entry_ids - already):
            decision = audit_decision_strict(eid)
            if decision.active_hard_event_sha256:
                verdict = next((r.split(":", 1)[1]
                                for r in decision.blocking_reasons
                                if r.startswith("unresolved_first_hard_verdict:")),
                               "RULE_VIOLATION")
                out.append({"entry_id": eid, "recorded": "authority-event",
                            "verdict": verdict,
                            "_clear_key": decision.active_hard_event_sha256,
                            "_audit_event_sha256": decision.active_hard_event_sha256})
    except Exception:
        out.append({"_clear_key": "AUDIT-AUTHORITY-MISSING-OR-MALFORMED|?",
                    "verdict": "RULE_VIOLATION"})
    return out


def issue_permit(st, direction, mode, shape, impl, ledger, prediction, plan_ref,
                 predict_min=None, predict_max=None, *, campaign_id,
                 target_sha256, counter_evidence, prediction_kind,
                 family=None):
    """Validate and build a scientific permit *request*.

    The name is retained for CLI compatibility. The returned object is not an
    authoritative permit until the trusted controller binds and consumes it.
    """
    if PERMIT.exists():
        return None, "a permit is already ARMED — one attempt at a time"
    if INFLIGHT.exists():
        return None, ("legacy in_flight.json exists; it is not authority and must "
                      "be reconciled/migrated against the controller journal")
    if USED.exists() and any(f.name.startswith("claim.") for f in USED.iterdir()):
        return None, ("a stranded reconciliation claim exists in permits_used/ "
                      "— investigate and restore it before new permits")
    if st.get("pending_screen_judgment"):
        return None, ("the previous screening attempt has no recorded hit/miss "
                      "judgment — run `screen-judge` first")
    if mode not in SCIENTIFIC_MODES:
        return None, f"scientific plan mode must be one of {SCIENTIFIC_MODES}"
    try:
        campaign = active_campaign(st, campaign_id)
        authority_pending = _pending_authority_reason(st)
        if authority_pending:
            raise GateRefusal(authority_pending)
        family = family or trusted_family(st, direction)
        shape = int(str(shape), 10)
        if not 1 <= shape <= 14 or family["shape"] != shape:
            raise GateRefusal("shape must equal the trusted family's shape")
        if not HEX64.fullmatch(str(target_sha256 or "")):
            raise GateRefusal("--target-sha256 must be the immutable current champion SHA")
    except (GateRefusal, TypeError, ValueError) as exc:
        return None, str(exc)
    if campaign.get("stalled"):
        return None, ("campaign stall brake is engaged; only diagnostics/research "
                      "and a controller-authorized recovery using an untried family may proceed")
    if campaign.get("scientific_attempts", 0) >= campaign["max_total_attempts"]:
        return None, "campaign attempt budget exhausted"
    hard = unacked_hard_verdicts(st)
    violations = [h for h in hard if h.get("verdict") == "RULE_VIOLATION"]
    retests = [h for h in hard if h.get("verdict") == "RETEST"]
    if violations:
        keys = [h.get("_clear_key", "?") for h in violations[:3]]
        return None, (f"{len(violations)} uncleared RULE_VIOLATION verdict(s) "
                      f"freeze ALL new permits (e.g. {keys}). This unlock is "
                      "OWNER-ONLY: obtain a signed controller capability for "
                      "the exact resolution. Workspace quotes have no authority.")
    if retests and mode != "confirmation":
        keys = [h.get("_clear_key", "?") for h in retests[:3]]
        return None, (f"{len(retests)} unsatisfied RETEST verdict(s) "
                      f"(e.g. {keys}) — satisfy them FIRST: run a "
                      "confirmation-mode attempt on the SAME bytes+shape in "
                      "the PRIMARY journal, then verdict-clear --kind retest "
                      "--confirm-entry <new row id>. Until then ONLY "
                      "confirmations bound to the retested bytes issue.")
    retest_open = bool(retests)
    retest_targets = set()  # (impl_sha256, shape_id) pairs being retested
    if retest_open:
        for h in retests:
            row = _journal_row(str(h.get("entry_id")))
            sha = (row or {}).get("impl", {}).get("sha256")
            sid = (row or {}).get("shape_id")
            if sha and sid is not None:
                retest_targets.add((sha, int(sid)))
        if mode == "confirmation" and not retest_targets:
            return None, ("outstanding RETEST on entries absent from the "
                          "primary journal — no mechanical path exists; "
                          "surface it through an exact controller-authorized resolution.")
    if st.get("pending_postmortem"):
        return None, (f"postmortem debt outstanding for "
                      f"{st['pending_postmortem']} — a research step with "
                      "--postmortem must come first")
    gkey = f"{direction}|{shape}"
    grp = st.setdefault("groups", {}).setdefault(gkey, {})
    snapshot = grp.get("budget_snapshot")
    if snapshot is None:
        snapshot = {
            "family_id": family["family_id"],
            "budget_attempts": family["budget_attempts"],
            "budget_minutes": family["budget_minutes"],
            "catalog_sha256": sha_file(CATALOG),
        }
        grp["budget_snapshot"] = snapshot
    elif (not isinstance(snapshot, dict)
          or snapshot.get("family_id") != family["family_id"]
          or type(snapshot.get("budget_attempts")) is not int
          or type(snapshot.get("budget_minutes")) is not int):
        return None, "immutable family budget snapshot is malformed"
    if grp.get("scientific_attempts", 0) >= snapshot["budget_attempts"]:
        return None, (f"immutable family attempt budget exhausted for {gkey} "
                      f"({grp.get('scientific_attempts', 0)}/"
                      f"{snapshot['budget_attempts']})")
    if grp.get("closed"):
        return None, f"group {gkey} is CLOSED (reopen needs controller authority)"
    if mode == "confirmation":
        # A confirmation that exists to satisfy an outstanding RETEST must
        # never deadlock on the politeness budget — but ONLY for the
        # retested shape (bytes are verified at the impl check below).
        if grp.get("nonstrike_budget", 2) <= 0 and not (
                retest_open and mode == "confirmation"
                and any(sid == shape for _, sid in retest_targets)):
            return None, (f"non-strike budget exhausted for {gkey} — further "
                          "attempts must be optimization or screening (which "
                          "can strike)")
    impl_p = (ROOT / impl).resolve() if impl else None
    if impl_p is None or not impl_p.exists():
        return None, f"impl file not found: {impl}"
    try:
        impl_p.relative_to(ROOT)
    except ValueError:
        return None, "impl must live inside the repository (fail closed)"
    impl_sha = sha_file(impl_p)
    if mode == "optimization":
        if impl_sha in grp.get("primary_attempted_shas", []):
            return None, ("optimization refused: these exact bytes already had "
                          "a primary attempt; use confirmation honestly")
    if mode == "confirmation" and not retest_open:
        # Scratch membership is intentionally irrelevant. Only a prior clean
        # primary optimization authorizes primary confirmation.
        if impl_sha not in grp.get("primary_attempted_shas", []):
            return None, ("confirmation requires a prior PRIMARY optimization "
                          "attempt for these exact bytes; scratch evidence is "
                          "not a promotion credential")
    if retest_open and mode == "confirmation":
        if (impl_sha, shape) not in retest_targets:
            return None, ("outstanding RETEST: confirmation permits are "
                          "bound to the retested (candidate bytes, shape) "
                          "pairs only — no cross-shape or cross-candidate "
                          "confirmations")
        # Bounded sampling: re-rolling a flaky candidate until it passes is
        # not confirmation. Existing satisfying evidence must be cleared.
        att = st.setdefault("retest_confirm_attempts", {})
        for h in retests:
            row = _journal_row(str(h.get("entry_id")))
            if not row:
                continue
            if (row.get("impl", {}).get("sha256"),
                    int(row.get("shape_id", -1))) != (impl_sha, shape):
                continue
            sat = _retest_satisfying_row(row, h.get("recorded"))
            if sat:
                return None, ("satisfying confirmation evidence already "
                              f"exists (row {sat}) — run verdict-clear "
                              f"--kind retest --confirm-entry {sat} "
                              "instead of re-rolling the dice")
            k = h.get("_clear_key")
            if att.get(k, 0) >= 3:
                return None, (f"confirmation retry budget exhausted for "
                              f"retest {k} (3 attempts, none satisfied) "
                              "— this now needs an exact controller-authorized resolution.")
            att[k] = att.get(k, 0) + 1
    expected_ledger = DEFAULT_JOURNAL.resolve()
    if mode in SCRATCH_MODES:
        expected_ledger = (SCRATCH_LEDGERS / campaign["campaign_id"]
                           / family["family_id"] / f"shape-{shape}.jsonl").resolve()
    ledger_p = Path(ledger).resolve() if ledger else expected_ledger
    if ledger_p != expected_ledger:
        namespace = "PRIMARY" if mode in PRIMARY_MODES else "trusted SCRATCH"
        return None, f"{mode} must use its computed {namespace} ledger: {expected_ledger}"
    if (mode in PRIMARY_MODES and ledger_p != DEFAULT_JOURNAL.resolve()):
        return None, ("optimization/confirmation permits must use the PRIMARY "
                      "journal — champion-grade and retest evidence never "
                      "comes from scratch ledgers")
    try:
        profile = _profile_record(st, counter_evidence, family, target_sha256)
        if mode in ("optimization", "screening"):
            predict_min, predict_max = _validate_prediction(
                campaign, grp, profile, prediction_kind,
                predict_min, predict_max)
        elif prediction_kind == "characterization":
            predict_min, predict_max = _validate_prediction(
                campaign, grp, profile, prediction_kind,
                predict_min, predict_max)
    except GateRefusal as exc:
        return None, str(exc)
    pre_lines = (len([l for l in ledger_p.read_text().splitlines() if l.strip()])
                 if ledger_p.exists() else 0)
    request = {
        "schema_version": 1,
        "request_id": secrets.token_hex(16),
        "request_kind": "scientific_attempt",
        "campaign_id": campaign["campaign_id"],
        "direction_id": direction,
        "family": {
            "family_id": family["family_id"], "mechanism": family["mechanism"],
            "bottleneck": family["bottleneck"],
            "changed_resource": family["changed_resource"],
            "parent_family_id": family["parent_family_id"],
        },
        "mode": mode,
        "shape": shape,
        "impl_path": str(impl_p.relative_to(ROOT)) if impl_p else None,
        "impl_sha256": impl_sha if impl_p else None,
        "target_sha256": target_sha256,
        "ledger": str(ledger_p),
        "ledger_namespace": "primary" if mode in PRIMARY_MODES else "scratch",
        "ledger_pre_lines": pre_lines,
        "prediction": prediction,
        "prediction_kind": prediction_kind,
        "predict_min": predict_min,
        "predict_max": predict_max,
        "counter_evidence_id": counter_evidence,
        "counter_evidence_sha256": profile["artifact_sha256"],
        "budget_snapshot": snapshot,
        # No gate request is itself a promotion credential. Optimization may
        # become champion-eligible only after authority measurement + packet +
        # independent audit are reconciled.
        "promotion_allowed": False,
        "champion_consideration_after_audit": mode == "optimization",
        "candidate_authorized": True,
        "scientific_strike_eligible": mode in ("optimization", "screening", "correctness"),
        # The SAME snapshot the brake evaluated above (set by
        # unacked_hard_verdicts): the guard refuses to consume this permit
        # if the count changed (a verdict landed after issuance — re-plan).
        "verdict_lines": st.get("_verdict_lines_snapshot", -1),
        "plan_ref": plan_ref,
        "issued": now(),
        "expires_epoch": time.time() + PERMIT_TTL_S,
    }
    return request, None


def arm_permit(request, artifact=None) -> tuple[str, str]:
    """Persist the immutable request, then write a transport projection.

    The controller must issue/consume authority from its own hash-chained
    journal. Possession or mutation of ``permit.json`` grants nothing.
    """
    LOOP.mkdir(parents=True, exist_ok=True)
    rel, digest = artifact or _request_artifact(request)
    projection = dict(request)
    projection.update({
        "authority": "transport-only-not-a-permit",
        "request_path": rel,
        "request_sha256": digest,
    })
    PERMIT.write_text(json.dumps(projection, indent=1, sort_keys=True))
    return rel, digest


def cmd_research(args) -> int:
    st = load_state_strict()
    try:
        campaign = active_campaign(st, args.campaign)
    except GateRefusal as exc:
        print(f"REFUSED: {exc}")
        return 1
    if PERMIT.exists() or INFLIGHT.exists():
        print("REFUSED: an attempt is armed or in flight — finish and "
              "reconcile it before starting a new research cycle.")
        return 1
    pending = st.get("pending_postmortem", [])
    if pending and len((args.postmortem or "").strip()) < 200:
        print(f"REFUSED: direction(s) {pending} were CLOSED. A >=200-char "
              "--postmortem (predicted vs happened, why it failed, what it "
              "rules out) is mandatory before any new research cycle. It "
              "MUST also state a CASE FOR REVIVAL — either 'none: the idea "
              "itself was disproven' or the specific evidence that would "
              "justify a critic-appealed reopen (the critic weighs this on "
              "any future appeal).")
        return 1
    if pending and "revival" not in (args.postmortem or "").lower():
        print("REFUSED: the postmortem must contain a 'case for revival' "
              "statement (even if it is 'revival case: none — disproven').")
        return 1
    if args.index_hash != index_hash():
        print(f"REFUSED: --index-hash mismatch (current: {index_hash()}). "
              "Read Project/research/INDEX.md THIS cycle.")
        return 1
    notes = [n.strip() for n in args.notes.split(",") if n.strip()]
    missing = [n for n in notes if not (ROOT / "Project" / "research" / n).exists()]
    if len(notes) < 2 or missing:
        print(f"REFUSED: >=2 existing research-base files (missing: {missing}).")
        return 1
    if len(args.summary.strip()) < 200:
        print("REFUSED: summary under 200 chars.")
        return 1
    st["research_cycle"] = st.get("research_cycle", 0) + 1
    st["research_open"] = True
    if pending:
        st["pending_postmortem"] = []
    entry = {"ts": now(), "step": "research", "cycle": st["research_cycle"],
             "campaign_id": campaign["campaign_id"],
             "index_hash": args.index_hash, "notes": notes, "summary": args.summary}
    if pending:
        entry["postmortem_for"] = pending
        entry["postmortem"] = args.postmortem
    commit(st, entry)
    print(f"RESEARCH accepted (cycle {st['research_cycle']}). Next: plan (new "
          "direction card required).")
    return 0


def _recover_stall_if_authorized(st, args, family, profile) -> dict | None:
    campaign = active_campaign(st, args.campaign)
    if not campaign.get("stalled"):
        return None
    if family["family_id"] in campaign.get("families_tried", []):
        raise GateRefusal("stall recovery requires a mechanism family not yet tried in this campaign")
    if float(profile["created_epoch"]) <= float(campaign.get("stalled_epoch", 0)):
        raise GateRefusal("stall recovery requires a profile captured after the stall")
    if st.get("research_cycle", 0) <= campaign.get("stall_research_cycle", 0):
        raise GateRefusal("stall recovery requires a fresh research cycle")
    subject = {
        "campaign_id": campaign["campaign_id"],
        "stall_nonce": campaign.get("stall_nonce"),
        "new_family_id": family["family_id"],
        "profile_record_id": profile["profile_record_id"],
    }
    verified = verify_controller_receipt(
        args.stall_receipt, "resume_stalled_campaign", subject, st)
    consume_controller_receipt(st, verified)
    campaign["stalled"] = False
    campaign["stall_nonce"] = None
    campaign["stall_recovered_by"] = verified["authority_event_id"]
    return verified


def cmd_plan(args) -> int:
    st = load_state_strict()
    if not st.get("research_open"):
        print("REFUSED: research step required first (this cycle). Two steps, in order.")
        return 1
    try:
        campaign = active_campaign(st, args.campaign)
        cards = open_cards()
        family = trusted_family(st, args.direction)
    except GateRefusal as exc:
        print(f"REFUSED: {exc}")
        return 1
    card = cards.get(args.direction)
    if card is None:
        print(f"REFUSED: --direction must name an OPEN card family in "
              f"{CARDS} (found: {sorted(cards)}). Open the card first — the "
              "card describes the direction; trusted admission supplies identity.")
        return 1
    if args.mode not in SCIENTIFIC_MODES:
        print(f"REFUSED: --mode must be one of {SCIENTIFIC_MODES}; diagnostics "
              "and calibration have dedicated commands.")
        return 1
    if args.bottleneck != family["bottleneck"]:
        print("REFUSED: --bottleneck must equal the trusted family's catalog assignment.")
        return 1
    if family["shape"] != int(args.shape):
        print("REFUSED: plan shape differs from the trusted family shape.")
        return 1
    if not re.search(r"\d", args.prediction):
        print("REFUSED: numeric prediction required.")
        return 1
    if len(args.hypothesis.strip()) < 50 or len(args.kill.strip()) < 20:
        print("REFUSED: hypothesis >=50 chars and kill criteria >=20 chars.")
        return 1
    citations, err = parse_citations(args.sources or "")
    if err or not citations:
        print(f"REFUSED: valid --sources citations required ({err}).")
        return 1
    if len(args.reasoning.strip()) < 100:
        print("REFUSED: --reasoning >=100 chars.")
        return 1
    if (len(args.falsifier.strip()) < 60
            or len(args.falsifier_kill.strip()) < 20):
        print("REFUSED: a cheap --falsifier (>=60 chars) and its "
              "--falsifier-kill threshold (>=20 chars) are required.")
        return 1
    expected_prior = prior_family_verdict(args.direction, int(args.shape))
    if args.prior_family_verdict != expected_prior:
        print(f"REFUSED: --prior-family-verdict must cite the durable latest "
              f"family outcome exactly (expected: {expected_prior}).")
        return 1
    try:
        profile = _profile_record(st, args.counter_evidence, family,
                                  args.target_sha256)
        stall_auth = _recover_stall_if_authorized(st, args, family, profile)
    except GateRefusal as exc:
        print(f"REFUSED: {exc}")
        return 1
    plan_id = secrets.token_hex(6)
    request, perr = issue_permit(
        st, args.direction, args.mode, args.shape, args.impl, args.ledger,
        args.prediction, plan_id, args.predict_min, args.predict_max,
        campaign_id=args.campaign, target_sha256=args.target_sha256,
        counter_evidence=args.counter_evidence,
        prediction_kind=args.prediction_kind, family=family)
    if perr:
        print(f"REFUSED: {perr}")
        return 1
    request.update({
        "bottleneck": args.bottleneck,
        "expected_counter_change": family["expected_counter_change"],
        "falsifier": args.falsifier,
        "falsifier_kill": args.falsifier_kill,
        "prior_family_verdict": args.prior_family_verdict,
    })
    artifact = _request_artifact(request)
    st["research_open"] = False
    st.setdefault("request_shas", []).append(artifact[1])
    event = {"ts": now(), "step": "plan", "plan_id": plan_id,
         "campaign_id": campaign["campaign_id"],
         "direction": args.direction, "mode": args.mode, "shape": args.shape,
         "family": request["family"], "target_sha256": args.target_sha256,
         "bottleneck": args.bottleneck,
         "hypothesis": args.hypothesis, "prediction": args.prediction,
         "prediction_kind": args.prediction_kind,
         "predict_min": request["predict_min"], "predict_max": request["predict_max"],
         "counter_evidence_id": args.counter_evidence,
         "falsifier": args.falsifier, "falsifier_kill": args.falsifier_kill,
         "prior_family_verdict": args.prior_family_verdict,
         "kill": args.kill, "citations": citations, "reasoning": args.reasoning,
         "request_id": request["request_id"], "request_path": artifact[0],
         "request_sha256": artifact[1]}
    if stall_auth:
        event["stall_recovery_authority_event"] = stall_auth["authority_event_id"]
    commit(st, event)
    arm_permit(request, artifact)
    print(f"PLAN accepted. Request {request['request_id']} emitted for ONE run: "
          f"direction={args.direction} mode={args.mode} shape={args.shape}.")
    return 0


def cmd_delta(args) -> int:
    """Concise continuation within an open direction — no research packet,
    but still: what changed + numeric prediction + one permit."""
    st = load_state_strict()
    try:
        active_campaign(st, args.campaign)
        cards = open_cards()
        family = trusted_family(st, args.direction)
    except GateRefusal as exc:
        print(f"REFUSED: {exc}")
        return 1
    card = cards.get(args.direction)
    if card is None:
        print(f"REFUSED: no open card for '{args.direction}'.")
        return 1
    had_plan = False
    for e in _strict_log_rows():
        if (e.get("step") == "plan" and e.get("direction") == args.direction
                and e.get("campaign_id") == args.campaign):
            had_plan = True
            break
    if not had_plan:
        print("REFUSED: this direction has never had a FULL plan step — "
              "deltas only continue an already-planned direction.")
        return 1
    if len(args.changed.strip()) < 40 or not re.search(r"\d", args.prediction):
        print("REFUSED: --changed (>=40 chars, the exact delta from the last "
              "attempt) and a numeric --prediction are required.")
        return 1
    expected_prior = prior_family_verdict(args.direction, int(args.shape))
    if args.prior_family_verdict != expected_prior:
        print(f"REFUSED: --prior-family-verdict must be {expected_prior}.")
        return 1
    plan_id = secrets.token_hex(6)
    request, perr = issue_permit(
        st, args.direction, args.mode, args.shape, args.impl, args.ledger,
        args.prediction, plan_id, args.predict_min, args.predict_max,
        campaign_id=args.campaign, target_sha256=args.target_sha256,
        counter_evidence=args.counter_evidence,
        prediction_kind=args.prediction_kind, family=family)
    if perr:
        print(f"REFUSED: {perr}")
        return 1
    request["prior_family_verdict"] = args.prior_family_verdict
    artifact = _request_artifact(request)
    st.setdefault("request_shas", []).append(artifact[1])
    commit(st, {"ts": now(), "step": "delta", "plan_id": plan_id,
         "campaign_id": args.campaign,
         "direction": args.direction, "mode": args.mode, "shape": args.shape,
         "changed": args.changed, "prediction": args.prediction,
         "prediction_kind": args.prediction_kind,
         "predict_min": request["predict_min"], "predict_max": request["predict_max"],
         "counter_evidence_id": args.counter_evidence,
         "prior_family_verdict": args.prior_family_verdict,
         "request_id": request["request_id"], "request_path": artifact[0],
         "request_sha256": artifact[1]})
    arm_permit(request, artifact)
    print(f"DELTA accepted. Request {request['request_id']} emitted for ONE run.")
    return 0


def cmd_campaign_open(args) -> int:
    st = load_state_strict()
    if st.get("active_campaign"):
        print("REFUSED: an active campaign already exists.")
        return 1
    try:
        spec = _strict_json_file(Path(args.spec), "campaign spec")
        required = {"schema_version", "campaign_id", "max_total_attempts",
                    "max_calibrations_per_shape", "max_total_calibrations",
                    "stall_window", "timing_config", "score_scenarios"}
        if set(spec) != required or spec.get("schema_version") != 1:
            raise GateRefusal("campaign spec has unknown/missing fields")
        if not re.fullmatch(r"[A-Za-z0-9._-]{3,96}", str(spec["campaign_id"])):
            raise GateRefusal("campaign_id is invalid")
        policy = load_catalog()["campaign_policy"]
        bounds = (
            ("max_total_attempts", 1, policy["maximum_total_attempts"]),
            ("max_calibrations_per_shape", 1,
             policy["maximum_calibrations_per_shape"]),
            ("max_total_calibrations", 1,
             policy["maximum_total_calibrations"]),
            ("stall_window", 2, policy["stall_window"]),
        )
        for key, low, high in bounds:
            if type(spec.get(key)) is not int or not low <= spec[key] <= high:
                raise GateRefusal(f"campaign {key} must be in [{low}, {high}]")
        timing = spec.get("timing_config")
        if (not isinstance(timing, dict)
                or set(timing) != {"warmup", "repeats", "rounds"}
                or any(type(timing.get(k)) is not int or timing[k] <= 0
                       for k in timing)):
            raise GateRefusal("campaign timing_config must bind positive warmup/repeats/rounds")
        # A campaign's timing_config is not a free dial. The trusted
        # controller stamps its OWN protocol constant into every measurement
        # it records, and reconcile requires
        # measurement.timing_args == campaign.timing_config. Any other value
        # opens a campaign that can never reconcile a calibration, and a
        # campaign-bound calibration is a hard prerequisite for `plan` -- so
        # the mismatch wedges the gate permanently. Refuse here, where it is
        # free and recoverable, and name both values.
        protocol, protocol_source = controller_timing_protocol()
        controller_sha = sha_file(CONTROLLER)
        if protocol is not None and timing != protocol:
            raise GateRefusal(
                "campaign timing_config does not match the trusted "
                "controller's timing protocol -- this campaign could never "
                "reconcile a calibration and `plan` would refuse forever.\n"
                f"          spec timing_config: {json.dumps(timing, sort_keys=True)}\n"
                f"  controller protocol ({protocol_source}): "
                f"{json.dumps(protocol, sort_keys=True)}\n"
                "  Re-open the spec with the controller's values.")
        if (not isinstance(spec.get("score_scenarios"), list)
                or not spec["score_scenarios"]
                or any(not isinstance(x, str) or not x for x in spec["score_scenarios"])):
            raise GateRefusal("campaign needs one or more predeclared score scenarios")
        verified = verify_controller_receipt(
            args.authority_receipt, "open_campaign", spec, st)
    except GateRefusal as exc:
        print(f"REFUSED: {exc}")
        return 1
    cid = spec["campaign_id"]
    if cid in st["campaigns"]:
        print("REFUSED: campaign_id already exists; campaigns are immutable.")
        return 1
    campaign = dict(spec)
    campaign.update({
        "status": "active", "opened": now(), "opened_epoch": time.time(),
        "authority_event_id": verified["authority_event_id"],
        "authority_event_sha256": verified["authority_event_sha256"],
        "scientific_attempts": 0, "calibration_requests": 0,
        "calibration_requests_by_shape": {}, "calibrations": {},
        "production_outcomes": [], "families_tried": [],
        "side_evaluation_requests": 0, "side_evaluations": [],
        "stalled": False, "stall_nonce": None, "stalled_epoch": None,
        "stall_research_cycle": st.get("research_cycle", 0),
        "timing_protocol_binding": {
            "verified_against_controller": protocol is not None,
            "source": protocol_source,
            "controller_sha256": controller_sha,
        },
    })
    consume_controller_receipt(st, verified)
    st["campaigns"][cid] = campaign
    st["active_campaign"] = cid
    commit(st, {"ts": now(), "step": "campaign_open", "campaign_id": cid,
                "spec_sha256": sha_json(spec),
                "timing_config": timing,
                "timing_protocol_source": protocol_source,
                "timing_protocol_verified": protocol is not None,
                "controller_sha256": campaign["timing_protocol_binding"]["controller_sha256"],
                "authority_event_id": verified["authority_event_id"],
                "authority_event_sha256": verified["authority_event_sha256"]})
    if protocol is None:
        # Not a silent default: the campaign is opened on the operator's
        # value, but the state and the log both record that no controller
        # protocol was available to check it against.
        print(f"WARNING: {protocol_source} — timing_config "
              f"{json.dumps(timing, sort_keys=True)} could NOT be verified "
              "against a controller protocol constant. If the controller "
              "records different timing_args, reconcile will refuse and the "
              "chain will need `quarantine`.")
    print(f"Campaign {cid} opened under controller authority.")
    return 0


def cmd_family_register(args) -> int:
    st = load_state_strict()
    try:
        active_campaign(st, args.campaign)
        family = _strict_json_file(Path(args.family_spec), "family spec")
        catalog = load_catalog()
        _validate_family(family, catalog)
        if family["admission"] != "controller-authorized" or not family["allow_new_attempts"]:
            raise GateRefusal("new family must be controller-authorized and active")
        fid = family["family_id"]
        if fid in st["family_registry"] or fid in catalog["legacy_families"]:
            raise GateRefusal("family_id already exists and cannot be replaced")
        same_mechanism = [f for f in st["family_registry"].values()
                          if f.get("shape") == family["shape"]
                          and f.get("mechanism") == family["mechanism"]]
        parent = family.get("parent_family_id")
        if parent is None:
            if same_mechanism:
                raise GateRefusal("same shape+mechanism already exists; this variant must inherit that family")
            action = "register_family"
            subject = {"campaign_id": args.campaign, "family": family}
        else:
            trusted_family(st, parent, require_active=False)
            if len((args.novelty_basis or "").strip()) < 100:
                raise GateRefusal("child family requires >=100-char material novelty basis")
            action = "resolve_family_novelty"
            subject = {"campaign_id": args.campaign, "family": family,
                       "parent_family_id": parent,
                       "novelty_basis": args.novelty_basis.strip()}
        verified = verify_controller_receipt(
            args.authority_receipt, action, subject, st)
    except GateRefusal as exc:
        print(f"REFUSED: {exc}")
        return 1
    consume_controller_receipt(st, verified)
    st["family_registry"][fid] = family
    st["family_admissions"][fid] = {
        "admitted": now(), "admitted_epoch": time.time(),
        "action": action, "subject_sha256": sha_json(subject),
        "authority_event_id": verified["authority_event_id"],
        "authority_event_sha256": verified["authority_event_sha256"],
    }
    commit(st, {"ts": now(), "step": "family_register",
                "campaign_id": args.campaign, "family_id": fid,
                "mechanism": family["mechanism"], "shape": family["shape"],
                "parent_family_id": parent, "authority_action": action,
                "authority_event_id": verified["authority_event_id"],
                "subject_sha256": sha_json(subject)})
    print(f"Trusted family {fid} registered; variants must inherit this id.")
    return 0


def _request_preconditions(st: dict) -> str | None:
    if PERMIT.exists():
        return "a request transport projection already exists"
    if INFLIGHT.exists():
        return ("legacy in_flight.json exists; migrate it against the "
                "controller authority journal")
    if USED.exists() and any(f.name.startswith("claim.") for f in USED.iterdir()):
        return "a stranded reconciliation claim exists"
    try:
        authority_pending = _pending_authority_reason(st)
    except GateRefusal as exc:
        return str(exc)
    if authority_pending:
        return authority_pending
    hard = unacked_hard_verdicts(st)
    violations = [h for h in hard if h.get("verdict") == "RULE_VIOLATION"]
    if violations:
        return f"{len(violations)} unresolved integrity violation(s) pause execution"
    return None


def cmd_diagnostic(args) -> int:
    st = load_state_strict()
    try:
        campaign = active_campaign(st, args.campaign)
        blocked = _request_preconditions(st)
        if blocked:
            raise GateRefusal(blocked)
        shape = int(args.shape)
        if not 1 <= shape <= 14 or not HEX64.fullmatch(args.target_sha256):
            raise GateRefusal("diagnostic requires shape 1..14 and a 64-hex target SHA")
        catalog = load_catalog()
        bottlenecks = [x.strip() for x in args.supports.split(",") if x.strip()]
        if not bottlenecks or any(x not in catalog["bottlenecks"] for x in bottlenecks):
            raise GateRefusal("--supports must list known bottleneck ids")
        if any(args.tool not in catalog["bottlenecks"][b]["evidence_tools"]
               for b in bottlenecks):
            raise GateRefusal("diagnostic tool is inappropriate for a declared bottleneck")
        if len(args.question.strip()) < 40 or len(args.route.strip()) < 3:
            raise GateRefusal("diagnostic needs a concrete question and route")
        # Snapshot the catalog terms this diagnostic is being authorized
        # against, into the content-addressed (and controller-bound) request.
        # The scientific path already snapshots catalog_sha256 into the
        # immutable budget snapshot; the diagnostic path used to re-read the
        # LIVE catalog at reconcile time, so adding one required_metric after
        # a profile had already been captured retroactively invalidated it
        # and wedged the chain forever. Evidence is judged against the
        # contract that was in force when it was authorized.
        contract = {
            b: {"required_metrics": sorted(catalog["bottlenecks"][b]["required_metrics"]),
                "evidence_tools": sorted(catalog["bottlenecks"][b]["evidence_tools"])}
            for b in bottlenecks
        }
    except (GateRefusal, TypeError, ValueError) as exc:
        print(f"REFUSED: {exc}")
        return 1
    record_id = "profile-" + secrets.token_hex(12)
    # The trusted profile namespace must exist before the controller's
    # profiler worker is told to write into it; reconcile later refuses any
    # artifact whose path escapes this directory.
    PROFILE_EVIDENCE.mkdir(parents=True, exist_ok=True)
    output = PROFILE_EVIDENCE / f"{record_id}.json"
    request = {
        "schema_version": 1, "request_id": secrets.token_hex(16),
        "request_kind": "diagnostic", "campaign_id": campaign["campaign_id"],
        "mode": "diagnostic", "shape": shape,
        "target_sha256": args.target_sha256, "tool": args.tool,
        "question": args.question.strip(), "route": args.route.strip(),
        "supported_bottlenecks": bottlenecks,
        "bottleneck_contract": contract,
        "catalog_sha256": sha_file(CATALOG),
        "profile_record_id": record_id,
        "profile_output": str(output.relative_to(ROOT)),
        "impl_path": None, "impl_sha256": None, "ledger": None,
        "ledger_namespace": "diagnostic", "ledger_pre_lines": None,
        "promotion_allowed": False, "candidate_authorized": False,
        "scientific_strike_eligible": False,
        "verdict_lines": st.get("_verdict_lines_snapshot", -1),
        "issued": now(), "expires_epoch": time.time() + PERMIT_TTL_S,
    }
    artifact = _request_artifact(request)
    st.setdefault("request_shas", []).append(artifact[1])
    commit(st, {"ts": now(), "step": "diagnostic_request",
                "campaign_id": campaign["campaign_id"], "shape": shape,
                "target_sha256": args.target_sha256, "tool": args.tool,
                "profile_record_id": record_id, "request_id": request["request_id"],
                "catalog_sha256": request["catalog_sha256"],
                "request_path": artifact[0], "request_sha256": artifact[1]})
    arm_permit(request, artifact)
    print(f"DIAGNOSTIC request {request['request_id']} emitted; it cannot authorize candidate bytes or promotion.")
    return 0


def cmd_calibrate(args) -> int:
    st = load_state_strict()
    try:
        campaign = active_campaign(st, args.campaign)
        blocked = _request_preconditions(st)
        if blocked:
            raise GateRefusal(blocked)
        shape = int(args.shape)
        if not 1 <= shape <= 14:
            raise GateRefusal("shape must be 1..14")
        if str(shape) in campaign.get("calibrations", {}):
            raise GateRefusal("shape already has an immutable designated campaign calibration")
        by_shape = campaign["calibration_requests_by_shape"]
        used = int(by_shape.get(str(shape), 0))
        if used >= campaign["max_calibrations_per_shape"]:
            raise GateRefusal("per-shape calibration request cap exhausted")
        if campaign["calibration_requests"] >= campaign["max_total_calibrations"]:
            raise GateRefusal("campaign calibration request cap exhausted")
        machine = Path(args.machine_state).resolve()
        machine.relative_to(ROOT)
        machine_obj = _strict_json_file(machine, "machine-state artifact")
        if not machine_obj:
            raise GateRefusal("machine-state artifact cannot be empty")
    except (GateRefusal, TypeError, ValueError) as exc:
        print(f"REFUSED: {exc}")
        return 1
    request = {
        "schema_version": 1, "request_id": secrets.token_hex(16),
        "request_kind": "calibration", "campaign_id": campaign["campaign_id"],
        "mode": "calibration", "shape": shape,
        "timing_config": campaign["timing_config"],
        "machine_state_path": str(machine.relative_to(ROOT)),
        "machine_state_sha256": sha_file(machine),
        "impl_path": None, "impl_sha256": None,
        "ledger": None, "ledger_namespace": "calibration",
        "ledger_pre_lines": None,
        "promotion_allowed": False, "candidate_authorized": False,
        "scientific_strike_eligible": False,
        "verdict_lines": st.get("_verdict_lines_snapshot", -1),
        "issued": now(), "expires_epoch": time.time() + PERMIT_TTL_S,
    }
    artifact = _request_artifact(request)
    campaign["calibration_requests"] += 1
    campaign["calibration_requests_by_shape"][str(shape)] = used + 1
    st.setdefault("request_shas", []).append(artifact[1])
    commit(st, {"ts": now(), "step": "calibration_request",
                "campaign_id": campaign["campaign_id"], "shape": shape,
                "timing_config": campaign["timing_config"],
                "machine_state_sha256": request["machine_state_sha256"],
                "request_id": request["request_id"], "request_path": artifact[0],
                "request_sha256": artifact[1]})
    arm_permit(request, artifact)
    print(f"CALIBRATION request {request['request_id']} emitted ({used + 1}/"
          f"{campaign['max_calibrations_per_shape']} for shape {shape}).")
    return 0


def cmd_side_evaluate(args) -> int:
    """Emit immutable side-lane evidence for the dedicated shape-6/14 lanes.

    Side evaluation spends the campaign attempt budget and is independently
    audited, but can never authorize a primary champion or write a primary or
    scratch ledger. Final-evidence selection remains a separate authority step.
    """
    st = load_state_strict()
    try:
        campaign = active_campaign(st, args.campaign)
        blocked = _request_preconditions(st)
        if blocked:
            raise GateRefusal(blocked)
        hard = unacked_hard_verdicts(st)
        if hard:
            raise GateRefusal("side evaluation is paused by unresolved hard audit verdicts")
        if campaign.get("stalled"):
            raise GateRefusal("campaign stall brake is engaged")
        if st.get("pending_postmortem"):
            raise GateRefusal("family postmortem debt must be resolved first")
        if campaign["scientific_attempts"] >= campaign["max_total_attempts"]:
            raise GateRefusal("campaign attempt budget exhausted")
        shape = int(args.shape)
        if shape not in (6, 14):
            raise GateRefusal("side evaluation supports only shape 6 or 14")
        source = Path(args.submission).resolve()
        expected_source = (ROOT / "Project" / "submission"
                           / "torch_transformer_benchmark_submission.py").resolve()
        if source.is_symlink() or not source.is_file():
            raise GateRefusal("generated submission must be a regular non-symlink file")
        if source != expected_source:
            raise GateRefusal(f"side evaluation requires exact generated submission: "
                              f"{expected_source}")
        source_sha = sha_file(source)
        if args.submission_sha256 and args.submission_sha256 != source_sha:
            raise GateRefusal("--submission-sha256 disagrees with generated submission bytes")
    except (GateRefusal, OSError, TypeError, ValueError) as exc:
        print(f"REFUSED: {exc}")
        return 1
    mode = f"shape{shape}"
    request = {
        "schema_version": 1, "request_id": secrets.token_hex(16),
        "request_kind": "side_evaluation", "campaign_id": campaign["campaign_id"],
        "mode": mode, "shape": shape,
        "impl_path": str(source.relative_to(ROOT)), "impl_sha256": source_sha,
        "ledger": None, "ledger_namespace": "side-evidence",
        "ledger_pre_lines": None, "candidate_authorized": True,
        "promotion_allowed": False, "scientific_strike_eligible": False,
        "champion_candidate": False, "final_evidence_candidate": True,
        "verdict_lines": st.get("_verdict_lines_snapshot", -1),
        "issued": now(), "expires_epoch": time.time() + PERMIT_TTL_S,
    }
    artifact = _request_artifact(request)
    campaign["side_evaluation_requests"] += 1
    st["request_shas"].append(artifact[1])
    commit(st, {
        "ts": now(), "step": "side_evaluation_request",
        "campaign_id": campaign["campaign_id"], "shape": shape, "mode": mode,
        "candidate_sha256": source_sha, "request_id": request["request_id"],
        "request_path": artifact[0], "request_sha256": artifact[1],
        "champion_candidate": False, "final_evidence_candidate": True,
    })
    arm_permit(request, artifact)
    print(f"SIDE EVALUATION request {request['request_id']} emitted for {mode}; "
          "it cannot authorize primary champion promotion.")
    return 0


def gate_lock():
    """One shared advisory lock serializing EVERY state transition (CLI and
    watcher). IDEMPOTENT within a process (flock on a second fd of the same
    file would self-deadlock). Blocks up to 30s, then fails CLOSED."""
    import fcntl
    if _LOCK_REF:
        return _LOCK_REF[0]
    LOOP.mkdir(parents=True, exist_ok=True)
    fh = open(LOOP / ".gate.lock", "w")
    deadline = time.time() + 30
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _LOCK_REF.append(fh)
            return fh
        except OSError:
            if time.time() > deadline:
                raise SystemExit("REFUSED: gate lock busy >30s — fail closed.")
            time.sleep(0.2)


def _settle_authority_request(st: dict, chain: dict, terminal: dict) -> None:
    terminal_sha = terminal.get("event_sha256")
    request_sha = chain["request_sha256"]
    if not HEX64.fullmatch(str(terminal_sha or "")):
        raise GateRefusal("authority terminal event has a malformed SHA-256")
    if terminal_sha in st["reconciled_authority_event_shas"]:
        raise GateRefusal("authority terminal event was already reconciled")
    if request_sha in st["settled_request_shas"]:
        raise GateRefusal("gate request was already settled")
    st["reconciled_authority_event_shas"].append(terminal_sha)
    st["settled_request_shas"].append(request_sha)


def _quarantine_settled_transport(st: dict) -> None:
    """Remove only UX transport after its request is durably settled.

    The move is recoverable and has no authority effect. Malformed transport
    remains visible and blocks issuance instead of being silently discarded.
    """
    if not PERMIT.exists():
        return
    projection = _strict_json_file(PERMIT, "request transport projection")
    request_sha = projection.get("request_sha256")
    if request_sha not in st.get("settled_request_shas", []):
        return
    request = _gate_request_from_sha(request_sha)
    if any(projection.get(key) != value for key, value in request.items()):
        raise GateRefusal("transport projection disagrees with its immutable request")
    USED.mkdir(parents=True, exist_ok=True)
    target = USED / f"transport.{request_sha}.{secrets.token_hex(4)}.json"
    PERMIT.replace(target)


def _family_group(st: dict, request: dict) -> tuple[str, dict, dict]:
    family_id = request.get("direction_id")
    shape = request.get("shape")
    family = trusted_family(st, family_id)
    if family["shape"] != shape:
        raise GateRefusal("reconciled request no longer matches its trusted family")
    campaign = active_campaign(st, request.get("campaign_id"))
    gkey = f"{family_id}|{shape}"
    grp = st.setdefault("groups", {}).setdefault(
        gkey, {"best_speedup": None, "strikes": 0, "exec_failures": 0,
               "closed": False, "closure_nonce": None, "scientific_attempts": 0,
               "nonstrike_budget": 2, "last_attempt_sha": None})
    for key, default in (("best_speedup", None), ("strikes", 0),
                         ("exec_failures", 0), ("closed", False),
                         ("closure_nonce", None), ("scientific_attempts", 0),
                         ("nonstrike_budget", 2)):
        grp.setdefault(key, default)
    return gkey, grp, campaign


def _close_group_if_needed(st: dict, gkey: str, grp: dict,
                           outcome: dict) -> None:
    if grp.get("strikes", 0) < MAX_STRIKES:
        return
    grp["closed"] = True
    grp["closure_nonce"] = grp.get("closure_nonce") or secrets.token_hex(8)
    if gkey not in st["pending_postmortem"]:
        st["pending_postmortem"].append(gkey)
    outcome["closed"] = True
    outcome["closure_nonce"] = grp["closure_nonce"]


def _reconcile_authority_failure(st: dict, chain: dict) -> None:
    request = chain["request"]
    failed = chain["failed"]
    outcome = {
        "ts": now(), "step": "authority_reconcile",
        "result": "infrastructure_failure", "request_id": request["request_id"],
        "request_sha256": chain["request_sha256"],
        "permit_id": chain["permit"]["permit_id"],
        "controller_issue_event_id": chain["issued"]["event_id"],
        "terminal_event_id": failed["event_id"],
        "terminal_event_sha256": failed["event_sha256"],
        "campaign_id": request["campaign_id"], "shape": request["shape"],
        "mode": request["mode"], "scientific_strike": False,
        "reason": failed["payload"]["reason"],
    }
    if request["request_kind"] == "scientific_attempt":
        gkey, grp, campaign = _family_group(st, request)
        # Infrastructure failures consume the immutable experiment budget but
        # never count as scientific falsification/strikes.
        grp["scientific_attempts"] += 1
        campaign["scientific_attempts"] += 1
        grp["exec_failures"] += 1
        outcome.update({"direction": request["direction_id"], "group": gkey,
                        "exec_failures": grp["exec_failures"]})
        if grp["exec_failures"] >= MAX_EXEC_FAILURES:
            grp["infrastructure_paused"] = True
            outcome["infrastructure_paused"] = True
    elif request["request_kind"] == "side_evaluation":
        campaign = active_campaign(st, request["campaign_id"])
        campaign["scientific_attempts"] += 1
        outcome["side_evidence_only"] = True
    _settle_authority_request(st, chain, failed)
    commit(st, outcome)


def _reconcile_authority_diagnostic(st: dict, chain: dict) -> None:
    request = chain["request"]
    measurement = chain["measurement"]
    packet = chain["packet"]
    path = (ROOT / request["profile_output"]).resolve()
    try:
        path.relative_to(PROFILE_EVIDENCE.resolve())
    except ValueError as exc:
        raise GateRefusal("diagnostic profile escaped its trusted namespace") from exc
    artifact = _strict_json_file(path, "diagnostic profile artifact")
    required = {
        "schema_version", "profile_record_id", "request_id", "campaign_id",
        "shape", "target_sha256", "tool", "tool_version", "created_epoch",
        "machine_state_sha256", "route", "metrics", "supported_bottlenecks",
        "raw_artifacts", "gate_request_sha256",
    }
    if set(artifact) != required or artifact.get("schema_version") != 1:
        raise GateRefusal("diagnostic artifact has an unknown/missing field")
    digest = sha_file(path)
    if (measurement["payload"].get("diagnostic_profile_sha256") != digest
            or packet["payload"].get("diagnostic_profile_sha256") != digest):
        raise GateRefusal("diagnostic profile is not bound by controller authority")
    bindings = {
        "profile_record_id": request["profile_record_id"],
        "request_id": request["request_id"],
        "campaign_id": request["campaign_id"],
        "shape": request["shape"],
        "target_sha256": request["target_sha256"],
        "tool": request["tool"], "route": request["route"],
        "supported_bottlenecks": request["supported_bottlenecks"],
        "gate_request_sha256": chain["request_sha256"],
    }
    if any(artifact.get(key) != value for key, value in bindings.items()):
        raise GateRefusal("diagnostic artifact disagrees with its authority/request bindings")
    if (not isinstance(artifact["metrics"], dict) or not artifact["metrics"]
            or not isinstance(artifact["raw_artifacts"], list)
            or not isinstance(artifact["created_epoch"], (int, float))
            or isinstance(artifact["created_epoch"], bool)
            or not math.isfinite(float(artifact["created_epoch"]))
            or not HEX64.fullmatch(str(artifact["machine_state_sha256"]))):
        raise GateRefusal("diagnostic metrics/raw artifacts/machine state are malformed")
    for raw in artifact["raw_artifacts"]:
        if (not isinstance(raw, dict) or set(raw) != {"path", "sha256"}
                or not isinstance(raw.get("path"), str)
                or not HEX64.fullmatch(str(raw.get("sha256", "")))):
            raise GateRefusal("diagnostic raw artifact reference is malformed")
        raw_path = (ROOT / raw["path"]).resolve()
        try:
            raw_path.relative_to(PROFILE_EVIDENCE.resolve())
        except ValueError as exc:
            raise GateRefusal("diagnostic raw artifact escaped its namespace") from exc
        if not raw_path.is_file() or sha_file(raw_path) != raw["sha256"]:
            raise GateRefusal("diagnostic raw artifact is missing or changed")
    # The catalog is mutable. Judge already-collected evidence against the
    # terms pinned into the immutable request, so a later catalog edit cannot
    # retroactively invalidate a profile that was correct when captured.
    # Requests emitted before the snapshot existed fall back to the live
    # catalog (the old behaviour) rather than becoming unreconcilable.
    contract = request.get("bottleneck_contract")
    if isinstance(contract, dict) and contract:
        contract_source = "the catalog terms pinned in the request"
    else:
        live = load_catalog()["bottlenecks"]
        contract = {b: {"required_metrics": live[b]["required_metrics"],
                        "evidence_tools": live[b]["evidence_tools"]}
                    for b in artifact["supported_bottlenecks"] if b in live}
        contract_source = "the live mechanism catalog (request predates pinning)"
    for bottleneck in artifact["supported_bottlenecks"]:
        terms = contract.get(bottleneck)
        if (not isinstance(terms, dict)
                or not isinstance(terms.get("evidence_tools"), list)
                or not isinstance(terms.get("required_metrics"), list)
                or artifact["tool"] not in terms["evidence_tools"]
                or any(metric not in artifact["metrics"]
                       for metric in terms["required_metrics"])):
            raise GateRefusal(
                f"diagnostic evidence is insufficient for {bottleneck} "
                f"(checked against {contract_source})")
    record = {
        "profile_record_id": artifact["profile_record_id"],
        "artifact_path": str(path.relative_to(ROOT)), "artifact_sha256": digest,
        "shape": artifact["shape"], "target_sha256": artifact["target_sha256"],
        "tool": artifact["tool"],
        "supported_bottlenecks": artifact["supported_bottlenecks"],
        "metrics": artifact["metrics"], "created_epoch": artifact["created_epoch"],
        "reconciled_state_seq": st.get("seq", 0) + 1,
    }
    if record["profile_record_id"] in st["profiles"]:
        raise GateRefusal("profile_record_id already exists")
    st["profiles"][record["profile_record_id"]] = record
    _settle_authority_request(st, chain, measurement)
    commit(st, {
        "ts": now(), "step": "diagnostic_reconcile",
        "campaign_id": request["campaign_id"], "shape": request["shape"],
        "profile_record_id": record["profile_record_id"],
        "artifact_sha256": digest, "tool": record["tool"],
        "request_id": request["request_id"],
        "catalog_sha256": request.get("catalog_sha256"),
        "bottleneck_contract_source": contract_source,
        "request_sha256": chain["request_sha256"],
        "permit_id": chain["permit"]["permit_id"],
        "measurement_event_sha256": measurement["event_sha256"],
        "packet_sha256": packet["sha256"],
        "scientific_attempt": False, "champion_eligible": False,
    })


def _reconcile_authority_calibration(st: dict, chain: dict) -> None:
    request = chain["request"]
    measurement = chain["measurement"]
    payload = measurement["payload"]
    campaign = active_campaign(st, request["campaign_id"])
    noise = payload.get("calibrated_noise")
    threshold = payload.get("promotion_threshold")
    if payload.get("timing_args") != request["timing_config"]:
        # Campaign open now binds timing_config to the controller's protocol,
        # so this can only mean the CONTROLLER changed after the campaign was
        # opened. Say so with both values instead of one opaque message: the
        # campaign is immutable, so the fix is a new campaign (or, for a chain
        # already measured under the old protocol, `quarantine`).
        raise GateRefusal(
            "controller timing protocol drifted away from the campaign after "
            "it was opened -- this measurement can never bind.\n"
            f"    campaign timing_config: {json.dumps(request['timing_config'], sort_keys=True)}\n"
            f"  controller timing_args : {json.dumps(payload.get('timing_args'), sort_keys=True, default=str)}")
    if (not isinstance(noise, (int, float)) or isinstance(noise, bool)
            or not math.isfinite(float(noise)) or noise <= 0
            or not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
            or not math.isfinite(float(threshold)) or threshold <= 1
            or payload.get("performance_eligible") is not False):
        raise GateRefusal("controller calibration has invalid noise/timing bindings")
    key = str(request["shape"])
    if key in campaign["calibrations"]:
        raise GateRefusal("designated campaign calibration already exists; no reroll")
    environment_sha = sha_json(payload.get("worker_environment"))
    campaign["calibrations"][key] = {
        "entry_id": payload["run_id"], "noise": float(noise),
        "promotion_threshold": float(threshold),
        "timing_config": request["timing_config"],
        "machine_state_sha256": request["machine_state_sha256"],
        "worker_environment_sha256": environment_sha,
        "request_id": request["request_id"],
        "request_sha256": chain["request_sha256"],
        "measurement_event_sha256": measurement["event_sha256"],
    }
    _settle_authority_request(st, chain, measurement)
    commit(st, {
        "ts": now(), "step": "calibration_reconcile", "result": "bound",
        "campaign_id": request["campaign_id"], "shape": request["shape"],
        "request_id": request["request_id"],
        "request_sha256": chain["request_sha256"],
        "permit_id": chain["permit"]["permit_id"],
        "entry_id": payload["run_id"], "noise": noise,
        "promotion_threshold": threshold,
        "machine_state_sha256": request["machine_state_sha256"],
        "worker_environment_sha256": environment_sha,
        "measurement_event_sha256": measurement["event_sha256"],
        "packet_sha256": chain["packet"]["sha256"],
        "champion_eligible": False,
    })


def _reconcile_authority_scientific(st: dict, chain: dict) -> None:
    request = chain["request"]
    measurement = chain["measurement"]
    payload = measurement["payload"]
    packet = chain["packet"]
    mode = request["mode"]
    gkey, grp, campaign = _family_group(st, request)
    grp["scientific_attempts"] += 1
    campaign["scientific_attempts"] += 1
    if request["direction_id"] not in campaign["families_tried"]:
        campaign["families_tried"].append(request["direction_id"])
    sha_key = ("primary_attempted_shas" if mode in PRIMARY_MODES
               else "scratch_attempted_shas")
    if request["impl_sha256"] not in grp.setdefault(sha_key, []):
        grp[sha_key].append(request["impl_sha256"])
    if mode == "confirmation":
        grp["nonstrike_budget"] = grp.get("nonstrike_budget", 2) - 1
    grp["exec_failures"] = 0
    speed = float(payload["supporting_timing"]["event_speedup"])
    correct = payload["controller_correctness"]["passed"] is True
    clean = payload["supporting_timing"]["suspicious"] is False
    outcome = {
        "ts": now(), "step": "reconcile", "request_id": request["request_id"],
        "request_sha256": chain["request_sha256"],
        "permit_id": chain["permit"]["permit_id"],
        "controller_issue_event_id": chain["issued"]["event_id"],
        "campaign_id": request["campaign_id"],
        "direction": request["direction_id"], "group": gkey,
        "shape": request["shape"], "mode": mode,
        "entry_id": payload["run_id"], "speedup": speed,
        "correct": correct, "clean": clean,
        "lane": payload["lane"],
        "measurement_event_sha256": measurement["event_sha256"],
        "packet_sha256": packet["sha256"], "champion_eligible": False,
    }
    if mode == "optimization":
        entry_id = payload["run_id"]
        if entry_id in st["pending_audit_decisions"]:
            raise GateRefusal("optimization measurement entry_id is duplicated")
        st["pending_audit_decisions"][entry_id] = {
            "entry_id": entry_id, "campaign_id": request["campaign_id"],
            "group": gkey, "family_id": request["direction_id"],
            "shape": request["shape"], "request_id": request["request_id"],
            "request_sha256": chain["request_sha256"],
            "candidate_sha256": request["impl_sha256"], "speedup": speed,
            "correct": correct, "clean": clean,
            "controller_performance_eligible": payload["performance_eligible"],
            "measurement_event_id": measurement["event_id"],
            "measurement_event_sha256": measurement["event_sha256"],
            "packet_sha256": packet["sha256"],
            "packet_binding_event_sha256": chain["binding"]["event_sha256"],
            "eligibility_scope": "primary_champion",
        }
        outcome["result"] = "pending_bound_independent_audit"
    elif mode == "screening":
        if st.get("pending_screen_judgment") is not None:
            raise GateRefusal("another screening judgment is already pending")
        outcome["result"] = "pending_computed_screen_judgment"
        outcome["declared_prediction"] = request.get("prediction")
        st["pending_screen_judgment"] = {
            "permit_id": chain["permit"]["permit_id"], "group": gkey,
            "observed_speedup": speed, "row_correct": correct,
            "predict_min": request.get("predict_min"),
            "predict_max": request.get("predict_max"),
            "prediction_kind": request.get("prediction_kind"),
            "measurement_event_sha256": measurement["event_sha256"],
        }
    elif mode == "correctness":
        outcome["result"] = "pass" if correct else "falsified"
        if not correct:
            grp["strikes"] += 1
            outcome["scientific_strike"] = True
    elif mode == "confirmation":
        outcome["result"] = "confirmation_recorded"
    _close_group_if_needed(st, gkey, grp, outcome)
    _settle_authority_request(st, chain, measurement)
    commit(st, outcome)


def _reconcile_authority_side(st: dict, chain: dict) -> None:
    request = chain["request"]
    measurement = chain["measurement"]
    payload = measurement["payload"]
    packet = chain["packet"]
    campaign = active_campaign(st, request["campaign_id"])
    campaign["scientific_attempts"] += 1
    entry_id = payload["entry_id"]
    if entry_id in st["pending_audit_decisions"]:
        raise GateRefusal("side-evaluation entry_id is duplicated")
    st["pending_audit_decisions"][entry_id] = {
        "entry_id": entry_id, "campaign_id": request["campaign_id"],
        "group": None, "family_id": None, "shape": request["shape"],
        "request_id": request["request_id"],
        "request_sha256": chain["request_sha256"],
        "candidate_sha256": request["impl_sha256"],
        "speedup": None,
        "correct": payload["controller_validation"]["passed"] is True,
        "clean": True,
        "controller_performance_eligible":
            payload["evidence_eligible_pre_audit"] is True,
        "measurement_event_id": measurement["event_id"],
        "measurement_event_sha256": measurement["event_sha256"],
        "packet_sha256": packet["sha256"],
        "side_evidence_sha256": payload["side_evidence_sha256"],
        "packet_binding_event_sha256": chain["binding"]["event_sha256"],
        "eligibility_scope": "final_evidence_only",
    }
    _settle_authority_request(st, chain, measurement)
    commit(st, {
        "ts": now(), "step": "side_evaluation_reconcile",
        "result": "pending_bound_independent_audit",
        "campaign_id": request["campaign_id"], "shape": request["shape"],
        "mode": request["mode"], "entry_id": entry_id,
        "candidate_sha256": request["impl_sha256"],
        "request_id": request["request_id"],
        "request_sha256": chain["request_sha256"],
        "measurement_event_sha256": measurement["event_sha256"],
        "packet_sha256": packet["sha256"],
        "champion_eligible": False, "final_evidence_eligible": False,
    })


def _dispatch_reconciler(st: dict, chain: dict) -> None:
    """Route one complete chain to the reconciler that owns its kind."""
    request = chain["request"]
    if chain.get("failed") is not None:
        _reconcile_authority_failure(st, chain)
    elif request["request_kind"] == "diagnostic":
        _reconcile_authority_diagnostic(st, chain)
    elif request["request_kind"] == "calibration":
        _reconcile_authority_calibration(st, chain)
    elif request["request_kind"] == "side_evaluation":
        _reconcile_authority_side(st, chain)
    else:
        _reconcile_authority_scientific(st, chain)


def _reconcile_probe_reason(st: dict, chain: dict) -> str | None:
    """The CURRENT refusal text for one chain, or None if it would settle.

    Runs the real reconciler against a deep copy of state marked ``_dry_run``,
    which turns ``commit`` into a no-op, so the probe cannot write the gate
    log, the state file, or any artifact. A quarantine record therefore
    carries the reconciler's own words rather than operator prose.
    """
    probe = copy.deepcopy(st)
    probe["_dry_run"] = True
    try:
        _dispatch_reconciler(probe, chain)
    except GateRefusal as exc:
        return str(exc)
    return None


def cmd_reconcile(_args) -> int:
    """Crash-safe reconciliation from authority events, never workspace rows."""
    gate_lock()
    st = load_state_strict()
    try:
        chains = _authority_chains(st)
        settled = set(st["settled_request_shas"])
        progressed = False
        for chain in chains:
            if chain["request_sha256"] in settled:
                continue
            terminal = chain.get("terminal")
            if terminal is None:
                # A signed but never-consumed permit is harmless after expiry;
                # a consumed/incomplete run remains closed for controller recovery.
                if (chain.get("consumed") is None
                        and _parse_epoch(chain["permit"]["expires_at"], "permit expiry")
                            <= time.time()):
                    _settle_authority_request(st, chain, chain["issued"])
                    commit(st, {
                        "ts": now(), "step": "authority_reconcile",
                        "result": "expired_unconsumed", "scientific_strike": False,
                        "request_id": chain["request"]["request_id"],
                        "request_sha256": chain["request_sha256"],
                        "permit_id": chain["permit"]["permit_id"],
                        "terminal_event_sha256": chain["issued"]["event_sha256"],
                    })
                    settled.add(chain["request_sha256"])
                    progressed = True
                continue
            request = chain["request"]
            try:
                _dispatch_reconciler(st, chain)
            except GateRefusal as exc:
                # A refusal here is TERMINAL for this chain: it never enters
                # settled_request_shas, so _pending_authority_reason keeps
                # reporting it and _request_preconditions refuses every
                # request kind from now on. Nothing this run mutated is
                # committed. Say which chain, and name the owner-gated exit
                # for causes that cannot be repaired (a deleted raw profiler
                # artifact, a catalog term that moved).
                print(f"REFUSED: {exc}")
                print(f"  request_id     : {request.get('request_id')}")
                print(f"  request_sha256 : {chain['request_sha256']}")
                print(f"  kind/mode      : {request.get('request_kind')}"
                      f" / {request.get('mode')}")
                print(f"  permit_id      : {chain['permit']['permit_id']}")
                print("  Repair the cause and re-run `reconcile`. If the cause "
                      "is unrecoverable, the owner can settle this one chain "
                      "as unreconcilable with a signed capability:")
                print("    python3 Project/tools/run_gate.py quarantine "
                      f"--request-sha256 {chain['request_sha256']} "
                      "--authority-receipt <receipt.json>")
                return 1
            settled.add(chain["request_sha256"])
            progressed = True
        # Unissued request artifacts become inert at their immutable expiry.
        issued = {chain["request_sha256"] for chain in chains}
        for request_sha in st["request_shas"]:
            if request_sha in settled or request_sha in issued:
                continue
            request = _gate_request_from_sha(request_sha)
            if float(request.get("expires_epoch", 0)) <= time.time():
                st["settled_request_shas"].append(request_sha)
                commit(st, {"ts": now(), "step": "request_expired_unissued",
                            "request_id": request["request_id"],
                            "request_sha256": request_sha})
                settled.add(request_sha)
                progressed = True
        _quarantine_settled_transport(st)
        if not progressed:
            pending = _pending_authority_reason(st)
            if pending:
                print(f"PENDING: {pending}")
        return 0
    except GateRefusal as exc:
        print(f"REFUSED: {exc}")
        return 1


def cmd_quarantine(args) -> int:
    """OWNER-ONLY: settle ONE chain the reconciler can never settle itself.

    A GateRefusal raised inside a reconciler is terminal by construction. The
    chain never enters ``settled_request_shas``, ``_pending_authority_reason``
    keeps reporting it, and ``_request_preconditions`` then refuses
    ``diagnostic``, ``calibrate``, ``side-evaluate``, ``plan`` and ``delta``
    from that moment on. Some causes are genuinely unrepairable in the
    workspace -- a multi-GB nsys report cleaned up after the fact, a catalog
    term that moved under an already-collected profile -- so an exit has to
    exist or a careful operator can lose the gate permanently.

    It is capability-gated exactly like ``reopen``: a controller-verified
    owner capability bound to this exact subject. The agent cannot self-serve
    it, and workspace prose has no authority. Sign this object:

        {"request_sha256", "request_id", "terminal_event_sha256",
         "reason", "resolution"}

    ``reason`` is not operator prose: the gate re-runs the real reconciler
    against a throwaway copy of state and binds ITS refusal text, so the
    owner signs the failure that actually happened. Run the command once
    without a valid receipt and it prints the subject and its SHA-256.

    A quarantine SETTLES and RECORDS. It never records a measurement, never
    promotes, never satisfies or clears a strike, and never binds a
    calibration. A quarantined scientific chain still spends its experiment
    budget, so quarantining cannot launder GPU time back into the campaign.
    """
    st = load_state_strict()
    resolution = (args.resolution or "").strip()
    try:
        request_sha = str(args.request_sha256 or "").strip().lower()
        if not HEX64.fullmatch(request_sha):
            raise GateRefusal("--request-sha256 must be a 64-hex gate request hash")
        if request_sha not in st.get("request_shas", []):
            raise GateRefusal("that request hash is not in the gate's request registry")
        if request_sha in st.get("settled_request_shas", []):
            raise GateRefusal("that request is already settled; nothing to quarantine")
        try:
            chains = {c["request_sha256"]: c for c in _authority_chains(st)}
        except GateRefusal as exc:
            raise GateRefusal(
                f"the controller authority chain itself fails closed ({exc}). "
                "Quarantine settles a reconciler refusal, not a broken chain; "
                "this needs controller/owner recovery of the journal.")
        chain = chains.get(request_sha)
        if chain is None:
            raise GateRefusal(
                "that request was never issued a controller permit, so there "
                "is nothing to settle; it goes inert at its own expiry via "
                "`reconcile`")
        terminal = chain.get("terminal")
        if terminal is None:
            raise GateRefusal(
                "that request has no terminal controller event (still in "
                "flight, or consumed without a bound measurement packet). "
                "Quarantine only settles a chain the reconciler has actually "
                "refused; an incomplete chain stays open for controller "
                "recovery.")
        terminal_sha = terminal.get("event_sha256")
        if not HEX64.fullmatch(str(terminal_sha or "")):
            raise GateRefusal("authority terminal event has a malformed SHA-256")
        reason = _reconcile_probe_reason(st, chain)
        if reason is None:
            raise GateRefusal(
                "that request reconciles cleanly right now — run `reconcile`, "
                "not `quarantine`")
        request = chain["request"]
        subject = {"request_sha256": request_sha,
                   "request_id": request["request_id"],
                   "terminal_event_sha256": terminal_sha,
                   "reason": reason, "resolution": resolution}
    except GateRefusal as exc:
        print(f"REFUSED: {exc}")
        return 1
    try:
        verified = verify_controller_receipt(
            args.authority_receipt, "quarantine_request", subject, st)
    except GateRefusal as exc:
        print(f"REFUSED: {exc}")
        print("The owner capability must be bound to this exact subject:")
        print(f"  subject        : {json.dumps(subject, sort_keys=True)}")
        print(f"  subject_sha256 : {sha_json(subject)}")
        return 1
    # Charge the experiment budget the way the infrastructure-failure path
    # does: the run happened, so it is spent -- but nothing scientific is
    # recorded (no strike, no improvement, no promotion, no calibration).
    # Quarantining must never become a way to run experiments off-budget.
    # Diagnostics and calibrations spend no attempt budget either way.
    kind = request.get("request_kind")
    budget = {"accounted": False, "group": None,
              "note": "this request kind spends no attempt budget"}
    if kind in ("scientific_attempt", "side_evaluation"):
        try:
            campaign = active_campaign(st, request.get("campaign_id"))
            group_key = None
            if kind == "scientific_attempt":
                group_key, grp, campaign = _family_group(st, request)
                grp["scientific_attempts"] = grp.get("scientific_attempts", 0) + 1
            campaign["scientific_attempts"] = campaign.get("scientific_attempts", 0) + 1
            budget = {"accounted": True, "group": group_key, "note": None}
        except GateRefusal as exc:
            # The chain outlived its campaign/family record. Settle it anyway
            # -- that is the whole point -- and record that the budget could
            # not be charged instead of pretending it was.
            budget = {"accounted": False, "group": None, "note": str(exc)}
    try:
        _settle_authority_request(st, chain, terminal)
    except GateRefusal as exc:
        print(f"REFUSED: {exc}")
        return 1
    consume_controller_receipt(st, verified)
    record = {
        "request_sha256": request_sha, "request_id": request["request_id"],
        "request_kind": kind, "mode": request.get("mode"),
        "campaign_id": request.get("campaign_id"), "shape": request.get("shape"),
        "direction_id": request.get("direction_id"),
        "permit_id": chain["permit"]["permit_id"],
        "controller_issue_event_id": chain["issued"]["event_id"],
        "terminal_event_id": terminal.get("event_id"),
        "terminal_event_sha256": terminal_sha,
        "reason": reason, "resolution": resolution,
        "quarantined_at": now(), "quarantined_epoch": time.time(),
        "budget_accounted": budget["accounted"],
        "budget_note": budget["note"], "group": budget["group"],
        "measurement_recorded": False, "champion_eligible": False,
        "performance_eligible": False, "promotion_eligible": False,
        "scientific_strike": False, "strike_satisfied": False,
        "authority_event_id": verified["authority_event_id"],
        "authority_event_sha256": verified["authority_event_sha256"],
        "subject_sha256": sha_json(subject),
    }
    st.setdefault("quarantined_requests", []).append(record)
    commit(st, {"ts": now(), "step": "authority_reconcile",
                "result": "unreconcilable", "quarantined": True, **record})
    try:
        _quarantine_settled_transport(st)
    except GateRefusal as exc:
        print(f"NOTE: the transport projection was left in place ({exc}); "
              "`reconcile` will retry it.")
    print(f"QUARANTINED {request_sha} as unreconcilable on controller-verified "
          "owner authority.")
    print(f"  reason recorded : {reason}")
    print("  No measurement, promotion, calibration or strike was recorded.")
    return 0


def cmd_audit_finalize(args) -> int:
    """Derive eligibility only from one measurement+packet+audit binding."""
    st = load_state_strict()
    pending = st.get("pending_audit_decisions", {}).get(args.entry_id)
    if not isinstance(pending, dict):
        print("REFUSED: no pending bound audit for that entry.")
        return 1
    try:
        chains = _authority_chains(st)
        chain = next((item for item in chains
                      if item["request_sha256"] == pending.get("request_sha256")), None)
        authority_entry = (chain["measurement"]["payload"].get("entry_id")
                           if chain and chain.get("measurement")
                           and pending.get("eligibility_scope") == "final_evidence_only"
                           else chain["measurement"]["payload"].get("run_id")
                           if chain and chain.get("measurement") else None)
        if (chain is None or chain.get("measurement") is None
                or chain.get("packet") is None
                or authority_entry != pending.get("entry_id")
                or chain["measurement"]["event_sha256"]
                    != pending.get("measurement_event_sha256")
                or chain["packet"]["sha256"] != pending.get("packet_sha256")
                or chain["binding"]["event_sha256"]
                    != pending.get("packet_binding_event_sha256")
                or chain["measurement"]["payload"].get("candidate_sha256")
                    != pending.get("candidate_sha256")):
            raise GateRefusal("pending audit is not bound to the live authority chain")
        decision = audit_decision_strict(
            pending["entry_id"], pending["candidate_sha256"],
            pending["packet_sha256"])
    except GateRefusal as exc:
        print(f"REFUSED: {exc}")
        return 1
    terminal_integrity = decision.integrity_status == "RULE_VIOLATION"
    if not decision.promotion_eligible and not terminal_integrity:
        print("REFUSED: bound audit keeps eligibility pending — "
              + ", ".join(decision.blocking_reasons))
        return 1
    campaign = active_campaign(st, pending["campaign_id"])
    if pending.get("eligibility_scope") == "final_evidence_only":
        eligible = bool(decision.promotion_eligible
                        and pending.get("controller_performance_eligible") is True
                        and pending.get("correct") is True
                        and pending.get("clean") is True)
        campaign["side_evaluations"].append({
            "request_id": pending["request_id"], "shape": pending["shape"],
            "entry_id": pending["entry_id"],
            "candidate_sha256": pending["candidate_sha256"],
            "measurement_event_sha256": pending["measurement_event_sha256"],
            "packet_sha256": pending["packet_sha256"],
            "side_evidence_sha256": pending.get("side_evidence_sha256"),
            "audit_event_sha256": decision.effective_event_sha256,
            "final_evidence_eligible": eligible,
        })
        del st["pending_audit_decisions"][args.entry_id]
        commit(st, {
            "ts": now(), "step": "audit_finalize",
            "campaign_id": pending["campaign_id"], "shape": pending["shape"],
            "entry_id": pending["entry_id"],
            "candidate_sha256": pending["candidate_sha256"],
            "measurement_event_sha256": pending["measurement_event_sha256"],
            "packet_sha256": pending["packet_sha256"],
            "side_evidence_sha256": pending.get("side_evidence_sha256"),
            "audit": decision.as_dict(), "champion_eligible": False,
            "final_evidence_eligible": eligible,
        })
        print(f"AUDIT FINALIZED {args.entry_id}: side evidence "
              f"{'eligible' if eligible else 'ineligible'}; never a primary champion.")
        return 0
    grp = st.get("groups", {}).get(pending["group"])
    if not isinstance(grp, dict):
        print("REFUSED: pending audit's family group is missing.")
        return 1
    prev_best = grp.get("best_speedup")
    champion_eligible = (decision.promotion_eligible
                  and pending.get("controller_performance_eligible") is True
                  and pending["correct"] and pending["clean"]
                  and isinstance(pending["speedup"], (int, float))
                  and math.isfinite(pending["speedup"]))
    if champion_eligible and (prev_best is None or pending["speedup"] > prev_best):
        grp["best_speedup"] = pending["speedup"]
    improved = bool(champion_eligible and
                    (prev_best is None
                     or pending["speedup"] > prev_best * IMPROVE_MARGIN))
    if improved:
        grp["strikes"] = 0
    else:
        grp["strikes"] = grp.get("strikes", 0) + 1
    campaign["production_outcomes"].append({
        "request_id": pending["request_id"],
        "family_id": pending["family_id"], "shape": pending["shape"],
        "entry_id": pending["entry_id"],
        "meaningful_improvement": improved, "speedup": pending["speedup"],
        "measurement_event_sha256": pending["measurement_event_sha256"],
        "packet_sha256": pending["packet_sha256"],
        "audit_event_sha256": decision.effective_event_sha256,
        "champion_eligible": champion_eligible,
    })
    event = {"ts": now(), "step": "audit_finalize",
             "campaign_id": pending["campaign_id"],
             "direction": pending["family_id"], "shape": pending["shape"],
             "entry_id": pending["entry_id"],
             "candidate_sha256": pending["candidate_sha256"],
             "measurement_event_sha256": pending["measurement_event_sha256"],
             "packet_sha256": pending["packet_sha256"],
             "audit": decision.as_dict(), "prev_best": prev_best,
             "champion_eligible": champion_eligible,
             "final_evidence_eligible": champion_eligible,
             "improved": improved, "strikes": grp["strikes"]}
    window = campaign["production_outcomes"][-campaign["stall_window"]:]
    if (len(window) == campaign["stall_window"]
            and not any(x["meaningful_improvement"] for x in window)):
        campaign["stalled"] = True
        campaign["stall_nonce"] = secrets.token_hex(16)
        campaign["stalled_epoch"] = time.time()
        campaign["stall_research_cycle"] = st.get("research_cycle", 0)
        event["campaign_stalled"] = True
        event["stall_nonce"] = campaign["stall_nonce"]
    _close_group_if_needed(st, pending["group"], grp, event)
    del st["pending_audit_decisions"][args.entry_id]
    commit(st, event)
    print(f"AUDIT FINALIZED {args.entry_id}: "
          f"{'champion-eligible improvement' if improved else 'not a qualifying improvement'}; "
          f"strikes {grp['strikes']}/{MAX_STRIKES}.")
    return 0


def cmd_screen_judge(args) -> int:
    """COMPUTE the screening hit/miss from the bounds stored at permit time
    against the reconciled row. The agent supplies --observed only as an
    attention check; it never declares the result (Track-2 lesson: the
    beneficiary of a judgment must not be its author)."""
    st = load_state_strict()  # locked + seq-checked from the first read
    pend = st.get("pending_screen_judgment")
    gkey = f"{args.direction}|{int(args.shape)}"
    if not pend or pend.get("group") != gkey:
        print("REFUSED: no pending screening judgment for this group.")
        return 1
    obs = pend.get("observed_speedup")
    if obs is None:
        # No recorded measurement -> NEVER judged from a caller-typed number
        # (that would be self-declared). It is an execution failure.
        st["pending_screen_judgment"] = None
        grp = st.setdefault("groups", {}).setdefault(
            gkey, {"best_speedup": None, "strikes": 0, "exec_failures": 0,
                   "closed": False, "closure_nonce": None})
        grp["exec_failures"] = grp.get("exec_failures", 0) + 1
        paused_now = False
        if grp["exec_failures"] >= MAX_EXEC_FAILURES:
            grp["infrastructure_paused"] = True
            paused_now = True
        commit(st, {"ts": now(), "step": "screen_judge", "group": gkey,
                    "result": "exec_failure", "computed": True,
                    "exec_failures": grp["exec_failures"],
                    "infrastructure_paused": paused_now})
        print(f"screening attempt has NO recorded speedup — logged as an "
              f"execution failure for {gkey} (no strike, no hit; "
              f"{grp['exec_failures']}/{MAX_EXEC_FAILURES})."
              + (" INFRASTRUCTURE PAUSED; scientific strikes unchanged."
                 if paused_now else ""))
        return 0
    try:
        stated = float(args.observed)
    except ValueError:
        print("REFUSED: --observed must be the numeric speedup from the row.")
        return 1
    if abs(stated - obs) > max(0.01 * abs(obs), 1e-6):
        print(f"REFUSED: --observed {stated} does not match the reconciled "
              f"row's speedup {obs}.")
        return 1
    pmin, pmax = pend.get("predict_min"), pend.get("predict_max")
    if pmin is None or pmax is None:
        print("REFUSED: this pending screening attempt carries no structured "
              "bounds (predates the computed-judgment gate) — resolve it with "
              "the owner; results are never self-declared.")
        return 1
    basis = obs
    result = ("hit" if (pend.get("row_correct", False)
                        and float(pmin) <= basis <= float(pmax)) else "miss")
    st["pending_screen_judgment"] = None
    grp = st.setdefault("groups", {}).setdefault(
        gkey, {"best_speedup": None, "strikes": 0, "exec_failures": 0,
               "closed": False, "closure_nonce": None})
    if not pend.get("row_correct", False):
        print("NOTE: the row failed correctness — MISS regardless of range.")
    characterization = pend.get("prediction_kind") == "characterization"
    if result == "miss" and not characterization:
        grp["strikes"] += 1
        if grp["strikes"] >= MAX_STRIKES:
            grp["closed"] = True
            grp["closure_nonce"] = secrets.token_hex(8)
            st.setdefault("pending_postmortem", []).append(gkey)
    commit(st, {"ts": now(), "step": "screen_judge", "group": gkey,
         "result": result, "computed": True, "observed": basis,
         "predict_min": pmin, "predict_max": pmax,
         "prediction_kind": pend.get("prediction_kind", "win"),
         "scientific_strike": result == "miss" and not characterization,
         "strikes": grp["strikes"], "closed": grp["closed"]})
    print(f"screening JUDGED {result} for {gkey} — computed from bounds "
          f"[{pmin}, {pmax}] vs observed {basis} "
          f"(strikes {grp['strikes']}/{MAX_STRIKES}).")
    return 0


def _ts_compact(ts):
    """Normalize any of our timestamp spellings (ISO with offset, entry-id
    prefix YYYYMMDD-HHMMSS) to a comparable 14-digit YYYYMMDDHHMMSS string."""
    return re.sub(r"[^0-9]", "", str(ts))[:14]


def _row_ts(row):
    t = (row or {}).get("timestamp")
    if t:
        return _ts_compact(t)
    return _ts_compact(str((row or {}).get("entry_id", ""))[:15])


def _retest_satisfying_row(orig, recorded):
    """entry_id of a journal row that would mechanically clear a RETEST on
    orig (same bytes+shape, passed, newer than both the original row and
    the verdict, with a reconciled confirmation-mode witness in the gate
    log) — else None."""
    if not orig:
        return None
    witnesses = set()
    try:
        for line in LOG.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("step") == "reconcile" and e.get("mode") == "confirmation":
                witnesses.add(e.get("entry_id"))
    except Exception:
        return None
    try:
        for line in DEFAULT_JOURNAL.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("entry_id") != orig.get("entry_id")
                    and r.get("entry_id") in witnesses
                    and r.get("impl", {}).get("sha256")
                    == orig.get("impl", {}).get("sha256")
                    and r.get("shape") == orig.get("shape")
                    and bool(r.get("correctness", {}).get("passed"))
                    and _row_ts(r) > _row_ts(orig)
                    and _row_ts(r) > _ts_compact(recorded)):
                return r.get("entry_id")
    except Exception:
        return None
    return None


def _journal_row(entry_id):
    try:
        for line in DEFAULT_JOURNAL.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("entry_id") == entry_id:
                return r
    except Exception:
        return None
    return None


def cmd_verdict_clear(args) -> int:
    """Lift the brake for ONE hard verdict. NO AI-owned free-text unlock
    exists (Track-2 lesson; authority-design.md rule 4):
      --kind retest    -> MECHANICAL: point at a newer journal row that
                          reruns the SAME bytes + shape and passed. The gate
                          verifies it; no prose involved.
      --kind violation -> OWNER-ONLY: requires a controller-verified signed
                          capability bound to this verdict and resolution.
                          Workspace quotes have no authority."""
    st = load_state_strict()
    key = f"{args.entry_id}|{args.recorded}"
    outstanding = {h.get("_clear_key"): h for h in unacked_hard_verdicts(st)}
    if key not in outstanding:
        print("REFUSED: that (entry-id, recorded) pair is not an outstanding "
              "hard verdict (already cleared, pre-gate, or nonexistent).")
        return 1
    kind = outstanding[key].get("verdict")
    if args.kind == "retest":
        if kind != "RETEST":
            print(f"REFUSED: verdict {key} is {kind}, not RETEST.")
            return 1
        if not args.confirm_entry:
            print("REFUSED: --confirm-entry <entry_id of the confirmation "
                  "row> is required for a retest clear.")
            return 1
        orig = _journal_row(args.entry_id)
        conf = _journal_row(args.confirm_entry)
        if orig is None or conf is None:
            print("REFUSED: original or confirmation row not found in the "
                  "primary journal. A retest of an entry outside the journal "
                  "is an owner decision (--kind violation path).")
            return 1
        if conf.get("entry_id") == orig.get("entry_id"):
            print("REFUSED: the confirmation must be a NEW row, not the "
                  "retested row itself.")
            return 1
        same_bytes = (conf.get("impl", {}).get("sha256")
                      and conf.get("impl", {}).get("sha256")
                      == orig.get("impl", {}).get("sha256"))
        same_shape = conf.get("shape") == orig.get("shape")
        passed = bool(conf.get("correctness", {}).get("passed"))
        newer = _row_ts(conf) > _row_ts(orig)
        # The confirmation must postdate the VERDICT, not just the original
        # row — a pre-existing rerun cannot satisfy a reviewer's request.
        after_verdict = _row_ts(conf) > _ts_compact(args.recorded)
        if not (same_bytes and same_shape and passed and newer and after_verdict):
            print(f"REFUSED: confirmation row fails the mechanical check "
                  f"(same_bytes={bool(same_bytes)} same_shape={same_shape} "
                  f"passed={passed} newer={newer} "
                  f"after_verdict={after_verdict}).")
            return 1
        # The confirmation must have been produced under a reconciled
        # confirmation-mode PERMIT — a matching row alone (however it got
        # into the journal) is not enough. The gate's own transition log is
        # the witness.
        recon_ok = False
        try:
            for line in LOG.read_text().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                if (e.get("step") == "reconcile"
                        and e.get("mode") == "confirmation"
                        and e.get("entry_id") == args.confirm_entry):
                    recon_ok = True
                    break
        except Exception:
            recon_ok = False
        if not recon_ok:
            print("REFUSED: no gate record of a reconciled confirmation-mode "
                  "permit producing that row — retest evidence must come "
                  "from a permitted, reconciled confirmation run.")
            return 1
        st.setdefault("cleared_verdicts", []).append(key)
        commit(st, {"ts": now(), "step": "verdict_clear", "kind": "retest",
                    "verdict_key": key, "confirm_entry": args.confirm_entry})
        print(f"RETEST {key} satisfied by {args.confirm_entry} "
              "(verified mechanically).")
        return 0
    if args.kind == "violation":
        if kind == "RETEST":
            attempts = st.get("retest_confirm_attempts", {}).get(key, 0)
            if _journal_row(args.entry_id) is not None and attempts < 3:
                print("REFUSED: this RETEST has a mechanical path (--kind "
                      "retest). Controller resolution applies to "
                      "RULE_VIOLATIONs, and to RETESTs only when the "
                      "original row is outside the primary journal or the "
                      "3-attempt confirmation budget is exhausted.")
                return 1
        if kind not in ("RULE_VIOLATION", "RETEST"):
            print(f"REFUSED: verdict {key} is {kind}.")
            return 1
        resolution = (args.resolution or "").strip()
        if len(resolution) < 30:
            print("REFUSED: --resolution must explain the exact disposition (>=30 chars).")
            return 1
        subject = {"verdict_key": key, "entry_id": args.entry_id,
                   "recorded": args.recorded, "verdict": kind,
                   "resolution": resolution}
        try:
            verified = verify_controller_receipt(
                args.authority_receipt, "resolve_integrity_verdict", subject, st)
        except GateRefusal as exc:
            print(f"REFUSED: {exc}")
            return 1
        consume_controller_receipt(st, verified)
        st.setdefault("cleared_verdicts", []).append(key)
        commit(st, {"ts": now(), "step": "verdict_clear", "kind": "violation",
                    "verdict_key": key, "resolution": resolution,
                    "authority_event_id": verified["authority_event_id"],
                    "authority_event_sha256": verified["authority_event_sha256"],
                    "subject_sha256": sha_json(subject)})
        print(f"RULE_VIOLATION {key} resolved by controller-verified owner authority.")
        return 0
    print("REFUSED: --kind must be retest or violation.")
    return 1


def cmd_reopen(args) -> int:
    st = load_state_strict()
    grp = st.get("groups", {}).get(args.group)
    if not grp or not grp.get("closed"):
        print("Nothing closed under that group key.")
        return 1
    nonce = grp.get("closure_nonce")
    resolution = (args.resolution or "").strip()
    if not nonce or len(resolution) < 50:
        print("REFUSED: closure nonce and a concrete >=50-char resolution are required.")
        return 1
    subject = {"group": args.group, "closure_nonce": nonce,
               "resolution": resolution}
    try:
        verified = verify_controller_receipt(
            args.authority_receipt, "reopen_family", subject, st)
    except GateRefusal as exc:
        print(f"REFUSED: {exc}")
        return 1
    consume_controller_receipt(st, verified)
    grp["closed"] = False
    grp["strikes"] = 0
    grp["exec_failures"] = 0
    grp["closure_nonce"] = None
    commit(st, {"ts": now(), "step": "reopen", "group": args.group,
         "consumed_nonce": nonce, "resolution": resolution,
         "authority_event_id": verified["authority_event_id"],
         "authority_event_sha256": verified["authority_event_sha256"],
         "subject_sha256": sha_json(subject)})
    print(f"Reopened {args.group} on controller-verified authority.")
    return 0


def cmd_init(_args) -> int:
    if STATE.exists():
        print("REFUSED: state already exists — init is first-time only.")
        return 1
    if LOG.exists() and LOG.stat().st_size > 0:
        print("REFUSED: gate history exists (gate_log.jsonl) — a missing "
              "state file must be brought back from version control, "
              "never re-initialized.")
        return 1
    if USED.exists() and any(USED.iterdir()):
        print("REFUSED: consumed permits exist — bring state back from "
              "version control.")
        return 1
    st = {"research_cycle": 0, "research_open": False, "groups": {},
          "family_registry": {}, "family_admissions": {},
          "campaigns": {}, "active_campaign": None, "profiles": {},
          "pending_postmortem": [], "consumed_nonces": [],
          "consumed_capability_nonces": [], "request_shas": [],
          "reconciled_authority_event_shas": [], "settled_request_shas": [],
          "pending_audit_decisions": {},
          "pending_screen_judgment": None, "seq": 0,
          "quarantined_requests": [],
          "created": now(), "cleared_verdicts": []}
    commit(st, {"ts": now(), "step": "init"})
    print("Gate state initialized CLOSED (event logged).")
    return 0


def cmd_status(_args) -> int:
    st = load_json(STATE, {})
    if isinstance(st, dict):
        _ensure_state_schema(st)
    st["_index_hash_now"] = index_hash()
    st["_catalog_sha256"] = sha_file(CATALOG)
    st["_permit_armed"] = PERMIT.exists()
    st["_permit_file_authoritative"] = False
    st["_legacy_in_flight_projection_present"] = INFLIGHT.exists()
    print(json.dumps(st, indent=1, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run gate v5 (competence request policy)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    co = sub.add_parser("campaign-open")
    co.add_argument("--spec", required=True)
    co.add_argument("--authority-receipt", required=True)
    fr = sub.add_parser("family-register")
    fr.add_argument("--campaign", required=True)
    fr.add_argument("--family-spec", required=True)
    fr.add_argument("--authority-receipt", required=True)
    fr.add_argument("--novelty-basis", default=None)
    r = sub.add_parser("research")
    r.add_argument("--campaign", required=True)
    r.add_argument("--index-hash", required=True)
    r.add_argument("--notes", required=True)
    r.add_argument("--summary", required=True)
    r.add_argument("--postmortem", default=None)
    p = sub.add_parser("plan")
    for a, req in (("--campaign", True), ("--direction", True),
                   ("--mode", True), ("--shape", True),
                   ("--impl", False), ("--ledger", False),
                   ("--hypothesis", True), ("--prediction", True),
                   ("--prediction-kind", True), ("--target-sha256", True),
                   ("--bottleneck", True), ("--counter-evidence", True),
                   ("--falsifier", True), ("--falsifier-kill", True),
                   ("--prior-family-verdict", True),
                   ("--kill", True), ("--sources", True), ("--reasoning", True)):
        p.add_argument(a, required=req, default=None)
    p.add_argument("--predict-min", default=None)
    p.add_argument("--predict-max", default=None)
    p.add_argument("--stall-receipt", default=None)
    d = sub.add_parser("delta")
    for a, req in (("--campaign", True), ("--direction", True),
                   ("--mode", True), ("--shape", True),
                   ("--impl", False), ("--ledger", False),
                   ("--changed", True), ("--prediction", True),
                   ("--prediction-kind", True), ("--target-sha256", True),
                   ("--counter-evidence", True),
                   ("--prior-family-verdict", True)):
        d.add_argument(a, required=req, default=None)
    d.add_argument("--predict-min", default=None)
    d.add_argument("--predict-max", default=None)
    diag = sub.add_parser("diagnostic")
    diag.add_argument("--campaign", required=True)
    diag.add_argument("--shape", required=True)
    diag.add_argument("--target-sha256", required=True)
    diag.add_argument("--tool", required=True)
    diag.add_argument("--supports", required=True)
    diag.add_argument("--question", required=True)
    diag.add_argument("--route", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--campaign", required=True)
    cal.add_argument("--shape", required=True)
    cal.add_argument("--machine-state", required=True)
    side = sub.add_parser("side-evaluate")
    side.add_argument("--campaign", required=True)
    side.add_argument("--shape", required=True, choices=("6", "14"))
    side.add_argument("--submission", required=True)
    side.add_argument("--submission-sha256", default=None)
    sub.add_parser("reconcile")
    af = sub.add_parser("audit-finalize")
    af.add_argument("--entry-id", required=True)
    sj = sub.add_parser("screen-judge")
    sj.add_argument("--direction", required=True)
    sj.add_argument("--shape", required=True)
    sj.add_argument("--observed", required=True)
    va = sub.add_parser("verdict-clear")
    va.add_argument("--entry-id", required=True)
    va.add_argument("--recorded", required=True)
    va.add_argument("--kind", required=True, choices=("retest", "violation"))
    va.add_argument("--confirm-entry", default=None)
    va.add_argument("--resolution", default=None)
    va.add_argument("--authority-receipt", default=None)
    ro = sub.add_parser("reopen")
    ro.add_argument("--group", required=True)
    ro.add_argument("--resolution", required=True)
    ro.add_argument("--authority-receipt", required=True)
    qa = sub.add_parser("quarantine")
    qa.add_argument("--request-sha256", required=True)
    qa.add_argument("--authority-receipt", required=True)
    qa.add_argument("--resolution", default=None)
    sub.add_parser("status")
    sub.add_parser("init")
    args = ap.parse_args()
    return {"campaign-open": cmd_campaign_open,
            "family-register": cmd_family_register,
            "research": cmd_research, "plan": cmd_plan, "delta": cmd_delta,
            "diagnostic": cmd_diagnostic, "calibrate": cmd_calibrate,
            "side-evaluate": cmd_side_evaluate,
            "reconcile": cmd_reconcile, "audit-finalize": cmd_audit_finalize,
            "screen-judge": cmd_screen_judge,
            "verdict-clear": cmd_verdict_clear,
            "reopen": cmd_reopen, "quarantine": cmd_quarantine,
            "status": cmd_status,
            "init": cmd_init}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
