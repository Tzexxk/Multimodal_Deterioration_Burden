"""Step 05: publish a recoverable, evidence-linked V2 audit, never a frozen cohort."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow.parquet as pq

from src.cohort.build_candidates import (
    CANDIDATE_COLUMNS,
    OBSERVATION_COLUMNS,
    active_intervals,
    build_candidate,
    deduplicate,
    observation,
    prepare_intervals,
)
from src.cohort.burden_events import (
    EVENT_COLUMNS,
    EVIDENCE_COLUMNS,
    aggregate_medications,
    classify_uncertainty,
    evaluate,
    infusion_evidence,
    ventilation_episodes,
)
from src.cohort.burden_io import (
    COMPONENTS,
    SUPPLEMENTARY_COLUMNS,
    TABLE_COLUMNS,
    ParquetSink,
    clinical_time,
    confirmed,
    frequencies,
    identifier,
    ingest,
    open_store,
    record_key,
    rows,
)
from src.cohort.common import (
    _config_hash,
    _csv_reader,
    _file_hash,
    _file_version,
    _git_commit,
    _json_file,
    _OutputRun,
    _progress,
    _protect_outputs,
    _resolve_config_path,
    _under_root,
    _verify_run,
    _write_csv,
)

WIDE_COLUMNS = list(dict.fromkeys(OBSERVATION_COLUMNS + EVENT_COLUMNS))
CONFLICT_POLICY = "non_icu_transfer_with_active_icustay"
IDENTITY_POLICY = "candidate_medication_identity_policy"
REVIEW_PATHS = {
    "cohort_decisions_path": "cohort_decisions_sha256",
    "event_decisions_path": "event_decisions_sha256",
}
SUPPLEMENTARY_REASONS = {
    "Medication identity cannot be uniquely confirmed",
    "Pre-admission medication record with unresolved identity",
    "Invasive ventilation evidence or onset cannot be verified",
    "Conflicting care-unit sources at t0",
}
IDENTITY_EXCLUSION_COLUMNS = (
    EVIDENCE_COLUMNS
    + (
        "emar_id original_medication resolved_medication medication_name_source "
        "pharmacy_lookup_status pharmacy_match_count prescriptions_lookup_status "
        "prescriptions_match_count identity_reason reviewer policy_version "
        "candidate_exclusion_reason automated_candidate_status"
    ).split()
)
DRUGS = {
    "norepinephrine",
    "epinephrine",
    "vasopressin",
    "phenylephrine",
    "dopamine",
    "dobutamine",
}
MAPPING_ACTIONS = {
    "transfer_event_actions": ("action", {"location", "exclude"}),
    "careunits": (
        "care_level",
        {"ward", "stepdown_intermediate", "icu_ccu", "other", "unknown"},
    ),
    "administration_types": ("action", {"infusion", "exclude", "review"}),
    "medication_event_actions": (
        "action",
        {"start", "restart", "continue", "stop", "given", "exclude", "review"},
    ),
    "procedure_status_actions": ("action", {"include", "exclude"}),
    "iv_routes": ("action", {"iv", "non_iv", "missing"}),
}


def validate_rules(cohort: dict, ehr: dict, imaging: dict) -> dict:
    rules = cohort["burden_v2"]
    if cohort["sample_unit"] != "study" or rules["outcome_version"] != "burden_v2":
        raise ValueError("Step 05 requires study-level burden_v2")
    if (
        rules["audit_horizons_hours"] != [48, 72]
        or rules["minimum_age"] != 18
        or (rules["spo2_threshold"] != 85)
    ):
        raise ValueError("V2 audit requires H=[48,72], minimum_age=18 and SpO2<=85")
    if not rules["rules_version"] or not rules["dictionary_version"]:
        raise ValueError("Rule and dictionary versions must be nonempty")
    if not ehr["burden_v2"]["memory_limit"] or set(ehr["burden_v2"]["tables"]) != set(
        TABLE_COLUMNS
    ):
        raise ValueError("Incomplete V2 source configuration")
    if (
        set(imaging["raw_audit"]["frontal_views"]) != {"AP", "PA"}
        or not (imaging["raw_audit"]["decode_pixels"])
    ):
        raise ValueError("V2 requires decoded AP/PA selection")
    expected = {
        "225792": ("Invasive Ventilation", "procedureevents"),
        "225794": ("Non-invasive Ventilation", "procedureevents"),
        "220277": ("O2 saturation pulseoxymetry", "chartevents"),
    }
    if set(rules["items"]) != set(expected):
        raise ValueError("Unexpected V2 primary item dictionary")
    for key, (label, table) in expected.items():
        if (rules["items"][key]["label"], rules["items"][key]["linksto"]) != (
            label,
            table,
        ):
            raise ValueError("Primary item meaning differs from V2")
    for family in (*MAPPING_ACTIONS, "medications", "spo2_units", "items"):
        if not isinstance(rules[family], dict):
            raise ValueError(f"Expected a mapping for {family}")
        for name, value in rules[family].items():
            if not isinstance(value, dict) or type(value.get("confirmed")) is not bool:
                raise ValueError(f"Invalid mapping confirmation: {family}/{name}")
            if value["confirmed"] and (
                not value.get("reviewer") or not value.get("mapping_reason")
            ):
                raise ValueError(
                    f"Confirmed mapping needs reviewer and mapping_reason: {family}/{name}"
                )
            if family in MAPPING_ACTIONS:
                field, allowed = MAPPING_ACTIONS[family]
                if value.get(field) not in allowed:
                    raise ValueError(f"Invalid mapping action: {family}/{name}")
            if family == "careunits" and (
                type(value.get("exclude_at_t0", False)) is not bool
                or (value.get("exclude_at_t0", False) and not value["confirmed"])
            ):
                raise ValueError(
                    "Care-unit exclude_at_t0 must be boolean and confirmed"
                )
            if family == "iv_routes" and (
                (name == "") != (value["action"] == "missing")
            ):
                raise ValueError("Only the blank route can have action=missing")
            if family == "medications" and (
                name != name.strip().casefold()
                or (
                    value.get("canonical_drug") is not None
                    and value["canonical_drug"] not in DRUGS
                )
            ):
                raise ValueError(
                    "Medication keys must be normalized exact names with a V2 canonical drug or null"
                )
    handling_reviews = rules.get("dictionary_handling_reviews", {})
    if not isinstance(handling_reviews, dict) or set(handling_reviews) - {
        "careunits",
        "medications",
    }:
        raise ValueError("Unsupported dictionary handling review family")
    for family, reviews in handling_reviews.items():
        if not isinstance(reviews, dict):
            raise ValueError("Dictionary handling reviews must be exact-name mappings")
        for name, review in reviews.items():
            if (
                not isinstance(review, dict)
                or type(review.get("handling_confirmed")) is not bool
                or review.get("handling") != "preserve_unknown"
                or family == "medications"
                and name != name.strip().casefold()
            ):
                raise ValueError("Invalid dictionary handling review")
            mapping = rules[family].get(name, {})
            if (
                mapping.get("confirmed")
                or (
                    family == "careunits"
                    and mapping.get("care_level", "unknown") != "unknown"
                )
                or (
                    family == "medications"
                    and mapping.get("canonical_drug") is not None
                )
            ):
                raise ValueError(
                    "Preserve-unknown handling cannot replace a classification"
                )
            if review["handling_confirmed"]:
                if any(
                    not isinstance(review.get(field), str) or not review[field].strip()
                    for field in ("reviewer", "reviewed_on", "rationale", "version")
                ):
                    raise ValueError(
                        "Confirmed handling review requires review provenance"
                    )
                datetime.strptime(review["reviewed_on"], "%Y-%m-%d")
    policies = rules.get("candidate_conflict_policies", {})
    if not isinstance(policies, dict) or set(policies) - {CONFLICT_POLICY}:
        raise ValueError("Unsupported candidate conflict policy")
    identity_policy = rules.get(IDENTITY_POLICY)
    if identity_policy is not None and not isinstance(identity_policy, dict):
        raise ValueError("Invalid candidate medication identity policy")
    for policy in [
        *policies.values(),
        *([identity_policy] if identity_policy is not None else []),
    ]:
        if (
            not isinstance(policy, dict)
            or type(policy.get("confirmed")) is not bool
            or policy.get("action") != "exclude"
        ):
            raise ValueError(
                "Candidate conflict policies require a confirmed flag and action=exclude"
            )
        if policy["confirmed"] and any(
            not isinstance(policy.get(field), str) or not policy[field].strip()
            for field in ("reviewer", "reason", "rationale", "version")
        ):
            raise ValueError(
                "Confirmed candidate conflict policy requires review provenance"
            )
    for source in ("pharmacy", "prescriptions"):
        lookup = rules.get(f"{source}_identity_lookup")
        if lookup is not None:
            if (
                not isinstance(lookup, dict)
                or type(lookup.get("confirmed")) is not bool
            ):
                raise ValueError(f"Invalid {source} identity lookup confirmation")
            if lookup["confirmed"] and any(
                not isinstance(lookup.get(field), str) or not lookup[field].strip()
                for field in ("reviewer", "mapping_reason", "version")
            ):
                raise ValueError(
                    f"Confirmed {source} identity lookup requires review provenance"
                )
    return rules


def save_audit_snapshot(outputs: Path, destination: Path | None = None) -> Path:
    """Preserve the original manifest bytes; snapshots never bind a later audit."""
    _verify_run(outputs, 5)
    manifest = outputs / "audit/step_05_manifest.json"
    record = _json_file(manifest)
    target_outputs = destination or outputs
    base = target_outputs / "audit/burden_v2/review_snapshots" / record["run_id"]
    index = base / "snapshot.json"
    if index.exists():
        value = _json_file(index)
        if value["source_manifest_sha256"] != _file_hash(manifest):
            raise ValueError("Immutable review snapshot manifest changed")
        verify_snapshot(index)
        return index
    sources = {**record["files"], str(manifest.resolve()): _file_hash(manifest)}
    with _OutputRun(target_outputs, 5, publish_manifest=False) as run:
        files = {}
        for number, (name, digest) in enumerate(sorted(sources.items())):
            source = Path(name)
            if _file_hash(source) != digest:
                raise ValueError("Review snapshot source changed")
            target = base / f"{number:03d}_{source.name}"
            shutil.copy2(source, run.path(target))
            if _file_hash(run.files[target.resolve()]) != digest:
                raise ValueError("Review snapshot copy changed")
            files[name] = {"path": str(target.resolve()), "sha256": digest}
        run.path(index).write_text(
            json.dumps(
                {
                    "run_id": record["run_id"],
                    "source_manifest_sha256": _file_hash(manifest),
                    "files": files,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _verify_run(outputs, 5)
    return index


def verify_snapshot(index: Path) -> dict:
    snapshot = _json_file(index)
    if not re.fullmatch(r"[a-f0-9]{32}", snapshot.get("run_id", "")):
        raise ValueError("Invalid review snapshot run ID")
    if not snapshot.get("files"):
        raise ValueError("Empty review snapshot")
    manifest = None
    for original, value in snapshot["files"].items():
        path = Path(value["path"])
        if not path.resolve().is_relative_to(index.parent.resolve()):
            raise ValueError("Review snapshot path escaped its directory")
        if _file_hash(path) != value["sha256"]:
            raise ValueError("Review snapshot evidence changed")
        if Path(original).name == "step_05_manifest.json":
            manifest = _json_file(path)
            if value["sha256"] != snapshot["source_manifest_sha256"]:
                raise ValueError("Review snapshot manifest checksum mismatch")
    if not manifest or manifest["run_id"] != snapshot["run_id"]:
        raise ValueError("Review snapshot run mismatch")
    expected = {
        p: v["sha256"]
        for p, v in snapshot["files"].items()
        if Path(p).name != "step_05_manifest.json"
    }
    if manifest["files"] != expected:
        raise ValueError("Review snapshot does not cover its original manifest")
    return snapshot


def review_identity(row: dict, event: bool, include_version: bool = True) -> tuple:
    target = (row["sample_id"],)
    if event:
        scope = row.get("override_scope") or "component"
        target += (identifier(row["horizon_hours"]), row["component"], scope)
        if scope == "source_record":
            target += (tuple(sorted(json.loads(row["source_record_keys"]))),)
    else:
        target += (row["reason"],)
    return ((row["version"],) + target) if include_version else target


def load_review_decisions(
    path: Path | None,
    samples: set[str],
    event: bool,
    legacy: list[dict],
    check: dict | None = None,
) -> list[dict]:
    rows = []
    for old in legacy:
        rows.append(
            {
                **old,
                "decision": (
                    "exclude_source"
                    if event and old.get("override_scope") == "source_record"
                    else "set_event"
                    if event
                    else "exclude"
                ),
                "legacy_review": True,
            }
        )
    if path:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = _csv_reader(stream, check)
            required = {
                "sample_id",
                "source_record_keys",
                "reviewer",
                "rationale",
                "version",
                "decision",
                "reviewed_on",
                "source_run_id",
                "source_manifest_sha256",
                "supersedes",
            }
            required |= (
                {
                    "horizon_hours",
                    "component",
                    "event_flag",
                    "first_event_time",
                    "override_scope",
                }
                if event
                else {"reason"}
            )
            if (
                not reader.fieldnames
                or required - set(reader.fieldnames)
                or len(reader.fieldnames) != len(set(reader.fieldnames))
            ):
                raise ValueError("Incomplete difficult-sample decision CSV")
            for row in reader:
                allowed = (
                    {"pending", "set_event", "exclude_source"}
                    if event
                    else {"pending", "resolve_blocker", "exclude"}
                )
                if row["sample_id"] not in samples or row["decision"] not in allowed:
                    raise ValueError("Invalid difficult-sample decision target")
                keys = json.loads(row["source_record_keys"] or "[]")
                if (
                    not isinstance(keys, list)
                    or any(not isinstance(k, str) or not k.strip() for k in keys)
                    or len(keys) != len(set(keys))
                ):
                    raise ValueError("Invalid difficult-sample source key array")
                if event:
                    if (
                        row["component"] not in COMPONENTS
                        or identifier(row["horizon_hours"]) not in (48, 72)
                        or row["override_scope"] not in {"component", "source_record"}
                    ):
                        raise ValueError("Invalid difficult-sample event scope")
                    if row["decision"] == "pending" and (
                        row["event_flag"] or row["first_event_time"]
                    ):
                        raise ValueError(
                            "Pending event decisions must leave labels blank"
                        )
                    if row["decision"] == "set_event" and (
                        row["event_flag"] not in {"0", "1"}
                        or row["override_scope"] != "component"
                    ):
                        raise ValueError(
                            "set_event requires a binary component decision"
                        )
                    if row["decision"] == "set_event" and (
                        (row["event_flag"] == "0" and row["first_event_time"])
                        or (
                            row["event_flag"] == "1"
                            and not clinical_time(row["first_event_time"])
                        )
                    ):
                        raise ValueError(
                            "Event time must agree with the binary decision"
                        )
                    if row["decision"] == "exclude_source":
                        if (
                            row["event_flag"] not in {"", "0"}
                            or row["first_event_time"]
                            or row["override_scope"] != "source_record"
                        ):
                            raise ValueError("exclude_source is not a component label")
                        row["event_flag"] = "0"
                elif not row["reason"].strip():
                    raise ValueError("Cohort decision requires a specific blocker")
                if row["decision"] != "pending":
                    if not keys or any(
                        not row[k].strip()
                        for k in (
                            "reviewer",
                            "rationale",
                            "version",
                            "reviewed_on",
                            "source_run_id",
                            "source_manifest_sha256",
                        )
                    ):
                        raise ValueError(
                            "Final decisions require complete review provenance"
                        )
                    datetime.strptime(row["reviewed_on"], "%Y-%m-%d")
                    if not re.fullmatch(
                        r"[a-f0-9]{32}", row["source_run_id"]
                    ) or not re.fullmatch(
                        r"[a-f0-9]{64}", row["source_manifest_sha256"]
                    ):
                        raise ValueError("Invalid decision audit binding")
                elif row["supersedes"]:
                    raise ValueError(
                        "Pending decisions cannot supersede final decisions"
                    )
                rows.append(row)
    final = [r for r in rows if r["decision"] != "pending"]
    indexed = {}
    for row in final:
        key = review_identity(row, event)
        if key in indexed:
            raise ValueError("Duplicate versioned review decision")
        indexed[key] = row
    removed, edges = set(), {}
    for key, row in indexed.items():
        refs = json.loads(row.get("supersedes") or "[]")
        if not isinstance(refs, list):
            raise ValueError("supersedes must be a JSON array of decision references")
        edges[key] = []
        for ref in refs:
            if not isinstance(ref, dict):
                raise ValueError("Invalid supersedes reference")
            old = review_identity(ref, event)
            end = 4 if event else 3
            if old not in indexed or old in removed or old[1:end] != key[1:end]:
                raise ValueError("Missing, branching or unrelated supersedes reference")
            removed.add(old)
            edges[key].append(old)

    def visit(key, ancestors):
        if key in ancestors:
            raise ValueError("Cyclic decision supersession")
        for old in edges[key]:
            visit(old, ancestors | {key})

    for key in edges:
        visit(key, set())
    active = [r for k, r in indexed.items() if k not in removed]
    targets = set()
    for row in active:
        target = review_identity(row, event, False)
        if target in targets:
            raise ValueError("Conflicting active review decisions require supersedes")
        targets.add(target)
    if event:
        for row in active:
            peers = [
                r
                for r in active
                if r is not row
                and review_identity(r, True, False)[:3]
                == review_identity(row, True, False)[:3]
            ]
            if peers and (
                (row.get("override_scope") or "component") == "component"
                or any(
                    set(json.loads(row["source_record_keys"]))
                    & set(json.loads(r["source_record_keys"]))
                    for r in peers
                )
            ):
                raise ValueError("Overlapping active event decisions")
    return active


def raw_source_index(values: list[dict]) -> dict:
    result = {}

    def collect(value, owner=None):
        if isinstance(value, dict):
            owner = value.get("hadm_id") or owner
            key = value.get("source_record_key")
            if key:
                table = key.split(":", 1)[0]
                fields = TABLE_COLUMNS.get(table, "").split()
                if table == "ventilation_episode":
                    fields = [
                        "subject_id",
                        "hadm_id",
                        "source_record_keys",
                        "start",
                        "end",
                    ]
                if fields:
                    from src.cohort.burden_io import SUPPLEMENTARY_COLUMNS

                    fields += [
                        f
                        for f in SUPPLEMENTARY_COLUMNS.get(table, "").split()
                        if f in value
                    ]
                    raw = {
                        f: str(value.get(f)) if f in {"start", "end"} else value.get(f)
                        for f in fields
                    }
                    result[key] = {
                        "content": record_key(table, raw),
                        "subject_id": identifier(value.get("subject_id")),
                        "hadm_id": identifier(owner),
                    }
            for nested in value.values():
                if isinstance(nested, (list, dict)):
                    collect(nested, owner)
        elif isinstance(value, list):
            for nested in value:
                collect(nested, owner)

    collect(values)
    return result


def validate_review_binding(
    row: dict,
    candidate: dict,
    context: dict,
    outputs: Path,
    input_versions: dict,
    rules: dict,
) -> Path:
    index = (
        outputs
        / "audit/burden_v2/review_snapshots"
        / row["source_run_id"]
        / "snapshot.json"
    )
    snapshot = verify_snapshot(index)
    if snapshot["source_manifest_sha256"] != row["source_manifest_sha256"]:
        raise ValueError("Decision references a stale source manifest")
    saved, original = [], None
    for name, entry in snapshot["files"].items():
        basename = Path(name).name
        if basename == "run_metadata.json":
            from src.cohort import build_labels

            metadata = _json_file(Path(entry["path"]))
            if metadata["input_versions"] != input_versions:
                raise ValueError("Decision raw source versions changed; review again")
            dictionary = {
                name: rules[name]
                for name in (*MAPPING_ACTIONS, "medications", "spo2_units", "items")
            }
            if (
                metadata["dictionary_sha256"] != build_labels._json_hash(dictionary)
                or metadata["code_sha256"] != build_labels._audit_code_hash()
                or any(
                    metadata[name] != rules[name]
                    for name in ("rules_version", "dictionary_version")
                )
            ):
                raise ValueError(
                    "Decision rules, dictionary or audit code changed; review again"
                )
        elif basename == "candidate_studies.parquet":
            values = pq.read_table(
                entry["path"], filters=[("sample_id", "=", candidate["sample_id"])]
            ).to_pylist()
            if len(values) != 1:
                raise ValueError("Decision sample missing from source snapshot")
            original = values[0]
        elif basename in {
            "candidate_evidence.parquet",
            "event_evidence.parquet",
            "supplementary_event_evidence.parquet",
        }:
            values = pq.read_table(
                entry["path"], filters=[("sample_id", "=", candidate["sample_id"])]
            ).to_pylist()
            for value in values:
                raw = json.loads(value["raw_evidence"])
                if isinstance(raw, dict) and "source_record_key" not in raw:
                    raw["source_record_key"] = value["source_record_key"]
                saved.append(raw)
        elif basename == "medication_identity_audit.parquet":
            values = pq.read_table(
                entry["path"],
                filters=[
                    ("subject_id", "=", candidate["subject_id"]),
                    ("hadm_id", "=", candidate["hadm_id"]),
                ],
            ).to_pylist()
            saved += [json.loads(v["raw_evidence"]) for v in values]
    if original is None or any(
        original[f] != candidate[f]
        for f in ("subject_id", "hadm_id", "study_id", "t0", "admittime", "dischtime")
    ):
        raise ValueError("Decision patient, admission or window changed")
    if row["decision"] == "resolve_blocker" and row["reason"] not in original[
        "exclusion_reason"
    ].split(";"):
        raise ValueError("Decision does not resolve a blocker in its source audit")
    if row["decision"] == "set_event" and row["event_flag"] == "1":
        time = clinical_time(row["first_event_time"])
        t0, discharge = candidate["t0"], candidate["dischtime"]
        end = min(t0 + timedelta(hours=identifier(row["horizon_hours"])), discharge)
        if time is None or not t0 < time <= end:
            raise ValueError("Decision event time is outside its observation window")
    current = [
        *context["transfers"],
        *context["stays"],
        *context["medication_events"],
        *context["ventilation_episodes"],
        *context["ventilation_rejected"],
        *context["chartevents"],
        *context["procedureevents"],
        {
            **context["admission"],
            "source_record_key": record_key("admissions", context["admission"]),
        },
    ]
    for table in ("chartevents", "procedureevents"):
        current += [
            {**v, "source_record_key": record_key(table, v)} for v in context[table]
        ]
    old, new = raw_source_index(saved), raw_source_index(current)
    for key in json.loads(row["source_record_keys"]):
        if (
            key not in old
            or key not in new
            or old[key] != new[key]
            or (
                new[key]["subject_id"] != candidate["subject_id"]
                or new[key]["hadm_id"] != candidate["hadm_id"]
            )
        ):
            raise ValueError(
                "Decision evidence changed or belongs to another admission"
            )
    return index


def freeze_endpoint(
    root: Path,
    paths_file: Path,
    cohort_file: Path,
    ehr_file: Path,
    imaging_file: Path,
    review_file: Path,
) -> int:
    """Freeze a reviewed current audit without changing its source manifest."""
    from src.cohort import build_labels

    configs = [paths_file, cohort_file, ehr_file, imaging_file]
    paths, cohort, ehr, imaging = map(_json_file, configs)
    rules = validate_rules(cohort, ehr, imaging)
    processed = _resolve_config_path(paths["processed"], root, raw=False)
    outputs = _resolve_config_path(paths["outputs"], root, raw=False)
    _protect_outputs(paths, root, processed, outputs)
    review = _json_file(review_file)
    required = {
        "decision_version",
        "reviewer",
        "reviewed_on",
        "rationale",
        "rules_revision",
        "source_run_id",
        "source_manifest_sha256",
    }
    if (
        review.get("confirmed") is not True
        or any(
            not isinstance(review.get(k), str) or not review[k].strip()
            for k in required
        )
        or review.get("horizon_hours") != 72
        or type(review.get("horizon_hours")) is not int
    ):
        raise ValueError(
            "Endpoint freeze requires an explicit reviewed 72-hour decision"
        )
    datetime.strptime(review["reviewed_on"], "%Y-%m-%d")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", review["decision_version"]):
        raise ValueError("Unsafe endpoint decision version")
    refs = review.get("review_records")
    if (
        not isinstance(refs, list)
        or not refs
        or any(not isinstance(p, str) for p in refs)
    ):
        raise ValueError("Endpoint freeze requires reviewed record references")
    review_hashes = {
        review_file.resolve().relative_to(root).as_posix(): _file_hash(review_file)
    }
    for name in refs:
        path = _resolve_config_path(name, root, raw=False)
        review_hashes[path.relative_to(root).as_posix()] = _file_hash(path)
    for step in (3, 4, 5):
        _verify_run(outputs, step)
    manifest_path = outputs / "audit/step_05_manifest.json"
    audit_dir = outputs / "audit/burden_v2"
    meta_path = audit_dir / "run_metadata.json"
    meta = _json_file(meta_path)
    if (
        meta["run_id"] != review["source_run_id"]
        or _file_hash(manifest_path) != review["source_manifest_sha256"]
    ):
        raise ValueError("Endpoint review does not reference the latest Step 05 audit")
    if not all(meta.get(k) is True for k in build_labels.READY_FLAGS):
        raise ValueError(
            "Audit still has unresolved labels, candidates or dictionary handling"
        )
    extras = []
    for field, hash_field in (
        ("overrides_path", "override_sha256"),
        ("candidate_exclusions_path", "candidate_exclusions_sha256"),
        *REVIEW_PATHS.items(),
    ):
        path = (
            _resolve_config_path(rules[field], root, raw=False)
            if rules.get(field)
            else None
        )
        if (_file_hash(path) if path else None) != meta.get(hash_field):
            raise ValueError("Audit decision input changed before freeze")
        if path:
            extras.append(path)
    if (
        _config_hash(configs + extras) != meta["config_sha256"]
        or build_labels._audit_code_hash() != meta["code_sha256"]
    ):
        raise ValueError("Audit code or configuration changed before freeze")
    iv = _resolve_config_path(paths["mimic_iv"], root, raw=True)
    if any(
        _file_version(_under_root(iv, p)) != meta["input_versions"][n]
        for n, p in ehr["burden_v2"]["tables"].items()
    ):
        raise ValueError("Raw sources changed before endpoint freeze")
    for name, digest in meta.get("review_snapshot_bindings", {}).items():
        index = _under_root(root, name)
        if _file_hash(index) != digest:
            raise ValueError("Review snapshot binding changed before freeze")
        verify_snapshot(index)
    artifacts = {
        "candidates": processed / "cohort/burden_v2/candidate_studies.parquet",
        "candidate_evidence": processed / "cohort/burden_v2/candidate_evidence.parquet",
        "observations": processed
        / "cohort/burden_v2/window_72h/candidate_observation.parquet",
        "wide": processed / "cohort/burden_v2/window_72h/study_event_audit.parquet",
        "evidence": processed / "cohort/burden_v2/window_72h/event_evidence.parquet",
    }
    with tempfile.TemporaryDirectory(prefix=".freeze_05_", dir=outputs) as scratch:
        connection = open_store(Path(scratch), ehr["burden_v2"]["memory_limit"])
        try:
            counts = build_labels.validate_labels(connection, artifacts, 72)
        finally:
            connection.close()
    source = {
        "run_id": meta["run_id"],
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": _file_hash(manifest_path),
        "metadata_path": meta_path.relative_to(root).as_posix(),
        "metadata_sha256": _file_hash(meta_path),
        **{
            k: meta[k]
            for k in (
                "config_sha256",
                "code_sha256",
                "input_versions",
                "upstream",
                "csv_tail_checks",
                *build_labels.READY_FLAGS,
            )
        },
    }
    window_path = audit_dir / "window_decision.md"
    decision_path = audit_dir / "endpoint_decision.json"
    if decision_path.exists():
        existing = _json_file(decision_path)
        if (
            existing.get("review_records_sha256") == review_hashes
            and existing["source_audit"] == source
        ):
            build_labels.verify_frozen_inputs(root, configs)
            print("Endpoint already frozen under the same reviewed evidence")
            return 0
        raise ValueError(
            "An endpoint is already frozen here; use a new isolated revision"
        )
    text = (
        "# Reviewed supplementary endpoint\n\n"
        f"Reviewer: {review['reviewer']}\n\nReviewed on: {review['reviewed_on']}\n\n"
        f"Rationale: {review['rationale']}\n\nRule revision: {review['rules_revision']}\n\n"
        "H=72; window (t0, min(t0+72h, dischtime)]. All six events must be determined.\n"
        "SpO2 <=85%; T=max(ICU,Stepdown); R=max(IMV,hypoxemia); S=T+2R+2C+3D.\n"
        "Early 3-hour events remain included. Live discharge truncates observation.\n"
    )
    decision = {
        "decision_version": review["decision_version"],
        "decision_status": "frozen",
        **{
            k: review[k]
            for k in ("reviewer", "reviewed_on", "rationale", "rules_revision")
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outcome_version": "burden_v2",
        "sample_unit": "study",
        "horizon_hours": 72,
        "ehr_lookback_hours": 24,
        "event_window": "(t0, min(t0+H, dischtime)]",
        "transfer_weight": 1,
        "respiratory_weight": 2,
        "circulatory_weight": 2,
        "death_weight": 3,
        "domain_capping": True,
        "target_column": "burden_score",
        "target_range": [0, 8],
        "target_formula": "T + 2*R + 2*C + 3*D; T=max(icu,stepdown), R=max(imv,hypoxemia), C=infusion, D=death",
        "phys_death_rule": "int(R == 1 or C == 1 or D == 1)",
        "high_burden_threshold": 3,
        "hypoxemia_source": "MIMIC-IV ICU chartevents only",
        "hypoxemia_itemid": 220277,
        "hypoxemia_threshold_percent": 85,
        "early_3h_events_included": True,
        "live_discharge_rule": "Retain events in (t0, dischtime] when alive discharge precedes t0+H; absent recorded events are operational zeros.",
        "horizon_frozen": True,
        "label_rules_frozen": True,
        "formal_labels_published": False,
        "common_cohort_frozen": False,
        "source_audit": source,
        **{
            k: meta.get(k)
            for k in (
                "rules_version",
                "dictionary_version",
                "rules_config_sha256",
                "dictionary_sha256",
                "dictionary_handling_reviews_sha256",
                "override_sha256",
                "candidate_exclusions_sha256",
                *REVIEW_PATHS.values(),
            )
        },
        "review_snapshot_bindings": meta.get("review_snapshot_bindings", {}),
        "baseline_reference_bindings": meta.get("baseline_reference_bindings", {}),
        "review_records_sha256": review_hashes,
        "selection_record": {
            "path": review_file.relative_to(root).as_posix(),
            "sha256": _file_hash(review_file),
        },
        "window_decision_path": window_path.relative_to(root).as_posix(),
        "window_decision_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "selected_window_artifact_sha256": {
            p.relative_to(root).as_posix(): _file_hash(p) for p in artifacts.values()
        },
        "window_comparison": {"72": counts},
    }
    with _OutputRun(outputs, 5, processed, publish_manifest=False) as run:
        run.path(window_path).write_text(text, encoding="utf-8", newline="\n")
        run.path(decision_path).write_text(
            json.dumps(decision, indent=2) + "\n", encoding="utf-8"
        )
        build_labels.verify_frozen_inputs(
            root,
            configs,
            decision_file=run.files[decision_path.resolve()],
            window_file=run.files[window_path.resolve()],
        )
        _verify_run(outputs, 5)
        if any(_file_hash(_under_root(root, p)) != h for p, h in review_hashes.items()):
            raise ValueError("Endpoint review changed during publication")
    build_labels.verify_frozen_inputs(root, configs)
    print(f"Reviewed endpoint frozen: {decision_path}")
    return 0


def medication_identity_exclusion(
    candidate: dict, context: dict, rules: dict
) -> tuple[dict | None, list[dict]]:
    """Exclude eligible studies using the union of existing window identity reviews."""
    policy = rules.get(IDENTITY_POLICY, {})
    if not policy.get("confirmed") or candidate["candidate_status"] != "eligible":
        return None, []
    triggers = []
    for horizon in (48, 72):
        study = observation(candidate, horizon)
        records = infusion_evidence(study, context)
        classify_uncertainty(study, records, bool(rules.get("supplementary_audit")))
        triggers += [
            row
            for row in records
            if row["qualifies"] is None
            and row["blocks_event"]
            and row["exclusion_reason"] == "unconfirmed_medication_name"
        ]
    if not triggers:
        return None, []
    keys = sorted(
        {key for row in triggers for key in json.loads(row["source_record_keys"])}
    )
    return {
        **policy,
        "sample_id": candidate["sample_id"],
        "policy_name": IDENTITY_POLICY,
        "source_record_keys": json.dumps(keys),
    }, triggers


def load_upstream(
    processed: Path, outputs: Path, run: _OutputRun | None = None
) -> tuple[list[dict], dict, dict]:
    selection_path = processed / "index/study_image_selection.parquet"
    match_path = processed / "index/cxr_admission_matches.parquet"
    _verify_run(outputs, 3, selection_path)
    _verify_run(outputs, 4, match_path)
    selections = pq.read_table(selection_path).to_pylist()
    matches = pq.read_table(match_path).to_pylist()
    selected = {(row["subject_id"], row["study_id"]): row for row in selections}
    keys = [(row["subject_id"], row["study_id"]) for row in matches]
    if (
        len(selected) != len(selections)
        or len(set(keys)) != len(matches)
        or set(keys) != set(selected)
    ):
        raise ValueError("Step 03/04 study keys do not match uniquely")
    unique = []
    for row in matches:
        chosen = selected[row["subject_id"], row["study_id"]]
        if (
            row["dicom_id"] != chosen["selected_dicom_id"]
            or row["t0"] != chosen["acquisition_datetime"]
        ):
            raise ValueError("Match and selection disagree on image or t0")
        if row["match_status"] not in {"unique_match", "no_match", "multiple_match"}:
            raise ValueError("Unexpected match_status")
        if row["match_status"] == "unique_match":
            if any(
                type(row[field]) is not int
                for field in ("subject_id", "hadm_id", "study_id")
            ):
                raise ValueError("Invalid candidate identifiers")
            unique.append(row)
    provenance = {
        "selection_sha256": _file_hash(selection_path),
        "matches_sha256": _file_hash(match_path),
    }
    summary_path = outputs / "audit/admission_match_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as stream:
        summary = {
            row["metric"]: row["value"]
            for row in _csv_reader(stream, run.check_csv(summary_path) if run else None)
        }
    if summary["selection_input_sha256"] != provenance["selection_sha256"]:
        raise ValueError("Step 04 used a different selection table; rerun matching")
    provenance["step_04_admissions_sha256"] = summary["admissions_input_sha256"]
    return unique, selected, provenance


def load_overrides(
    path: Path | None, sample_ids: set[str], check: dict | None = None
) -> dict:
    if path is None:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = _csv_reader(stream, check)
        required = {
            "sample_id",
            "horizon_hours",
            "component",
            "event_flag",
            "first_event_time",
            "source_record_keys",
            "reviewer",
            "rationale",
            "version",
        }
        if not reader.fieldnames or required - set(reader.fieldnames):
            raise ValueError("Incomplete override CSV schema")
        result = {}
        for row in reader:
            scope = row.get("override_scope") or "component"
            if (
                scope not in {"component", "source_record"}
                or scope == "source_record"
                and (
                    identifier(row["event_flag"]) != 0
                    or row["first_event_time"]
                    or row.get("baseline_status")
                )
            ):
                raise ValueError("Invalid override scope or source-record exclusion")
            horizon = identifier(row["horizon_hours"])
            key = row["sample_id"], horizon, row["component"]
            if (
                row["sample_id"] not in sample_ids
                or horizon not in {48, 72}
                or (row["component"] not in COMPONENTS or key in result)
            ):
                raise ValueError("Unknown or duplicate override target")
            result[key] = row
        return result


def load_candidate_exclusions(
    path: Path | None, sample_ids: set[str], check: dict | None = None
) -> dict:
    """Load study-specific exclusions; these decisions cannot grant eligibility."""
    if path is None:
        return {}
    required = {
        "sample_id",
        "reason",
        "source_record_keys",
        "reviewer",
        "rationale",
        "version",
    }
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = _csv_reader(stream, check)
        if not reader.fieldnames or required - set(reader.fieldnames):
            raise ValueError("Incomplete candidate exclusion CSV schema")
        for row in reader:
            sample = row["sample_id"]
            if sample not in sample_ids or sample in result:
                raise ValueError("Unknown or duplicate candidate exclusion target")
            if any(not (row[name] or "").strip() for name in required):
                raise ValueError(
                    "Candidate exclusions require complete review provenance"
                )
            keys = json.loads(row["source_record_keys"])
            if (
                not isinstance(keys, list)
                or not keys
                or any(not isinstance(key, str) or not key.strip() for key in keys)
            ):
                raise ValueError(
                    "Candidate exclusions require a nonempty source key array"
                )
            result[sample] = row
    return result


def candidate_conflict_exclusion(
    candidate: dict, context: dict, rules: dict
) -> dict | None:
    """Apply only the reviewed non-ICU transfer versus active ICU-stay policy."""
    policy = confirmed(rules.get("candidate_conflict_policies", {}), CONFLICT_POLICY)
    if (
        not policy
        or candidate["candidate_status"] != "needs_review"
        or "transfers_icustays_conflict" not in candidate["exclusion_reason"].split(";")
        or candidate["care_level_at_t0"] not in {"ward", "stepdown_intermediate"}
        or not candidate["careunit_at_t0"]
    ):
        return None
    current = active_intervals(context["transfers"], candidate["t0"])
    stays = active_intervals(context["stays"], candidate["t0"])
    if not current or not stays:
        return None
    return {
        "sample_id": candidate["sample_id"],
        **{
            field: policy[field]
            for field in ("reason", "reviewer", "rationale", "version")
        },
        "source_record_keys": json.dumps(
            sorted({row["source_record_key"] for row in current + stays})
        ),
        "policy_name": CONFLICT_POLICY,
    }


def query_rows(connection, query: str, params: list | None = None) -> list[dict]:
    result = connection.execute(query, params or [])
    names = [column[0] for column in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


def write_query(connection, run: _OutputRun, path: Path, query: str) -> None:
    """Stream SQL results to CSV with headers even for an empty result."""
    result = connection.execute(query)
    with run.path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([column[0] for column in result.description])
        while batch := result.fetchmany(8192):
            writer.writerows(batch)


def dictionary_handling_fields(
    rules: dict, family: str, name: str, mapping: dict | None
) -> dict:
    """Handling approval describes uncertainty management, never clinical identity."""
    review = rules.get("dictionary_handling_reviews", {}).get(family, {}).get(name, {})
    approved = review.get("handling_confirmed") is True
    return {
        "handling_confirmed": bool(mapping) or approved,
        "handling_policy": "confirmed_classification"
        if mapping
        else review.get("handling", "unreviewed"),
        "handling_reviewer": mapping.get("reviewer", "")
        if mapping
        else review.get("reviewer", ""),
        "handling_reviewed_on": review.get("reviewed_on", ""),
        "handling_rationale": mapping.get("mapping_reason", "")
        if mapping
        else review.get("rationale", ""),
        "handling_version": rules["dictionary_version"]
        if mapping
        else review.get("version", ""),
    }


def dictionary_reports(
    connection, run: _OutputRun, directory: Path, rules: dict
) -> tuple[bool, bool]:
    observed = {
        "transfer_event_actions": [
            (row["value"], row["record_count"])
            for row in frequencies(connection, "transfers", ["eventtype"])
        ],
        "careunits": [
            (row["value"], row["record_count"])
            for row in frequencies(connection, "transfers", ["careunit"])
        ],
        "medications": [
            (row["value"], row["record_count"])
            for table, field in (
                ("emar", "medication"),
                ("pharmacy", "medication"),
                ("prescriptions", "drug"),
            )
            for row in frequencies(connection, table, [field])
        ],
        "administration_types": [
            (row["value"], row["record_count"])
            for row in frequencies(connection, "emar_detail", ["administration_type"])
        ],
        "medication_event_actions": [
            (row["value"], row["record_count"])
            for row in frequencies(connection, "emar", ["event_txt"])
        ],
        "procedure_status_actions": [
            (row["value"], row["record_count"])
            for row in frequencies(connection, "procedureevents", ["statusdescription"])
        ],
        "iv_routes": [
            (row["value"], row["record_count"])
            for row in frequencies(connection, "emar_detail", ["route"])
        ],
        "spo2_units": query_rows(
            connection,
            "SELECT coalesce(valueuom,'') AS \"value\", count(*) record_count "
            "FROM chartevents WHERE itemid='220277' GROUP BY valueuom",
        ),
    }
    observed["spo2_units"] = [
        (row["value"], row["record_count"]) for row in observed["spo2_units"]
    ]
    medication_counts = defaultdict(int)
    for name, count in observed["medications"]:
        medication_counts[name] += count
    observed["medications"] = sorted(medication_counts.items())
    inventory, careunits = [], []
    for family, values in observed.items():
        for raw, count in values:
            name = raw.strip().casefold() if family == "medications" else raw
            mapping = confirmed(rules[family], name)
            inventory.append(
                dict(
                    family=family,
                    value_raw=raw,
                    lookup_key=name,
                    record_count=count,
                    confirmed=bool(mapping),
                    mapping=json.dumps(rules[family].get(name, {})),
                    dictionary_version=rules["dictionary_version"],
                    **dictionary_handling_fields(rules, family, name, mapping),
                )
            )
            if family == "careunits":
                level = mapping["care_level"] if mapping else "unknown"
                careunits.append(
                    dict(
                        careunit_raw=raw,
                        care_level=level,
                        is_icu_ccu=int(level == "icu_ccu"),
                        is_stepdown_intermediate=int(level == "stepdown_intermediate"),
                        exclude_at_t0=int(
                            bool(mapping and mapping.get("exclude_at_t0", False))
                        ),
                        mapping_reason=mapping.get("mapping_reason", "")
                        if mapping
                        else "unconfirmed",
                        reviewer=mapping.get("reviewer", "") if mapping else "",
                        dictionary_version=rules["dictionary_version"],
                        record_count=count,
                    )
                )
    local_items = {row["itemid"]: row for row in rows(connection, "d_items")}
    item_reports = []
    for key, expected in rules["items"].items():
        local = local_items.get(key)
        if not local or (local["label"], local["linksto"]) != (
            expected["label"],
            expected["linksto"],
        ):
            raise ValueError(f"Local d_items differs from V2 item {key}")
        mapped = bool(confirmed(rules["items"], key))
        inventory.append(
            dict(
                family="items",
                value_raw=key,
                lookup_key=key,
                record_count=1,
                confirmed=mapped,
                mapping=json.dumps(expected),
                dictionary_version=rules["dictionary_version"],
                **dictionary_handling_fields(
                    rules, "items", key, confirmed(rules["items"], key)
                ),
            )
        )
        item_reports.append(
            {
                **local,
                "confirmed": mapped,
                "dictionary_version": rules["dictionary_version"],
            }
        )
    _write_csv(
        run.path(directory / "dictionary_inventory.csv"),
        list(inventory[0])
        if inventory
        else [
            "family",
            "value_raw",
            "lookup_key",
            "record_count",
            "confirmed",
            "mapping",
            "dictionary_version",
        ],
        inventory,
    )
    _write_csv(
        run.path(directory / "careunit_dictionary.csv"),
        [
            "careunit_raw",
            "care_level",
            "is_icu_ccu",
            "is_stepdown_intermediate",
            "exclude_at_t0",
            "mapping_reason",
            "reviewer",
            "dictionary_version",
            "record_count",
        ],
        careunits,
    )
    _write_csv(
        run.path(directory / "item_dictionary.csv"),
        ["itemid", "label", "linksto", "confirmed", "dictionary_version"],
        item_reports,
    )
    frequency_specs = {
        "medication_type_counts.csv": [
            ("emar", ["medication", "event_txt"]),
            ("pharmacy", ["medication"]),
            ("prescriptions", ["drug", "drug_type"]),
            (
                "emar_detail",
                [
                    "administration_type",
                    "complete_dose_not_given",
                    "route",
                    "infusion_complete",
                    "new_iv_bag_hung",
                    "continued_infusion_in_other_location",
                ],
            ),
        ],
        "transfer_value_counts.csv": [("transfers", ["careunit", "eventtype"])],
        "ventilation_item_status_counts.csv": [
            ("procedureevents", ["itemid", "statusdescription"])
        ],
        "spo2_unit_warning_counts.csv": [("chartevents", ["valueuom", "warning"])],
    }
    for filename, specs in frequency_specs.items():
        values = [
            row
            for table, fields in specs
            for row in frequencies(connection, table, fields)
        ]
        _write_csv(
            run.path(directory / filename),
            ["source_table", "field", "value", "record_count"],
            values,
        )
    return (
        all(row["confirmed"] for row in inventory),
        all(row["handling_confirmed"] for row in inventory),
    )


def count_query(where: str) -> str:
    return (
        "SELECT count(*) study_count, count(DISTINCT hadm_id) admission_count, "
        f"count(DISTINCT subject_id) patient_count FROM wide WHERE {where}"
    )


def reports(
    connection,
    run: _OutputRun,
    directory: Path,
    data_dir: Path,
    dictionary_status: tuple[bool, bool],
) -> dict:
    wide_paths = [
        str(run.files[data_dir / f"window_{h}h/study_event_audit.parquet"])
        for h in (48, 72)
    ]
    evidence_paths = [
        str(run.files[data_dir / f"window_{h}h/event_evidence.parquet"])
        for h in (48, 72)
    ]
    candidate_path = str(run.files[data_dir / "candidate_studies.parquet"])
    connection.read_parquet(wide_paths).create_view("wide")
    connection.read_parquet(evidence_paths).create_view("evidence")
    connection.read_parquet(candidate_path).create_view("candidates")
    write_query(
        connection,
        run,
        directory / "candidate_counts.csv",
        "SELECT candidate_status, count(*) study_count, count(DISTINCT hadm_id) admission_count, "
        "count(DISTINCT subject_id) patient_count FROM candidates GROUP BY candidate_status ORDER BY candidate_status",
    )
    for filename, condition in (
        ("excluded_studies.csv", "candidate_status='excluded'"),
        ("candidate_time_conflicts.csv", "review_status='pending'"),
    ):
        write_query(
            connection,
            run,
            directory / filename,
            f"SELECT * FROM candidates WHERE {condition} ORDER BY sample_id",
        )
    for horizon in (48, 72):
        target = directory / f"window_{horizon}h"
        summaries = []
        for component in COMPONENTS:
            for state, condition in (
                ("candidate", "true"),
                ("eligible", "candidate_status='eligible'"),
                ("assessable", f"{component} IS NOT NULL"),
                ("positive", f"{component}=1"),
                (
                    "no_qualifying_record",
                    f"{component}_event_status='no_qualifying_record'",
                ),
                ("preexisting_only", f"{component}_event_status='preexisting_only'"),
                (
                    "unresolved",
                    f"{component}_event_status='unresolved' AND candidate_status<>'excluded'",
                ),
                ("excluded", "candidate_status='excluded'"),
            ):
                counts = query_rows(
                    connection,
                    count_query(f"horizon_hours={horizon} AND ({condition})"),
                )[0]
                independent = connection.execute(
                    "SELECT count(DISTINCT source_record_key) FROM evidence WHERE horizon_hours=? "
                    "AND component=? AND qualifies=1",
                    [horizon, component],
                ).fetchall()[0][0]
                summaries.append(
                    dict(
                        horizon_hours=horizon,
                        component=component,
                        group=state,
                        **counts,
                        independent_positive_event_count=(
                            independent if state == "positive" else None
                        ),
                    )
                )
        _write_csv(
            run.path(target / "component_audit.csv"), list(summaries[0]), summaries
        )
        flags = ", ".join(COMPONENTS)
        write_query(
            connection,
            run,
            target / "component_overlap.csv",
            f"SELECT {flags}, count(*) study_count, count(DISTINCT hadm_id) admission_count, "
            f"count(DISTINCT subject_id) patient_count FROM wide WHERE horizon_hours={horizon} "
            f"AND label_status='complete' GROUP BY {flags} ORDER BY {flags}",
        )
        overlap_groups = []
        component_sum = " + ".join(COMPONENTS)
        for label, condition in [
            (f"only_{component}", f"{component}=1 AND ({component_sum})=1")
            for component in COMPONENTS
        ] + [
            ("at_least_two_components", f"({component_sum})>=2"),
            ("physiological_event_or_death", "phys_death=1"),
        ]:
            overlap_groups.append(
                dict(
                    horizon_hours=horizon,
                    group=label,
                    **query_rows(
                        connection,
                        count_query(
                            f"horizon_hours={horizon} AND label_status='complete' AND ({condition})"
                        ),
                    )[0],
                )
            )
        _write_csv(
            run.path(target / "component_overlap_groups.csv"),
            list(overlap_groups[0]),
            overlap_groups,
        )
        distribution = []
        for label, condition in [
            (str(value), f"burden_score={value}") for value in range(9)
        ] + [
            ("ge3", "burden_score>=3"),
            ("unresolved", "label_status='unresolved'"),
            ("excluded", "label_status='excluded'"),
            ("phys_death", "phys_death=1"),
            ("high_burden", "high_burden=1"),
        ]:
            distribution.append(
                dict(
                    horizon_hours=horizon,
                    group=label,
                    **query_rows(
                        connection,
                        count_query(f"horizon_hours={horizon} AND ({condition})"),
                    )[0],
                )
            )
        _write_csv(
            run.path(target / "score_distribution.csv"),
            list(distribution[0]),
            distribution,
        )
        write_query(
            connection,
            run,
            target / "unresolved_cases.csv",
            f"SELECT * FROM wide WHERE horizon_hours={horizon} AND label_status='unresolved' ORDER BY sample_id",
        )
        early_flags = []
        for component in COMPONENTS:
            early_flags.append(
                f"CASE WHEN {component} IS NULL THEN NULL ELSE cast("
                f"coalesce({component}_first_event_time <= least(t0 + INTERVAL '3 hours', "
                f"observation_end), false) AS INTEGER) END early_3h_{component}"
            )
        early = ", ".join(early_flags)
        e = [f"early_3h_{component}" for component in COMPONENTS]
        early_formula = (
            f"greatest({e[0]},{e[1]}) + 2*greatest({e[2]},{e[3]}) + 2*{e[4]} + 3*{e[5]}"
        )
        time_columns = ", ".join(
            f"{component}_first_event_time" for component in COMPONENTS
        )
        write_query(
            connection,
            run,
            target / "early_3h_events.csv",
            f"WITH early AS (SELECT *, {early}, least({time_columns}) first_early_event_time "
            f"FROM wide WHERE horizon_hours={horizon}) SELECT sample_id, subject_id, hadm_id, "
            f"study_id, t0, horizon_hours, observation_end, {', '.join(e)}, first_early_event_time, "
            f"CASE WHEN label_status='complete' THEN {early_formula} END early_3h_score, "
            "burden_score, label_status FROM early WHERE first_early_event_time <= "
            "least(t0+INTERVAL '3 hours', observation_end) ORDER BY sample_id",
        )
        write_query(
            connection,
            run,
            target / "early_discharge.csv",
            f"SELECT * FROM wide WHERE horizon_hours={horizon} AND early_discharge=1 ORDER BY sample_id",
        )
        stats = query_rows(
            connection,
            f"SELECT label_status, count(*) n FROM wide WHERE "
            f"horizon_hours={horizon} GROUP BY label_status ORDER BY label_status",
        )
        identity_counts = query_rows(
            connection,
            "SELECT count(DISTINCT sample_id) study_count, "
            "count(DISTINCT hadm_id) admission_count, "
            "count(DISTINCT subject_id) patient_count, "
            "count(DISTINCT source_record_key) emar_record_count FROM identity_exclusions",
        )[0]
        run.path(target / "audit_report.md").write_text(
            f"# V2 {horizon}-hour audit\n\nThis is an audit, not a frozen training cohort.\n\n"
            + "\n".join(f"- {row['label_status']}: {row['n']} studies" for row in stats)
            + f"\n\nDictionary classification complete: {dictionary_status[0]}. "
            f"Dictionary handling review complete: {dictionary_status[1]}. "
            "Unknown identity may remain after its handling policy is reviewed. "
            "Handling approval does not assign a care level or medication identity, "
            "resolve candidate or event conflicts, or freeze H. See "
            "../dictionary_inventory.csv for exact values and review provenance."
            + "\n\nEvents at t0 are excluded; the cutoff is inclusive. Early events remain in "
            "the outcome. No recorded event is an operational zero, not proof of absence. "
            "Candidate exclusion reasons use multiple-reason counting; status totals count "
            "each study once. Unknown or conflicting evidence is not scored.\n\n"
            "Medication identity exclusions use the union of both windows and apply to "
            "both: "
            + ", ".join(f"{key}={value}" for key, value in identity_counts.items())
            + ". See ../medication_identity_exclusions.csv for triggering windows, "
            "lookup statuses and original evidence. Patient counts describe affected "
            "patients; their other studies may remain eligible. These exclusions do not "
            "classify unknown medication as non-target.\n",
            encoding="utf-8",
        )
    component_counts = ", ".join(
        f"count(*) FILTER (WHERE {component}=1) {component}_study_count, "
        f"count(DISTINCT subject_id) FILTER (WHERE {component}=1) {component}_patient_count"
        for component in COMPONENTS
    )
    write_query(
        connection,
        run,
        directory / "window_comparison.csv",
        "SELECT horizon_hours, count(*) study_count, count(DISTINCT hadm_id) admission_count, "
        "count(DISTINCT subject_id) patient_count, count(*) FILTER (WHERE label_status='complete') complete_count, "
        "count(*) FILTER (WHERE label_status='unresolved') unresolved_count, "
        "count(*) FILTER (WHERE burden_score>0) nonzero_count, count(*) FILTER (WHERE phys_death=1) phys_death_count, "
        "count(*) FILTER (WHERE high_burden=1) high_burden_count, "
        "count(DISTINCT subject_id) FILTER (WHERE burden_score>0) nonzero_patient_count, "
        "count(DISTINCT subject_id) FILTER (WHERE phys_death=1) phys_death_patient_count, "
        "count(DISTINCT subject_id) FILTER (WHERE high_burden=1) high_burden_patient_count, "
        "count(*) FILTER (WHERE early_discharge=1) early_discharge_count, "
        "count(*) FILTER (WHERE spo2_source_available=0) no_spo2_count, "
        "count(*) FILTER (WHERE emar_source_available=0) no_emar_count, "
        "min(observed_hours) min_observed_hours, median(observed_hours) median_observed_hours, "
        f"max(observed_hours) max_observed_hours, {component_counts} FROM wide GROUP BY horizon_hours ORDER BY horizon_hours",
    )
    write_query(
        connection,
        run,
        directory / "window_changes.csv",
        "SELECT a.sample_id, a.subject_id, a.hadm_id, a.study_id, a.burden_score score_48h, "
        "b.burden_score score_72h, a.label_status status_48h, b.label_status status_72h, "
        "a.burden_score=0 AND b.burden_score>0 became_nonzero, b.burden_score>a.burden_score score_increased, "
        "a.in_hospital_death=0 AND b.in_hospital_death=1 death_added, "
        "a.early_discharge discharge_48h, b.early_discharge discharge_72h "
        "FROM wide a JOIN wide b USING (sample_id) WHERE a.horizon_hours=48 AND b.horizon_hours=72 ORDER BY a.sample_id",
    )
    for filename, component in (
        ("transfer_conflicts.csv", COMPONENTS[0]),
        ("ventilation_review_cases.csv", COMPONENTS[2]),
        ("infusion_onset_review.csv", COMPONENTS[4]),
        ("death_time_conflicts.csv", COMPONENTS[5]),
    ):
        condition = f"component='{component}'"
        if component == COMPONENTS[0]:
            condition = f"component IN ('{COMPONENTS[0]}','{COMPONENTS[1]}')"
        write_query(
            connection,
            run,
            directory / filename,
            f"SELECT * FROM evidence WHERE qualifies IS NULL AND {condition} ORDER BY sample_id, horizon_hours, source_record_key",
        )
    write_query(
        connection,
        run,
        directory / "transfer_transition_counts.csv",
        f"SELECT horizon_hours, component, json_extract_string(raw_evidence,'$.from_level') from_level, "
        "json_extract_string(raw_evidence,'$.to_level') to_level, qualifies, "
        "count(DISTINCT sample_id) study_count, count(DISTINCT subject_id) patient_count "
        f"FROM evidence WHERE component IN ('{COMPONENTS[0]}','{COMPONENTS[1]}') "
        "GROUP BY horizon_hours,component,from_level,to_level,qualifies ORDER BY ALL",
    )
    for filename, field in (
        ("spo2_coverage_summary.csv", "spo2_source_available"),
        ("emar_coverage_summary.csv", "emar_source_available"),
        ("procedure_coverage_summary.csv", "procedure_source_available"),
    ):
        grouping = (
            ", hypoxemia_evidence_status" if field == "spo2_source_available" else ""
        )
        write_query(
            connection,
            run,
            directory / filename,
            f"SELECT horizon_hours, anchor_year_group, {field}, label_status{grouping}, count(*) study_count, "
            "count(DISTINCT hadm_id) admission_count, count(DISTINCT subject_id) patient_count "
            f"FROM wide GROUP BY horizon_hours,anchor_year_group,{field},label_status{grouping} ORDER BY ALL",
        )
    write_query(
        connection,
        run,
        directory / "spo2_capture_timing.csv",
        "SELECT horizon_hours, CASE WHEN minutes_from_first_icu_transfer_to_hypoxemia IS NULL THEN "
        "'unavailable' WHEN minutes_from_first_icu_transfer_to_hypoxemia<0 THEN 'before_icu_transfer' "
        "ELSE 'at_or_after_icu_transfer' END capture_group, count(*) study_count, "
        "count(DISTINCT subject_id) patient_count FROM wide WHERE new_severe_hypoxemia=1 "
        "GROUP BY horizon_hours,capture_group ORDER BY ALL",
    )
    write_query(
        connection,
        run,
        directory / "spo2_value_qc.csv",
        f"SELECT horizon_hours,exclusion_reason,qualifies,count(DISTINCT source_record_key) unique_record_count, "
        "count(DISTINCT sample_id) study_count,count(DISTINCT subject_id) patient_count "
        f"FROM evidence WHERE component='{COMPONENTS[3]}' "
        "GROUP BY horizon_hours,exclusion_reason,qualifies ORDER BY ALL",
    )
    write_query(
        connection,
        run,
        directory / "early_discharge_summary.csv",
        "SELECT horizon_hours,early_discharge,label_status,count(*) study_count, "
        "count(DISTINCT subject_id) patient_count, avg(cast(burden_score=0 AS INTEGER)) zero_score_proportion, "
        "min(observed_hours) min_hours,median(observed_hours) median_hours,max(observed_hours) max_hours "
        "FROM wide GROUP BY horizon_hours,early_discharge,label_status ORDER BY ALL",
    )
    write_query(
        connection,
        run,
        directory / "early_discharge_feature_summary.csv",
        "SELECT horizon_hours,early_discharge,care_level_at_t0,count(*) study_count, "
        "count(DISTINCT subject_id) patient_count,avg(age_at_t0) mean_age_at_t0 "
        "FROM wide WHERE candidate_status='eligible' GROUP BY horizon_hours,early_discharge,care_level_at_t0 ORDER BY ALL",
    )
    write_query(
        connection,
        run,
        directory / "patient_study_counts.csv",
        "SELECT subject_id,candidate_status,count(*) study_count,count(DISTINCT hadm_id) admission_count "
        "FROM candidates GROUP BY subject_id,candidate_status ORDER BY ALL",
    )
    write_query(
        connection,
        run,
        directory / "event_study_counts.csv",
        "SELECT horizon_hours,component,source_record_key,count(DISTINCT sample_id) study_count, "
        "count(DISTINCT subject_id) patient_count FROM evidence WHERE qualifies=1 "
        "GROUP BY horizon_hours,component,source_record_key ORDER BY ALL",
    )
    write_query(
        connection,
        run,
        directory / "review_sample.csv",
        "SELECT * EXCLUDE (review_rank) FROM (SELECT *, row_number() OVER "
        "(PARTITION BY horizon_hours,component,event_status ORDER BY sha256('42:'||sample_id||source_record_key)) "
        "AS review_rank FROM evidence) WHERE review_rank<=20 ORDER BY horizon_hours,component,event_status,review_rank",
    )
    bad = connection.execute(
        "SELECT count(*) FROM wide WHERE label_status='complete' AND ("
        "burden_score <> T+2*R+2*C+3*D OR burden_score NOT BETWEEN 0 AND 8 "
        f"OR T<>greatest({COMPONENTS[0]},{COMPONENTS[1]}) OR R<>greatest({COMPONENTS[2]},{COMPONENTS[3]}) "
        f"OR C<>{COMPONENTS[4]} OR D<>{COMPONENTS[5]} "
        "OR phys_death<>cast(R=1 OR C=1 OR D=1 AS INTEGER) OR high_burden<>cast(burden_score>=3 AS INTEGER) "
        "OR candidate_status<>'eligible')"
    ).fetchall()[0][0]
    outside = connection.execute(
        "SELECT count(*) FROM evidence e JOIN wide w USING (sample_id,horizon_hours) "
        "WHERE qualifies=1 AND (e.event_time IS NULL OR e.event_time<=w.t0 OR e.event_time>w.observation_end)"
    ).fetchall()[0][0]
    if bad or outside:
        raise ValueError("Audit score or positive-event boundary validation failed")
    for component in COMPONENTS:
        inconsistent = connection.execute(
            "WITH decisions AS (SELECT sample_id,horizon_hours,"
            "bool_or(qualifies IS NULL AND blocks_event=1) pending,coalesce(max(qualifies),0) event_flag,"
            "min(first_event_time) FILTER (WHERE qualifies=1) first_time FROM evidence "
            "WHERE component=? GROUP BY sample_id,horizon_hours) SELECT count(*) FROM wide w "
            "LEFT JOIN decisions d USING(sample_id,horizon_hours) WHERE candidate_status<>'excluded' "
            "AND observation_end IS NOT NULL AND ("
            f"w.{component} IS DISTINCT FROM (CASE WHEN coalesce(d.pending,false) THEN NULL "
            "ELSE coalesce(d.event_flag,0) END) OR "
            f"w.{component}_first_event_time IS DISTINCT FROM (CASE WHEN coalesce(d.pending,false) "
            "THEN NULL ELSE d.first_time END))",
            [component],
        ).fetchall()[0][0]
        if inconsistent:
            raise ValueError(
                f"Evidence does not reproduce wide-table component: {component}"
            )
    return {
        "candidate_counts": query_rows(
            connection,
            "SELECT candidate_status,count(*) n FROM candidates GROUP BY candidate_status ORDER BY candidate_status",
        ),
        "label_counts": query_rows(
            connection,
            "SELECT horizon_hours,label_status,count(*) n FROM wide GROUP BY horizon_hours,label_status ORDER BY ALL",
        ),
        "unresolved_evidence_count": connection.execute(
            "SELECT count(*) FROM evidence WHERE qualifies IS NULL"
        ).fetchall()[0][0],
    }


def audit_burden_labels(
    root: Path, paths_file: Path, cohort_file: Path, ehr_file: Path, imaging_file: Path
) -> int:
    started = datetime.now(timezone.utc)
    paths, cohort, ehr, imaging = map(
        _json_file, (paths_file, cohort_file, ehr_file, imaging_file)
    )
    rules = validate_rules(cohort, ehr, imaging)
    iv = _resolve_config_path(paths["mimic_iv"], root, raw=True)
    cxr = _resolve_config_path(paths["mimic_cxr"], root, raw=True)
    processed = _resolve_config_path(paths["processed"], root, raw=False)
    outputs = _resolve_config_path(paths["outputs"], root, raw=False)
    _protect_outputs(paths, root, processed, outputs)
    supplementary = rules.get("supplementary_audit", {})
    baseline, review_scope = {}, set()
    if supplementary:
        baseline_path = _resolve_config_path(
            supplementary["baseline_exclusions_path"], root, raw=False
        )
        with baseline_path.open(encoding="utf-8-sig", newline="") as stream:
            baseline_rows = list(_csv_reader(stream))
        baseline = {r["sample_id"]: r for r in baseline_rows}
        if len(baseline) != len(baseline_rows):
            raise ValueError("Duplicate baseline exclusion sample")
        review_scope = {
            k
            for k, r in baseline.items()
            if r["exclusion_reason"] in SUPPLEMENTARY_REASONS
        }
        if (len(review_scope), len(baseline) - len(review_scope)) != tuple(
            supplementary.get("expected_scope_counts", [221, 841])
        ):
            raise ValueError("Supplementary scope must match the confirmed baseline")
        source_outputs = _resolve_config_path(
            supplementary["baseline_outputs"], root, raw=False
        )
        baseline_bindings = {
            baseline_path.relative_to(root).as_posix(): _file_hash(baseline_path)
        }
        for field in (
            "baseline_labels_path",
            "baseline_split_path",
            "baseline_image_manifest_path",
            "baseline_acceptance_path",
        ):
            if supplementary.get(field):
                path = _resolve_config_path(supplementary[field], root, raw=False)
                baseline_bindings[path.relative_to(root).as_posix()] = _file_hash(path)
        if source_outputs == outputs:
            raise ValueError("Supplementary audit must use isolated output roots")
        save_audit_snapshot(source_outputs, outputs)
        if (outputs / "audit/step_05_manifest.json").exists():
            save_audit_snapshot(outputs)
    config_files = [paths_file, cohort_file, ehr_file, imaging_file]
    override_path = (
        _resolve_config_path(rules["overrides_path"], root, raw=False)
        if rules.get("overrides_path")
        else None
    )
    if override_path:
        config_files.append(override_path)
    candidate_exclusions_path = (
        _resolve_config_path(rules["candidate_exclusions_path"], root, raw=False)
        if rules.get("candidate_exclusions_path")
        else None
    )
    if candidate_exclusions_path:
        config_files.append(candidate_exclusions_path)
    review_paths = {}
    for field in REVIEW_PATHS:
        path = (
            _resolve_config_path(rules[field], root, raw=False)
            if rules.get(field)
            else None
        )
        review_paths[field] = path
        if path:
            config_files.append(path)
    config_hash = _config_hash(config_files)
    source_names = ehr["burden_v2"]["tables"]
    source_paths = {
        name: _under_root(iv, value) for name, value in source_names.items()
    }
    versions = {name: _file_version(path) for name, path in source_paths.items()}
    data_dir, audit_dir = processed / "cohort/burden_v2", outputs / "audit/burden_v2"
    with _OutputRun(outputs, 5, processed) as run, ExitStack() as stack:
        matches, selected, upstream = load_upstream(processed, outputs, run)
        if source_names["admissions"] != cohort["admission_matching"][
            "admissions_path"
        ] or (
            _file_hash(source_paths["admissions"])
            != upstream["step_04_admissions_sha256"]
        ):
            raise ValueError("Admissions changed since Step 04; rerun matching")
        sample_ids = {
            f"p{row['subject_id']}_h{row['hadm_id']}_s{row['study_id']}"
            for row in matches
        }
        overrides = load_overrides(
            override_path,
            sample_ids,
            run.check_csv(override_path) if override_path else None,
        )
        candidate_exclusions = load_candidate_exclusions(
            candidate_exclusions_path,
            sample_ids,
            run.check_csv(candidate_exclusions_path)
            if candidate_exclusions_path
            else None,
        )
        cohort_decisions = load_review_decisions(
            review_paths["cohort_decisions_path"],
            sample_ids,
            False,
            list(candidate_exclusions.values()),
            run.check_csv(review_paths["cohort_decisions_path"])
            if review_paths["cohort_decisions_path"]
            else None,
        )
        event_decisions = load_review_decisions(
            review_paths["event_decisions_path"],
            sample_ids,
            True,
            list(overrides.values()),
            run.check_csv(review_paths["event_decisions_path"])
            if review_paths["event_decisions_path"]
            else None,
        )
        overrides = defaultdict(list)
        for decision in event_decisions:
            overrides[
                decision["sample_id"],
                identifier(decision["horizon_hours"]),
                decision["component"],
            ].append(decision)
        snapshot_bindings = {}
        scope_report, lineage = [], {}
        outputs.mkdir(parents=True, exist_ok=True)
        scratch = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(prefix=".burden_v2_", dir=outputs)
            )
        )
        connection = open_store(scratch, ehr["burden_v2"]["memory_limit"])
        stack.callback(connection.close)
        ingest(
            connection,
            iv,
            ehr["burden_v2"],
            matches,
            run,
            supplementary=bool(supplementary),
        )
        for table, field in (
            ("admissions", "hadm_id"),
            ("patients", "subject_id"),
            ("d_items", "itemid"),
        ):
            if connection.execute(
                f"SELECT count(*) FROM (SELECT {field} FROM {table} GROUP BY {field} HAVING count(*)>1)"
            ).fetchall()[0][0]:
                raise ValueError(f"Duplicate source key: {table}/{field}")
        dictionaries_confirmed, handling_confirmed = dictionary_reports(
            connection, run, audit_dir, rules
        )
        interval_fields = (
            "source_record_key intime outtime start end admission_overlap_start "
            "admission_overlap_end starts_before_admission ends_after_discharge "
            "outtime_imputed imputation_basis reason"
        ).split()
        interval_audit = csv.DictWriter(
            stack.enter_context(
                run.path(audit_dir / "care_interval_audit.csv").open(
                    "w", encoding="utf-8", newline=""
                )
            ),
            fieldnames=[
                "subject_id",
                "hadm_id",
                "record_subject_id",
                "record_hadm_id",
                "source_file",
                "admittime",
                "dischtime",
                *interval_fields,
                "raw_evidence",
            ],
        )
        interval_audit.writeheader()
        identity_exclusions_stream = stack.enter_context(
            run.path(audit_dir / "medication_identity_exclusions.csv").open(
                "w", encoding="utf-8", newline=""
            )
        )
        identity_exclusions_writer = csv.DictWriter(
            identity_exclusions_stream, fieldnames=IDENTITY_EXCLUSION_COLUMNS
        )
        identity_exclusions_writer.writeheader()
        interval_counts = dict(
            admission_interval_count=0, accepted_boundary_crossing_count=0
        )
        sinks = {}

        def sink(key: str, destination: Path, columns: Sequence[str]) -> ParquetSink:
            value = ParquetSink(run.path(destination), columns)
            stack.callback(value.close)
            sinks[key] = value
            return value

        candidate_sink = sink(
            "candidates", data_dir / "candidate_studies.parquet", CANDIDATE_COLUMNS
        )
        if supplementary:
            sink(
                "supplementary_wide",
                data_dir / "supplementary_event_audit.parquet",
                WIDE_COLUMNS,
            )
            sink(
                "supplementary_evidence",
                data_dir / "supplementary_event_evidence.parquet",
                EVIDENCE_COLUMNS,
            )
        candidate_evidence_sink = sink(
            "candidate_evidence",
            data_dir / "candidate_evidence.parquet",
            [
                "sample_id",
                "subject_id",
                "hadm_id",
                "study_id",
                "source_file",
                "source_record_key",
                "raw_evidence",
            ],
        )
        for horizon in (48, 72):
            base = data_dir / f"window_{horizon}h"
            sink(
                f"observation_{horizon}",
                base / "candidate_observation.parquet",
                OBSERVATION_COLUMNS,
            )
            sink(f"wide_{horizon}", base / "study_event_audit.parquet", WIDE_COLUMNS)
            sink(
                f"evidence_{horizon}", base / "event_evidence.parquet", EVIDENCE_COLUMNS
            )
        detail_files = {
            "transfer_events": COMPONENTS[:2],
            "hypoxemia_records": (COMPONENTS[3],),
            "vaso_inotrope_events": (COMPONENTS[4],),
            "death_events": (COMPONENTS[5],),
        }
        for name in detail_files:
            sink(name, data_dir / f"{name}.parquet", EVIDENCE_COLUMNS)
        episode_columns = (
            "subject_id hadm_id source_record_key source_record_keys episode_start episode_end exclusion_reason raw_evidence"
        ).split()
        medication_columns = (
            "subject_id hadm_id emar_id source_record_key source_record_keys event_time canonical_drug action exclusion_reason raw_evidence"
        ).split()
        sink("episodes", data_dir / "ventilation_episodes.parquet", episode_columns)
        sink(
            "medications",
            data_dir / "medication_administrations.parquet",
            medication_columns,
        )
        identity_columns = (
            "subject_id hadm_id emar_id pharmacy_id original_medication resolved_medication "
            "medication_name_source pharmacy_lookup_status pharmacy_lookup_version "
            "pharmacy_match_count medication_mapping_confirmed canonical_drug "
            "prescriptions_lookup_status prescriptions_lookup_version prescriptions_match_count "
            "event_time source_record_key source_record_keys raw_evidence"
        ).split()
        identity_sink = sink(
            "medication_identity",
            data_dir / "medication_identity_audit.parquet",
            identity_columns,
        )
        groups = defaultdict(list)
        for match in matches:
            groups[match["subject_id"], match["hadm_id"]].append(match)
        consumed_overrides = set()
        policy_exclusion_count = 0
        identity_exclusion_count = 0
        with _progress(
            "Audit V2 studies", unit="study", total=len(matches)
        ) as progress:
            for (subject, hadm), studies in sorted(groups.items()):
                admissions = rows(connection, "admissions", subject)
                current = [
                    row for row in admissions if identifier(row["hadm_id"]) == hadm
                ]
                patient_rows = rows(connection, "patients", subject)
                if len(current) != 1:
                    raise ValueError("Candidate admission does not resolve uniquely")
                admission, patient = current[0], patient_rows[0] if patient_rows else {}
                context = {
                    "admission": admission,
                    "admissions": admissions,
                    "source_names": source_names,
                }
                for table in (
                    "transfers",
                    "icustays",
                    "procedureevents",
                    "chartevents",
                    "emar",
                    "pharmacy",
                    "prescriptions",
                ):
                    context["stays" if table == "icustays" else table] = rows(
                        connection, table, subject, hadm
                    )
                context["transfers"] = prepare_intervals(
                    "transfers", context["transfers"], admission, admissions, rules
                )
                context["stays"] = prepare_intervals(
                    "icustays", context["stays"], admission, admissions, rules
                )
                for table, values in (
                    ("transfers", context["transfers"]),
                    ("icustays", context["stays"]),
                ):
                    for row in values:
                        interval_audit.writerow(
                            {
                                "subject_id": subject,
                                "hadm_id": hadm,
                                "record_subject_id": row.get("subject_id"),
                                "record_hadm_id": row.get("hadm_id"),
                                "source_file": source_names[table],
                                "admittime": admission["admittime"],
                                "dischtime": admission["dischtime"],
                                **{name: row.get(name) for name in interval_fields},
                                "raw_evidence": json.dumps(
                                    row, sort_keys=True, default=str
                                ),
                            }
                        )
                        interval_counts["admission_interval_count"] += 1
                        interval_counts["accepted_boundary_crossing_count"] += int(
                            not row["reason"]
                            and bool(
                                row["starts_before_admission"]
                                or row["ends_after_discharge"]
                            )
                        )
                context["ventilation_episodes"], context["ventilation_rejected"] = (
                    ventilation_episodes(context, rules)
                )
                detail = query_rows(
                    connection,
                    "SELECT d.* FROM emar_detail d WHERE EXISTS (SELECT 1 FROM emar e "
                    "WHERE e.emar_id=d.emar_id AND try_cast(e.subject_id AS BIGINT)=? AND "
                    "(try_cast(e.hadm_id AS BIGINT)=? OR coalesce(e.hadm_id,'')=''))",
                    [subject, hadm],
                )
                context["medication_events"] = aggregate_medications(
                    context, detail, rules
                )
                identity_sink.write(
                    [
                        {
                            **{name: row.get(name) for name in identity_columns},
                            "subject_id": subject,
                            "hadm_id": hadm,
                            "original_medication": row.get("medication") or "",
                            "event_time": row["time"],
                            "medication_mapping_confirmed": str(
                                row["medication_mapping_confirmed"]
                            ),
                            "source_record_keys": json.dumps(row["source_record_keys"]),
                            "raw_evidence": json.dumps(
                                row, sort_keys=True, default=str
                            ),
                        }
                        for row in context["medication_events"]
                    ]
                )
                sinks["episodes"].write(
                    [
                        dict(
                            subject_id=subject,
                            hadm_id=hadm,
                            source_record_key=row["source_record_key"],
                            source_record_keys=json.dumps(
                                row.get(
                                    "source_record_keys", [row["source_record_key"]]
                                )
                            ),
                            episode_start=row["start"],
                            episode_end=row["end"],
                            exclusion_reason=row["reason"],
                            raw_evidence=json.dumps(row, sort_keys=True, default=str),
                        )
                        for row in context["ventilation_episodes"]
                        + context["ventilation_rejected"]
                    ]
                )
                # Conflicting IDs are represented once with all conflicting raw records.
                medications = defaultdict(list)
                for row in context["medication_events"]:
                    medications[row["emar_id"]].append(row)
                sinks["medications"].write(
                    [
                        dict(
                            subject_id=subject,
                            hadm_id=hadm,
                            emar_id=key,
                            source_record_key=record_key(
                                "medication_administration",
                                {
                                    "keys": sorted(
                                        row["source_record_key"] for row in values
                                    )
                                },
                            ),
                            source_record_keys=json.dumps(
                                sorted(
                                    {
                                        key
                                        for row in values
                                        for key in row["source_record_keys"]
                                    }
                                )
                            ),
                            event_time=values[0]["time"] if len(values) == 1 else None,
                            canonical_drug=values[0]["canonical_drug"]
                            if len(values) == 1
                            else None,
                            action=values[0]["action"]
                            if len(values) == 1
                            else "review",
                            exclusion_reason=values[0]["reason"]
                            if len(values) == 1
                            else "conflicting_emar_id",
                            raw_evidence=json.dumps(
                                values, sort_keys=True, default=str
                            ),
                        )
                        for key, values in sorted(medications.items())
                    ]
                )
                for match in sorted(studies, key=lambda row: row["study_id"]):
                    if any(
                        clinical_time(match[name]) != clinical_time(admission[name])
                        for name in ("admittime", "dischtime")
                    ):
                        raise ValueError("Matched admission boundaries changed")
                    selection = selected[subject, match["study_id"]]
                    if selection["view_position"] not in {"AP", "PA"}:
                        raise ValueError("Selected image is not frontal")
                    candidate = build_candidate(
                        match,
                        selection,
                        admission,
                        patient,
                        context["transfers"],
                        context["stays"],
                        rules,
                        _under_root(cxr, selection["dicom_path"]).is_file(),
                    )
                    diagnostic_candidate = dict(candidate)
                    decisions = [
                        r
                        for r in cohort_decisions
                        if r["sample_id"] == candidate["sample_id"]
                    ]
                    for decision in decisions + [
                        r
                        for r in event_decisions
                        if r["sample_id"] == candidate["sample_id"]
                    ]:
                        if decision.get("legacy_review"):
                            continue
                        index = validate_review_binding(
                            decision, candidate, context, outputs, versions, rules
                        )
                        snapshot_bindings[index.relative_to(root).as_posix()] = (
                            _file_hash(index)
                        )
                    resolved = {
                        r["reason"]
                        for r in decisions
                        if r["decision"] == "resolve_blocker"
                    }
                    if (
                        resolved
                        and candidate["sample_id"] in baseline
                        and candidate["sample_id"] not in review_scope
                    ):
                        raise ValueError(
                            "A decision attempts to reopen a protected baseline exclusion"
                        )
                    for decision in decisions:
                        if decision["decision"] != "resolve_blocker":
                            continue
                        if decision["reason"] == "Conflicting care-unit sources at t0":
                            level = decision.get("care_level_at_t0")
                            if level not in {
                                "ward",
                                "stepdown_intermediate",
                                "icu_ccu",
                            }:
                                raise ValueError(
                                    "Care-unit resolution requires structured care_level_at_t0"
                                )
                            candidate["care_level_at_t0"] = level
                            reasons = set(candidate["exclusion_reason"].split(";")) - {
                                "",
                                "transfers_icustays_conflict",
                            }
                            if level == "icu_ccu":
                                reasons.add("icu_at_t0")
                            candidate["exclusion_reason"] = ";".join(sorted(reasons))
                            candidate["candidate_status"] = (
                                "excluded"
                                if "icu_at_t0" in reasons
                                else "needs_review"
                                if reasons
                                else "eligible"
                            )
                            candidate["review_status"] = "reviewed"
                    if (
                        candidate["sample_id"] in baseline
                        and candidate["sample_id"] not in review_scope
                    ):
                        candidate.update(
                            candidate_status="excluded",
                            exclusion_reason=baseline[candidate["sample_id"]][
                                "exclusion_reason"
                            ],
                        )
                    candidate_sources = [
                        ("admissions", record_key("admissions", admission), admission),
                        ("patients", record_key("patients", patient), patient),
                    ]
                    candidate_sources += [
                        ("transfers", row["source_record_key"], row)
                        for row in context["transfers"]
                    ]
                    candidate_sources += [
                        ("icustays", row["source_record_key"], row)
                        for row in context["stays"]
                    ]
                    exclusions = [r for r in decisions if r["decision"] == "exclude"]
                    exclusion = exclusions[0] if exclusions else None
                    if len(exclusions) > 1:
                        exclusion = {
                            **exclusion,
                            "reason": ";".join(
                                sorted({r["reason"] for r in exclusions})
                            ),
                            "source_record_keys": json.dumps(
                                sorted(
                                    {
                                        k
                                        for r in exclusions
                                        for k in json.loads(r["source_record_keys"])
                                    }
                                )
                            ),
                        }
                    exclusion_source = candidate_exclusions_path
                    if exclusion and not exclusion.get("legacy_review"):
                        exclusion_source = review_paths["cohort_decisions_path"]
                    if not exclusion:
                        exclusion = candidate_conflict_exclusion(
                            candidate, context, rules
                        )
                        if exclusion and exclusion["reason"] in resolved:
                            exclusion = None
                        if exclusion:
                            exclusion_source = cohort_file
                            policy_exclusion_count += 1
                    if not exclusion:
                        exclusion, triggers = medication_identity_exclusion(
                            candidate, context, rules
                        )
                        if exclusion and exclusion["reason"] in resolved:
                            exclusion = None
                        if exclusion:
                            exclusion_source = cohort_file
                            identity_exclusion_count += 1
                            for row in triggers:
                                raw = json.loads(row["raw_evidence"])
                                lookup_statuses = {
                                    raw["pharmacy_lookup_status"],
                                    raw["prescriptions_lookup_status"],
                                }
                                identity_reason = (
                                    "unconfirmed_exact_name"
                                    if raw["resolved_medication"]
                                    else "multiple_identity_matches"
                                    if "ambiguous_match" in lookup_statuses
                                    else "missing_lookup_keys"
                                    if "missing_lookup_key" in lookup_statuses
                                    else "empty_lookup_drug_name"
                                    if any(
                                        value.startswith("blank_")
                                        for value in lookup_statuses
                                    )
                                    else "identity_not_resolved"
                                )
                                identity_exclusions_writer.writerow(
                                    {
                                        **row,
                                        "emar_id": raw["emar_id"],
                                        "original_medication": raw["medication"],
                                        **{
                                            field: raw[field]
                                            for field in (
                                                "resolved_medication",
                                                "medication_name_source",
                                                "pharmacy_lookup_status",
                                                "pharmacy_match_count",
                                                "prescriptions_lookup_status",
                                                "prescriptions_match_count",
                                            )
                                        },
                                        "identity_reason": identity_reason,
                                        "reviewer": exclusion["reviewer"],
                                        "policy_version": exclusion["version"],
                                        "candidate_exclusion_reason": exclusion[
                                            "reason"
                                        ],
                                        "automated_candidate_status": candidate[
                                            "candidate_status"
                                        ],
                                    }
                                )
                    if exclusion:
                        cited = set(json.loads(exclusion["source_record_keys"]))
                        candidate_sources.extend(
                            ("chartevents", key, raw)
                            for raw in context["chartevents"]
                            if (key := record_key("chartevents", raw)) in cited
                            and identifier(raw.get("subject_id"))
                            == candidate["subject_id"]
                            and identifier(raw.get("hadm_id")) == candidate["hadm_id"]
                        )
                        candidate_sources.extend(
                            (
                                "icustays" if table == "stays" else table,
                                raw["legacy_source_record_key"],
                                raw,
                            )
                            for table in ("transfers", "stays")
                            for raw in context[table]
                            if raw.get("legacy_source_record_key") in cited
                        )
                        for raw in context["medication_events"]:
                            if raw["reason"] == "other_admission" or identifier(
                                raw.get("hadm_id")
                            ) not in (None, candidate["hadm_id"]):
                                continue
                            for table, records in (
                                ("emar", [raw]),
                                ("emar_detail", raw["details"]),
                                ("pharmacy", raw["pharmacy_records"]),
                                ("prescriptions", raw["prescriptions_records"]),
                            ):
                                candidate_sources.extend(
                                    (table, value["source_record_key"], value)
                                    for value in records
                                    if value["source_record_key"] in cited
                                    and identifier(value.get("subject_id"))
                                    == candidate["subject_id"]
                                )
                                candidate_sources.extend(
                                    (table, value["legacy_source_record_key"], value)
                                    for value in records
                                    if value.get("legacy_source_record_key") in cited
                                    and identifier(value.get("subject_id"))
                                    == candidate["subject_id"]
                                )
                        available = {
                            key
                            for _, key, raw in candidate_sources
                            if raw.get("reason") != "other_admission"
                        }
                        if not cited <= available:
                            raise ValueError(
                                "Candidate exclusion cites unavailable source keys"
                            )
                        manual_key = record_key("candidate_exclusions", exclusion)
                        candidate_sources.append(
                            (
                                "candidate_exclusions",
                                manual_key,
                                {
                                    **exclusion,
                                    "automated_candidate_status": candidate[
                                        "candidate_status"
                                    ],
                                    "automated_exclusion_reason": candidate[
                                        "exclusion_reason"
                                    ],
                                    "automated_review_status": candidate[
                                        "review_status"
                                    ],
                                },
                            )
                        )
                        candidate.update(
                            candidate_status="excluded",
                            exclusion_reason=exclusion["reason"],
                            review_status="reviewed",
                            candidate_exclusion_reviewer=exclusion["reviewer"],
                            candidate_exclusion_rationale=exclusion["rationale"],
                            candidate_exclusion_version=exclusion["version"],
                            candidate_evidence_keys=json.dumps(
                                sorted(
                                    set(
                                        json.loads(candidate["candidate_evidence_keys"])
                                    )
                                    | cited
                                    | {manual_key}
                                )
                            ),
                        )
                    candidate_sink.write([candidate])
                    referenced = set(json.loads(candidate["candidate_evidence_keys"]))
                    candidate_evidence_sink.write(
                        [
                            {
                                **{
                                    name: candidate[name]
                                    for name in (
                                        "sample_id",
                                        "subject_id",
                                        "hadm_id",
                                        "study_id",
                                    )
                                },
                                "source_file": str(exclusion_source)
                                if table == "candidate_exclusions"
                                else source_names[table],
                                "source_record_key": key,
                                "raw_evidence": json.dumps(
                                    raw, sort_keys=True, default=str
                                ),
                            }
                            for table, key, raw in {
                                key: (table, key, raw)
                                for table, key, raw in candidate_sources
                            }.values()
                            if key in referenced
                            or (raw.get("reason") not in (None, "", "other_admission"))
                        ]
                    )
                    for horizon in (48, 72):
                        study = observation(candidate, horizon)
                        values, evidence_rows = evaluate(
                            study,
                            context,
                            rules,
                            {}
                            if supplementary and study["candidate_status"] == "excluded"
                            else overrides,
                        )
                        if supplementary and candidate["sample_id"] in review_scope:
                            diagnostic = observation(
                                candidate
                                if candidate["candidate_status"] != "excluded"
                                else diagnostic_candidate,
                                horizon,
                            )
                            preview, preview_evidence = evaluate(
                                diagnostic,
                                context,
                                rules,
                                overrides
                                if diagnostic["candidate_status"] == "eligible"
                                else {},
                            )
                            sinks["supplementary_wide"].write(
                                [{**diagnostic, **preview}]
                            )
                            sinks["supplementary_evidence"].write(preview_evidence)
                        if (
                            supplementary
                            and horizon == 72
                            and candidate["sample_id"] in baseline
                        ):
                            scope_report.append(
                                {
                                    "sample_id": candidate["sample_id"],
                                    "baseline_reason": baseline[candidate["sample_id"]][
                                        "exclusion_reason"
                                    ],
                                    "in_review_scope": int(
                                        candidate["sample_id"] in review_scope
                                    ),
                                    "disposition": "protected_exclusion"
                                    if candidate["sample_id"] not in review_scope
                                    else "restored"
                                    if values["label_status"] == "complete"
                                    else "still_excluded"
                                    if any(
                                        not r.get("legacy_review")
                                        and r["decision"] == "exclude"
                                        for r in decisions
                                    )
                                    else "pending",
                                    "current_reason": candidate["exclusion_reason"],
                                    "label_status": values["label_status"],
                                }
                            )
                        consumed_overrides |= {
                            (study["sample_id"], horizon, component)
                            for component in COMPONENTS
                            if (study["sample_id"], horizon, component) in overrides
                        }
                        sinks[f"observation_{horizon}"].write([study])
                        sinks[f"wide_{horizon}"].write([{**study, **values}])
                        sinks[f"evidence_{horizon}"].write(evidence_rows)
                        for name, components in detail_files.items():
                            sinks[name].write(
                                [
                                    row
                                    for row in evidence_rows
                                    if row["component"] in components
                                ]
                            )
                    progress.update(1)
        if consumed_overrides != set(overrides):
            raise ValueError("Unused override targets")
        if supplementary:
            if {r["sample_id"] for r in scope_report} != set(baseline):
                raise ValueError("Baseline exclusion coverage changed")
            _write_csv(
                run.path(audit_dir / "supplementary_scope.csv"),
                [
                    "sample_id",
                    "baseline_reason",
                    "in_review_scope",
                    "disposition",
                    "current_reason",
                    "label_status",
                ],
                scope_report,
            )
            for table in SUPPLEMENTARY_COLUMNS:
                for raw in deduplicate(table, rows(connection, table)):
                    if raw.get("legacy_source_record_key"):
                        lineage[raw["source_record_key"]] = {
                            "source_table": table,
                            "source_record_key": raw["source_record_key"],
                            "legacy_source_record_key": raw["legacy_source_record_key"],
                            "subject_id": raw.get("subject_id"),
                            "hadm_id": raw.get("hadm_id", ""),
                        }
            _write_csv(
                run.path(audit_dir / "source_key_lineage.csv"),
                [
                    "source_table",
                    "source_record_key",
                    "legacy_source_record_key",
                    "subject_id",
                    "hadm_id",
                ],
                list(lineage.values()),
            )
        for value in sinks.values():
            value.close()
        identity_exclusions_stream.flush()
        connection.read_csv(
            str(run.files[audit_dir / "medication_identity_exclusions.csv"]),
            header=True,
            all_varchar=True,
        ).create_view("identity_exclusions")
        identity_exclusion_summary = query_rows(
            connection,
            "SELECT count(DISTINCT sample_id) study_count, "
            "count(DISTINCT hadm_id) admission_count, "
            "count(DISTINCT subject_id) patient_count, "
            "count(DISTINCT source_record_key) emar_record_count "
            "FROM identity_exclusions",
        )[0]
        if identity_exclusion_summary["study_count"] != identity_exclusion_count:
            raise ValueError("Medication identity exclusion counts disagree")
        if candidate_sink.count != len(matches) or any(
            sinks[f"wide_{h}"].count != len(matches) for h in (48, 72)
        ):
            raise ValueError("V2 candidate/window row counts disagree")
        summary = reports(
            connection,
            run,
            audit_dir,
            data_dir,
            (dictionaries_confirmed, handling_confirmed),
        )
        connection.read_parquet(
            str(run.files[data_dir / "medication_identity_audit.parquet"])
        ).create_view("medication_identity")
        write_query(
            connection,
            run,
            audit_dir / "pharmacy_identity_audit.csv",
            "SELECT * FROM medication_identity WHERE trim(original_medication)='' "
            "ORDER BY subject_id,hadm_id,emar_id,source_record_key",
        )
        identity_summary_query = (
            "SELECT pharmacy_lookup_status,medication_name_source,medication_mapping_confirmed, "
            "count(*) context_record_count,count(DISTINCT source_record_key) emar_record_count, "
            "count(DISTINCT subject_id) patient_count FROM medication_identity "
            "GROUP BY ALL ORDER BY ALL"
        )
        identity_summary = query_rows(connection, identity_summary_query)
        write_query(
            connection,
            run,
            audit_dir / "pharmacy_identity_summary.csv",
            identity_summary_query,
        )
        write_query(
            connection,
            run,
            audit_dir / "prescriptions_identity_audit.csv",
            "SELECT * FROM medication_identity WHERE trim(original_medication)='' "
            "AND prescriptions_lookup_status<>'not_needed' "
            "ORDER BY subject_id,hadm_id,emar_id,source_record_key",
        )
        prescriptions_summary_query = (
            "SELECT prescriptions_lookup_status,pharmacy_lookup_status,medication_name_source, "
            "medication_mapping_confirmed,count(*) context_record_count, "
            "count(DISTINCT source_record_key) emar_record_count, "
            "count(DISTINCT subject_id) patient_count FROM medication_identity "
            "WHERE trim(original_medication)='' GROUP BY ALL ORDER BY ALL"
        )
        prescriptions_summary = query_rows(connection, prescriptions_summary_query)
        write_query(
            connection,
            run,
            audit_dir / "prescriptions_identity_summary.csv",
            prescriptions_summary_query,
        )
        ready = (
            bool(matches)
            and any(
                row["candidate_status"] == "eligible" and row["n"] > 0
                for row in summary["candidate_counts"]
            )
            and handling_confirmed
            and not connection.execute(
                "SELECT count(*) FROM wide WHERE candidate_status='eligible' AND label_status<>'complete'"
            ).fetchone()[0]
            and not any(
                row["candidate_status"] == "needs_review"
                for row in summary["candidate_counts"]
            )
        )
        metadata = {
            "csv_tail_checks": run.csv_tail_summary(),
            "run_id": run.run_id,
            "outcome_version": "burden_v2",
            "audit_horizons_hours": [48, 72],
            "event_window": "(t0, min(t0+H, dischtime)]",
            "target_formula": "max(icu,stepdown)+2*max(imv,hypoxemia)+2*vaso_inotrope+3*death",
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(root),
            "config_sha256": config_hash,
            "rules_config_sha256": _file_hash(cohort_file),
            "dictionary_sha256": hashlib.sha256(
                json.dumps(
                    {
                        name: rules[name]
                        for name in (
                            *MAPPING_ACTIONS,
                            "medications",
                            "spo2_units",
                            "items",
                        )
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "code_sha256": _config_hash(
                [
                    Path(__file__),
                    *[
                        Path(__file__).with_name(name)
                        for name in (
                            "burden_io.py",
                            "build_candidates.py",
                            "burden_events.py",
                            "common.py",
                        )
                    ],
                    Path(__file__).resolve().parents[2]
                    / "scripts/05_audit_burden_labels.py",
                ]
            ),
            "dictionary_version": rules["dictionary_version"],
            "rules_version": rules["rules_version"],
            "input_versions": versions,
            "source_tables": source_names,
            "pharmacy_identity_lookup": {
                "policy": rules.get("pharmacy_identity_lookup", {}),
                "report": "pharmacy_identity_audit.csv",
                "summary_report": "pharmacy_identity_summary.csv",
                "summary": identity_summary,
            },
            "prescriptions_identity_lookup": {
                "policy": rules.get("prescriptions_identity_lookup", {}),
                "report": "prescriptions_identity_audit.csv",
                "summary_report": "prescriptions_identity_summary.csv",
                "summary": prescriptions_summary,
            },
            "upstream": upstream,
            "override_sha256": _file_hash(override_path) if override_path else None,
            "override_count": len(overrides),
            **{
                hash_field: _file_hash(review_paths[field])
                if review_paths[field]
                else None
                for field, hash_field in REVIEW_PATHS.items()
            },
            "review_snapshot_bindings": snapshot_bindings,
            "baseline_reference_bindings": baseline_bindings if supplementary else {},
            "supplementary_scope": {
                "review_studies": len(review_scope),
                "protected_studies": len(baseline) - len(review_scope),
                "baseline_exclusions_sha256": _file_hash(baseline_path)
                if supplementary
                else None,
            },
            "candidate_exclusions_path": str(candidate_exclusions_path)
            if candidate_exclusions_path
            else None,
            "candidate_exclusions_sha256": _file_hash(candidate_exclusions_path)
            if candidate_exclusions_path
            else None,
            "candidate_exclusion_count": len(candidate_exclusions),
            "candidate_policy_exclusion_count": policy_exclusion_count,
            "candidate_exclusion_total_count": len(candidate_exclusions)
            + policy_exclusion_count
            + identity_exclusion_count,
            "candidate_medication_identity_exclusions": {
                "policy": rules.get(IDENTITY_POLICY, {}),
                "report": "medication_identity_exclusions.csv",
                "summary": identity_exclusion_summary,
            },
            "candidate_conflict_policies": rules.get("candidate_conflict_policies", {}),
            "dictionary_review_complete": dictionaries_confirmed,
            "dictionary_handling_review_complete": handling_confirmed,
            "dictionary_handling_reviews": rules.get("dictionary_handling_reviews", {}),
            "dictionary_handling_reviews_sha256": hashlib.sha256(
                json.dumps(
                    rules.get("dictionary_handling_reviews", {}), sort_keys=True
                ).encode()
            ).hexdigest(),
            "audit_checks_passed": True,
            "audit_ready_for_window_decision": ready,
            "ready_for_formal_labels": False,
            "horizon_frozen": False,
            "unique_match_studies": len(matches),
            "care_interval_audit": {
                "report": "care_interval_audit.csv",
                **interval_counts,
            },
            "source_retained_row_counts": {
                name: connection.execute(f"SELECT count(*) FROM {name}").fetchall()[0][
                    0
                ]
                for name in TABLE_COLUMNS
            },
            "emar_join_counts": {
                "retained_emar_rows": connection.execute(
                    "SELECT count(*) FROM emar"
                ).fetchall()[0][0],
                "unique_emar_ids": connection.execute(
                    "SELECT count(DISTINCT emar_id) FROM emar"
                ).fetchall()[0][0],
                "retained_detail_rows": connection.execute(
                    "SELECT count(*) FROM emar_detail"
                ).fetchall()[0][0],
                "aggregated_administration_rows": sinks["medications"].count,
            },
            **summary,
        }
        if _config_hash(config_files) != config_hash or any(
            _file_version(source_paths[name]) != value
            for name, value in versions.items()
        ):
            raise ValueError("An input or configuration changed during Step 05")
        _verify_run(outputs, 3)
        _verify_run(outputs, 4)
        run.path(audit_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    print(f"V2 audit complete: {len(matches)} unique-match studies; H=48 and 72 hours")
    if supplementary:
        save_audit_snapshot(outputs)
    print(f"Reports: {audit_dir}")
    print(
        "H remains unfrozen. Review dictionary_inventory.csv, conflicts and run_metadata.json before Step 06."
    )
    return 0
