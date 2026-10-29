"""Bound-cache laboratory aggregation and audited static features for Step 10B."""

from __future__ import annotations

import json
import math
import platform
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.burden_io import clinical_time, open_store
from src.cohort.common import (
    _file_hash,
    _file_version,
    _json_file,
    _OutputRun,
    _protect_outputs,
    _resolve_config_path,
    _under_root,
    _verify_run,
    _write_csv,
)
from src.ehr import build_ehr_features as upstream
from src.ehr import lab_audit as lab
from src.ehr import lab_review as review
from src.ehr.lab_dictionary_inputs import APPROVED_FEATURES

AGGREGATIONS = [
    "last_24h",
    "min_24h",
    "max_24h",
    "mean_24h",
    "count_24h",
    "hours_since_last",
]
STATIC = [
    "age_at_t0",
    "sex",
    "admission_type",
    "admission_location",
    "careunit_at_t0",
    "current_service",
    "hours_from_admission",
]
CATEGORICAL = [
    "sex",
    "admission_type",
    "admission_location",
    "careunit_at_t0",
    "current_service",
]
SOURCE_COLUMNS = {
    "patients": "subject_id gender anchor_age anchor_year",
    "admissions": "subject_id hadm_id admittime admission_type admission_location",
    "services": "subject_id hadm_id transfertime curr_service",
}
ISSUE_COLUMNS = ["sample_id", "source", "field", "reason", "evidence"]
MISSING = "__MISSING__"


def identity(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("Unsafe version identity")
    return value


def code_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    return lab.json_hash(
        {
            str(p.relative_to(root)): _file_hash(p)
            for p in [
                Path(__file__),
                root / "scripts/10b_build_ehr_features.py",
                root / "src/ehr/lab_dictionary_inputs.py",
            ]
        }
    )


def load_settings(path: Path) -> dict:
    settings = _json_file(path)
    identity(settings["audit_version"])
    if (
        settings["algorithm_version"] != "ehr_features_v1"
        or settings["lookback_hours"] != 24
        or settings["aggregations"] != AGGREGATIONS
        or settings["static_features"] != STATIC
        or settings["tables"] != {k: f"hosp/{k}.csv" for k in SOURCE_COLUMNS}
    ):
        raise ValueError("Unapproved Step 10B feature definitions")
    return settings


def validate_dictionary(table: pa.Table, version: str) -> list[dict]:
    rows = table.to_pylist()
    if not rows or any(r["dictionary_version"] != version for r in rows):
        raise ValueError("Dictionary version mismatch")
    retained = [r for r in rows if r["keep"]]
    expected = {int(k): v for k, v in APPROVED_FEATURES.items()}
    if {r["itemid"] for r in retained} != set(expected):
        raise ValueError("Unapproved laboratory item scope")
    for row in retained:
        name, unit = expected[row["itemid"]]
        allowed_units = (
            {"%", "g/dL"}
            if row["itemid"] == 51249
            else {""}
            if row["itemid"] == 51237
            else {unit}
        )
        if (
            row["feature_name"] != name
            or row["canonical_unit"] != unit
            or row["source_unit"] not in allowed_units
            or row["missing_unit_rule"]
            != ("explicit_item_unit" if row["itemid"] == 51237 else "")
            or json.loads(row["specimen_rule"]) != {"kind": "item_identity"}
            or review.duplicate_rule(row["duplicate_rule"])
            != {
                "kind": "latest_available",
                "conflicting_id": "error",
                "missing_specimen": "error",
            }
            or review.numeric_rule(row["numeric_result_rule"])
            != {
                "kind": "valuenum_only",
                "comparator": "exclude",
                "text_disagreement": "error",
            }
            or row["valid_min"] is not None
            or row["valid_max"] is not None
            or row["conversion_multiplier"] != 1
            or row["conversion_offset"] != 0
        ):
            raise ValueError("Dictionary rules differ from approved Step 10B scope")
    branches = [(r["itemid"], r["source_unit"]) for r in retained]
    if len(branches) != len(set(branches)) or {
        unit for item, unit in branches if item == 51249
    } != {"%", "g/dL"}:
        raise ValueError("Incomplete or overlapping approved unit branches")
    return sorted(retained, key=lambda r: (r["itemid"], r["source_unit"]))


def clean_labs(connection, rules: list[dict]) -> None:
    """Select the latest available revision before rejecting nonnumeric results."""
    review.validate_retained_conflicts(connection, rules)
    pieces, params = [], []
    for row in rules:
        pieces.append(
            "SELECT a.*,?::VARCHAR AS feature_name,?::VARCHAR AS canonical_unit,"
            "numeric_value*?+? AS converted_value FROM associations a WHERE itemid=? AND valueuom=? "
            "AND availability_reason='available' AND ownership_reason='owned'"
        )
        params.extend(
            [
                row["feature_name"],
                row["canonical_unit"],
                row["conversion_multiplier"],
                row["conversion_offset"],
                str(row["itemid"]),
                row["source_unit"],
            ]
        )
    connection.execute("CREATE TABLE selected AS " + " UNION ALL ".join(pieces), params)
    connection.execute(
        "CREATE TABLE lab_trace AS SELECT *, CASE WHEN revision_rank<>1 THEN 'superseded_revision' "
        "WHEN regexp_matches(value,'[<>≤≥]') THEN 'comparator' "
        "WHEN numeric_value IS NULL THEN 'no_finite_valuenum' ELSE 'retained' END AS disposition "
        "FROM (SELECT *,row_number() OVER(PARTITION BY sample_id,patient_id,assigned_hadm_id,itemid,specimen_id,chart_at "
        "ORDER BY stored_at DESC,try_cast(labevent_id AS BIGINT) DESC,lab_record_key DESC) AS revision_rank FROM selected)"
    )
    connection.execute(
        "CREATE TABLE cleaned AS SELECT * FROM lab_trace WHERE disposition='retained'"
    )
    if connection.execute(
        "SELECT count(*) FROM cleaned WHERE NOT isfinite(converted_value)"
    ).fetchone()[0]:
        raise ValueError("Non-finite laboratory conversion")


def schema_for(rules: list[dict], version: str) -> dict:
    units = {r["feature_name"]: r["canonical_unit"] for r in rules}
    names = sorted(units)
    numeric = [
        f"{name}_{aggregation}" for name in names for aggregation in AGGREGATIONS
    ] + ["age_at_t0", "hours_from_admission"]
    features = {}
    for name in names:
        ids = sorted({r["itemid"] for r in rules if r["feature_name"] == name})
        for aggregation in AGGREGATIONS:
            column = f"{name}_{aggregation}"
            features[column] = {
                "dtype": "float32",
                "unit": "count"
                if aggregation == "count_24h"
                else "hour"
                if aggregation == "hours_since_last"
                else units[name],
                "source": "hosp/labevents.csv",
                "itemids": ids,
                "aggregation": aggregation,
                "description": f"Reviewed {name}: {aggregation}; available in the inclusive pre-imaging 24-hour window.",
            }
    for name in STATIC:
        features[name] = {
            "dtype": "string" if name in CATEGORICAL else "float32",
            "unit": "year"
            if name == "age_at_t0"
            else "hour"
            if name == "hours_from_admission"
            else None,
            "source": "patients"
            if name in {"age_at_t0", "sex"}
            else "services"
            if name == "current_service"
            else "frozen cohort evidence"
            if name == "careunit_at_t0"
            else "admissions",
            "aggregation": "latest transfertime <= t0"
            if name == "current_service"
            else "static or t0 context",
            "description": name.replace("_", " "),
        }
    return {
        "schema_version": "1.0",
        "feature_set_version": version,
        "key_column": "sample_id",
        "lookback_hours": 24,
        "numeric_features": numeric,
        "categorical_features": CATEGORICAL,
        "categorical_missing_value": MISSING,
        "features": features,
        "limitations": [
            "Deidentified anchor ages cannot recover true ages above 89.",
            "Service availability uses transfertime; no independent service storetime is supplied.",
        ],
        "preprocessing_applied": False,
    }


def float32_table(rows: list[dict], schema: dict) -> pa.Table:
    columns = {"sample_id": pa.array([r["sample_id"] for r in rows], type=pa.string())}
    for name in schema["numeric_features"]:
        values = [float("nan") if r.get(name) is None else float(r[name]) for r in rows]
        if any(math.isinf(v) for v in values):
            raise ValueError(f"Non-finite input feature: {name}")
        array = pa.array(values, type=pa.float64()).cast(pa.float32())
        if any(math.isinf(v) for v in array.to_pylist()):
            raise ValueError(f"float32 overflow: {name}")
        columns[name] = array
    for name in schema["categorical_features"]:
        columns[name] = pa.array(
            [r.get(name) or MISSING for r in rows], type=pa.string()
        )
    return pa.table(columns)


def aggregate_labs(connection, cohort: pa.Table, rules: list[dict]) -> list[dict]:
    names = sorted({r["feature_name"] for r in rules})
    rows = {
        r["sample_id"]: {
            "sample_id": r["sample_id"],
            **{
                f"{n}_{a}": 0 if a == "count_24h" else None
                for n in names
                for a in AGGREGATIONS
            },
        }
        for r in cohort.to_pylist()
    }
    for result in lab.query_rows(
        connection,
        "SELECT sample_id,feature_name,"
        "first(converted_value ORDER BY chart_at DESC,stored_at DESC,try_cast(labevent_id AS BIGINT) DESC,lab_record_key DESC) AS last_24h,"
        "min(converted_value) AS min_24h,max(converted_value) AS max_24h,avg(converted_value) AS mean_24h,count(*) AS count_24h,"
        "epoch(max(t0)-max(chart_at))/3600.0 AS hours_since_last FROM cleaned GROUP BY sample_id,feature_name",
    ):
        rows[result["sample_id"]].update(
            {f"{result['feature_name']}_{a}": result[a] for a in AGGREGATIONS}
        )
    return [rows[k] for k in sorted(rows)]


def static_features(
    connection, cohort: pa.Table, evidence: pa.Table
) -> tuple[dict, list[dict]]:
    """Only cohort-filtered source subsets and known-at-t0 evidence enter Python."""
    sources = {
        name: lab.query_rows(connection, f"SELECT DISTINCT * FROM static_{name}")
        for name in SOURCE_COLUMNS
    }
    by_patient, by_admission, by_services = {}, {}, {}

    def number(value):
        return int(value) if value and re.fullmatch(r"[1-9][0-9]*", value) else None

    for row in sources["patients"]:
        by_patient.setdefault(number(row["subject_id"]), []).append(row)
    for row in sources["admissions"]:
        by_admission.setdefault(number(row["hadm_id"]), []).append(row)
    for row in sources["services"]:
        by_services.setdefault(number(row["hadm_id"]), []).append(row)
    known = {r["sample_id"]: r for r in evidence.to_pylist()}
    output, issues = {}, []
    for name, records in sources.items():
        for raw in records:
            if number(raw["subject_id"]) is None or (
                name != "patients" and number(raw["hadm_id"]) is None
            ):
                issues.append(
                    dict(
                        sample_id="",
                        source=name,
                        field="identity",
                        reason="invalid_source_identity",
                        evidence=json.dumps(raw, sort_keys=True),
                    )
                )
    for study in cohort.to_pylist():
        key, patient, admission, t0 = (
            study[k] for k in ["sample_id", "subject_id", "hadm_id", "t0"]
        )
        row = dict.fromkeys(STATIC)
        output[key] = row

        def issue(source, field, reason, raw):
            issues.append(
                dict(
                    sample_id=key,
                    source=source,
                    field=field,
                    reason=reason,
                    evidence=json.dumps(raw, default=str, sort_keys=True),
                )
            )

        patients = by_patient.get(patient, [])
        admissions = by_admission.get(admission, [])
        baseline = known.get(key)
        if (
            baseline is None
            or baseline["subject_id"] != patient
            or baseline["hadm_id"] != admission
            or baseline["t0"] != t0
        ):
            issue(
                "cohort", "identity", "missing_or_conflicting_frozen_evidence", baseline
            )
            continue
        row["age_at_t0"], row["careunit_at_t0"] = (
            baseline.get("age_at_t0"),
            baseline.get("careunit_at_t0"),
        )
        if len(patients) > 1:
            issue("patients", "identity", "conflicting_rows", patients)
        elif patients:
            p = patients[0]
            row["sex"] = (p["gender"] or "").strip() or None
            if row["sex"] not in {None, "F", "M"}:
                issue("patients", "sex", "unsupported_nonempty_value", p)
            raw_age, raw_year = p["anchor_age"], p["anchor_year"]
            if raw_age and raw_year:
                try:
                    age = int(raw_age) + t0.year - int(raw_year)
                    if age < 0 or age != row["age_at_t0"]:
                        raise ValueError("age mismatch")
                except (ValueError, TypeError):
                    issue("patients", "age_at_t0", "invalid_or_conflicting_age", p)
            elif any(v and not re.fullmatch(r"[0-9]+", v) for v in [raw_age, raw_year]):
                issue("patients", "age_at_t0", "invalid_nonempty_age_field", p)
        if len(admissions) > 1:
            issue("admissions", "identity", "conflicting_rows", admissions)
        elif admissions:
            a = admissions[0]
            if number(a["subject_id"]) != patient:
                issue("admissions", "identity", "patient_mismatch", a)
            admitted = clinical_time(a["admittime"])
            if (
                admitted is None
                or admitted != baseline.get("admittime")
                or admitted > t0
            ):
                issue("admissions", "admittime", "invalid_or_conflicting_admittime", a)
            else:
                row["hours_from_admission"] = (t0 - admitted).total_seconds() / 3600
            for field in ["admission_type", "admission_location"]:
                row[field] = (a[field] or "").strip() or None
        services = []
        for s in by_services.get(admission, []):
            time = clinical_time(s["transfertime"])
            if number(s["subject_id"]) != patient or time is None:
                issue(
                    "services", "identity_or_time", "invalid_or_conflicting_service", s
                )
            elif time <= t0:
                services.append((time, (s["curr_service"] or "").strip() or None))
        if services:
            latest = max(time for time, _ in services)
            values = {value for time, value in services if time == latest}
            if len(values) != 1:
                issue(
                    "services",
                    "current_service",
                    "conflicting_latest_service",
                    sorted(values, key=str),
                )
            else:
                row["current_service"] = next(iter(values))
    return output, issues


def validate_wide(table: pa.Table, cohort: pa.Table, schema: dict) -> None:
    numeric, categorical = schema["numeric_features"], schema["categorical_features"]
    names = ["sample_id", *numeric, *categorical]
    keys = table["sample_id"].to_pylist()
    if (
        len(names) != len(set(names))
        or table.column_names != names
        or set(schema["features"]) != set(names[1:])
        or len(keys) != len(set(keys))
        or set(keys) != set(cohort["sample_id"].to_pylist())
    ):
        raise ValueError("Wide table keys or whitelist mismatch")
    for name in numeric:
        if (
            table[name].type != pa.float32()
            or table[name].null_count
            or any(math.isinf(v) for v in table[name].to_pylist())
        ):
            raise ValueError(f"Invalid numeric feature: {name}")
    for name in categorical:
        if (
            table[name].type != pa.string()
            or table[name].null_count
            or any(not v for v in table[name].to_pylist())
        ):
            raise ValueError(f"Invalid categorical feature: {name}")


def feature_reports(
    connection, table: pa.Table, cohort: pa.Table, patients: pa.Table, schema: dict,
    cv_bundle: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    split_by_patient = {r["subject_id"]: r["split"] for r in patients.to_pylist()}
    subject = {r["sample_id"]: r["subject_id"] for r in cohort.to_pylist()}
    rows = table.to_pylist()
    groups = {
        group: [
            r
            for r in rows
            if group == "all" or split_by_patient[subject[r["sample_id"]]] == group
        ]
        for group in ["all", "train", "validation", "test"]
    }
    if cv_bundle is not None:
        by_sample = {r["sample_id"]: r for r in rows}
        master = cv_bundle["master"]
        for fold in cv_bundle["folds"]:
            for name, key in (("train", "train_indices"), ("validation", "validation_indices")):
                groups[f"fold_{fold['fold']}_{name}"] = [
                    by_sample[master[i]["sample_id"]] for i in fold[key]
                ]
    summary, missingness = [], []
    raw = {
        (r["sample_id"], r["feature_name"])
        for r in lab.query_rows(
            connection,
            "SELECT DISTINCT sample_id,feature_name FROM selected WHERE numeric_value IS NOT NULL",
        )
    }
    for group, subset in groups.items():
        patient_ids = {subject[r["sample_id"]] for r in subset}
        for name in schema["numeric_features"] + schema["categorical_features"]:
            values = [
                r[name]
                for r in subset
                if r[name] != MISSING
                and not (isinstance(r[name], float) and math.isnan(r[name]))
            ]
            numeric = name in schema["numeric_features"]
            summary.append(
                dict(
                    split=group,
                    feature=name,
                    studies=len(subset),
                    missing=len(subset) - len(values),
                    missing_rate=(len(subset) - len(values)) / len(subset)
                    if subset
                    else None,
                    minimum=min(values) if numeric and values else None,
                    maximum=max(values) if numeric and values else None,
                    distinct_values=len(set(values)),
                    all_missing=not values,
                    constant=len(set(values)) == 1,
                )
            )
        for name in sorted(
            {
                r["feature_name"]
                for r in lab.query_rows(
                    connection, "SELECT DISTINCT feature_name FROM selected"
                )
            }
            | {
                n.removesuffix("_last_24h")
                for n in schema["numeric_features"]
                if n.endswith("_last_24h")
            }
        ):
            before = {r["sample_id"] for r in subset if (r["sample_id"], name) in raw}
            after = {
                r["sample_id"] for r in subset if not math.isnan(r[f"{name}_last_24h"])
            }
            rate = 1 - len(after) / len(subset) if subset else None
            raw_rate = 1 - len(before) / len(subset) if subset else None

            def coverage_band(observed):
                # Integer comparisons preserve the approved inclusive thresholds.
                missing, total = len(subset) - observed, len(subset)
                return (
                    None
                    if not total
                    else 0
                    if 100 * missing < 30 * total
                    else 1
                    if 100 * missing <= 50 * total
                    else 2
                    if 100 * missing <= 80 * total
                    else 3
                )

            missingness.append(
                dict(
                    split=group,
                    feature=name,
                    studies=len(subset),
                    patients=len(patient_ids),
                    before_studies=len(before),
                    after_studies=len(after),
                    before_patients=len({subject[k] for k in before}),
                    after_patients=len({subject[k] for k in after}),
                    before_missing_rate=raw_rate,
                    after_missing_rate=rate,
                    crossed_band=coverage_band(len(before))
                    != coverage_band(len(after)),
                    all_missing=not after,
                )
            )
    return summary, missingness


def write_report(
    run, directory: Path, name: str, columns: list[str], rows: list[dict]
) -> None:
    _write_csv(run.path(directory / name), columns, rows)


def verify_sources(sources: dict, versions: dict) -> None:
    if {k: _file_version(v) for k, v in sources.items()} != versions:
        raise ValueError("Source versions changed during Step 10B")


def verify_audit(record: dict, binding: dict, roots: tuple[Path, ...]) -> None:
    if record["binding"] != binding or record["audit_fingerprint"] != lab.json_hash(
        binding
    ):
        raise ValueError("Stale EHR audit cache; use a new audit_version")
    for filename, digest in record["artifact_sha256"].items():
        path = Path(filename).resolve()
        if (
            not any(path.is_relative_to(r.resolve()) for r in roots)
            or _file_hash(path) != digest
        ):
            raise ValueError("EHR audit artifact checksum or path changed")


def publish_blocked_audit(run, path: Path, binding: dict, issues: list[dict]) -> int:
    record = {
        "binding": binding,
        "audit_fingerprint": lab.json_hash(binding),
        "audit_checks_passed": False,
        "blocking_conflicts": len(issues),
        "ehr_features_ready": False,
        "run_id": run.run_id,
        "cache_table": None,
        "artifact_sha256": {
            str(target): _file_hash(stage) for target, stage in run.files.items()
        },
    }
    upstream.write_json(run.path(path), record)
    print(
        "Audit published with blocking conflicts; inspect blocking_conflicts.csv. Build is unavailable."
    )
    return 0


def verify_releases(data: Path, registry_path: Path) -> dict:
    registry = _json_file(registry_path) if registry_path.exists() else {}
    for version, entry in registry.items():
        snapshot = data / "releases" / identity(version)
        record_path = snapshot / "ehr_features_freeze.json"
        if _file_hash(record_path) != entry["freeze_sha256"]:
            raise ValueError("Historical EHR freeze record changed")
        record = _json_file(record_path)
        if set(record["artifact_sha256"]) != {
            "ehr_features_raw.parquet",
            "feature_schema.json",
        }:
            raise ValueError("Incomplete historical EHR snapshot")
        for name, digest in record["artifact_sha256"].items():
            if _file_hash(snapshot / name) != digest:
                raise ValueError("Historical EHR snapshot changed")
    return registry


def publish_features(
    run,
    data: Path,
    directory: Path,
    table: pa.Table,
    schema: dict,
    binding: dict,
    version: str,
) -> None:
    registry_path = directory.parent / "ehr_features_releases.json"
    registry = verify_releases(data, registry_path)
    snapshot = data / "releases" / identity(version)
    record_path = snapshot / "ehr_features_freeze.json"
    if snapshot.exists() and any(snapshot.iterdir()) and not record_path.exists():
        raise ValueError("Incomplete immutable feature snapshot")
    if record_path.exists():
        record = _json_file(record_path)
        if record["binding"] != binding:
            raise ValueError(
                "Changed content cannot overwrite a frozen feature identity"
            )
        for filename, digest in record["artifact_sha256"].items():
            if (
                filename not in {"ehr_features_raw.parquet", "feature_schema.json"}
                or _file_hash(snapshot / filename) != digest
            ):
                raise ValueError("Frozen feature snapshot changed")
        if set(record["artifact_sha256"]) != {
            "ehr_features_raw.parquet",
            "feature_schema.json",
        }:
            raise ValueError("Incomplete frozen feature record")
        if _json_file(snapshot / "feature_schema.json") != schema:
            raise ValueError("Frozen feature schema differs")
        for name in record["artifact_sha256"]:
            shutil.copyfile(snapshot / name, run.path(data / name))
    else:
        pq.write_table(
            table, run.path(snapshot / "ehr_features_raw.parquet"), compression="zstd"
        )
        upstream.write_json(run.path(snapshot / "feature_schema.json"), schema)
        record = {
            "feature_set_version": version,
            "binding": binding,
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "original_freeze_run_id": run.run_id,
            "ehr_features_ready": True,
            "artifact_sha256": {
                name: _file_hash(run.path(snapshot / name))
                for name in ["ehr_features_raw.parquet", "feature_schema.json"]
            },
        }
        upstream.write_json(run.path(record_path), record)
        for name in record["artifact_sha256"]:
            shutil.copyfile(run.path(snapshot / name), run.path(data / name))
    upstream.write_json(run.path(directory / "ehr_features_freeze.json"), record)
    registry[version] = {
        "freeze_sha256": _file_hash(
            record_path if record_path.exists() else run.path(record_path)
        )
    }
    upstream.write_json(run.path(registry_path), registry)


def run_step10b(root: Path, args) -> int:
    settings = load_settings(args.feature_config)
    identity(args.dictionary_version)
    if args.mode not in {"audit", "build"} or (args.mode == "build") != bool(
        args.feature_set_version
    ):
        raise ValueError("Invalid Step 10B mode arguments")
    if args.feature_set_version:
        identity(args.feature_set_version)
    configs = [
        getattr(args, name).resolve() for name in ["paths", "cohort", "ehr", "imaging"]
    ]
    lab_settings = upstream.load_settings(args.lab_config)
    context = upstream.verify_frozen_inputs(
        root, configs, args.processing, args.split_config, lab_settings
    )
    if (
        settings["cohort_version"] != lab_settings["cohort_version"]
        or settings["split_version"] != lab_settings["split_version"]
    ):
        raise ValueError("EHR configuration differs from frozen identities")
    outputs, processed = context["outputs"], context["processed"]
    _verify_run(outputs, 10)
    upstream.dictionary_releases(context)
    data = processed / "ehr/burden_v2"
    frozen = _json_file(outputs / "audit/burden_v2/lab_dictionary_freeze.json")
    if (
        not frozen["dictionary_frozen"]
        or frozen["dictionary_version"] != args.dictionary_version
        or frozen["source_binding"]["code_sha256"] != upstream.code_hash()
    ):
        raise ValueError("Frozen dictionary identity or Step 10A code changed")
    dictionary_path = data / "lab_dictionary.parquet"
    if _file_hash(dictionary_path) != frozen["dictionary_sha256"]:
        raise ValueError("Frozen dictionary checksum changed")
    rules = validate_dictionary(pq.read_table(dictionary_path), args.dictionary_version)
    paths = _json_file(args.paths)
    raw = _resolve_config_path(paths["mimic_iv"], root, raw=True)
    lab_sources = {k: _under_root(raw, v) for k, v in lab_settings["tables"].items()}
    if lab_settings.get("vital_audit"):
        lab_sources.update({k: _under_root(raw,v)
                            for k,v in lab_settings["vital_audit"]["tables"].items()})
    lab_binding = upstream.source_binding(
        context, lab_sources, args.lab_config, lab_settings
    )
    lab_directory = outputs / "audit/burden_v2/labs" / lab_settings["audit_version"]
    lab_record = upstream.load_audit(
        lab_directory / "lab_audit_manifest.json",
        lab_binding,
        processed.parent,
        outputs,
    )
    if lab_record.get("vital_audit"):
        scope = frozen["source_binding"].get("feature_scope_review")
        if not scope or scope != upstream.validate_feature_scope(
                Path(scope["path"]), lab_record, pq.read_table(dictionary_path).to_pylist(),
                lab_settings, lab_directory / "lab_audit_manifest.json"):
            raise ValueError("Vital feature exclusion scope is missing or changed")
    if (
        frozen["source_binding"]["audit_manifest_sha256"]
        != _file_hash(lab_directory / "lab_audit_manifest.json")
        or frozen["source_binding"]["audit_fingerprint"]
        != lab_record["audit_fingerprint"]
    ):
        raise ValueError("Frozen dictionary belongs to different audit evidence")
    sources = {k: _under_root(raw, v) for k, v in settings["tables"].items()}
    versions = {k: _file_version(p) for k, p in sources.items()}
    if any(not v["exists"] for v in versions.values()):
        raise FileNotFoundError("Missing Step 10B static source")
    binding = {
        "cohort": context["cohort_identity"],
        "split": context["split_identity"],
        "dictionary_freeze_sha256": _file_hash(
            outputs / "audit/burden_v2/lab_dictionary_freeze.json"
        ),
        "dictionary_sha256": _file_hash(dictionary_path),
        "lab_audit_sha256": _file_hash(lab_directory / "lab_audit_manifest.json"),
        "source_versions": versions,
        "source_paths": {k: str(v) for k, v in sources.items()},
        "configuration_sha256": _file_hash(args.feature_config),
        "settings": settings,
        "code_sha256": code_hash(),
        "runtime_versions": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "pyarrow": pa.__version__,
        },
    }
    directory = outputs / "audit/burden_v2/ehr" / settings["audit_version"]
    cache_root = (
        _resolve_config_path(paths["cache"], root, raw=False)
        if paths.get("cache")
        else processed.parent / "cache"
    )
    cache = (cache_root / "burden_v2/ehr" / settings["audit_version"]).resolve()
    namespace = outputs / "step_10b" / args.mode
    _protect_outputs(paths, root, processed, outputs, cache)
    for path in [
        *configs,
        args.processing,
        args.split_config,
        args.lab_config,
        args.feature_config,
    ]:
        if any(
            path.resolve().is_relative_to(p.resolve())
            for p in (processed, outputs, cache)
        ):
            raise ValueError(
                "Input configuration must be outside generated directories"
            )
    audit_path = directory / "ehr_audit_manifest.json"
    cohort = context["cohort"]
    print(
        f"Step 10B {args.mode}: {settings['audit_version']}; dictionary {args.dictionary_version}"
    )
    with _OutputRun(namespace, 10, processed, outputs, cache) as run:
        verify_releases(data, directory.parent / "ehr_features_releases.json")
        if run.manifest.exists():
            _verify_run(namespace, 10)
        if audit_path.exists():
            record = _json_file(audit_path)
            verify_audit(record, binding, (processed.parent, outputs))
            _verify_run(outputs / "step_10b/audit", 10)
        elif args.mode == "build":
            raise ValueError("Run Step 10B audit before build")
        else:
            with tempfile.TemporaryDirectory(
                prefix="synthetic_or_user_step10b_"
            ) as temp:
                with open_store(Path(temp), settings["memory_limit"]) as connection:
                    lab.load_cache(
                        connection,
                        {k: Path(v) for k, v in lab_record["cache_paths"].items()},
                    )
                    lab.build_views(connection, cohort)
                    connection.register(
                        "cohort_patients", cohort.select(["subject_id"])
                    )
                    for name, columns in SOURCE_COLUMNS.items():
                        condition = "try_cast(r.subject_id AS BIGINT) IN (SELECT subject_id FROM cohort_patients)"
                        if name in {"admissions", "services"}:
                            # Also retain any incorrectly owned row bearing a cohort admission ID.
                            connection.register(
                                "cohort_admissions", cohort.select(["hadm_id"])
                            )
                            condition += " OR try_cast(r.hadm_id AS BIGINT) IN (SELECT hadm_id FROM cohort_admissions)"
                        lab.scan_csv(
                            connection,
                            sources[name],
                            f"static_{name}",
                            columns.split(),
                            [],
                            condition,
                            run,
                        )
                    static, issues = static_features(
                        connection, cohort, context["cohort_audit"]
                    )
                    lab_error = None
                    try:
                        clean_labs(connection, rules)
                    except ValueError as error:
                        lab_error = str(error)
                        issues.append(
                            dict(
                                sample_id="",
                                source="labevents",
                                field="reviewed_rules",
                                reason="laboratory_conflict",
                                evidence=lab_error,
                            )
                        )
                    write_report(
                        run,
                        directory,
                        "static_conflicts.csv",
                        ISSUE_COLUMNS,
                        [i for i in issues if i["source"] != "labevents"],
                    )
                    write_report(
                        run, directory, "blocking_conflicts.csv", ISSUE_COLUMNS, issues
                    )
                    if lab_error:
                        run.path(directory / "ehr_audit_report.md").write_text(
                            f"# Step 10B audit blocked\n\n{lab_error}\n",
                            encoding="utf-8",
                        )
                        verify_sources(sources, versions)
                        if (
                            upstream.source_binding(
                                context, lab_sources, args.lab_config, lab_settings
                            )
                            != lab_binding
                        ):
                            raise ValueError(
                                "Laboratory sources changed during blocked audit"
                            )
                        return publish_blocked_audit(run, audit_path, binding, issues)
                    schema = schema_for(rules, "ehr_features_v1")
                    rows = aggregate_labs(connection, cohort, rules)
                    for row in rows:
                        row.update(static[row["sample_id"]])
                    table = float32_table(rows, schema)
                    validate_wide(table, cohort, schema)
                    patients = pq.read_table(
                        processed / "splits/burden_v2" / upstream.split.split_table_names(
                            _json_file(args.split_config))[0]
                    )
                    cv_path = outputs / "audit/burden_v2/cross_validation_indices.json"
                    cv_bundle = (
                        _json_file(cv_path)
                        if _json_file(args.split_config).get("algorithm_version")
                        == upstream.split.CV_ALGORITHM else None
                    )
                    summary, missingness = feature_reports(
                        connection, table, cohort, patients, schema, cv_bundle
                    )
                    pq.write_table(
                        table,
                        run.path(cache / "ehr_features_candidate.parquet"),
                        compression="zstd",
                    )
                    lab.copy_parquet(
                        connection,
                        "lab_trace",
                        run.path(cache / "lab_cleaning_trace.parquet"),
                    )
                    for name in SOURCE_COLUMNS:
                        lab.copy_parquet(
                            connection,
                            f"static_{name}",
                            run.path(cache / f"{name}.parquet"),
                        )
                    write_report(
                        run,
                        directory,
                        "ehr_feature_summary.csv",
                        list(summary[0]),
                        summary,
                    )
                    write_report(
                        run,
                        directory,
                        "lab_missingness.csv",
                        list(missingness[0]),
                        missingness,
                    )
                    lab.write_query_csv(
                        connection,
                        run,
                        directory / "lab_cleaning_summary.csv",
                        "SELECT feature_name,disposition,count(*) AS study_associations,count(DISTINCT lab_record_key) AS source_records FROM lab_trace GROUP BY ALL ORDER BY ALL",
                    )
                    lab.write_query_csv(
                        connection,
                        run,
                        directory / "lab_window_rejections.csv",
                        "SELECT itemid,availability_reason,ownership_reason,count(*) AS study_associations,count(DISTINCT lab_record_key) AS source_records FROM associations WHERE availability_reason<>'available' OR ownership_reason<>'owned' GROUP BY ALL ORDER BY ALL",
                    )
                    upstream.write_json(
                        run.path(directory / "feature_schema_candidate.json"), schema
                    )
                    report = (
                        "# Step 10B EHR audit\n\n"
                        f"Studies: {len(rows)}. Numeric features: {len(schema['numeric_features'])}. Categorical features: {len(CATEGORICAL)}. Blocking static conflicts: {len(issues)}.\n\n"
                        "All approved items are retained regardless of cleaned missingness. Inspect train all-missing, constant and changed coverage bands in the CSV reports. Ordinary missing values are preserved. No imputation, scaling, category vocabulary or model fitting was performed.\n\n"
                        "The lab trace binds each sample, source record, specimen, charttime, storetime and revision disposition. Window source counts and study associations have separate denominators. Reference ranges do not establish validity limits.\n\n"
                        "Service availability uses transfertime because no storetime is supplied. Deidentified ages do not recover true ages above 89. No final EHR table is published by audit.\n"
                    )
                    run.path(directory / "ehr_audit_report.md").write_text(
                        report, encoding="utf-8"
                    )
            verify_sources(sources, versions)
            if (
                upstream.source_binding(
                    context, lab_sources, args.lab_config, lab_settings
                )
                != lab_binding
            ):
                raise ValueError("Laboratory sources changed during audit")
            record = {
                "binding": binding,
                "audit_fingerprint": lab.json_hash(binding),
                "audit_checks_passed": not issues,
                "blocking_conflicts": len(issues),
                "ehr_features_ready": False,
                "run_id": run.run_id,
                "cache_table": str(cache / "ehr_features_candidate.parquet"),
                "artifact_sha256": {
                    str(target): _file_hash(stage)
                    for target, stage in run.files.items()
                },
            }
            upstream.write_json(run.path(audit_path), record)
        if args.mode == "build":
            if not record["audit_checks_passed"] or record["blocking_conflicts"]:
                raise ValueError("Unresolved audit conflicts block feature publication")
            schema = _json_file(directory / "feature_schema_candidate.json")
            schema["feature_set_version"] = args.feature_set_version
            table = pq.read_table(record["cache_table"])
            validate_wide(table, cohort, schema)
            publish_features(
                run,
                data,
                directory,
                table,
                schema,
                {**binding, "audit_manifest_sha256": _file_hash(audit_path)},
                args.feature_set_version,
            )
        verify_sources(sources, versions)
        if (
            upstream.source_binding(context, lab_sources, args.lab_config, lab_settings)
            != lab_binding
        ):
            raise ValueError("Laboratory sources changed before publication")
    print(
        "Audit complete; inspect the reports before build."
        if args.mode == "audit"
        else "EHR features frozen; Steps 11-12 remain pending."
    )
    return 0
