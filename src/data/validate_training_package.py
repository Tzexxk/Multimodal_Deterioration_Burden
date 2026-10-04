"""Independent Step 12 checks. Only provenance/publication helpers are shared."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import stat
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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

EVENTS = (
    "new_icu_ccu_transfer",
    "new_stepdown_transfer",
    "new_invasive_ventilation",
    "new_severe_hypoxemia",
    "new_iv_vaso_inotrope_infusion",
    "in_hospital_death",
)
SPLITS = ("train", "validation", "test")
AGGREGATIONS = (
    "last_24h",
    "min_24h",
    "max_24h",
    "mean_24h",
    "count_24h",
    "hours_since_last",
)
CATEGORIES = [
    "sex",
    "admission_type",
    "admission_location",
    "careunit_at_t0",
    "current_service",
]
SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        *[(k, pa.int64()) for k in ("subject_id", "hadm_id", "study_id")],
        *[(k, pa.string()) for k in ("dicom_id", "split", "image_relpath")],
        ("burden_score", pa.float32()),
        *[(k, pa.uint8()) for k in (*EVENTS, "phys_death", "high_burden")],
        ("t0", pa.timestamp("us")),
        ("observation_end", pa.timestamp("us")),
        ("observed_hours", pa.float32()),
        ("early_discharge", pa.uint8()),
    ]
)
CHECKS = (
    "provenance",
    "samples",
    "ehr_contract",
    "laboratory_reconstruction",
    "static_reconstruction",
    "images",
    "metadata_summary",
    "zip_relocation",
    "source_stability",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def code_hash():
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), root / "scripts/12_validate_training_package.py"]
    return hashlib.sha256(
        json.dumps(
            {p.relative_to(root).as_posix(): _file_hash(p) for p in files},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def timestamp(value):
    if isinstance(value, datetime):
        return value if value.tzinfo is None else None
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d{1,6})?", value
    ):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def integer(value):
    try:
        return int(value) if re.fullmatch(r"[0-9]+", str(value)) else None
    except (ValueError, TypeError):
        return None


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def indexed(table, key):
    rows = table.to_pylist()
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows) and None not in result, f"Duplicate or null {key}")
    return result


def validate_samples(samples, cohort, split, settings):
    require(
        samples.schema == SAMPLE_SCHEMA
        and samples.num_rows > 0
        and all(not c.null_count for c in samples.columns),
        "Invalid sample types or nulls",
    )
    rows = indexed(samples, "sample_id")
    original = indexed(cohort, "sample_id")
    assignments = indexed(split, "subject_id")
    require(set(rows) == set(original), "Common cohort sample keys changed")
    require(
        len({r["study_id"] for r in rows.values()}) == len(rows)
        and len({r["dicom_id"] for r in rows.values()}) == len(rows),
        "Study or image key is not unique",
    )
    require(
        set(assignments) == {r["subject_id"] for r in rows.values()},
        "Patient coverage changed",
    )
    for key, row in rows.items():
        source = original[key]
        ids = [row[k] for k in ("subject_id", "hadm_id", "study_id")]
        require(
            all(v > 0 for v in ids) and key == f"p{ids[0]}_h{ids[1]}_s{ids[2]}",
            "Invalid sample identity",
        )
        require(
            row["split"] in SPLITS and row["split"] == assignments[ids[0]]["split"],
            "Patient split changed",
        )
        expected_path = f"images/p{str(ids[0])[:2]}/p{ids[0]}/s{ids[2]}.npy"
        require(
            row["image_relpath"] == expected_path == source["npy_relpath"],
            "Image path identity changed",
        )
        for name in SAMPLE_SCHEMA.names:
            if name not in {"split", "image_relpath"}:
                require(
                    row[name] == source[name],
                    f"Frozen sample value changed: {key}/{name}",
                )
        require(
            all(
                row[k] in (0, 1)
                for k in (*EVENTS, "phys_death", "high_burden", "early_discharge")
            ),
            "Nonbinary label",
        )
        transfer = max(row[EVENTS[0]], row[EVENTS[1]])
        respiratory = max(row[EVENTS[2]], row[EVENTS[3]])
        circulatory, death = row[EVENTS[4]], row[EVENTS[5]]
        score = transfer + 2 * respiratory + 2 * circulatory + 3 * death
        require(
            row["burden_score"] == score
            and row["phys_death"] == int(bool(respiratory or circulatory or death))
            and row["high_burden"] == int(score >= 3),
            "Target formula mismatch",
        )
        end = min(row["t0"] + timedelta(hours=72), source["dischtime"])
        require(
            source["admittime"] <= row["t0"] < end and row["observation_end"] == end,
            "Observation cutoff mismatch",
        )
        hours = np.float32((end - row["t0"]).total_seconds() / 3600).item()
        require(
            row["observed_hours"] == hours
            and row["early_discharge"]
            == int(
                source["dischtime"] < row["t0"] + timedelta(hours=72)
                and str(source.get("hospital_expire_flag")) == "0"
                and "death_time_conflict" not in source.get("exclusion_reason", "")
            ),
            "Observed duration or discharge mismatch",
        )
        for event in EVENTS:
            time = source[event + "_first_event_time"]
            require(
                not row[event] or time is not None and row["t0"] < time <= end,
                "Positive event outside outcome window",
            )
    counts = {
        "studies": len(rows),
        "admissions": len({r["hadm_id"] for r in rows.values()}),
        "patients": len(assignments),
    }
    require(counts == settings["expected_counts"], "Frozen cohort counts changed")
    for group in SPLITS:
        part = [r for r in rows.values() if r["split"] == group]
        require(
            len(part) == settings["expected_split_studies"][group]
            and len({r["subject_id"] for r in part})
            == settings["expected_split_patients"][group],
            "Frozen split counts changed",
        )
    return counts


def retained_rules(dictionary):
    rules = [r for r in dictionary.to_pylist() if r["keep"]]
    require(bool(rules), "No retained dictionary rules")
    seen = set()
    for r in rules:
        key = (str(r["itemid"]), r["source_unit"])
        require(key not in seen, "Overlapping dictionary branches")
        seen.add(key)
        require(
            json.loads(r["specimen_rule"]) == {"kind": "item_identity"},
            "Unsupported specimen selector",
        )
        require(
            json.loads(r["duplicate_rule"])
            == {
                "kind": "latest_available",
                "conflicting_id": "error",
                "missing_specimen": "error",
            },
            "Unsupported duplicate selector",
        )
        require(
            json.loads(r["numeric_result_rule"])
            == {
                "kind": "valuenum_only",
                "comparator": "exclude",
                "text_disagreement": "error",
            },
            "Unsupported numeric selector",
        )
        require(
            r["valid_min"] is None
            and r["valid_max"] is None
            and bool(r["valid_range_basis"]),
            "Unexpected validity treatment",
        )
        require(
            finite(r["conversion_multiplier"]) is not None
            and finite(r["conversion_offset"]) is not None,
            "Invalid conversion",
        )
        require(
            bool(r["canonical_unit"])
            and (
                bool(r["source_unit"]) or r["missing_unit_rule"] == "explicit_item_unit"
            ),
            "Unreviewed missing unit",
        )
    return rules


def validate_ehr_contract(table, samples, schema, rules):
    numeric = [
        f"{n}_{a}"
        for n in sorted({r["feature_name"] for r in rules})
        for a in AGGREGATIONS
    ] + ["age_at_t0", "hours_from_admission"]
    require(
        schema["numeric_features"] == numeric
        and schema["categorical_features"] == CATEGORIES
        and set(schema["features"]) == set(numeric + CATEGORIES)
        and schema["key_column"] == "sample_id"
        and schema["lookback_hours"] == 24
        and schema["categorical_missing_value"] == "__MISSING__"
        and schema["preprocessing_applied"] is False,
        "Feature whitelist or missing-value contract changed",
    )
    require(
        table.column_names == ["sample_id", *numeric, *CATEGORIES]
        and table["sample_id"].type == pa.string(),
        "EHR columns changed",
    )
    require(
        set(indexed(table, "sample_id")) == set(samples["sample_id"].to_pylist()),
        "EHR alignment changed",
    )
    for name in numeric + CATEGORIES:
        col = table[name]
        require(
            col.type == (pa.float32() if name in numeric else pa.string())
            and not col.null_count,
            f"Invalid EHR type: {name}",
        )
        require(
            all(
                not math.isinf(v) if name in numeric else bool(v) and v == v.strip()
                for v in col.to_pylist()
            ),
            f"Invalid EHR value: {name}",
        )
    return {"numeric_features": len(numeric), "categorical_features": len(CATEGORIES)}


def compare_features(actual, expected, columns):
    rows = indexed(actual, "sample_id")
    require(set(rows) == set(expected), "Reconstructed feature keys changed")
    for key, source in expected.items():
        for column in columns:
            want, got = source[column], rows[key][column]
            if isinstance(want, (float, int, np.number)):
                want = np.float32(want).item()
                equal = math.isnan(want) and math.isnan(got) or want == got
                if (
                    column.endswith("_mean_24h")
                    and math.isfinite(want)
                    and math.isfinite(got)
                ):
                    equal = abs(want - got) <= abs(float(np.spacing(np.float32(want))))
            else:
                equal = want == got
            require(equal, f"Reconstruction mismatch: {key}/{column}")


def reconstruct_labs(lab_path, admission_table, cohort, rules, actual):
    """Stream the uncleaned bound subset; never trust published cleaning traces."""
    by_patient = defaultdict(list)
    for row in cohort.to_pylist():
        by_patient[row["subject_id"]].append(row)
    admissions = defaultdict(dict)
    for row in admission_table.to_pylist():
        patient, admission = integer(row["subject_id"]), integer(row["hadm_id"])
        pair = (timestamp(row["admittime"]), timestamp(row["dischtime"]))
        if patient is not None and admission is not None:
            require(
                admission not in admissions[patient]
                or admissions[patient][admission] == pair,
                "Conflicting admission intervals",
            )
            admissions[patient][admission] = pair
    rule_index = {(str(r["itemid"]), r["source_unit"]): r for r in rules}
    revisions, identifiers, selected_ids = {}, defaultdict(set), set()
    counts = Counter()
    for batch in pq.ParquetFile(lab_path).iter_batches(batch_size=8192):
        for raw in batch.to_pylist():
            counts["subset_records_scanned"] += 1
            rule = rule_index.get((raw["itemid"], raw["valueuom"]))
            patient = integer(raw["subject_id"])
            chart, stored = timestamp(raw["charttime"]), timestamp(raw["storetime"])
            if chart is None or stored is None or stored < chart:
                counts["invalid_or_missing_time"] += 1
                continue
            provided = raw["hadm_id"]
            if provided.strip():
                assigned = integer(provided)
                if assigned not in admissions[patient]:
                    counts["ownership_rejected"] += 1
                    continue
            else:
                membership = {
                    a
                    for a, (start, end) in admissions[patient].items()
                    if start is not None and end is not None and start <= chart <= end
                }
                if len(membership) != 1:
                    counts["ambiguous_or_absent_admission"] += 1
                    continue
                assigned = next(iter(membership))
            for study in by_patient.get(patient, []):
                t0, key = study["t0"], study["sample_id"]
                if (
                    assigned != study["hadm_id"]
                    or not t0 - timedelta(hours=24) <= chart <= t0
                    or stored > t0
                ):
                    continue
                identity = (key, raw["labevent_id"])
                identifiers[identity].add(raw["lab_record_key"])
                if rule is None:
                    continue
                counts["eligible_associations"] += 1
                require(
                    re.fullmatch(r"[1-9][0-9]*", raw["specimen_id"]),
                    f"Missing specimen in retained branch: {key}/{raw['lab_record_key']}",
                )
                numeric, text = finite(raw["valuenum"]), finite(raw["value"])
                require(
                    numeric is None or text is None or numeric == text,
                    f"Numeric/text disagreement in eligible revision: {key}/{raw['lab_record_key']}",
                )
                require(
                    re.fullmatch(r"[1-9][0-9]*", raw["labevent_id"]),
                    "Invalid retained labevent ID",
                )
                selected_ids.add(identity)
                group = (
                    key,
                    patient,
                    assigned,
                    raw["itemid"],
                    raw["specimen_id"],
                    chart,
                )
                order = (
                    stored,
                    integer(raw["labevent_id"]) or -1,
                    raw["lab_record_key"],
                )
                if group not in revisions or order > revisions[group][0]:
                    revisions[group] = (order, raw, rule, t0)
    require(
        all(len(identifiers[key]) == 1 for key in selected_ids),
        "Conflicting eligible labevent IDs",
    )
    values = defaultdict(list)
    for group, (order, raw, rule, t0) in revisions.items():
        numeric = finite(raw["valuenum"])
        if re.search(r"[<>≤≥]", raw["value"]) or numeric is None:
            counts["selected_nonnumeric_or_comparator"] += 1
            continue
        converted = numeric * rule["conversion_multiplier"] + rule["conversion_offset"]
        require(math.isfinite(converted), "Nonfinite converted value")
        values[(group[0], rule["feature_name"])].append(
            (group[-1], order, converted, t0)
        )
        counts["retained_associations"] += 1
    names = sorted({r["feature_name"] for r in rules})
    expected = {}
    for study in cohort.to_pylist():
        key = study["sample_id"]
        expected[key] = {}
        for name in names:
            records = values[(key, name)]
            result = [math.nan, math.nan, math.nan, math.nan, 0, math.nan]
            if records:
                last = max(records, key=lambda r: (r[0], r[1]))
                nums = [r[2] for r in records]
                result = [
                    last[2],
                    min(nums),
                    max(nums),
                    math.fsum(nums) / len(nums),
                    len(nums),
                    (study["t0"] - last[0]).total_seconds() / 3600,
                ]
            expected[key].update(
                {f"{name}_{a}": v for a, v in zip(AGGREGATIONS, result, strict=True)}
            )
    compare_features(
        actual, expected, [f"{n}_{a}" for n in names for a in AGGREGATIONS]
    )
    return dict(counts)


def reconstruct_static(cohort, evidence, patients, admissions, services, actual):
    known = indexed(evidence, "sample_id")
    p_index, a_index, s_index = defaultdict(list), defaultdict(list), defaultdict(list)
    for table, index, field in [
        (patients, p_index, "subject_id"),
        (admissions, a_index, "hadm_id"),
        (services, s_index, "hadm_id"),
    ]:
        unique = {json.dumps(r, sort_keys=True): r for r in table.to_pylist()}
        for row in unique.values():
            require(
                integer(row["subject_id"]) is not None
                and integer(row[field]) is not None,
                "Invalid static identity",
            )
            index[integer(row[field])].append(row)
    expected = {}
    for study in cohort.to_pylist():
        key, patient, admission, t0 = (
            study[k] for k in ("sample_id", "subject_id", "hadm_id", "t0")
        )
        base = known[key]
        require(
            all(
                study[k] == base[k]
                for k in (
                    "subject_id",
                    "hadm_id",
                    "t0",
                    "age_at_t0",
                    "careunit_at_t0",
                    "admittime",
                )
            ),
            "Static frozen evidence mismatch",
        )
        require(
            len(p_index[patient]) <= 1 and len(a_index[admission]) <= 1,
            "Conflicting static rows",
        )
        result = dict.fromkeys(CATEGORIES, "__MISSING__")
        result.update(
            age_at_t0=base["age_at_t0"] if base["age_at_t0"] is not None else math.nan,
            hours_from_admission=math.nan,
            careunit_at_t0=base["careunit_at_t0"] or "__MISSING__",
        )
        if p_index[patient]:
            p = p_index[patient][0]
            result["sex"] = p["gender"].strip() or "__MISSING__"
            require(result["sex"] in ("F", "M", "__MISSING__"), "Invalid sex")
            if p["anchor_age"] and p["anchor_year"]:
                require(
                    int(p["anchor_age"]) + t0.year - int(p["anchor_year"])
                    == base["age_at_t0"],
                    "Age reconstruction mismatch",
                )
            else:
                require(
                    all(
                        not p[k] or re.fullmatch(r"[0-9]+", p[k])
                        for k in ("anchor_age", "anchor_year")
                    ),
                    "Invalid nonempty anchor age/year",
                )
        if a_index[admission]:
            a = a_index[admission][0]
            start = timestamp(a["admittime"])
            require(
                integer(a["subject_id"]) == patient
                and start == base["admittime"]
                and start <= t0,
                "Admission static conflict",
            )
            result["hours_from_admission"] = (t0 - start).total_seconds() / 3600
            result.update(
                {
                    k: a[k].strip() or "__MISSING__"
                    for k in ("admission_type", "admission_location")
                }
            )
        eligible = []
        for service in s_index[admission]:
            time = timestamp(service["transfertime"])
            require(
                time is not None and integer(service["subject_id"]) == patient,
                "Invalid service identity or timestamp",
            )
            if time <= t0:
                eligible.append(
                    (time, service["curr_service"].strip() or "__MISSING__")
                )
        if eligible:
            latest = max(r[0] for r in eligible)
            options = {r[1] for r in eligible if r[0] == latest}
            require(len(options) == 1, f"Conflicting latest service: {key}")
            result["current_service"] = options.pop()
        expected[key] = result
    compare_features(
        actual, expected, ["age_at_t0", "hours_from_admission", *CATEGORIES]
    )
    return {
        "studies_reconstructed": len(expected),
        "service_availability_proxy": "transfertime <= t0",
    }


def validate_pixels(path):
    image = np.load(path, allow_pickle=False)
    require(
        image.dtype == np.dtype("float32")
        and image.shape == (224, 224)
        and np.isfinite(image).all()
        and (image >= 0).all()
        and (image <= 1).all(),
        "Invalid NPY pixels, shape or dtype",
    )


def validate_images(
    package, samples, cohort, manifest, qc, cxr, image_root, processing
):
    original, images, quality = (
        indexed(t, "sample_id") for t in (cohort, manifest, qc)
    )
    rows = samples.to_pylist()
    require(
        set(images) == {r["sample_id"] for r in rows},
        "Image manifest coverage mismatch",
    )
    for row in rows:
        key = row["sample_id"]
        source, image, checked = original[key], images[key], quality[key]
        require(
            checked["qc_status"] == "pass"
            and checked["processing_status"] == "complete",
            "Image QC not accepted",
        )
        for name in (
            "subject_id",
            "hadm_id",
            "study_id",
            "dicom_id",
            "t0",
            "npy_relpath",
            "npy_sha256",
            "source_dicom_sha256",
            "processing_config_sha256",
            "processing_version",
            "qc_status",
            "view_position",
        ):
            require(
                source[name] == image[name] == checked[name],
                f"Image provenance mismatch: {name}",
            )
        require(
            image["split"] == row["split"]
            and image["npy_relpath"] == row["image_relpath"]
            and image["processing_config_sha256"] == _file_hash(processing),
            "Image configuration or assignment changed",
        )
        dicom = _under_root(cxr, source["dicom_path"])
        npy = _under_root(image_root, row["image_relpath"])
        require(
            Path(image["dicom_path"]).resolve() == dicom
            and Path(image["npy_path"]).resolve() == npy,
            "Image source path mismatch",
        )
        require(
            _file_hash(dicom) == image["source_dicom_sha256"], "DICOM checksum changed"
        )
        target = _under_root(package, row["image_relpath"])
        require(
            _file_hash(npy) == _file_hash(target) == image["npy_sha256"],
            "NPY checksum changed",
        )
        validate_pixels(target)
    return {"images_checked": len(rows), "source_dicoms_checked": len(rows)}


def expected_summary(samples):
    result = []
    for group in ("all", *SPLITS):
        rows = [r for r in samples.to_pylist() if group == "all" or r["split"] == group]
        patients = {r["subject_id"] for r in rows}

        def add(name, count, unit, denominator):
            result.append(
                dict(
                    split=group,
                    metric=name,
                    value=count,
                    unit=unit,
                    denominator=denominator,
                )
            )

        for name, ids, unit in [
            ("studies", {r["sample_id"] for r in rows}, "study"),
            ("admissions", {r["hadm_id"] for r in rows}, "admission"),
            ("patients", patients, "patient"),
        ]:
            add(name, len(ids), unit, len(ids))
        maximum = {
            p: max(r["burden_score"] for r in rows if r["subject_id"] == p)
            for p in patients
        }
        for score in range(9):
            add(
                f"score_{score}_studies",
                sum(r["burden_score"] == score for r in rows),
                "study",
                len(rows),
            )
            add(
                f"maximum_score_{score}_patients",
                sum(v == score for v in maximum.values()),
                "patient",
                len(patients),
            )
        for event in (*EVENTS, "phys_death", "high_burden", "early_discharge"):
            add(event + "_studies", sum(r[event] for r in rows), "study", len(rows))
            add(
                event + "_patients",
                len({r["subject_id"] for r in rows if r[event]}),
                "patient",
                len(patients),
            )
    return result


def validate_metadata(
    package, samples, schema, freeze, settings, decision, split_settings, image_settings
):
    metadata = _json_file(package / "dataset_metadata.json")
    for name in (
        "interface_version",
        "release_id",
        "cohort_version",
        "split_version",
        "dictionary_version",
        "feature_set_version",
    ):
        require(metadata[name] == settings[name], f"Metadata identity changed: {name}")
    for name in (
        "target_formula",
        "hypoxemia_source",
        "live_discharge_rule",
        "outcome_version",
    ):
        require(metadata[name] == decision[name], f"Metadata endpoint mismatch: {name}")
    require(
        metadata["horizon_hours"] == 72
        and metadata["ehr_lookback_hours"] == 24
        and metadata["target_column"] == "burden_score"
        and metadata["target_range"] == [0, 8]
        and metadata["split_column"] == "split"
        and metadata["split_values"] == list(SPLITS)
        and metadata["split_unit"] == "subject_id",
        "Metadata interface changed",
    )
    require(
        metadata["ehr_preprocessing_applied"] is False
        and schema["preprocessing_applied"] is False
        and metadata["independent_validation_completed"] is False,
        "Historical export status or preprocessing changed",
    )
    require(
        metadata["image"]
        == {
            "format": "npy",
            "dtype": "float32",
            "shape": [224, 224],
            "value_range": [0.0, 1.0],
            "model_normalization_applied": False,
        },
        "Image contract changed",
    )
    require(
        all(metadata["image_config"][k] == v for k, v in image_settings.items()),
        "Image settings changed",
    )
    require(
        metadata["split_rules"]
        == {
            k: split_settings[k]
            for k in (
                "algorithm_version",
                "seed",
                "weights",
                "primary_stratum",
                "require_high_burden_in_each_split",
            )
        },
        "Split settings changed",
    )
    require(
        metadata["configuration_sha256"] == freeze["binding"]["configuration_sha256"]
        and metadata["upstream_file_sha256"] == freeze["binding"]["source_sha256"]
        and metadata["export_code_sha256"] == freeze["binding"]["code_sha256"]
        and metadata["file_sha256"]
        == {
            k: v
            for k, v in freeze["package_file_sha256"].items()
            if k != "dataset_metadata.json"
        },
        "Metadata checksum binding changed",
    )
    require(
        metadata["feature_counts"]
        == {
            "numeric": len(schema["numeric_features"]),
            "categorical": len(schema["categorical_features"]),
        },
        "Metadata feature counts changed",
    )
    summaries = expected_summary(samples)
    with (package / "summary.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(
            reader.fieldnames == ["split", "metric", "value", "unit", "denominator"],
            "Summary header changed",
        )
        actual = list(reader)
    require(
        actual == [{k: str(v) for k, v in r.items()} for r in summaries],
        "Summary values changed",
    )
    counts = {
        g: {
            r["metric"]: r["value"]
            for r in summaries
            if r["split"] == g and r["metric"] in ("studies", "admissions", "patients")
        }
        for g in ("all", *SPLITS)
    }
    require(metadata["split_counts"] == counts, "Metadata split counts changed")

    def portable(value):
        if isinstance(value, dict):
            for item in value.values():
                portable(item)
        elif isinstance(value, list):
            for item in value:
                portable(item)
        elif isinstance(value, str):
            require(
                not re.search(r"(?:[A-Za-z]:[/\\]|\\\\|^/)", value),
                "Absolute path in portable metadata",
            )

    portable(metadata)
    portable(schema)
    return {"summary_rows_checked": len(summaries)}


def relocate_archive(archive, package, release, hashes):
    """Validate every member before extracting into an owned temporary directory."""
    import zipfile

    with zipfile.ZipFile(archive) as zipped:
        infos = zipped.infolist()
        expected = {f"{release}/{name}": name for name in hashes}
        require(
            len(infos) == len(expected)
            and {r.filename for r in infos} == set(expected)
            and len({r.filename.casefold() for r in infos}) == len(infos),
            "ZIP duplicate or unexpected inventory",
        )
        for member in infos:
            require(
                "\\" not in member.filename
                and ":" not in member.filename
                and not member.is_dir()
                and not member.flag_bits & 1
                and stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFREG),
                "Unsafe ZIP entry type or encryption",
            )
            name = expected[member.filename]
            path = _under_root(package, name)
            require(
                member.filename == f"{release}/{path.relative_to(package).as_posix()}"
                and member.file_size == path.stat().st_size,
                "Unsafe ZIP path or decompressed size",
            )
        with tempfile.TemporaryDirectory(prefix="icu-risk-step12-") as temporary:
            root = Path(temporary)
            for member in infos:
                target = _under_root(root, member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("wb") as sink:
                    remaining = member.file_size
                    while chunk := source.read(min(1024 * 1024, remaining + 1)):
                        remaining -= len(chunk)
                        require(remaining >= 0, "ZIP expands beyond declared bound")
                        sink.write(chunk)
                require(
                    remaining == 0
                    and _file_hash(target) == hashes[expected[member.filename]],
                    "ZIP member checksum mismatch",
                )
            data_root = root / release
            samples = pq.read_table(data_root / "samples.parquet")
            features = pq.read_table(data_root / "ehr_features.parquet")
            schema = _json_file(data_root / "feature_schema.json")
            require(
                samples.equals(pq.read_table(package / "samples.parquet")),
                "Relocated samples changed",
            )
            aligned = indexed(features, "sample_id")
            require(
                set(aligned) == set(samples["sample_id"].to_pylist()),
                "Relocated features do not align",
            )
            columns = schema["numeric_features"] + schema["categorical_features"]
            for row in samples.to_pylist():
                require(
                    all(c in aligned[row["sample_id"]] for c in columns),
                    "Relocated whitelist unavailable",
                )
                validate_pixels(_under_root(data_root, row["image_relpath"]))
            counts = {
                "relocated_members": len(infos),
                "relocated_studies": samples.num_rows,
                "relocated_images": samples.num_rows,
                "data_root_only": True,
                "temporary_directory_removed": True,
            }
        require(not root.exists(), "Relocation temporary directory was not removed")
        return counts


def load_settings(path):
    settings = _json_file(path)
    expected = {
        "validation_version",
        "interface_version",
        "release_id",
        "cohort_version",
        "split_version",
        "dictionary_version",
        "feature_set_version",
        "expected_counts",
        "expected_split_studies",
        "expected_split_patients",
        "relocation_policy",
        "failure_detail_limit",
    }
    require(
        set(settings) == expected
        and settings["interface_version"] == "1.0"
        and settings["relocation_policy"] == "temporary_directory"
        and type(settings["failure_detail_limit"]) is int
        and 1 <= settings["failure_detail_limit"] <= 100,
        "Invalid independent validation configuration",
    )
    for name in (
        "validation_version",
        "release_id",
        "cohort_version",
        "split_version",
        "dictionary_version",
        "feature_set_version",
    ):
        require(
            isinstance(settings[name], str)
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", settings[name]),
            "Unsafe validation identity",
        )
    for field, keys in [
        ("expected_counts", {"studies", "admissions", "patients"}),
        ("expected_split_studies", set(SPLITS)),
        ("expected_split_patients", set(SPLITS)),
    ]:
        require(
            set(settings[field]) == keys
            and all(type(v) is int and v > 0 for v in settings[field].values()),
            "Invalid expected counts",
        )
    return settings


def verify_provenance(root, args, settings, processed, outputs):
    """Verify immutable records directly, without upstream business validators."""
    from src.cohort import create_patient_split, freeze_cohort
    from src.data import export_training_package
    from src.ehr import build_ehr_features, step10b

    watched, raw_versions = {}, {}

    def digest(path, expected=None):
        path = Path(path).resolve()
        require(
            path.is_relative_to(processed.parent.resolve())
            or path.is_relative_to(outputs.resolve())
            or path.is_relative_to(root.resolve()),
            "Evidence path outside project",
        )
        value = _file_hash(path)
        require(
            expected is None or value == expected,
            f"Evidence checksum changed: {path.name}",
        )
        watched[str(path)] = value
        return value

    def record(path):
        digest(path)
        return _json_file(path)

    for step in range(1, 12):
        if not (outputs / f"audit/step_{step:02d}_manifest.json").is_file():
            raise FileNotFoundError(f"Missing Step {step:02d} manifest")
        _verify_run(outputs, step)
        manifest = record(outputs / f"audit/step_{step:02d}_manifest.json")
        for path, expected in manifest["files"].items():
            digest(path, expected)
    for mode in ("audit", "build"):
        namespace = outputs / "step_10b" / mode
        if not (namespace / "audit/step_10_manifest.json").is_file():
            raise FileNotFoundError(f"Missing Step 10B {mode} manifest")
        _verify_run(namespace, 10)
        manifest = record(namespace / "audit/step_10_manifest.json")
        for path, expected in manifest["files"].items():
            digest(path, expected)
    audit = outputs / "audit/burden_v2"
    data = processed / "ehr/burden_v2"
    release = settings["release_id"]
    package = processed / "training" / release
    archive = processed / "training" / f"{release}.zip"
    package_record = record(audit / "training/training_package_freeze.json")
    require(
        package_record["release_id"] == release
        and package_record["training_package_exported"] is True,
        "Package is not frozen",
    )
    package_hash = digest(audit / "training/training_package_freeze.json")
    digest(
        audit / "training/releases" / release / "training_package_freeze.json",
        package_hash,
    )
    registry = record(audit / "training/training_package_releases.json")
    require(
        registry[release]["freeze_sha256"] == package_hash, "Package registry mismatch"
    )
    export = _json_file(args.export_config)
    require(
        package_record["binding"]["settings"] == export
        and all(export[k] == settings[k] for k in export),
        "Configured package identities changed",
    )
    require(
        package_record["binding"]["code_sha256"] == export_training_package.code_hash(),
        "Frozen export code changed",
    )
    configs = [
        args.paths,
        args.cohort,
        args.ehr,
        args.imaging,
        args.processing,
        args.split_config,
        args.lab_config,
        args.feature_config,
        args.export_config,
    ]
    require(
        {p.name: digest(p) for p in configs}
        == package_record["binding"]["configuration_sha256"],
        "Frozen configuration changed",
    )
    digest(args.validation_config)
    files = {
        "cohort": processed / "cohort/burden_v2/cohort.parquet",
        "cohort_freeze": audit / "cohort_freeze.json",
        "split_freeze": audit / "split_freeze.json",
        "endpoint_decision": audit / "endpoint_decision.json",
        "split": processed / "splits/burden_v2/patient_split.parquet",
        "ehr": data / "ehr_features_raw.parquet",
        "schema": data / "feature_schema.json",
        "feature_freeze": audit
        / "ehr"
        / _json_file(args.feature_config)["audit_version"]
        / "ehr_features_freeze.json",
        "dictionary": data / "lab_dictionary.parquet",
        "dictionary_freeze": audit / "lab_dictionary_freeze.json",
        "lab_audit": audit
        / "labs"
        / _json_file(args.lab_config)["audit_version"]
        / "lab_audit_manifest.json",
    }
    require(
        set(files) == set(package_record["binding"]["source_sha256"]),
        "Incomplete package source binding",
    )
    for name, path in files.items():
        digest(path, package_record["binding"]["source_sha256"][name])
    frozen_cohort, frozen_split = (
        record(files["cohort_freeze"]),
        record(files["split_freeze"]),
    )
    require(
        frozen_cohort["common_cohort_frozen"] is True
        and frozen_cohort["release_version"] == settings["cohort_version"]
        and frozen_cohort["horizon_hours"] == 72
        and frozen_cohort["source_binding"]["code_sha256"]
        == freeze_cohort._code_hash(),
        "Cohort freeze identity changed",
    )
    require(
        frozen_split["patient_split_frozen"] is True
        and frozen_split["split_version"] == settings["split_version"]
        and frozen_split["cohort_version"] == settings["cohort_version"]
        and frozen_split["source_binding"]["code_sha256"]
        == create_patient_split._code_hash()
        and frozen_split["source_binding"]["split_config_sha256"]
        == digest(args.split_config),
        "Split freeze identity changed",
    )
    split_settings = _json_file(args.split_config)
    require(
        split_settings["seed"] == 42 and split_settings["weights"] == [70, 15, 15],
        "Frozen split policy changed",
    )
    for frozen, version, table_dir, table_names, freeze_name in [
        (
            frozen_cohort,
            settings["cohort_version"],
            processed / "cohort/burden_v2",
            {"cohort.parquet", "cohort_audit.parquet"},
            "cohort_freeze.json",
        ),
        (
            frozen_split,
            settings["split_version"],
            processed / "splits/burden_v2",
            {"patient_split.parquet"},
            "split_freeze.json",
        ),
    ]:
        digest(audit / "releases" / version / freeze_name, digest(audit / freeze_name))
        for name, expected in frozen["artifact_sha256"].items():
            base = table_dir if name in table_names else audit
            digest(_under_root(base, name), expected)
            digest(_under_root(base / "releases" / version, name), expected)
    lab_audit = record(files["lab_audit"])
    lab_binding = lab_audit["source_binding"]
    feature_freeze = record(files["feature_freeze"])
    feature_audit_path = files["feature_freeze"].parent / "ehr_audit_manifest.json"
    feature_audit = record(feature_audit_path)
    require(
        feature_audit["audit_checks_passed"] is True
        and not feature_audit["blocking_conflicts"]
        and feature_freeze["ehr_features_ready"] is True
        and feature_freeze["feature_set_version"] == settings["feature_set_version"]
        and feature_freeze["binding"]
        == {
            **feature_audit["binding"],
            "audit_manifest_sha256": digest(feature_audit_path),
        },
        "Feature audit/freeze binding mismatch",
    )
    require(
        feature_audit["binding"]["code_sha256"] == step10b.code_hash()
        and feature_audit["binding"]["configuration_sha256"]
        == digest(args.feature_config)
        and lab_binding["code_sha256"] == build_ehr_features.code_hash()
        and lab_binding["configuration_sha256"] == digest(args.lab_config),
        "Frozen audit code/configuration changed",
    )
    for evidence in (lab_audit, feature_audit):
        binding = evidence.get("binding", evidence.get("source_binding"))
        require(
            binding["cohort"] == package_record["binding"]["cohort"]
            and binding["split"] == package_record["binding"]["split"],
            "Audit cohort/split identity mismatch",
        )
        for path, expected in evidence["artifact_sha256"].items():
            digest(path, expected)
        for name, version in binding["source_versions"].items():
            path = Path(binding["source_paths"][name])
            require(_file_version(path) == version, f"Stale source: {name}")
            raw_versions[str(path)] = version
    raw_root = _resolve_config_path(_json_file(args.paths)["mimic_iv"], root, raw=True)
    decision = record(files["endpoint_decision"])
    for name, version in decision["source_audit"]["input_versions"].items():
        path = _under_root(raw_root, _json_file(args.ehr)["burden_v2"]["tables"][name])
        require(
            _file_version(path) == version, f"Frozen outcome source changed: {name}"
        )
        raw_versions[str(path)] = version
    require(
        decision["horizon_hours"] == 72
        and decision["horizon_frozen"] is True
        and decision["label_rules_frozen"] is True,
        "Outcome decision changed",
    )
    require(
        _file_hash(Path(lab_binding["source_paths"]["d_labitems"]))
        == lab_binding["dictionary_source_sha256"],
        "Raw lab dictionary changed",
    )
    dictionary_freeze = record(files["dictionary_freeze"])
    dictionary_registry = record(audit / "lab_dictionary_releases.json")[
        settings["dictionary_version"]
    ]
    require(
        dictionary_freeze["dictionary_frozen"] is True
        and dictionary_freeze["dictionary_version"] == settings["dictionary_version"]
        and dictionary_freeze["dictionary_sha256"] == digest(files["dictionary"])
        and dictionary_freeze["source_binding"]["audit_manifest_sha256"]
        == digest(files["lab_audit"])
        and dictionary_freeze["source_binding"]["audit_fingerprint"]
        == lab_audit["audit_fingerprint"]
        and dictionary_freeze["source_binding"]["code_sha256"]
        == build_ehr_features.code_hash(),
        "Dictionary binding changed",
    )
    digest(
        dictionary_registry["dictionary_path"], dictionary_freeze["dictionary_sha256"]
    )
    digest(dictionary_registry["freeze_path"], digest(files["dictionary_freeze"]))
    require(
        dictionary_registry["freeze_sha256"] == digest(files["dictionary_freeze"]),
        "Dictionary registry mismatch",
    )
    for path, expected in dictionary_freeze["source_binding"][
        "review_files_sha256"
    ].items():
        digest(path, expected)
    feature_snapshot = data / "releases" / settings["feature_set_version"]
    digest(
        feature_snapshot / "ehr_features_freeze.json", digest(files["feature_freeze"])
    )
    require(
        record(audit / "ehr/ehr_features_releases.json")[
            settings["feature_set_version"]
        ]["freeze_sha256"]
        == digest(files["feature_freeze"]),
        "Feature registry mismatch",
    )
    for name, expected in feature_freeze["artifact_sha256"].items():
        digest(data / name, expected)
        digest(feature_snapshot / name, expected)
    for name, field in [
        ("dictionary", "dictionary_sha256"),
        ("dictionary_freeze", "dictionary_freeze_sha256"),
        ("lab_audit", "lab_audit_sha256"),
    ]:
        require(
            feature_audit["binding"][field] == digest(files[name]),
            "Feature evidence binding changed",
        )
    inventory = {
        p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()
    }
    require(
        inventory == set(package_record["package_file_sha256"]),
        "Package file inventory changed",
    )
    require(
        not any(p.is_symlink() for p in package.rglob("*")),
        "Package symlink is forbidden",
    )
    for name, expected in package_record["package_file_sha256"].items():
        digest(_under_root(package, name), expected)
    digest(archive, package_record["zip_sha256"])
    image_manifest = processed / "imaging/burden_v2/image_manifest.parquet"
    digest(image_manifest, package_record["image_manifest_sha256"])
    digest(
        audit / "training/releases" / release / "image_manifest.parquet",
        package_record["image_manifest_sha256"],
    )
    return dict(
        package=package,
        archive=archive,
        freeze=package_record,
        watched=watched,
        raw_versions=raw_versions,
        provenance_code_hashes={
            "cohort": freeze_cohort._code_hash(),
            "split": create_patient_split._code_hash(),
            "labs": build_ehr_features.code_hash(),
            "ehr": step10b.code_hash(),
            "export": export_training_package.code_hash(),
        },
        raw_hashes={
            lab_binding["source_paths"]["d_labitems"]: lab_binding[
                "dictionary_source_sha256"
            ]
        },
        files=files,
        lab_audit=lab_audit,
        static_cache=Path(feature_audit["cache_table"]).parent,
        cohort=pq.read_table(files["cohort"]),
        split=pq.read_table(files["split"]),
        samples=pq.read_table(package / "samples.parquet"),
        ehr=pq.read_table(package / "ehr_features.parquet"),
        schema=_json_file(package / "feature_schema.json"),
        rules=retained_rules(pq.read_table(files["dictionary"])),
        evidence=pq.read_table(processed / "cohort/burden_v2/cohort_audit.parquet"),
        image_manifest=image_manifest,
        processed=processed,
        outputs=outputs,
        decision=decision,
        split_settings=split_settings,
        image_settings=_json_file(args.processing),
    )


def verify_stability(context, initial_code):
    require(
        all(_file_hash(Path(p)) == digest for p, digest in context["watched"].items()),
        "Evidence changed during validation",
    )
    require(
        all(
            _file_version(Path(p)) == version
            for p, version in context["raw_versions"].items()
        ),
        "Raw source changed during validation",
    )
    require(code_hash() == initial_code, "Validator code changed during validation")
    if "provenance_code_hashes" in context:
        from src.cohort import create_patient_split, freeze_cohort
        from src.data import export_training_package
        from src.ehr import build_ehr_features, step10b

        require(
            context["provenance_code_hashes"]
            == {
                "cohort": freeze_cohort._code_hash(),
                "split": create_patient_split._code_hash(),
                "labs": build_ehr_features.code_hash(),
                "ehr": step10b.code_hash(),
                "export": export_training_package.code_hash(),
            },
            "Upstream code changed during validation",
        )
    for path, digest in context.get("raw_hashes", {}).items():
        require(
            _file_hash(Path(path)) == digest, "Raw dictionary changed during validation"
        )
    require(
        {
            p.relative_to(context["package"]).as_posix()
            for p in context["package"].rglob("*")
            if p.is_file()
        }
        == set(context["freeze"]["package_file_sha256"]),
        "Package inventory changed during validation",
    )
    for row in pq.read_table(context["image_manifest"]).to_pylist():
        require(
            _file_hash(Path(row["npy_path"])) == row["npy_sha256"]
            and _file_hash(Path(row["dicom_path"])) == row["source_dicom_sha256"],
            "Source image changed during validation",
        )
    return {
        "evidence_files_rechecked": len(context["watched"]),
        "raw_source_versions_rechecked": len(context["raw_versions"]),
    }


def publish_reports(run, outputs, settings, checks, binding):
    for entry in checks.values():
        entry["affected_rows"] = 0 if entry["status"] == "pass" else None
        entry["affected_rows_count_status"] = (
            "complete" if entry["status"] == "pass" else "not_computed"
        )
    passed = all(checks[k]["status"] == "pass" for k in CHECKS)
    directory = outputs / "audit/burden_v2"
    report = {
        "validation_version": settings["validation_version"],
        "release_id": settings["release_id"],
        "run_id": run.run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "independent_validation_completed": all(
            checks[k]["status"] not in ("blocked",) for k in CHECKS
        ),
        "validation_passed": passed,
        "training_package_accepted": passed,
        "binding": binding,
    }
    history = (
        directory
        / "validation"
        / settings["validation_version"]
        / settings["release_id"]
        / run.run_id
    )
    write_json(run.path(history / "validation_report.json"), report)
    write_json(run.path(directory / "leakage_checks.json"), report)
    write_json(run.path(directory / "training_package_acceptance.json"), report)
    write_json(run.path(history / "training_package_acceptance.json"), report)
    columns = ["check", "status", "mandatory", "affected_rows", "detail", "evidence"]
    rows = [
        dict(
            check=k,
            status=v["status"],
            mandatory=k in CHECKS,
            affected_rows=0 if v["status"] == "pass" else None,
            detail=v.get("detail", ""),
            evidence=json.dumps(v.get("evidence", {}), sort_keys=True),
        )
        for k, v in checks.items()
    ]
    _write_csv(run.path(directory / "cross_file_checks.csv"), columns, rows)
    _write_csv(run.path(history / "cross_file_checks.csv"), columns, rows)
    failures = [
        dict(check=r["check"], status=r["status"], detail=r["detail"])
        for r in rows
        if r["status"] in ("fail", "blocked")
    ][: settings["failure_detail_limit"]]
    _write_csv(
        run.path(history / "failure_details.csv"),
        ["check", "status", "detail"],
        failures,
    )
    text = (
        "# Independent training validation\n\n"
        + f"Release: {settings['release_id']}. Run: {run.run_id}.\n\nAccepted: {passed}.\n\n"
    )
    text += "\n".join(f"- {r['check']}: {r['status']}. {r['detail']}" for r in rows)
    text += "\n\nFrozen export metadata remains a historical Step 11 record. This external acceptance applies only to the exact bound package bytes. Model preprocessing is not yet applicable. Failed checks stop at their first diagnostic; a blank affected-row count means the total was not computed.\n"
    run.path(history / "validation_report.md").write_text(text, encoding="utf-8")
    write_json(run.path(directory / "step_12_run_metadata.json"), report)
    return 0 if passed else 1


def run_validation(root, args):
    settings = load_settings(args.validation_config)
    paths = _json_file(args.paths)
    outputs = _resolve_config_path(paths["outputs"], root, raw=False)
    processed = _resolve_config_path(paths["processed"], root, raw=False)
    _protect_outputs(paths, root, outputs)
    require(
        not outputs.resolve().is_relative_to((processed / "training").resolve()),
        "Reports must not be inside package storage",
    )
    initial_code = code_hash()
    checks = {
        k: {"status": "blocked", "detail": "Prerequisite checks not completed"}
        for k in CHECKS
    }
    checks["train_only_model_preprocessing"] = {
        "status": "not_applicable",
        "detail": "No fitted model preprocessor exists; train medians/scaling/vocabularies and validation thresholds are deferred to modeling",
    }
    binding = {
        "validator_code_sha256": initial_code,
        "validation_configuration_sha256": _file_hash(args.validation_config),
    }

    def check(name, operation):
        print(f"Step 12: {name}", flush=True)
        try:
            evidence = operation()
            checks[name] = {"status": "pass", "evidence": evidence or {}}
            return True
        except Exception as error:
            checks[name] = {
                "status": "blocked" if isinstance(error, FileNotFoundError) else "fail",
                "detail": f"{type(error).__name__}: {error}"[:2000],
            }
            return False

    # Enter recovery before reading evidence. Failed validation still publishes a complete report transaction.
    with _OutputRun(outputs, 12) as run:
        context = {}

        def provenance():
            context.update(verify_provenance(root, args, settings, processed, outputs))
            binding.update(
                package_freeze_sha256=_file_hash(
                    outputs / "audit/burden_v2/training/training_package_freeze.json"
                ),
                zip_sha256=context["freeze"]["zip_sha256"],
                package_file_sha256=context["freeze"]["package_file_sha256"],
                upstream_binding=context["freeze"]["binding"],
                evidence_file_sha256=context["watched"],
                source_versions=context["raw_versions"],
            )
            return {
                "completed_steps": list(range(1, 12)),
                "step_10b_audit_and_build": True,
            }

        if check("provenance", provenance):
            cohort, samples, actual, rules = (
                context[k] for k in ("cohort", "samples", "ehr", "rules")
            )
            check(
                "samples",
                lambda: validate_samples(samples, cohort, context["split"], settings),
            )
            contract_ok = check(
                "ehr_contract",
                lambda: validate_ehr_contract(
                    actual, samples, context["schema"], rules
                ),
            )
            cache = context["static_cache"]
            if contract_ok:
                check(
                    "laboratory_reconstruction",
                    lambda: reconstruct_labs(
                        Path(context["lab_audit"]["cache_paths"]["labs"]),
                        pq.read_table(
                            context["lab_audit"]["cache_paths"]["admissions"]
                        ),
                        cohort,
                        rules,
                        actual,
                    ),
                )
                check(
                    "static_reconstruction",
                    lambda: reconstruct_static(
                        cohort,
                        context["evidence"],
                        pq.read_table(cache / "patients.parquet"),
                        pq.read_table(cache / "admissions.parquet"),
                        pq.read_table(cache / "services.parquet"),
                        actual,
                    ),
                )
            check(
                "images",
                lambda: validate_images(
                    context["package"],
                    samples,
                    cohort,
                    pq.read_table(context["image_manifest"]),
                    pq.read_table(
                        processed / "imaging/burden_v2/image_qc_status.parquet"
                    ),
                    _resolve_config_path(paths["mimic_cxr"], root, raw=True),
                    processed / "imaging/burden_v2",
                    args.processing,
                ),
            )
            check(
                "metadata_summary",
                lambda: validate_metadata(
                    context["package"],
                    samples,
                    context["schema"],
                    context["freeze"],
                    settings,
                    context["decision"],
                    context["split_settings"],
                    context["image_settings"],
                ),
            )
            check(
                "zip_relocation",
                lambda: relocate_archive(
                    context["archive"],
                    context["package"],
                    settings["release_id"],
                    context["freeze"]["package_file_sha256"],
                ),
            )
            check("source_stability", lambda: verify_stability(context, initial_code))
        result = publish_reports(run, outputs, settings, checks, binding)
    print(f"Step 12 reports published; training package accepted: {result == 0}")
    return result
