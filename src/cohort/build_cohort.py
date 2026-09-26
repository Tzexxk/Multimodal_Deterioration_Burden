"""Build the 48-hour candidate cohort audit without freezing model inputs."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.cohort.apply_exclusion import (
    build_icu_timeline,
    cross_audit,
    exclusion_reason,
    icu_signature,
)
from src.cohort.audit_raw_data import (
    _config_hash,
    _file_hash,
    _git_commit,
    _json_file,
    _resolve_config_path,
    _under_root,
)
from src.cohort.build_labels import comparable, evaluate_label, parse_time

ICU_CAREUNITS = frozenset(
    {
        "Surgical Intensive Care Unit (SICU)",
        "Medical Intensive Care Unit (MICU)",
        "Cardiac Vascular Intensive Care Unit (CVICU)",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)",
        "Coronary Care Unit (CCU)",
        "Trauma SICU (TSICU)",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)",
    }
)
MATCH_COLUMNS = {
    "subject_id",
    "study_id",
    "hadm_id",
    "dicom_id",
    "t0",
    "admittime",
    "dischtime",
    "match_status",
}
SELECTION_COLUMNS = {
    "subject_id",
    "study_id",
    "selected_dicom_id",
    "dicom_path",
    "view_position",
    "acquisition_datetime",
    "acquisition_time_source",
}
INDEX_COLUMNS = {"subject_id", "study_id", "dicom_id", "is_readable", "is_frontal"}

AUDIT_SCHEMA = pa.schema(
    [
        pa.field("subject_id", pa.int64(), nullable=False),
        pa.field("hadm_id", pa.int64()),
        pa.field("study_id", pa.int64(), nullable=False),
        pa.field("dicom_id", pa.string(), nullable=False),
        pa.field("dicom_path", pa.string(), nullable=False),
        pa.field("sample_id", pa.string()),
        pa.field("t0", pa.string(), nullable=False),
        pa.field("age", pa.int32()),
        pa.field("sex", pa.string()),
        pa.field("admittime", pa.string()),
        pa.field("dischtime", pa.string()),
        pa.field("deathtime", pa.string()),
        pa.field("hospital_expire_flag", pa.int8()),
        pa.field("view_position", pa.string()),
        pa.field("first_icu_time", pa.string()),
        pa.field("first_icu_transfer_id", pa.int64()),
        pa.field("icustays_first_icu_time", pa.string()),
        pa.field("icustays_first_stay_id", pa.int64()),
        pa.field("transfer_icu_class", pa.string(), nullable=False),
        pa.field("icustays_icu_class", pa.string(), nullable=False),
        pa.field("icu_mixed_stay_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("hours_to_icu", pa.float64()),
        pa.field("label", pa.int8()),
        pa.field("label_status", pa.string(), nullable=False),
        pa.field("prediction_window", pa.string(), nullable=False),
        pa.field("prediction_max_hours", pa.int16(), nullable=False),
        pa.field("eligible_age", pa.bool_(), nullable=False),
        pa.field("eligible_admission_match", pa.bool_(), nullable=False),
        pa.field("eligible_frontal", pa.bool_(), nullable=False),
        pa.field("icu_at_t0", pa.bool_(), nullable=False),
        pa.field("prior_icu", pa.bool_(), nullable=False),
        pa.field("icu_within_6h", pa.bool_(), nullable=False),
        pa.field("death_within_H_without_icu", pa.bool_(), nullable=False),
        pa.field("dicom_valid", pa.bool_(), nullable=False),
        pa.field("icu_timeline_valid", pa.bool_(), nullable=False),
        pa.field("icu_timeline_issue", pa.string()),
        pa.field("icu_source_disagreement", pa.bool_(), nullable=False),
        pa.field("icu_source_changes_decision", pa.bool_(), nullable=False),
        pa.field("icu_audit_status", pa.string(), nullable=False),
        pa.field("final_eligible", pa.bool_(), nullable=False),
        pa.field("exclusion_reason", pa.string()),
    ]
)


def _temp_path(directory: Path, suffix: str) -> Path:
    handle, name = tempfile.mkstemp(prefix=".cohort_05a_", suffix=suffix, dir=directory)
    os.close(handle)
    return Path(name)


def _load_parquet(path: Path, required: set[str]) -> list[dict[str, Any]]:
    source = pq.ParquetFile(path)
    missing = required - set(source.schema_arrow.names)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {', '.join(sorted(missing))}"
        )
    result: list[dict[str, Any]] = []
    for batch in source.iter_batches(batch_size=8192, columns=sorted(required)):
        result.extend(batch.to_pylist())
    return result


def _unique_studies(
    rows: list[dict[str, Any]], name: str
) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["subject_id"], row["study_id"])
        if type(key[0]) is not int or type(key[1]) is not int or key in result:
            raise ValueError(f"Invalid or duplicate study key in {name}: {key}")
        result[key] = row
    return result


def _read_relevant_csv(
    path: Path,
    required: set[str],
    filter_field: str,
    selected_ids: set[int],
    unique_field: str | None,
) -> dict[int, Any]:
    result: dict[int, Any] = {} if unique_field else defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {', '.join(sorted(missing))}"
            )
        with tqdm(
            desc=f"Read {path.name}",
            unit="row",
            unit_scale=True,
            file=sys.stderr,
            disable=not sys.stderr.isatty(),
        ) as progress:
            for line_number, row in enumerate(reader, 2):
                progress.update(1)
                if None in row:
                    raise ValueError(f"Malformed {path.name} row at line {line_number}")
                try:
                    filter_id = int(row[filter_field])
                except (TypeError, ValueError):
                    continue
                if filter_id not in selected_ids:
                    continue
                if unique_field:
                    if filter_id in result:
                        raise ValueError(
                            f"Duplicate {unique_field} in {path.name}: {filter_id}"
                        )
                    result[filter_id] = row
                else:
                    result[filter_id].append(row)
    return result


def _patient_age(patient: dict[str, str] | None, t0: datetime | None) -> int | None:
    if patient is None or t0 is None:
        return None
    try:
        anchor_age = int(patient["anchor_age"])
        anchor_year = int(patient["anchor_year"])
    except (TypeError, ValueError):
        return None
    age = anchor_age + t0.year - anchor_year
    return age if age >= 0 else None


def _expire_flag(admission: dict[str, str] | None) -> int | None:
    if admission is None:
        return None
    try:
        value = int(admission["hospital_expire_flag"])
    except (TypeError, ValueError):
        return None
    return value if value in (0, 1) else None


def _assess_study(
    match: dict[str, Any],
    selected: dict[str, Any],
    indexed: dict[str, Any],
    patient: dict[str, str] | None,
    admission: dict[str, str] | None,
    transfers: list[dict[str, str]],
    stays: list[dict[str, str]],
    careunits: frozenset[str],
    min_hours: int,
    max_hours: int,
) -> dict[str, Any]:
    subject_id, study_id = match["subject_id"], match["study_id"]
    if (
        selected["selected_dicom_id"] != match["dicom_id"]
        or selected["acquisition_datetime"] != match["t0"]
        or indexed["dicom_id"] != match["dicom_id"]
    ):
        raise ValueError(f"Stale image or admission match for study {study_id}")
    t0 = parse_time(match["t0"])
    if t0 is None:
        raise ValueError(f"Invalid t0 for study {study_id}")
    age = _patient_age(patient, t0)
    matched = match["match_status"] == "unique_match" and match["hadm_id"] is not None
    if matched and admission is None:
        raise ValueError(
            f"Matched admission is missing from raw CSV: {match['hadm_id']}"
        )
    if matched:
        assert admission is not None
        if (
            int(admission["subject_id"]) != subject_id
            or admission["admittime"] != match["admittime"]
            or admission["dischtime"] != match["dischtime"]
        ):
            raise ValueError(
                f"Admission source changed since step 04: {match['hadm_id']}"
            )
    admittime = parse_time(match["admittime"]) if matched else None
    dischtime = parse_time(match["dischtime"]) if matched else None
    valid_admission = bool(
        matched
        and admittime
        and dischtime
        and comparable(admittime, t0, dischtime)
        and admittime <= t0 < dischtime
    )
    deathtime_text: str | None = None
    if admission is not None:
        deathtime_text = admission["deathtime"] or None
    deathtime = parse_time(deathtime_text)
    expire_flag = _expire_flag(admission)

    transfer_timeline = build_icu_timeline(transfers, careunits, "transfers")
    stay_timeline = build_icu_timeline(stays, careunits, "icustays")
    signature = (
        icu_signature(transfer_timeline, t0, min_hours, max_hours)
        if valid_admission
        else "unknown"
    )
    stay_signature = (
        icu_signature(stay_timeline, t0, min_hours, max_hours)
        if valid_admission
        else "unknown"
    )
    timeline_valid = bool(transfers) and signature != "unknown"
    if not transfers and valid_admission:
        timeline_issue = "missing_transfer_rows"
    elif transfer_timeline.invalid_reason:
        timeline_issue = transfer_timeline.invalid_reason
    elif valid_admission and signature == "unknown":
        timeline_issue = "incomparable_or_open_icu_interval"
    else:
        timeline_issue = None
    if valid_admission and timeline_valid:
        audit_status, changed = cross_audit(
            transfer_timeline,
            stay_timeline,
            t0,
            min_hours,
            max_hours,
            dischtime,
            deathtime,
            expire_flag,
        )
    else:
        audit_status, changed = "not_assessed", False
    first_icu = transfer_timeline.first_time if valid_admission else None
    stay_first_icu = stay_timeline.first_time if valid_admission else None
    label_result = (
        evaluate_label(
            t0,
            dischtime,
            first_icu,
            deathtime,
            expire_flag,
            min_hours,
            max_hours,
        )
        if valid_admission and timeline_valid and dischtime is not None
        else None
    )
    eligible_frontal = bool(
        selected["view_position"] in ("AP", "PA")
        and indexed["is_frontal"] is True
        and selected["acquisition_time_source"]
        in ("AcquisitionDateTime", "StudyDate+StudyTime")
    )
    flags = {
        "eligible_admission_match": valid_admission,
        "eligible_frontal": eligible_frontal,
        "dicom_valid": indexed["is_readable"] is True,
        "eligible_age": age is not None and age >= 18,
        "icu_timeline_valid": timeline_valid,
        "icu_at_t0": signature == "at_t0",
        "prior_icu": signature == "prior"
        or (signature == "at_t0" and first_icu is not None and first_icu < t0),
        "icu_within_6h": signature == "early",
        "icu_source_changes_decision": changed,
        "death_within_H_without_icu": bool(
            label_result and label_result.death_within_H_without_icu
        ),
    }
    outcome_status = label_result.status if label_result else "unavailable"
    reason = exclusion_reason(flags, outcome_status)
    label = label_result.label if reason is None and label_result else None
    return {
        "subject_id": subject_id,
        "hadm_id": match["hadm_id"] if matched else None,
        "study_id": study_id,
        "dicom_id": match["dicom_id"],
        "dicom_path": selected["dicom_path"],
        "sample_id": (
            f"{subject_id}-{match['hadm_id']}-{study_id}" if matched else None
        ),
        "t0": match["t0"],
        "age": age,
        "sex": patient["gender"] if patient else None,
        "admittime": match["admittime"] if matched else None,
        "dischtime": match["dischtime"] if matched else None,
        "deathtime": deathtime_text,
        "hospital_expire_flag": expire_flag,
        "view_position": selected["view_position"],
        "first_icu_time": first_icu.isoformat() if first_icu else None,
        "first_icu_transfer_id": (
            transfer_timeline.first_event_id if first_icu else None
        ),
        "icustays_first_icu_time": (
            stay_first_icu.isoformat() if stay_first_icu is not None else None
        ),
        "icustays_first_stay_id": stay_timeline.first_event_id,
        "transfer_icu_class": signature,
        "icustays_icu_class": stay_signature,
        "icu_mixed_stay_ids": [item.event_id for item in stay_timeline.mixed_intervals],
        "hours_to_icu": label_result.hours_to_icu if label_result else None,
        "label": label,
        "label_status": outcome_status if reason is None else "excluded",
        "prediction_window": f"{min_hours}-{max_hours}h",
        "prediction_max_hours": max_hours,
        "eligible_age": flags["eligible_age"],
        "eligible_admission_match": valid_admission,
        "eligible_frontal": eligible_frontal,
        "icu_at_t0": flags["icu_at_t0"],
        "prior_icu": flags["prior_icu"],
        "icu_within_6h": flags["icu_within_6h"],
        "death_within_H_without_icu": flags["death_within_H_without_icu"],
        "dicom_valid": flags["dicom_valid"],
        "icu_timeline_valid": timeline_valid,
        "icu_timeline_issue": timeline_issue,
        "icu_source_disagreement": audit_status not in ("agree", "not_assessed"),
        "icu_source_changes_decision": changed,
        "icu_audit_status": audit_status,
        "final_eligible": reason is None,
        "exclusion_reason": reason,
    }


def _validate_and_count(rows: list[dict[str, Any]], n_matched: int) -> dict[str, Any]:
    if len({(row["subject_id"], row["study_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate study in candidate cohort audit")
    if sum(row["eligible_admission_match"] for row in rows) != n_matched:
        raise ValueError("Admission match count changed during cohort build")
    positives = [row for row in rows if row["label"] == 1]
    negatives = [row for row in rows if row["label"] == 0]
    eligible = [row for row in rows if row["final_eligible"]]
    if len(eligible) != len(positives) + len(negatives):
        raise ValueError("Eligible cohort count does not match binary labels")
    for row in rows:
        if row["final_eligible"] != (row["exclusion_reason"] is None):
            raise ValueError("Final eligibility and exclusion reason disagree")
        if row["final_eligible"] and (
            not row["sample_id"] or row["label"] not in (0, 1)
        ):
            raise ValueError("Eligible study has no sample key or binary label")
        if not row["final_eligible"] and row["label"] is not None:
            raise ValueError("Excluded study has a training label")
        if row["label"] == 1 and not 6 < row["hours_to_icu"] <= 48:
            raise ValueError("Positive study is outside the 6-48 hour window")
        if row["label"] == 0 and row["hours_to_icu"] is not None:
            if row["hours_to_icu"] <= 48:
                raise ValueError("Negative study entered ICU within 48 hours")
    if len({row["sample_id"] for row in eligible}) != len(eligible):
        raise ValueError("Duplicate eligible sample_id")
    exclusion_counts = Counter(
        row["exclusion_reason"] for row in rows if row["exclusion_reason"]
    )
    if len(eligible) + sum(exclusion_counts.values()) != len(rows):
        raise ValueError("Candidate exclusion counts do not cover every study")

    def entities(stage_rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "studies": len(stage_rows),
            "admissions": len(
                {row["hadm_id"] for row in stage_rows if row["hadm_id"] is not None}
            ),
            "subjects": len({row["subject_id"] for row in stage_rows}),
            "observed_first_icu_events": len(
                {
                    (row["hadm_id"], row["first_icu_time"])
                    for row in stage_rows
                    if row["hadm_id"] is not None and row["first_icu_time"] is not None
                }
            ),
        }

    stage_counts: dict[str, dict[str, int]] = {}
    stage = rows
    for name, predicate in (
        ("selected_cxr", lambda row: True),
        ("readable_dicom", lambda row: row["dicom_valid"]),
        ("frontal_with_time", lambda row: row["eligible_frontal"]),
        ("unique_admission", lambda row: row["eligible_admission_match"]),
        ("adult", lambda row: row["eligible_age"]),
        ("reliable_icu_timeline", lambda row: row["icu_timeline_valid"]),
        ("not_in_icu_at_t0", lambda row: not row["icu_at_t0"]),
        ("no_prior_icu", lambda row: not row["prior_icu"]),
        ("outside_early_icu_window", lambda row: not row["icu_within_6h"]),
        ("final_eligible", lambda row: row["final_eligible"]),
    ):
        stage = [row for row in stage if predicate(row)]
        stage_counts[name] = entities(stage)
    if stage_counts["final_eligible"]["studies"] != len(eligible):
        raise ValueError("Stage flow does not close on final eligible studies")
    return {
        "audited_studies": len(rows),
        "unique_match_studies": n_matched,
        "eligible_studies": len(eligible),
        "eligible_admissions": len({row["hadm_id"] for row in eligible}),
        "eligible_subjects": len({row["subject_id"] for row in eligible}),
        "negative_studies": len(negatives),
        "positive_studies": len(positives),
        "positive_admissions": len({row["hadm_id"] for row in positives}),
        "positive_subjects": len({row["subject_id"] for row in positives}),
        "positive_independent_icu_events": len(
            {(row["hadm_id"], row["first_icu_time"]) for row in positives}
        ),
        "icu_source_disagreement_studies": sum(
            row["icu_source_disagreement"] for row in rows
        ),
        "icu_source_conflict_studies": sum(
            row["icu_source_changes_decision"] for row in rows
        ),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "stage_counts": stage_counts,
    }


def build_candidate_cohort(root: Path, paths_file: Path, cohort_file: Path) -> int:
    """Write the preliminary 48-hour audit; never write a formal cohort."""
    started = datetime.now(timezone.utc)
    paths = _json_file(paths_file)
    config = _json_file(cohort_file)
    prediction = config["prediction"]
    if config["sample_unit"] != "study" or prediction != {
        "min_hours": 6,
        "max_hours": 48,
    }:
        raise ValueError("05a requires study-level sampling and a 6-48h window")
    if config["positive_count_unit"] != "unique_subject_id":
        raise ValueError("positive_count_unit must be unique_subject_id")
    configured_units = config["icu_careunits"]
    if (
        not isinstance(configured_units, list)
        or len(configured_units) != len(ICU_CAREUNITS)
        or frozenset(configured_units) != ICU_CAREUNITS
    ):
        raise ValueError("05a ICU careunit list differs from the agreed seven units")
    iv_root = _resolve_config_path(paths["mimic_iv"], root, raw=True)
    processed = _resolve_config_path(paths["processed"], root, raw=False)
    outputs = _resolve_config_path(paths["outputs"], root, raw=False)
    if any(
        target == iv_root or target.is_relative_to(iv_root)
        for target in (processed, outputs)
    ):
        raise ValueError("Generated outputs must be outside the raw MIMIC-IV root")
    source_config = config["cohort_sources"]
    source_paths = {
        "patients": _under_root(iv_root, source_config["patients_path"]),
        "admissions": _under_root(
            iv_root, config["admission_matching"]["admissions_path"]
        ),
        "transfers": _under_root(iv_root, source_config["transfers_path"]),
        "icustays": _under_root(iv_root, source_config["icustays_path"]),
    }
    input_paths = {
        "matches": processed / "index" / "cxr_admission_matches.parquet",
        "selection": processed / "index" / "study_image_selection.parquet",
        "cxr_index": processed / "index" / "cxr_index.parquet",
        **source_paths,
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} input: {path}")
    matches = _unique_studies(
        _load_parquet(input_paths["matches"], MATCH_COLUMNS), "matches"
    )
    selections = _unique_studies(
        _load_parquet(input_paths["selection"], SELECTION_COLUMNS), "selection"
    )
    if not matches or matches.keys() != selections.keys():
        raise ValueError(
            "Admission matches and selections do not cover the same studies"
        )
    for match in matches.values():
        if match["match_status"] not in ("unique_match", "no_match", "multiple_match"):
            raise ValueError("Unexpected admission match status")
        if match["match_status"] == "unique_match":
            if type(match["hadm_id"]) is not int:
                raise ValueError("Unique admission match has no valid hadm_id")
        elif match["hadm_id"] is not None:
            raise ValueError("Non-unique admission match must not specify hadm_id")
    selected_dicoms = {
        (row["subject_id"], row["study_id"], row["selected_dicom_id"])
        for row in selections.values()
    }
    indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in _load_parquet(input_paths["cxr_index"], INDEX_COLUMNS):
        key = (row["subject_id"], row["study_id"], row["dicom_id"])
        if key in selected_dicoms:
            if key in indexed:
                raise ValueError(f"Duplicate selected DICOM in image index: {key}")
            indexed[key] = row
    if indexed.keys() != selected_dicoms:
        raise ValueError("Selected DICOM is missing from the image index")
    subjects = {key[0] for key in matches}
    matched_rows = [
        row for row in matches.values() if row["match_status"] == "unique_match"
    ]
    hadms = {row["hadm_id"] for row in matched_rows}
    if None in hadms:
        raise ValueError("Unique match has no hadm_id")
    patients = _read_relevant_csv(
        source_paths["patients"],
        {"subject_id", "gender", "anchor_age", "anchor_year"},
        "subject_id",
        subjects,
        "subject_id",
    )
    admissions = _read_relevant_csv(
        source_paths["admissions"],
        {
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "hospital_expire_flag",
        },
        "hadm_id",
        hadms,
        "hadm_id",
    )
    transfers = _read_relevant_csv(
        source_paths["transfers"],
        {"subject_id", "hadm_id", "transfer_id", "careunit", "intime", "outtime"},
        "hadm_id",
        hadms,
        None,
    )
    stays = _read_relevant_csv(
        source_paths["icustays"],
        {
            "subject_id",
            "hadm_id",
            "stay_id",
            "first_careunit",
            "last_careunit",
            "intime",
            "outtime",
        },
        "hadm_id",
        hadms,
        None,
    )
    for hadm_id in hadms:
        subject_id = next(
            row["subject_id"] for row in matched_rows if row["hadm_id"] == hadm_id
        )
        for row in [*transfers.get(hadm_id, []), *stays.get(hadm_id, [])]:
            if int(row["subject_id"]) != subject_id:
                raise ValueError(
                    f"ICU source subject_id disagrees for hadm_id {hadm_id}"
                )
    audit_rows: list[dict[str, Any]] = []
    with tqdm(
        total=len(matches),
        desc="Assess candidate studies",
        unit="study",
        file=sys.stderr,
        disable=not sys.stderr.isatty(),
    ) as progress:
        for key in sorted(matches):
            match = matches[key]
            selected = selections[key]
            image_key = (key[0], key[1], selected["selected_dicom_id"])
            hadm_id = match["hadm_id"]
            audit_rows.append(
                _assess_study(
                    match,
                    selected,
                    indexed[image_key],
                    patients.get(key[0]),
                    admissions.get(hadm_id),
                    transfers.get(hadm_id, []),
                    stays.get(hadm_id, []),
                    ICU_CAREUNITS,
                    6,
                    48,
                )
            )
            progress.update(1)
    counts = _validate_and_count(audit_rows, len(matched_rows))
    audit_dir = outputs / "audit" / "window_48h"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "cohort_audit.parquet"
    counts_path = audit_dir / "positive_case_counts.json"
    temp_audit = _temp_path(audit_dir, ".parquet")
    temp_counts = _temp_path(audit_dir, ".json")
    try:
        pq.write_table(
            pa.Table.from_pylist(audit_rows, schema=AUDIT_SCHEMA),
            temp_audit,
            compression="zstd",
        )
        if pq.ParquetFile(temp_audit).metadata.num_rows != len(audit_rows):
            raise ValueError("Written candidate audit row count is incorrect")
        summary = {
            "mode": "candidate_48h",
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(root),
            "config_sha256": _config_hash([paths_file, cohort_file]),
            "input_sha256": {
                name: _file_hash(path) for name, path in input_paths.items()
            },
            "audit_output_sha256": _file_hash(temp_audit),
            "prediction_window": "6-48h",
            "positive_count_unit": "unique_subject_id",
            **counts,
            "meets_80_positive_subject_threshold": counts["positive_subjects"] >= 80,
            "meets_150_positive_subject_threshold": counts["positive_subjects"] >= 150,
        }
        with temp_counts.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temp_audit, audit_path)
        os.replace(temp_counts, counts_path)
    finally:
        temp_audit.unlink(missing_ok=True)
        temp_counts.unlink(missing_ok=True)
    print(
        f"Candidate 6-48h audit: {counts['audited_studies']} studies, "
        f"{counts['eligible_studies']} eligible, "
        f"{counts['positive_subjects']} unique positive subjects"
    )
    print(f"Audit: {audit_path}")
    print(f"Counts: {counts_path}")
    return 0
