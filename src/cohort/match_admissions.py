"""Match selected CXR studies to reliable admission intervals."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.common import (
    _config_hash,
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
)

SELECTION_COLUMNS = {
    "subject_id",
    "study_id",
    "selected_dicom_id",
    "acquisition_datetime",
}
ADMISSION_COLUMNS = {"subject_id", "hadm_id", "admittime", "dischtime"}
MATCH_STATUSES = ("unique_match", "no_match", "multiple_match")

MATCH_SCHEMA = pa.schema(
    [
        pa.field("subject_id", pa.int64(), nullable=False),
        pa.field("hadm_id", pa.int64()),
        pa.field("study_id", pa.int64(), nullable=False),
        pa.field("dicom_id", pa.string(), nullable=False),
        pa.field("t0", pa.string(), nullable=False),
        pa.field("admittime", pa.string()),
        pa.field("dischtime", pa.string()),
        pa.field("match_status", pa.string(), nullable=False),
        pa.field("failure_reason", pa.string()),
        pa.field("n_confirmed_matches", pa.int32(), nullable=False),
        pa.field("n_uncertain_candidates", pa.int32(), nullable=False),
    ]
)
FAILURE_SCHEMA = pa.schema(
    [
        pa.field("subject_id", pa.int64(), nullable=False),
        pa.field("study_id", pa.int64(), nullable=False),
        pa.field("dicom_id", pa.string(), nullable=False),
        pa.field("t0", pa.string(), nullable=False),
        pa.field("match_status", pa.string(), nullable=False),
        pa.field("failure_reason", pa.string(), nullable=False),
        pa.field("candidate_hadm_id", pa.int64()),
        pa.field("candidate_admittime", pa.string()),
        pa.field("candidate_dischtime", pa.string()),
        pa.field("candidate_status", pa.string()),
        pa.field("candidate_reason", pa.string()),
    ]
)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _compatible(left: datetime, right: datetime) -> bool:
    return (left.tzinfo is None) == (right.tzinfo is None)


def _load_selections(path: Path) -> list[dict[str, Any]]:
    source = pq.ParquetFile(path)
    missing = SELECTION_COLUMNS - set(source.schema_arrow.names)
    if missing:
        raise ValueError(
            f"Selection table is missing columns: {', '.join(sorted(missing))}"
        )
    selections: list[dict[str, Any]] = []
    keys: set[tuple[int, int]] = set()
    study_subjects: dict[int, int] = {}
    with _progress(
        "Read selected CXR", unit="study", total=source.metadata.num_rows
    ) as progress:
        for batch in source.iter_batches(
            batch_size=8192, columns=sorted(SELECTION_COLUMNS)
        ):
            for row in batch.to_pylist():
                subject_id, study_id = row["subject_id"], row["study_id"]
                dicom_id, time_text = (
                    row["selected_dicom_id"],
                    row["acquisition_datetime"],
                )
                if (
                    type(subject_id) is not int
                    or type(study_id) is not int
                    or not isinstance(dicom_id, str)
                    or not dicom_id
                    or not isinstance(time_text, str)
                    or "T" not in time_text
                ):
                    raise ValueError(
                        "Selected CXR has an invalid key or acquisition time"
                    )
                t0 = _parse_datetime(time_text)
                if t0 is None:
                    raise ValueError("Selected CXR has an invalid acquisition time")
                key = (subject_id, study_id)
                if key in keys:
                    raise ValueError(f"Duplicate selected CXR study: {key}")
                keys.add(key)
                previous_subject = study_subjects.setdefault(study_id, subject_id)
                if previous_subject != subject_id:
                    raise ValueError(
                        f"Study ID belongs to multiple subjects: {study_id}"
                    )
                row["_t0"] = t0
                selections.append(row)
            progress.update(batch.num_rows)
    if not selections:
        raise ValueError("The selected CXR table is empty")
    return selections


def _load_relevant_admissions(
    path: Path, subjects: set[int]
) -> tuple[dict[int, list[dict[str, Any]]], Counter[str]]:
    admissions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    seen_hadm_ids: set[int] = set()
    counts["duplicate_hadm_id_rows"] = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        missing = ADMISSION_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Admissions CSV is missing columns: {', '.join(sorted(missing))}"
            )
        with _progress("Read admissions") as progress:
            for row_number, row in enumerate(reader, start=2):
                counts["admissions_rows_scanned"] += 1
                progress.update(1)
                if None in row:
                    raise ValueError(
                        f"Malformed admissions CSV row at line {row_number}"
                    )
                try:
                    subject_id = int(row["subject_id"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid admissions subject_id at line {row_number}"
                    ) from error
                hadm_text = row["hadm_id"]
                try:
                    hadm_id = int(hadm_text) if hadm_text else None
                except (TypeError, ValueError):
                    hadm_id = None
                if hadm_id is not None:
                    if hadm_id in seen_hadm_ids:
                        raise ValueError(
                            f"Duplicate admissions hadm_id {hadm_id} at line {row_number}"
                        )
                    seen_hadm_ids.add(hadm_id)
                if subject_id not in subjects:
                    continue
                counts["relevant_admission_rows"] += 1
                if hadm_id is None:
                    counts["invalid_hadm_id_rows"] += 1
                start_text, end_text = row["admittime"], row["dischtime"]
                start, end = _parse_datetime(start_text), _parse_datetime(end_text)
                if start is None:
                    counts["missing_or_invalid_admittime_rows"] += 1
                if end is None:
                    counts["missing_or_invalid_dischtime_rows"] += 1
                if start is not None and end is not None:
                    if not _compatible(start, end):
                        counts["incomparable_admission_boundaries"] += 1
                    elif end <= start:
                        counts["reversed_or_zero_intervals"] += 1
                admissions[subject_id].append(
                    {
                        "hadm_id": hadm_id,
                        "admittime": start_text or None,
                        "dischtime": end_text or None,
                        "_start": start,
                        "_end": end,
                    }
                )
    counts["relevant_subjects_with_admissions"] = len(admissions)
    for subject_admissions in admissions.values():
        for aware in (False, True):
            valid = [
                row
                for row in subject_admissions
                if row["_start"] is not None
                and row["_end"] is not None
                and _compatible(row["_start"], row["_end"])
                and (row["_start"].tzinfo is not None) == aware
                and row["_start"] < row["_end"]
            ]
            latest_end: datetime | None = None
            for row in sorted(valid, key=lambda item: item["_start"]):
                start, end = row["_start"], row["_end"]
                if latest_end is not None and start < latest_end:
                    counts["overlapping_admission_rows"] += 1
                if latest_end is None or end > latest_end:
                    latest_end = end
    return admissions, counts


def _candidate_status(t0: datetime, admission: dict[str, Any]) -> tuple[str, str]:
    start, end = admission["_start"], admission["_end"]
    if start is not None and end is not None:
        if not _compatible(start, end):
            return "uncertain", "incomparable_admission_boundaries"
        if not _compatible(t0, start):
            return "uncertain", "timezone_mismatch"
        if end <= start:
            return "uncertain", "invalid_interval"
        if start <= t0 < end:
            if admission["hadm_id"] is None:
                return "uncertain", "invalid_hadm_id"
            return "confirmed", "within_interval"
        return "outside", "outside_interval"
    if start is not None:
        if not _compatible(t0, start):
            return "uncertain", "timezone_mismatch"
        if t0 < start:
            return "outside", "before_admission"
    if end is not None:
        if not _compatible(t0, end):
            return "uncertain", "timezone_mismatch"
        if t0 >= end:
            return "outside", "at_or_after_discharge"
    if start is None and end is None:
        return "uncertain", "missing_or_invalid_both_boundaries"
    if start is None:
        return "uncertain", "missing_or_invalid_admittime"
    return "uncertain", "missing_or_invalid_dischtime"


def _match_one(
    selected: dict[str, Any], admissions: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assessed = [
        (admission, *_candidate_status(selected["_t0"], admission))
        for admission in admissions
    ]
    confirmed = [item for item in assessed if item[1] == "confirmed"]
    uncertain = [item for item in assessed if item[1] == "uncertain"]
    if len(confirmed) == 1 and not uncertain:
        status, failure, chosen = "unique_match", None, confirmed[0][0]
    elif not confirmed:
        status = "no_match"
        failure = (
            "uncertain_interval"
            if uncertain
            else "outside_all_admissions"
            if admissions
            else "no_patient_admissions"
        )
        chosen = None
    else:
        status = "multiple_match"
        failure = (
            "multiple_confirmed"
            if len(confirmed) > 1
            else "uncertain_competing_admission"
        )
        chosen = None

    match_row = {
        "subject_id": selected["subject_id"],
        "hadm_id": chosen["hadm_id"] if chosen else None,
        "study_id": selected["study_id"],
        "dicom_id": selected["selected_dicom_id"],
        "t0": selected["acquisition_datetime"],
        "admittime": chosen["admittime"] if chosen else None,
        "dischtime": chosen["dischtime"] if chosen else None,
        "match_status": status,
        "failure_reason": failure,
        "n_confirmed_matches": len(confirmed),
        "n_uncertain_candidates": len(uncertain),
    }
    failures: list[dict[str, Any]] = []
    if status != "unique_match":
        for admission, candidate_status, candidate_reason in assessed:
            failures.append(
                {
                    "subject_id": selected["subject_id"],
                    "study_id": selected["study_id"],
                    "dicom_id": selected["selected_dicom_id"],
                    "t0": selected["acquisition_datetime"],
                    "match_status": status,
                    "failure_reason": failure,
                    "candidate_hadm_id": admission["hadm_id"],
                    "candidate_admittime": admission["admittime"],
                    "candidate_dischtime": admission["dischtime"],
                    "candidate_status": candidate_status,
                    "candidate_reason": candidate_reason,
                }
            )
        if not assessed:
            failures.append(
                {
                    "subject_id": selected["subject_id"],
                    "study_id": selected["study_id"],
                    "dicom_id": selected["selected_dicom_id"],
                    "t0": selected["acquisition_datetime"],
                    "match_status": status,
                    "failure_reason": failure,
                    "candidate_hadm_id": None,
                    "candidate_admittime": None,
                    "candidate_dischtime": None,
                    "candidate_status": None,
                    "candidate_reason": None,
                }
            )
    return match_row, failures


def _write_summary(path: Path, values: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerows((key, value) for key, value in values.items())


def match_admissions(root: Path, paths_file: Path, cohort_file: Path) -> int:
    """Match all selected studies and write complete, auditable outputs."""
    started = datetime.now(timezone.utc)
    paths = _json_file(paths_file)
    cohort = _json_file(cohort_file)
    if cohort["sample_unit"] != "study":
        raise ValueError("cohort.sample_unit must be study")
    iv_root = _resolve_config_path(paths["mimic_iv"], root, raw=True)
    processed = _resolve_config_path(paths["processed"], root, raw=False)
    outputs = _resolve_config_path(paths["outputs"], root, raw=False)
    _protect_outputs(paths, root, processed, outputs)
    with _OutputRun(outputs, 4, processed) as run:
        _verify_run(outputs, 3, processed / "index" / "study_image_selection.parquet")
        selected_path = processed / "index" / "study_image_selection.parquet"
        admissions_path = _under_root(
            iv_root, cohort["admission_matching"]["admissions_path"]
        )
        if not selected_path.is_file():
            raise FileNotFoundError(f"Missing selected CXR table: {selected_path}")
        if not admissions_path.is_file():
            raise FileNotFoundError(f"Missing admissions CSV: {admissions_path}")

        selections = _load_selections(selected_path)
        subjects = {row["subject_id"] for row in selections}
        admissions, admission_counts = _load_relevant_admissions(
            admissions_path, subjects
        )
        matches: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        statuses: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        with _progress(
            "Match CXR studies", unit="study", total=len(selections)
        ) as progress:
            for selected in sorted(
                selections, key=lambda row: (row["subject_id"], row["study_id"])
            ):
                match, detail = _match_one(
                    selected, admissions.get(selected["subject_id"], [])
                )
                matches.append(match)
                failures.extend(detail)
                statuses[match["match_status"]] += 1
                if match["failure_reason"]:
                    reasons[match["failure_reason"]] += 1
                progress.update(1)
        if len(matches) != len(selections) or sum(statuses.values()) != len(selections):
            raise ValueError("Match status counts do not cover every selected CXR")
        if set(statuses) - set(MATCH_STATUSES):
            raise ValueError("Unexpected admission match status")
        if len({(row["subject_id"], row["study_id"]) for row in matches}) != len(
            matches
        ):
            raise ValueError("Duplicate study in admission match output")
        if len({(row["subject_id"], row["study_id"]) for row in failures}) != (
            statuses["no_match"] + statuses["multiple_match"]
        ):
            raise ValueError("Failure details do not cover every non-unique match")
        for row in matches:
            if row["match_status"] == "unique_match":
                admitted = _parse_datetime(row["admittime"])
                t0 = _parse_datetime(row["t0"])
                discharged = _parse_datetime(row["dischtime"])
                if (
                    row["hadm_id"] is None
                    or row["n_confirmed_matches"] != 1
                    or row["n_uncertain_candidates"] != 0
                    or admitted is None
                    or t0 is None
                    or discharged is None
                ):
                    raise ValueError("Invalid unique admission match")
                if not admitted <= t0 < discharged:
                    raise ValueError("Unique admission interval does not contain t0")
            elif row["hadm_id"] is not None:
                raise ValueError("Non-unique admission match contains a hadm_id")

        index_dir = processed / "index"
        audit_dir = outputs / "audit"
        match_path = index_dir / "cxr_admission_matches.parquet"
        failure_path = audit_dir / "cxr_admission_match_failures.parquet"
        summary_path = audit_dir / "admission_match_summary.csv"
        temp_match = run.path(match_path)
        temp_failure = run.path(failure_path)
        pq.write_table(
            pa.Table.from_pylist(matches, schema=MATCH_SCHEMA),
            temp_match,
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(failures, schema=FAILURE_SCHEMA),
            temp_failure,
            compression="zstd",
        )
        if pq.ParquetFile(temp_match).metadata.num_rows != len(matches):
            raise ValueError("Written admission match row count is incorrect")
        if pq.ParquetFile(temp_failure).metadata.num_rows != len(failures):
            raise ValueError("Written admission failure row count is incorrect")
        finished = datetime.now(timezone.utc)
        summary: dict[str, Any] = {
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "git_commit": _git_commit(root),
            "config_sha256": _config_hash([paths_file, cohort_file]),
            "selection_input_sha256": _file_hash(selected_path),
            "admissions_input_sha256": _file_hash(admissions_path),
            "match_output_sha256": _file_hash(temp_match),
            "failure_output_sha256": _file_hash(temp_failure),
            "selection_input_bytes": _file_version(selected_path)["size_bytes"],
            "admissions_input_bytes": _file_version(admissions_path)["size_bytes"],
            "selected_studies": len(selections),
            "selected_subjects": len(subjects),
            "failure_detail_rows": len(failures),
            "studies_with_uncertain_candidates": sum(
                row["n_uncertain_candidates"] > 0 for row in matches
            ),
        }
        summary.update({status: statuses[status] for status in MATCH_STATUSES})
        summary.update(
            {
                f"failure_reason_{reason}": count
                for reason, count in sorted(reasons.items())
            }
        )
        summary.update(
            {
                f"admissions_{key}": count
                for key, count in sorted(admission_counts.items())
            }
        )
        temp_summary = run.path(summary_path)
        _write_summary(temp_summary, summary)
    print(
        f"Matched {len(selections)} selected CXR studies: "
        + ", ".join(f"{status}={statuses[status]}" for status in MATCH_STATUSES)
    )
    print(f"Matches: {match_path}")
    print(f"Failures: {failure_path}")
    print(f"Summary: {summary_path}")
    return 0
