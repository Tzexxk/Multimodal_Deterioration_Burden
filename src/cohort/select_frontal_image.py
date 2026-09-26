"""Choose one readable, reliably timed frontal image per CXR study."""

from __future__ import annotations

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

from src.cohort.audit_raw_data import (
    _config_hash,
    _file_hash,
    _file_version,
    _git_commit,
    _json_file,
    _resolve_config_path,
)

INPUT_COLUMNS = (
    "subject_id",
    "study_id",
    "dicom_id",
    "dicom_path",
    "view_position",
    "acquisition_datetime",
    "acquisition_time_source",
    "rows",
    "columns",
    "is_readable",
    "is_frontal",
)
RELIABLE_TIME_SOURCES = {"AcquisitionDateTime", "StudyDate+StudyTime"}
MISSING_TIME_SOURCES = {"missing", "not_assessed"}
FRONTAL_VIEWS = {"AP", "PA"}

SELECTION_SCHEMA = pa.schema(
    [
        pa.field("subject_id", pa.int64(), nullable=False),
        pa.field("study_id", pa.int64(), nullable=False),
        pa.field("selected_dicom_id", pa.string(), nullable=False),
        pa.field("dicom_path", pa.string(), nullable=False),
        pa.field("view_position", pa.string(), nullable=False),
        pa.field("acquisition_datetime", pa.string(), nullable=False),
        pa.field("acquisition_time_source", pa.string(), nullable=False),
        pa.field("n_images", pa.int32(), nullable=False),
        pa.field("n_frontal_images", pa.int32(), nullable=False),
        pa.field("n_eligible_frontal_images", pa.int32(), nullable=False),
        pa.field("selection_reason", pa.string(), nullable=False),
    ]
)


def _temporary_path(directory: Path, suffix: str) -> Path:
    descriptor, value = tempfile.mkstemp(
        prefix=".study_image_selection_", suffix=suffix, dir=directory
    )
    os.close(descriptor)
    return Path(value)


def _parse_image_time(row: dict[str, Any]) -> datetime | None:
    source = row["acquisition_time_source"]
    value = row["acquisition_datetime"]
    if source in MISSING_TIME_SOURCES:
        if value not in (None, ""):
            raise ValueError("Missing time source has a nonempty acquisition datetime")
        return None
    if (
        source not in RELIABLE_TIME_SOURCES
        or not isinstance(value, str)
        or "T" not in value
    ):
        raise ValueError("Invalid acquisition time source or datetime")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Invalid acquisition datetime") from error


def _pixel_area(row: dict[str, Any]) -> int:
    rows, columns = row["rows"], row["columns"]
    if (
        type(rows) is int
        and type(columns) is int
        and rows > 0
        and columns > 0
    ):
        return rows * columns
    return -1


def _load_studies(
    index_path: Path,
) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], int]:
    source = pq.ParquetFile(index_path)
    absent = set(INPUT_COLUMNS) - set(source.schema_arrow.names)
    if absent:
        raise ValueError(f"CXR index is missing columns: {', '.join(sorted(absent))}")
    studies: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    study_subjects: dict[int, int] = {}
    study_dicom_ids: set[tuple[int, int, str]] = set()
    row_count = 0
    with tqdm(
        total=source.metadata.num_rows,
        desc="Read CXR index",
        unit="image",
        unit_scale=True,
        file=sys.stderr,
        disable=not sys.stderr.isatty(),
    ) as progress:
        for batch in source.iter_batches(batch_size=8192, columns=list(INPUT_COLUMNS)):
            for row in batch.to_pylist():
                row_count += 1
                subject_id, study_id, dicom_id = (
                    row["subject_id"], row["study_id"], row["dicom_id"]
                )
                if (
                    type(subject_id) is not int
                    or type(study_id) is not int
                    or not isinstance(dicom_id, str)
                    or not dicom_id
                    or not isinstance(row["dicom_path"], str)
                    or not row["dicom_path"]
                    or type(row["is_readable"]) is not bool
                    or type(row["is_frontal"]) is not bool
                ):
                    raise ValueError(
                        f"Invalid required CXR index fields at row {row_count}"
                    )
                if row["is_frontal"] != (row["view_position"] in FRONTAL_VIEWS):
                    raise ValueError(f"Inconsistent frontal view at row {row_count}")
                previous_subject = study_subjects.setdefault(study_id, subject_id)
                if previous_subject != subject_id:
                    raise ValueError(
                        f"Study ID belongs to multiple subjects: {study_id}"
                    )
                image_key = (subject_id, study_id, dicom_id)
                if image_key in study_dicom_ids:
                    raise ValueError(f"Duplicate CXR index image at row {row_count}")
                study_dicom_ids.add(image_key)
                row["_parsed_time"] = _parse_image_time(row)
                studies[(subject_id, study_id)].append(row)
            progress.update(batch.num_rows)
    if not row_count:
        raise ValueError("CXR index contains no images")
    return studies, row_count


def _select_study(
    images: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    frontal = [row for row in images if row["is_frontal"]]
    if not frontal:
        return None, "no_frontal"
    readable = [row for row in frontal if row["is_readable"]]
    if not readable:
        return None, "no_readable_frontal"
    eligible = [row for row in readable if row["_parsed_time"] is not None]
    if not eligible:
        return None, "no_reliable_time"

    awareness = {row["_parsed_time"].tzinfo is not None for row in eligible}
    if len(awareness) != 1:
        raise ValueError("A study mixes timezone-aware and timezone-naive image times")
    if len(eligible) == 1:
        chosen = eligible[0]
        reason = "single_frontal" if len(frontal) == 1 else "single_eligible_frontal"
    else:
        earliest = min(row["_parsed_time"] for row in eligible)
        time_ties = [row for row in eligible if row["_parsed_time"] == earliest]
        if len(time_ties) == 1:
            chosen = time_ties[0]
            reason = "earliest_frontal"
        else:
            largest_area = max(_pixel_area(row) for row in time_ties)
            area_ties = [row for row in time_ties if _pixel_area(row) == largest_area]
            if len(area_ties) == 1:
                chosen = area_ties[0]
                reason = "largest_pixel_area"
            else:
                chosen = min(area_ties, key=lambda row: row["dicom_id"])
                reason = "dicom_id_tiebreak"

    return (
        {
            "subject_id": chosen["subject_id"],
            "study_id": chosen["study_id"],
            "selected_dicom_id": chosen["dicom_id"],
            "dicom_path": chosen["dicom_path"],
            "view_position": chosen["view_position"],
            "acquisition_datetime": chosen["acquisition_datetime"],
            "acquisition_time_source": chosen["acquisition_time_source"],
            "n_images": len(images),
            "n_frontal_images": len(frontal),
            "n_eligible_frontal_images": len(eligible),
            "selection_reason": reason,
        },
        reason,
    )


def select_frontal_images(root: Path, paths_file: Path, imaging_file: Path) -> int:
    """Write a deterministic study-level selection and aggregate audit summary."""
    started = datetime.now(timezone.utc)
    paths = _json_file(paths_file)
    imaging = _json_file(imaging_file)["raw_audit"]
    if set(imaging["frontal_views"]) != FRONTAL_VIEWS:
        raise ValueError("frontal_views must contain AP and PA")
    if imaging["time_priority"] != ["AcquisitionDateTime", "StudyDate+StudyTime"]:
        raise ValueError("Unsupported time_priority")
    cxr_root = _resolve_config_path(paths["mimic_cxr"], root, raw=True)
    processed = _resolve_config_path(paths["processed"], root, raw=False)
    outputs = _resolve_config_path(paths["outputs"], root, raw=False)
    raw_roots = [cxr_root]
    for name in ("mimic_iv", "mimic_note"):
        value = paths.get(name)
        if isinstance(value, str) and value and not value.startswith("SET_YOUR_"):
            raw_roots.append(_resolve_config_path(value, root, raw=True))
    for destination in (processed, outputs):
        if any(
            destination == raw_root or destination.is_relative_to(raw_root)
            for raw_root in raw_roots
        ):
            raise ValueError("Generated output must be outside all raw data roots")

    index_path = processed / "index" / "cxr_index.parquet"
    if not index_path.is_file():
        raise FileNotFoundError(f"CXR image index does not exist: {index_path}")
    studies, image_count = _load_studies(index_path)
    selections: list[dict[str, Any]] = []
    selected_reasons: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    with tqdm(
        total=len(studies),
        desc="Select study images",
        unit="study",
        unit_scale=True,
        file=sys.stderr,
        disable=not sys.stderr.isatty(),
    ) as progress:
        for study_key, images in sorted(studies.items()):
            try:
                selected, reason = _select_study(images)
            except ValueError as error:
                raise ValueError(
                    f"Study {study_key} could not be ranked: {error}"
                ) from error
            if selected is None:
                exclusions[reason] += 1
            else:
                if (
                    selected["view_position"] not in FRONTAL_VIEWS
                    or selected["acquisition_time_source"] not in RELIABLE_TIME_SOURCES
                    or selected["n_eligible_frontal_images"] < 1
                    or selected["n_frontal_images"]
                    < selected["n_eligible_frontal_images"]
                    or selected["n_images"] < selected["n_frontal_images"]
                ):
                    raise ValueError(f"Invalid selected image for study {study_key}")
                selections.append(selected)
                selected_reasons[reason] += 1
            progress.update(1)

    if len(selections) + sum(exclusions.values()) != len(studies):
        raise ValueError("Selected and excluded study counts do not close")
    if len({row["study_id"] for row in selections}) != len(selections):
        raise ValueError("More than one image selected for a study")
    selected_keys = {(row["subject_id"], row["study_id"]) for row in selections}
    if sum(row["n_images"] for row in selections) + sum(
        len(images) for key, images in studies.items()
        if key not in selected_keys
    ) != image_count:
        raise ValueError("Image counts do not close")

    index_dir = processed / "index"
    audit_dir = outputs / "audit"
    index_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    selection_path = index_dir / "study_image_selection.parquet"
    summary_path = audit_dir / "study_image_selection_summary.json"
    temp_selection = _temporary_path(index_dir, ".parquet")
    temp_summary: Path | None = None
    try:
        pq.write_table(
            pa.Table.from_pylist(selections, schema=SELECTION_SCHEMA),
            temp_selection,
            compression="zstd",
        )
        if pq.ParquetFile(temp_selection).metadata.num_rows != len(selections):
            raise ValueError(
                "Written selection row count differs from validated studies"
            )
        summary = {
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(root),
            "config_sha256": _config_hash([paths_file, imaging_file]),
            "input_version": _file_version(index_path),
            "input_parquet_sha256": _file_hash(index_path),
            "output_parquet_sha256": _file_hash(temp_selection),
            "counts": {
                "input_images": image_count,
                "input_studies": len(studies),
                "selected_studies": len(selections),
                "excluded_studies": sum(exclusions.values()),
            },
            "selection_reasons": dict(sorted(selected_reasons.items())),
            "exclusion_reasons": dict(sorted(exclusions.items())),
        }
        temp_summary = _temporary_path(audit_dir, ".json")
        temp_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_selection, selection_path)
        os.replace(temp_summary, summary_path)
    finally:
        temp_selection.unlink(missing_ok=True)
        if temp_summary is not None:
            temp_summary.unlink(missing_ok=True)

    print(
        f"Selected {len(selections)} of {len(studies)} CXR studies; "
        f"excluded {sum(exclusions.values())}."
    )
    print(f"Parquet: {selection_path}")
    print(f"Summary: {summary_path}")
    return 0
