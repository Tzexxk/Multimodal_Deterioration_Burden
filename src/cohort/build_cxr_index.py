"""Build a portable, one-row-per-local-DICOM CXR index."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.common import (
    _acquisition_time,
    _config_hash,
    _file_hash,
    _file_version,
    _git_commit,
    _imaging_settings,
    _indexed_dicom_path,
    _json_file,
    _OutputRun,
    _progress,
    _protect_outputs,
    _resolve_config_path,
    _scan_physical_dicom,
    _under_root,
    _verify_run,
)

INDEX_SCHEMA = pa.schema(
    [
        pa.field("subject_id", pa.int64(), nullable=False),
        pa.field("study_id", pa.int64(), nullable=False),
        pa.field("dicom_id", pa.string(), nullable=False),
        pa.field("dicom_path", pa.string(), nullable=False),
        pa.field("view_position", pa.string()),
        pa.field("acquisition_datetime", pa.string()),
        pa.field("acquisition_time_source", pa.string(), nullable=False),
        pa.field("rows", pa.int32()),
        pa.field("columns", pa.int32()),
        pa.field("is_readable", pa.bool_(), nullable=False),
        pa.field("decode_error", pa.string()),
        pa.field("is_frontal", pa.bool_(), nullable=False),
        pa.field("metadata_readable", pa.bool_(), nullable=False),
        pa.field("time_failure_reason", pa.string()),
        pa.field("index_path", pa.string(), nullable=False),
        pa.field("path_resolution", pa.string(), nullable=False),
    ]
)
RECORD_COLUMNS = {"subject_id", "study_id", "dicom_id", "path"}
STUDY_COLUMNS = {"subject_id", "study_id", "path"}
TIME_SOURCES = {"AcquisitionDateTime", "StudyDate+StudyTime", "missing", "not_assessed"}


def _checked_csv(path: Path, required: set[str]) -> tuple[Any, csv.DictReader]:
    stream = path.open("r", encoding="utf-8-sig", newline="")
    try:
        reader = csv.DictReader(stream, strict=True)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {', '.join(sorted(missing))}"
            )
        return stream, reader
    except Exception:
        stream.close()
        raise


def _study_keys(path: Path) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    stream, reader = _checked_csv(path, STUDY_COLUMNS)
    with stream:
        for line, row in enumerate(reader, start=2):
            if None in row or not row.get("path"):
                raise ValueError(f"Malformed study-list row at line {line}")
            try:
                keys.add((int(row["subject_id"]), int(row["study_id"])))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid study-list key at line {line}") from error
    if not keys:
        raise ValueError("The study list is empty")
    return keys


def _dimension(dataset: Any, name: str) -> int | None:
    try:
        value = dataset.get(name)
        result = int(value) if value is not None else None
        return result if result is not None and result > 0 else None
    except (TypeError, ValueError):
        return None


def _read_local_dicom(
    item: tuple[dict[str, str], Path, str], cxr_root: Path
) -> dict[str, Any]:
    index_row, path, strategy = item
    result: dict[str, Any] = {
        "subject_id": int(index_row["subject_id"]),
        "study_id": int(index_row["study_id"]),
        "dicom_id": index_row["dicom_id"],
        "dicom_path": path.relative_to(cxr_root).as_posix(),
        "view_position": None,
        "acquisition_datetime": None,
        "acquisition_time_source": "not_assessed",
        "rows": None,
        "columns": None,
        "is_readable": False,
        "decode_error": None,
        "is_frontal": False,
        "metadata_readable": False,
        "time_failure_reason": None,
        "index_path": index_row["path"],
        "path_resolution": strategy,
    }
    import pydicom

    try:
        dataset = pydicom.dcmread(str(path), force=False)
    except Exception as error:  # Malformed DICOMs can raise several exception types.
        result["decode_error"] = "metadata_read_error:" + type(error).__name__
        return result

    result["metadata_readable"] = True
    try:
        view = str(dataset.get("ViewPosition", "") or "").strip().upper()
        result["view_position"] = view or None
        result["is_frontal"] = view in {"AP", "PA"}
        result["rows"] = _dimension(dataset, "Rows")
        result["columns"] = _dimension(dataset, "Columns")
        value, source, failure = _acquisition_time(dataset)
        result["acquisition_datetime"] = value or None
        result["acquisition_time_source"] = source
        result["time_failure_reason"] = failure or None
    except Exception as error:
        result["acquisition_datetime"] = None
        result["acquisition_time_source"] = "missing"
        result["time_failure_reason"] = "time_parse_error:" + type(error).__name__

    if "PixelData" not in dataset:
        result["decode_error"] = "missing_pixel_data"
    else:
        try:
            if dataset.pixel_array.size == 0:
                result["decode_error"] = "empty_pixel_array"
            else:
                result["is_readable"] = True
        except Exception as error:  # Decoder plugins use different exception types.
            result["decode_error"] = "pixel_decode_error:" + type(error).__name__
    return result


def _validate_image(row: dict[str, Any]) -> None:
    source = row["acquisition_time_source"]
    if source not in TIME_SOURCES:
        raise ValueError(f"Unexpected acquisition time source: {source}")
    if bool(row["acquisition_datetime"]) != (
        source in {"AcquisitionDateTime", "StudyDate+StudyTime"}
    ):
        raise ValueError("Acquisition time and source disagree")
    if row["is_readable"] == bool(row["decode_error"]):
        raise ValueError("Readability and decode error disagree")
    if row["is_frontal"] != (row["view_position"] in {"AP", "PA"}):
        raise ValueError("Frontal flag and view disagree")
    if not row["metadata_readable"] and source != "not_assessed":
        raise ValueError("Unreadable metadata must have an unassessed time")


def _prior_audit_count(report_path: Path) -> int | None:
    if not report_path.is_file():
        return None
    match = re.search(
        r"^- Unique local DICOM files referenced by the index: (\d+)\s*$",
        report_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return int(match.group(1)) if match else None


def build_cxr_index(root: Path, paths_file: Path, imaging_file: Path) -> int:
    """Build the image index and aggregate summary; return zero on success."""
    started = datetime.now(timezone.utc)
    paths = _json_file(paths_file)
    imaging = _json_file(imaging_file)["raw_audit"]
    workers, batch_size, layout = _imaging_settings(imaging)
    if set(imaging["frontal_views"]) != {"AP", "PA"}:
        raise ValueError("frontal_views must contain AP and PA")
    if not imaging["decode_pixels"]:
        raise ValueError("Pixel decoding must be enabled for the CXR index")

    cxr_root = _resolve_config_path(paths["mimic_cxr"], root, raw=True)
    processed = _resolve_config_path(paths["processed"], root, raw=False)
    outputs = _resolve_config_path(paths["outputs"], root, raw=False)
    if not cxr_root.is_dir():
        raise FileNotFoundError(f"Missing MIMIC-CXR root: {cxr_root}")
    _protect_outputs(paths, root, processed, outputs)
    with _OutputRun(outputs, 2, processed) as run:
        prior_report = outputs / "audit" / "audit_report.md"
        if (
            prior_report.exists()
            or (outputs / "audit" / ".step_01_pending.json").exists()
        ):
            _verify_run(outputs, 1, prior_report)
        dicom_dir = _under_root(cxr_root, imaging["dicom_directory"])
        record_path = _under_root(cxr_root, imaging["record_list"])
        study_path = _under_root(cxr_root, imaging["study_list"])
        if not dicom_dir.is_dir():
            raise FileNotFoundError(f"Missing DICOM directory: {dicom_dir}")

        studies = _study_keys(study_path)
        scanned, walk_errors = _scan_physical_dicom(
            cxr_root, {suffix.lower() for suffix in imaging["dicom_suffixes"]}
        )
        if walk_errors:
            raise OSError("DICOM directory scan failed: " + "; ".join(walk_errors))
        root_files = cxr_root / "files"
        local_files = {
            path
            for path in scanned
            if path.is_relative_to(dicom_dir) or path.is_relative_to(root_files)
        }
        if not local_files:
            raise ValueError("No local DICOM files found in the configured CXR layouts")

        index_dir = processed / "index"
        audit_dir = outputs / "audit"
        index_path = index_dir / "cxr_index.parquet"
        summary_path = audit_dir / "cxr_index_summary.json"
        temp_index = run.path(index_path)
        counts: Counter[str] = Counter()
        time_sources: Counter[str] = Counter()
        time_failures: Counter[str] = Counter()
        decode_errors: Counter[str] = Counter()
        resolutions: Counter[str] = Counter()
        keys: set[tuple[int, int, str]] = set()
        matched_files: set[Path] = set()
        pending: list[tuple[dict[str, str], Path, str]] = []
        output_rows: list[dict[str, Any]] = []

        def write_rows(writer: pq.ParquetWriter) -> None:
            if output_rows:
                writer.write_table(
                    pa.Table.from_pylist(output_rows, schema=INDEX_SCHEMA)
                )
                output_rows.clear()

        def process_batch(pool: ThreadPoolExecutor, writer: pq.ParquetWriter) -> None:
            if not pending:
                return
            for row in pool.map(
                lambda item: _read_local_dicom(item, cxr_root), pending
            ):
                _validate_image(row)
                output_rows.append(row)
                counts["output_rows"] += 1
                counts["readable_images"] += int(row["is_readable"])
                counts["frontal_images"] += int(row["is_frontal"])
                counts["images_with_acquisition_time"] += int(
                    row["acquisition_datetime"] is not None
                )
                time_sources[row["acquisition_time_source"]] += 1
                if row["time_failure_reason"]:
                    time_failures[row["time_failure_reason"]] += 1
                if row["decode_error"]:
                    decode_errors[row["decode_error"]] += 1
                resolutions[row["path_resolution"]] += 1
                if len(output_rows) >= 4096:
                    write_rows(writer)
                image_progress.update(1)
            pending.clear()

        stream, reader = _checked_csv(record_path, RECORD_COLUMNS)
        with (
            stream,
            ThreadPoolExecutor(max_workers=workers) as pool,
            pq.ParquetWriter(temp_index, INDEX_SCHEMA, compression="zstd") as writer,
            _progress("Match CXR records") as progress,
            _progress(
                "Index local DICOM", unit="image", total=len(local_files)
            ) as image_progress,
        ):
            for line, row in enumerate(reader, start=2):
                counts["index_records"] += 1
                progress.update(1)
                if None in row or any(not row.get(field) for field in RECORD_COLUMNS):
                    raise ValueError(f"Malformed record-list row at line {line}")
                try:
                    subject_id = int(row["subject_id"])
                    study_id = int(row["study_id"])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid record-list key at line {line}"
                    ) from error
                key = (subject_id, study_id, row["dicom_id"])
                if key in keys:
                    raise ValueError(f"Duplicate CXR record key at line {line}")
                keys.add(key)
                if (subject_id, study_id) not in studies:
                    raise ValueError(
                        f"Record-list study absent from study-list at line {line}"
                    )
                try:
                    path, strategy, exists = _indexed_dicom_path(
                        cxr_root, dicom_dir, row, layout, local_files, probe_root=False
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Invalid DICOM index path at line {line}: {error}"
                    ) from error
                if not exists:
                    counts["missing_index_paths"] += 1
                    continue
                if path in matched_files:
                    raise ValueError(
                        f"Local DICOM referenced more than once at line {line}"
                    )
                if not path.is_file():
                    raise FileNotFoundError(f"Scanned DICOM disappeared: {path}")
                matched_files.add(path)
                pending.append((row, path, strategy))
                if len(pending) >= batch_size:
                    process_batch(pool, writer)
            process_batch(pool, writer)
            write_rows(writer)

        if not counts["index_records"] or not counts["output_rows"]:
            raise ValueError("No indexed local DICOM images were found")
        if counts["output_rows"] != len(matched_files):
            raise ValueError("Output rows do not match unique indexed local files")
        if counts["index_records"] != (
            counts["missing_index_paths"] + counts["output_rows"]
        ):
            raise ValueError("Index coverage totals are inconsistent")
        if pq.ParquetFile(temp_index).metadata.num_rows != counts["output_rows"]:
            raise ValueError(
                "Written Parquet row count differs from validated index rows"
            )

        prior_count = _prior_audit_count(audit_dir / "audit_report.md")
        summary = {
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(root),
            "config_sha256": _config_hash([paths_file, imaging_file]),
            "input_versions": {
                "record_list": _file_version(record_path),
                "study_list": _file_version(study_path),
            },
            "settings": {
                "dicom_path_layout": layout,
                "dicom_workers": workers,
                "dicom_batch_size": batch_size,
                "path_base": "mimic_cxr",
            },
            "counts": {
                "index_records": counts["index_records"],
                "local_dicom_files": len(local_files),
                "matched_local_files": len(matched_files),
                "output_rows": counts["output_rows"],
                "missing_index_paths": counts["missing_index_paths"],
                "unreferenced_local_files": len(local_files - matched_files),
                "readable_images": counts["readable_images"],
                "unreadable_images": counts["output_rows"] - counts["readable_images"],
                "frontal_images": counts["frontal_images"],
                "images_with_acquisition_time": counts["images_with_acquisition_time"],
            },
            "time_sources": dict(sorted(time_sources.items())),
            "time_failures": dict(sorted(time_failures.items())),
            "decode_errors": dict(sorted(decode_errors.items())),
            "path_resolutions": dict(sorted(resolutions.items())),
            "prior_audit_matched_files": prior_count,
            "difference_from_prior_audit": (
                len(matched_files) - prior_count if prior_count is not None else None
            ),
            "parquet_sha256": _file_hash(temp_index),
        }
        temp_summary = run.path(summary_path)
        temp_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Indexed {counts['output_rows']} local DICOM images; "
        f"{counts['missing_index_paths']} index paths are absent locally."
    )
    print(f"Parquet: {index_path}")
    print(f"Summary: {summary_path}")
    return 0
