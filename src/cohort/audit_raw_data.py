"""Read-only, streaming first-pass audit of local MIMIC-CXR and MIMIC-IV files.

All row-level CXR findings stay under the configured, Git-ignored outputs directory.
This module does not build an admission match, cohort, or EHR feature table.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

from src.cohort.common import (
    _acquisition_time,
    _config_hash,
    _csv_reader,
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
    _write_csv,
)

READABILITY_COLUMNS = [
    "subject_id",
    "study_id",
    "dicom_id",
    "index_path",
    "resolved_path",
    "path_resolution",
    "path_exists",
    "metadata_readable",
    "pixel_readable",
    "failure_reason",
    "view_position",
    "acquisition_datetime",
    "acquisition_time_source",
    "time_failure_reason",
]
FILE_COLUMNS = [
    "source",
    "file",
    "exists",
    "size_bytes",
    "header_readable",
    "row_readable",
    "required_columns_present",
    "missing_columns",
    "failure_reason",
]
SUMMARY_COLUMNS = ["category", "value", "count", "denominator", "proportion"]
EHR_COLUMNS = ["metric", "count", "denominator", "proportion", "detail"]


def _csv_screen(
    path: Path, source: str, required: list[str], check: dict | None = None
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": source,
        "file": str(path),
        "exists": path.is_file(),
        "size_bytes": "",
        "header_readable": False,
        "row_readable": False,
        "required_columns_present": False,
        "missing_columns": "",
        "failure_reason": "",
    }
    if not path.is_file():
        row["failure_reason"] = "missing_file"
        return row
    row["size_bytes"] = path.stat().st_size
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = _csv_reader(stream, check)
            columns = reader.fieldnames
            if not columns:
                row["failure_reason"] = "empty_or_missing_header"
                return row
            row["header_readable"] = True
            missing = sorted(set(required) - set(columns))
            row["missing_columns"] = ";".join(missing)
            row["required_columns_present"] = not missing
            first = next(reader, None)
            row["row_readable"] = first is not None and None not in first
            if missing:
                row["failure_reason"] = "missing_required_columns"
            elif not row["row_readable"]:
                row["failure_reason"] = "empty_or_malformed_first_row"
    except (OSError, UnicodeError, csv.Error) as error:
        row["failure_reason"] = type(error).__name__
    return row


def _dicom_row(
    index_row: dict[str, str],
    cxr_root: Path,
    dicom_dir: Path,
    layout: str,
    local_files: set[Path],
    decode_pixels: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {key: "" for key in READABILITY_COLUMNS}
    for key in ("subject_id", "study_id", "dicom_id"):
        result[key] = index_row.get(key, "")
    result["index_path"] = index_row.get("path", "")
    result["path_exists"] = False
    result["metadata_readable"] = False
    result["pixel_readable"] = False
    try:
        path, strategy, exists = _indexed_dicom_path(
            cxr_root, dicom_dir, index_row, layout, local_files
        )
    except ValueError as error:
        result["failure_reason"] = str(error)
        result["acquisition_time_source"] = "not_assessed"
        return result
    result["resolved_path"] = str(path)
    result["path_resolution"] = strategy
    if not exists or not path.is_file():
        result["failure_reason"] = "missing_path"
        result["acquisition_time_source"] = "not_assessed"
        return result
    result["path_exists"] = True
    import pydicom

    try:
        dataset = pydicom.dcmread(str(path), force=False)
    except Exception as error:  # Malformed DICOM can raise beyond InvalidDicomError.
        result["failure_reason"] = "metadata_read_error:" + type(error).__name__
        result["acquisition_time_source"] = "not_assessed"
        return result
    result["metadata_readable"] = True
    try:
        result["view_position"] = (
            str(dataset.get("ViewPosition", "") or "").strip().upper()
        )
        (
            result["acquisition_datetime"],
            result["acquisition_time_source"],
            result["time_failure_reason"],
        ) = _acquisition_time(dataset)
    except Exception as error:
        result["acquisition_time_source"] = "missing"
        result["time_failure_reason"] = "time_parse_error:" + type(error).__name__
    if "PixelData" not in dataset:
        result["failure_reason"] = "missing_pixel_data"
    elif decode_pixels:
        try:
            pixels = dataset.pixel_array
            if pixels.size == 0:
                result["failure_reason"] = "empty_pixel_array"
            else:
                result["pixel_readable"] = True
        except Exception as error:  # Decoder plugins raise several exception types.
            result["failure_reason"] = "pixel_decode_error:" + type(error).__name__
    else:
        result["failure_reason"] = "pixel_decode_not_attempted"
    return result


def _audit_index_row(
    index_row: dict[str, str],
    cxr_root: Path,
    dicom_dir: Path,
    layout: str,
    local_files: set[Path],
    decode_pixels: bool,
) -> dict[str, Any]:
    if None in index_row:
        result = {key: "" for key in READABILITY_COLUMNS}
        for key in ("subject_id", "study_id", "dicom_id"):
            result[key] = index_row.get(key, "")
        result["index_path"] = index_row.get("path", "")
        result["failure_reason"] = "malformed_index_row"
        result["acquisition_time_source"] = "not_assessed"
        return result
    try:
        return _dicom_row(
            index_row, cxr_root, dicom_dir, layout, local_files, decode_pixels
        )
    except Exception as error:
        result = {key: "" for key in READABILITY_COLUMNS}
        for key in ("subject_id", "study_id", "dicom_id"):
            result[key] = index_row.get(key, "")
        result["index_path"] = index_row.get("path", "")
        result["failure_reason"] = "audit_worker_error:" + type(error).__name__
        result["acquisition_time_source"] = "not_assessed"
        return result


def _proportion(count: int, denominator: int) -> str:
    return f"{count / denominator:.6f}" if denominator else ""


def _summary_row(
    category: str, value: str, count: int, denominator: int
) -> dict[str, Any]:
    return {
        "category": category,
        "value": value,
        "count": count,
        "denominator": denominator,
        "proportion": _proportion(count, denominator),
    }


def _screen_lab_times(
    path: Path, sample_limit: int, fields: list[str], check: dict | None = None
) -> tuple[list[dict[str, Any]], str]:
    """Inspect only the first N CSV records; never load/scan the full CSV."""
    counts: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    seen = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = _csv_reader(stream, check)
            if not set(fields).issubset(reader.fieldnames or []):
                return [], "missing_lab_time_columns"
            for row in islice(reader, sample_limit):
                seen += 1
                if None in row:
                    counts["malformed_row"] += 1
                    continue
                parsed: dict[str, datetime | None] = {}
                for field in fields:
                    value = (row.get(field) or "").strip()
                    if not value:
                        counts[field + "_missing"] += 1
                        parsed[field] = None
                        continue
                    try:
                        parsed_time = datetime.fromisoformat(value)
                        parsed[field] = parsed_time
                        counts[field + "_valid"] += 1
                        formats[
                            field
                            + "_"
                            + ("timezone_aware" if parsed_time.tzinfo else "naive")
                        ] += 1
                    except ValueError:
                        parsed[field] = None
                        counts[field + "_invalid"] += 1
                charttime = parsed.get(fields[0])
                storetime = parsed.get(fields[1])
                if charttime is not None and storetime is not None:
                    counts["both_valid"] += 1
                    try:
                        if storetime < charttime:
                            counts["storetime_before_charttime"] += 1
                        elif storetime == charttime:
                            counts["storetime_equal_charttime"] += 1
                        else:
                            counts["storetime_after_charttime"] += 1
                    except TypeError:
                        counts["incomparable_timezone_types"] += 1
    except (OSError, UnicodeError, csv.Error) as error:
        return [], type(error).__name__
    rows: list[dict[str, Any]] = []
    for field in fields:
        rows.append(
            {
                "metric": field + "_raw_csv_type_string",
                "count": seen,
                "denominator": seen,
                "proportion": _proportion(seen, seen),
                "detail": "CSV has no declared column types; parsed validity follows below",
            }
        )
    pairwise_metrics = {
        "storetime_before_charttime",
        "storetime_equal_charttime",
        "storetime_after_charttime",
    }
    for metric in [
        "sample_rows",
        "malformed_row",
        *(f + suffix for f in fields for suffix in ("_missing", "_invalid", "_valid")),
        "both_valid",
        "storetime_before_charttime",
        "storetime_equal_charttime",
        "storetime_after_charttime",
        "incomparable_timezone_types",
    ]:
        count = seen if metric == "sample_rows" else counts[metric]
        denominator = counts["both_valid"] if metric in pairwise_metrics else seen
        rows.append(
            {
                "metric": metric,
                "count": count,
                "denominator": denominator,
                "proportion": _proportion(count, denominator),
                "detail": f"first {sample_limit} CSV records; CSV fields are raw strings",
            }
        )
    for metric, count in sorted(formats.items()):
        rows.append(
            {
                "metric": metric,
                "count": count,
                "denominator": seen,
                "proportion": _proportion(count, seen),
                "detail": "datetime.fromisoformat parsed; no full-file inference",
            }
        )
    return rows, ""


def run_audit(root: Path, paths_file: Path, imaging_file: Path, ehr_file: Path) -> int:
    """Write the six section-2 outputs. Return 0, or 2 on incomplete inputs/audit."""
    started = datetime.now(timezone.utc)
    paths = _json_file(paths_file)
    imaging = _json_file(imaging_file)["raw_audit"]
    ehr = _json_file(ehr_file)["raw_audit"]
    if ehr["lab_time_fields"] != ["charttime", "storetime"]:
        raise ValueError("lab_time_fields must be charttime, storetime in that order")
    if type(ehr["lab_sample_rows"]) is not int or ehr["lab_sample_rows"] < 1:
        raise ValueError("lab_sample_rows must be a positive integer")
    workers, batch_size, layout = _imaging_settings(imaging)
    cxr_root = _resolve_config_path(paths["mimic_cxr"], root, raw=True)
    iv_root = _resolve_config_path(paths["mimic_iv"], root, raw=True)
    note_setting = paths.get("mimic_note", "")
    note_root = (
        _resolve_config_path(note_setting, root, raw=True)
        if isinstance(note_setting, str)
        and note_setting
        and not note_setting.startswith("SET_YOUR_")
        else None
    )
    output_root = _resolve_config_path(paths["outputs"], root, raw=False)
    _protect_outputs(paths, root, output_root)
    with _OutputRun(output_root, 1) as run:
        audit_dir = output_root / "audit"
        dicom_dir = _under_root(cxr_root, imaging["dicom_directory"])
        record_path = _under_root(cxr_root, imaging["record_list"])
        study_path = _under_root(cxr_root, imaging["study_list"])
        required = ehr["required_tables"]
        input_files = {"cxr_record_list": record_path, "cxr_study_list": study_path}
        file_rows = [
            _csv_screen(
                record_path,
                "MIMIC-CXR",
                [
                    "subject_id",
                    "study_id",
                    "dicom_id",
                    "path",
                ],
                run.check_csv(record_path),
            ),
            _csv_screen(
                study_path,
                "MIMIC-CXR",
                ["subject_id", "study_id", "path"],
                run.check_csv(study_path),
            ),
        ]
        for name, specification in required.items():
            path = _under_root(iv_root, specification["path"])
            input_files[name] = path
            file_rows.append(
                _csv_screen(
                    path, "MIMIC-IV", specification["columns"], run.check_csv(path)
                )
            )
        if note_root is not None:
            note_file = _under_root(note_root, "note/radiology.csv")
            input_files["radiology_optional"] = note_file
            file_rows.append(
                _csv_screen(
                    note_file,
                    "MIMIC-IV-NOTE (optional)",
                    ["subject_id", "charttime", "storetime"],
                    run.check_csv(note_file),
                )
            )
        local_files, walk_errors = _scan_physical_dicom(
            dicom_dir,
            {suffix.lower() for suffix in imaging["dicom_suffixes"]},
        )
        physical_count = len(local_files)
        file_rows.append(
            {
                "source": "MIMIC-CXR",
                "file": str(dicom_dir),
                "exists": dicom_dir.is_dir(),
                "size_bytes": "",
                "header_readable": "not_applicable",
                "row_readable": "not_applicable",
                "required_columns_present": "not_applicable",
                "missing_columns": "",
                "failure_reason": ";".join(walk_errors),
            }
        )
        _write_csv(
            run.path(audit_dir / "raw_file_summary.csv"), FILE_COLUMNS, file_rows
        )

        counts: Counter[str] = Counter()
        views: Counter[str] = Counter()
        readable_views: Counter[str] = Counter()
        times: Counter[str] = Counter()
        time_failures: Counter[str] = Counter()
        failures: Counter[str] = Counter()
        path_resolutions: Counter[str] = Counter()
        existing_files: set[str] = set()
        decoded_files: set[str] = set()
        index_error = ""
        with (run.path(audit_dir / "dicom_readability.csv")).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=READABILITY_COLUMNS)
            writer.writeheader()
            if (
                file_rows[0]["required_columns_present"]
                and file_rows[0]["row_readable"]
            ):
                try:
                    with record_path.open(
                        "r", encoding="utf-8-sig", newline=""
                    ) as index:
                        reader = _csv_reader(index, run.check_csv(record_path))

                        def process(index_row: dict[str, str]) -> dict[str, Any]:
                            return _audit_index_row(
                                index_row,
                                cxr_root,
                                dicom_dir,
                                layout,
                                local_files,
                                imaging["decode_pixels"],
                            )

                        with (
                            ThreadPoolExecutor(max_workers=workers) as pool,
                            _progress("Audit indexed DICOM", unit="image") as progress,
                        ):
                            while batch := list(islice(reader, batch_size)):
                                results = (
                                    map(process, batch)
                                    if workers == 1
                                    else pool.map(process, batch)
                                )
                                for row in results:
                                    counts["index_records"] += 1
                                    writer.writerow(row)
                                    if row["path_exists"]:
                                        counts["indexed_paths_existing"] += 1
                                        existing_files.add(row["resolved_path"])
                                        path_resolutions[row["path_resolution"]] += 1
                                    if row["metadata_readable"]:
                                        counts["metadata_readable"] += 1
                                        views[row["view_position"] or "MISSING"] += 1
                                    if row["pixel_readable"]:
                                        counts["pixel_readable"] += 1
                                        decoded_files.add(row["resolved_path"])
                                        readable_views[
                                            row["view_position"] or "MISSING"
                                        ] += 1
                                    times[row["acquisition_time_source"]] += 1
                                    if row["time_failure_reason"]:
                                        time_failures[row["time_failure_reason"]] += 1
                                    if row["failure_reason"]:
                                        failures[row["failure_reason"]] += 1
                                progress.update(len(batch))
                except (OSError, UnicodeError, csv.Error) as error:
                    index_error = type(error).__name__
            else:
                index_error = "index_header_or_first_row_failed"
        view_rows = [
            {
                "view_position": view,
                "metadata_readable_count": count,
                "metadata_readable_denominator": counts["metadata_readable"],
                "pixel_readable_count": readable_views[view],
                "pixel_readable_denominator": counts["pixel_readable"],
                "is_frontal": view in imaging["frontal_views"],
            }
            for view, count in sorted(views.items())
        ]
        _write_csv(
            run.path(audit_dir / "cxr_view_distribution.csv"),
            [
                "view_position",
                "metadata_readable_count",
                "metadata_readable_denominator",
                "pixel_readable_count",
                "pixel_readable_denominator",
                "is_frontal",
            ],
            view_rows,
        )
        time_rows = (
            [
                _summary_row(
                    "time_source", source, times[source], counts["metadata_readable"]
                )
                for source in ("AcquisitionDateTime", "StudyDate+StudyTime", "missing")
            ]
            + [
                _summary_row(
                    "time_source",
                    "not_assessed",
                    times["not_assessed"],
                    counts["index_records"],
                ),
            ]
            + [
                _summary_row("time_failure", reason, count, counts["metadata_readable"])
                for reason, count in sorted(time_failures.items())
            ]
        )
        _write_csv(
            run.path(audit_dir / "dicom_time_summary.csv"), SUMMARY_COLUMNS, time_rows
        )

        lab_file = input_files["labevents"]
        if (
            lab_file.is_file()
            and next(row for row in file_rows if row["file"] == str(lab_file))[
                "required_columns_present"
            ]
        ):
            ehr_rows, ehr_error = _screen_lab_times(
                lab_file,
                int(ehr["lab_sample_rows"]),
                ehr["lab_time_fields"],
                run.check_csv(lab_file),
            )
        else:
            ehr_rows, ehr_error = [], "labevents_file_or_columns_unavailable"
        if not ehr_rows:
            ehr_rows = [
                {
                    "metric": "sample_rows",
                    "count": 0,
                    "denominator": 0,
                    "proportion": "",
                    "detail": ehr_error or "empty_file",
                }
            ]
        _write_csv(
            run.path(audit_dir / "ehr_time_field_screen.csv"), EHR_COLUMNS, ehr_rows
        )

        indexed = counts["index_records"]
        present = counts["indexed_paths_existing"]
        decoded = counts["pixel_readable"]
        frontal = sum(readable_views[view] for view in imaging["frontal_views"])
        recoverable = times["AcquisitionDateTime"] + times["StudyDate+StudyTime"]
        critical = [
            row["file"] + ": " + row["failure_reason"]
            for row in file_rows
            if row["failure_reason"] and row["source"] != "MIMIC-IV-NOTE (optional)"
        ]
        if failures["malformed_index_row"]:
            critical.append(
                f"malformed CXR index rows: {failures['malformed_index_row']}"
            )
        if not indexed or not present:
            critical.append("no indexed local DICOM images available")
        if failures["index_path_key_mismatch"] or failures["unsafe_index_path"]:
            critical.append("invalid CXR index paths or keys")
        worker_errors = sum(
            count
            for reason, count in failures.items()
            if reason.startswith("audit_worker_error:")
        )
        if worker_errors:
            critical.append(f"unexpected per-image audit errors: {worker_errors}")
        if index_error:
            critical.append("record index scan: " + index_error)
        if ehr_error:
            critical.append("lab time sample: " + ehr_error)
        if not imaging["decode_pixels"]:
            critical.append("pixel decoding disabled; readability audit incomplete")
        report = [
            "# Raw Data Audit (Step 01)",
            "",
            f"- Started (UTC): {started.isoformat()}",
            f"- Git commit: {_git_commit(root)}",
            f"- Configuration SHA256: {_config_hash([paths_file, imaging_file, ehr_file])}",
            f"- DICOM worker threads: {workers}; index rows per batch: {batch_size}",
            f"- DICOM path layout: {layout}",
            f"- Status: {'incomplete' if critical else 'audit completed'}",
            "",
            "## CXR Denominators",
            "",
            f"- Local .dcm files under dataset: {physical_count} (filesystem scan)",
            f"- Rows in cxr-record-list: {indexed}",
            f"- Index rows with an existing local path: {present}/{indexed} "
            f"({_proportion(present, indexed)})",
            f"- Unique local DICOM files referenced by the index: {len(existing_files)}",
            f"- Existing paths by resolution rule: {dict(path_resolutions)}",
            f"- Metadata readable: {counts['metadata_readable']}/{present} "
            f"({_proportion(counts['metadata_readable'], present)})",
            f"- Pixel data decoded: {decoded}/{present} ({_proportion(decoded, present)})",
            f"- Unique DICOM files with decoded pixel data: {len(decoded_files)}",
            f"- Reliable acquisition time recovered: {recoverable}/"
            f"{counts['metadata_readable']} "
            f"({_proportion(recoverable, counts['metadata_readable'])})",
            f"- AP/PA frontal images: {frontal}/{decoded} "
            f"({_proportion(frontal, decoded)})",
            "",
            "All proportions use index rows; unique files are deduplicated by resolved local path.",
            "See dicom_readability.csv for per-image failures and dicom_time_summary.csv "
            "for time sources.",
            "ViewPosition values are converted to uppercase; view-specific denominators "
            "are in cxr_view_distribution.csv.",
            "",
            "## Initial EHR Screen",
            "",
            f"- Labevents sample: up to the first {ehr['lab_sample_rows']} CSV records; "
            "do not infer full-file missingness from this sample.",
            "- CSV fields are raw strings; a valid time is parseable by "
            "datetime.fromisoformat.",
            "- This step does not calculate admission matching, the formal 24-hour window, "
            "full-cohort lab missingness, or positive patient counts.",
            "- See ehr_time_field_screen.csv for EHR metrics and denominators.",
            "",
            "## CSV Tail Checks",
            "",
            f"- Results: {json.dumps(run.csv_tail_summary(), sort_keys=True)}",
            "- Incomplete final records are skipped when encountered. Review file sources.",
            "- A complete final record does not establish whole-dataset coverage.",
            "",
            "## Input Versions (Size and Modification Time, Not Content Checksums)",
            "",
            "```json",
            json.dumps(
                {key: _file_version(path) for key, path in input_files.items()},
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            "## Execution Issues",
            "",
            *(f"- {issue}" for issue in critical),
        ]
        if not critical:
            report.append(
                "- No input or read errors prevented completion; review per-image failures."
            )
        report_path = run.path(audit_dir / "audit_report.md")
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        run.write_csv_tail_report()
        output_files = [
            run.path(audit_dir / name)
            for name in (
                "raw_file_summary.csv",
                "dicom_readability.csv",
                "dicom_time_summary.csv",
                "cxr_view_distribution.csv",
                "ehr_time_field_screen.csv",
                "audit_report.md",
                "step_01_csv_tail_checks.csv",
            )
        ]
        log_dir = output_root / "logs"
        run_log = {
            "csv_tail_checks": run.csv_tail_summary(),
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(root),
            "config_sha256": _config_hash([paths_file, imaging_file, ehr_file]),
            "dicom_workers": workers,
            "dicom_batch_size": batch_size,
            "dicom_path_layout": layout,
            "input_versions": {
                key: _file_version(path) for key, path in input_files.items()
            },
            "output_sha256": {
                target.name: _file_hash(temp)
                for target, temp in run.files.items()
                if temp in output_files
            },
            "counts": {
                "physical_dicom_files": physical_count,
                "index_records": indexed,
                "indexed_paths_existing": present,
                "metadata_readable": counts["metadata_readable"],
                "unique_indexed_files_existing": len(existing_files),
                "path_resolutions": dict(path_resolutions),
                "pixel_readable": decoded,
                "unique_pixel_readable_files": len(decoded_files),
                "time_recoverable": recoverable,
                "frontal_pixel_readable": frontal,
                "failures": dict(failures),
            },
            "critical_issues": critical,
        }
        run.path(log_dir / f"01_audit_data_{started:%Y%m%dT%H%M%SZ}.json").write_text(
            json.dumps(run_log, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        run.status = "incomplete" if critical else "complete"
        return 2 if critical else 0
