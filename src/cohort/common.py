"""Shared validation and recoverable publication for the data pipeline."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from tqdm import tqdm

FULL_DT = re.compile(r"^\d{14}(?:\.\d{1,6})?(?:[+-]\d{4})?$")
FULL_TM = re.compile(r"^\d{6}(?:\.\d{1,6})?$")


def _json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _config_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _resolve_config_path(value: str, root: Path, *, raw: bool) -> Path:
    if not isinstance(value, str) or not value.strip() or value.startswith("SET_YOUR_"):
        raise ValueError("paths.local.json contains a missing or placeholder path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        if raw:
            raise ValueError(f"Raw data root must be absolute: {value}")
        path = root / path
    return path.resolve()


def _under_root(root: Path, relative: str) -> Path:
    """Reject absolute/traversal paths from untrusted index or config CSV text."""
    normalized = relative.replace("\\", "/")
    parts = normalized.split("/")
    if (
        not normalized
        or normalized.startswith("/")
        or ":" in parts[0]
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError("unsafe_relative_path")
    candidate = root.joinpath(*parts).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("unsafe_relative_path")
    return candidate


def _indexed_dicom_path(
    cxr_root: Path,
    dicom_dir: Path,
    index_row: dict[str, str],
    layout: str,
    local_files: set[Path],
    *,
    probe_root: bool = True,
) -> tuple[Path, str, bool]:
    """Resolve a validated index path; skip root probes after a full root scan."""
    value = index_row.get("path", "")
    parts = value.replace("\\", "/").split("/")
    if (
        not value
        or value.startswith(("/", "\\"))
        or ":" in parts[0]
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError("unsafe_index_path")

    standard = dicom_dir.joinpath(*parts)
    root_relative = cxr_root.joinpath(*parts)
    candidates = [(standard, "standard_dataset"), (root_relative, "standard_root")]
    if (
        len(parts) == 5
        and parts[0] == "files"
        and re.fullmatch(r"\d+", index_row.get("subject_id", ""))
        and re.fullmatch(r"\d+", index_row.get("study_id", ""))
        and re.fullmatch(r"[0-9a-fA-F-]+", index_row.get("dicom_id", ""))
        and parts[1] == "p" + index_row["subject_id"][:2]
        and parts[2] == "p" + index_row["subject_id"]
        and parts[3] == "s" + index_row["study_id"]
        and parts[4] == index_row["dicom_id"] + ".dcm"
    ):
        flattened = dicom_dir.joinpath(*parts[2:])
    else:
        raise ValueError("index_path_key_mismatch")

    if layout == "flattened":
        candidates = [(flattened, "flattened")]
    elif layout == "standard":
        candidates = [(standard, "standard_dataset")]
    elif layout == "auto":
        candidates.append((flattened, "flattened"))
    else:
        raise ValueError("Unsupported dicom_path_layout")

    for candidate, strategy in candidates:
        if candidate in local_files or (
            probe_root and strategy == "standard_root" and candidate.is_file()
        ):
            if not candidate.resolve().is_relative_to(cxr_root):
                raise ValueError("unsafe_index_path")
            return candidate, strategy, True
    candidate, strategy = candidates[0]
    return candidate, strategy, False


def _timestamp(value: object, *, dicom_date: bool = False) -> datetime | None:
    if isinstance(value, datetime) and not dicom_date:
        return value
    text = str(getattr(value, "original_string", value) or "").strip()
    if dicom_date:
        if not re.fullmatch(r"\d{8}", text):
            return None
        try:
            return datetime.strptime(text, "%Y%m%d")
        except ValueError:
            return None
    if not FULL_DT.fullmatch(text):
        return None
    suffix = re.search(r"[+-]\d{4}$", text)
    if suffix and _dicom_timezone(suffix.group()) is None:
        return None
    zone = "%z" if suffix else ""
    fraction = ".%f" if "." in text else ""
    try:
        return datetime.strptime(text, "%Y%m%d%H%M%S" + fraction + zone)
    except ValueError:
        return None


def _dicom_timezone(value: str) -> timezone | None:
    if not re.fullmatch(r"[+-]\d{4}", value) or value == "-0000":
        return None
    hours, minutes = int(value[1:3]), int(value[3:5])
    offset = (hours * 60 + minutes) * (1 if value[0] == "+" else -1)
    return (
        timezone(timedelta(minutes=offset))
        if minutes < 60 and -720 <= offset <= 840
        else None
    )


def _acquisition_time(dataset: Any) -> tuple[str, str, str]:
    first = dataset.get("AcquisitionDateTime")
    parsed, source = _timestamp(first), "AcquisitionDateTime"
    if parsed is None:
        date_value, time_value = (
            dataset.get("StudyDate", ""),
            dataset.get("StudyTime", ""),
        )
        date = str(getattr(date_value, "original_string", date_value) or "").strip()
        clock = str(getattr(time_value, "original_string", time_value) or "").strip()
        parsed = (
            _timestamp(date + clock)
            if _timestamp(date, dicom_date=True) and FULL_TM.fullmatch(clock)
            else None
        )
        source = "StudyDate+StudyTime"
    if parsed is None:
        failure = (
            "invalid_acquisition_datetime_and_no_valid_study_time"
            if first
            else "missing_or_invalid_study_date_time"
        )
        return "", "missing", failure
    offset = str(dataset.get("TimezoneOffsetFromUTC", "") or "").strip()
    if parsed.tzinfo is None and offset:
        zone = _dicom_timezone(offset)
        if zone is None:
            return "", "missing", "invalid_timezone_offset"
        parsed = parsed.replace(tzinfo=zone)
    return parsed.isoformat(), source, ""


def _file_version(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


CSV_TAIL_COLUMNS = [
    "file",
    "size_bytes",
    "status",
    "expected_fields",
    "actual_fields",
    "reason",
    "skipped_rows",
    "header_text",
    "last_line_text",
    "last_line_truncated",
]
CSV_TAIL_BYTES = 64 * 1024


def _tail_result(check: dict, status: str, actual: int | str, reason: str) -> None:
    changed = (check["status"], check["actual_fields"], check["reason"]) != (
        status,
        actual,
        reason,
    )
    check.update(status=status, actual_fields=actual, reason=reason)
    if changed:
        prefix = "WARNING" if status != "complete" else "CSV tail"
        print(
            f"{prefix}: {check['file']}: {status} ({reason or 'field count matches'})"
        )


def _check_csv_tail(path: Path) -> dict[str, Any]:
    """Inspect bounded input bytes; large logical boundaries need reader confirmation."""
    check = dict(
        file=str(path.resolve()),
        size_bytes="",
        status="pending",
        expected_fields="",
        actual_fields="",
        reason="",
        skipped_rows=0,
        header_text="",
        last_line_text="",
        last_line_truncated=False,
    )
    if not path.is_file():
        _tail_result(check, "missing", "", "missing_file")
        return check
    check["size_bytes"] = path.stat().st_size
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(iter(stream.readline, ""), strict=True), [])
            header_end = stream.tell()
        with path.open("rb") as stream:
            check["header_text"] = (
                stream.read(header_end).decode("utf-8-sig").rstrip("\r\n")
            )
            # Include the preceding byte to recognize a line exactly at the bound.
            tail_start = max(0, check["size_bytes"] - CSV_TAIL_BYTES - 1)
            stream.seek(tail_start)
            tail = stream.read().rstrip(b"\r\n")
        separator = tail.rfind(b"\n")
        last_line = tail[separator + 1 :]
        check["last_line_truncated"] = tail_start > 0 and separator < 0
        if check["last_line_truncated"]:
            last_line = last_line[-CSV_TAIL_BYTES:]
        check["last_line_text"] = last_line.decode("utf-8", errors="replace")
        check["expected_fields"] = len(header)
        if not header:
            _tail_result(check, "unreadable", "", "empty_or_missing_header")
        elif check["size_bytes"] <= CSV_TAIL_BYTES:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = _CsvDictReader(stream, check)
                list(reader)
            # Inspection identifies a bad tail but does not consume application rows.
            check["skipped_rows"] = 0
        else:
            try:
                check["actual_fields"] = len(
                    next(csv.reader([last_line.decode("utf-8")], strict=True))
                )
            except (csv.Error, UnicodeError, StopIteration):
                pass
            # A physical line can belong to a quoted field opened outside this suffix.
            _tail_result(
                check,
                "unverified",
                check["actual_fields"],
                "logical_record_boundary_requires_reader_confirmation",
            )
    except (OSError, UnicodeError, csv.Error) as error:
        _tail_result(check, "unreadable", "", type(error).__name__)
    return check


class _CsvDictReader(csv.DictReader):
    """Skip only a structurally incomplete record consumed at the actual EOF."""

    def __init__(self, stream: Any, check: dict) -> None:
        # readline preserves tell(), unlike TextIOWrapper's iteration protocol.
        super().__init__(iter(stream.readline, ""), strict=True)
        self.stream, self.check = stream, check
        self.last_fields: int | str = ""

    def __next__(self) -> dict:
        try:
            row = super().__next__()
        except csv.Error as error:
            if (
                str(error) != "unexpected end of data"
                or not self.check["expected_fields"]
                or not self._at_eof()
            ):
                raise
            self._skip("unterminated_quoted_record", "")
            raise StopIteration from error
        except StopIteration:
            if self.check["status"] not in {"incomplete", "invalid"}:
                _tail_result(
                    self.check,
                    "complete",
                    self.last_fields,
                    "header_only" if self.last_fields == "" else "",
                )
            raise
        expected = len(self.fieldnames or [])
        actual = sum(row.get(name) is not None for name in self.fieldnames or [])
        actual += len(row.get(None, []))
        self.last_fields = actual
        if actual < expected and self._at_eof():
            self._skip("missing_final_fields", actual)
            raise StopIteration
        if actual < expected:
            raise csv.Error(
                f"Missing CSV fields before EOF: expected {expected}, got {actual}"
            )
        if self._at_eof():
            _tail_result(
                self.check,
                "complete" if actual == expected else "invalid",
                actual,
                "" if actual == expected else "extra_final_fields",
            )
        return row

    def _at_eof(self) -> bool:
        return self.stream.tell() == self.check["size_bytes"]

    def _skip(self, reason: str, actual: int | str) -> None:
        self.check["skipped_rows"] = 1
        _tail_result(self.check, "incomplete", actual, reason)


def _csv_reader(stream: Any, check: dict | None = None) -> _CsvDictReader:
    return _CsvDictReader(
        stream, check if check is not None else _check_csv_tail(Path(stream.name))
    )


def _validate_csv_rejects(path: Path, check: dict, rejects: list[tuple]) -> None:
    """Validate DuckDB rejects against the raw EOF suffix before publishing rows."""
    records = {(position, raw) for position, raw in rejects}
    if not records:
        if check["status"] != "incomplete":
            _tail_result(check, "complete", check["expected_fields"], "")
        return
    if len(records) != 1:
        raise ValueError(f"Multiple malformed CSV records in {path}; refusing to skip")
    position, reported = records.pop()
    with path.open("rb") as stream:
        stream.seek(max(0, position - 1))
        suffix = stream.read(max(CSV_TAIL_BYTES, len(reported.encode("utf-8")) * 2 + 8))
        if stream.tell() != check["size_bytes"]:
            raise ValueError(f"Malformed CSV record is not at EOF in {path}")
    raw = suffix.decode("utf-8").strip("\r\n")
    # DuckDB diagnostics may escape embedded line endings and include a CRLF byte.
    reported = reported.strip("\r\n")
    alternatives = {reported, reported.replace("\\r", "\r").replace("\\n", "\n")}
    if raw not in alternatives:
        raise ValueError(f"Malformed CSV record is not the final record in {path}")
    try:
        rows = list(csv.reader(io.StringIO(raw, newline=""), strict=True))
        if len(rows) != 1 or len(rows[0]) >= check["expected_fields"]:
            raise ValueError(f"Final CSV error is not an incomplete record in {path}")
        actual, reason = len(rows[0]), "missing_final_fields"
    except csv.Error as error:
        if str(error) != "unexpected end of data":
            raise ValueError(f"Unsupported final CSV error in {path}") from error
        actual, reason = "", "unterminated_quoted_record"
    check["skipped_rows"] = 1
    _tail_result(check, "incomplete", actual, reason)


def _scan_physical_dicom(
    directory: Path, suffixes: set[str]
) -> tuple[set[Path], list[str]]:
    if not directory.is_dir():
        return set(), ["missing_dicom_directory"]
    files: set[Path] = set()
    errors: list[str] = []

    def on_error(error: OSError) -> None:
        errors.append(f"{type(error).__name__}: {error.filename}")

    with _progress("Scan local DICOM", unit="file") as progress:
        for current_dir, _, names in os.walk(
            directory, onerror=on_error, followlinks=False
        ):
            found = 0
            for name in names:
                if Path(name).suffix.lower() in suffixes:
                    files.add(Path(current_dir) / name)
                    found += 1
            progress.update(found)
    return files, errors


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _protect_outputs(paths: dict[str, Any], root: Path, *destinations: Path) -> None:
    raw_roots = [
        _resolve_config_path(paths[name], root, raw=True)
        for name in ("mimic_cxr", "mimic_iv")
    ]
    note = paths.get("mimic_note")
    if isinstance(note, str) and note and not note.startswith("SET_YOUR_"):
        raw_roots.append(_resolve_config_path(note, root, raw=True))
    if any(
        destination.resolve().is_relative_to(raw_root)
        for destination in destinations
        for raw_root in raw_roots
    ):
        raise ValueError("Generated output must be outside all raw data roots")


def _temporary_path(directory: Path, prefix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=prefix, dir=directory)
    os.close(descriptor)
    return Path(name)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = _temporary_path(path.parent, ".manifest_")
    try:
        temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _run_paths(outputs: Path, step: int) -> tuple[Path, Path]:
    directory = outputs / "audit"
    return (
        directory / f"step_{step:02d}_manifest.json",
        directory / f".step_{step:02d}_pending.json",
    )


def _verify_run(outputs: Path, step: int, required: Path | None = None) -> None:
    manifest, pending = _run_paths(outputs, step)
    if pending.exists():
        raise ValueError(
            f"Step {step:02d} has an interrupted publication; rerun that step"
        )
    if not manifest.is_file():
        raise ValueError(f"Missing step {step:02d} manifest; rerun that step")
    value = _json_file(manifest)
    files = value.get("files")
    if (
        value.get("step") != step
        or value.get("status") != "complete"
        or not isinstance(files, dict)
        or not files
    ):
        raise ValueError(f"Step {step:02d} is incomplete; rerun that step")
    if required is not None and str(required.resolve()) not in files:
        raise ValueError(f"Step {step:02d} manifest does not cover {required}")
    for name, digest in files.items():
        path = Path(name)
        if not path.is_file() or _file_hash(path) != digest:
            raise ValueError(f"Step {step:02d} output checksum mismatch: {path}")


class _OutputRun:
    """Stage files, then publish with a rollback journal and a completion marker."""

    # ponytail: one writer per step; add locking if concurrent runs are required.
    def __init__(self, outputs: Path, step: int, *roots: Path) -> None:
        self.manifest, self.pending = _run_paths(outputs, step)
        self.prefix = f".step_{step:02d}_"
        self.roots = tuple(path.resolve() for path in (outputs, *roots))
        self.step = step
        self.run_id = uuid4().hex
        self.status = "complete"
        self.files: dict[Path, Path] = {}
        self.csv_checks: dict[Path, dict] = {}

    def check_csv(self, path: Path) -> dict:
        path = path.resolve()
        if path not in self.csv_checks:
            self.csv_checks[path] = _check_csv_tail(path)
        return self.csv_checks[path]

    def csv_tail_summary(self) -> dict:
        values = list(self.csv_checks.values())
        return {
            "report": f"step_{self.step:02d}_csv_tail_checks.csv",
            "status": "checked" if values else "no_csv_inputs",
            "checked_file_count": len(values),
            "incomplete_file_count": sum(v["status"] == "incomplete" for v in values),
            "unverified_file_count": sum(v["status"] == "unverified" for v in values),
            "skipped_record_count": sum(v["skipped_rows"] for v in values),
        }

    def write_csv_tail_report(self) -> Path:
        target = self.manifest.parent / f"step_{self.step:02d}_csv_tail_checks.csv"
        values = list(self.csv_checks.values()) or [
            {
                **dict.fromkeys(CSV_TAIL_COLUMNS, ""),
                "status": "no_csv_inputs",
                "skipped_rows": 0,
            }
        ]
        _write_csv(self.path(target), CSV_TAIL_COLUMNS, values)
        return target

    def _checked_path(self, path: Path) -> Path:
        path = path.resolve()
        if not any(path.is_relative_to(root) for root in self.roots):
            raise ValueError(
                f"Output recovery path is outside configured output roots: {path}"
            )
        return path

    def _recover(self) -> None:
        if not self.pending.is_file():
            return
        entries = _json_file(self.pending)["files"]
        # Validate the entire journal before changing any files.
        for item in entries:
            target = self._checked_path(Path(item["target"]))
            staged = self._checked_path(Path(item["staged"]))
            if staged.parent != target.parent or not staged.name.startswith(
                self.prefix + "stage_"
            ):
                raise ValueError("Invalid output recovery staging path")
            if item["backup"] is not None:
                backup = self._checked_path(Path(item["backup"]))
                if backup.parent != target.parent or not backup.name.startswith(
                    self.prefix + "backup_"
                ):
                    raise ValueError("Invalid output recovery backup path")
                if not backup.is_file():
                    raise FileNotFoundError(f"Missing output recovery backup: {backup}")
        for item in entries:
            target = Path(item["target"])
            if item["backup"] is None:
                target.unlink(missing_ok=True)
            else:
                restored = _temporary_path(target.parent, self.prefix + "restore_")
                try:
                    shutil.copy2(item["backup"], restored)
                    os.replace(restored, target)
                finally:
                    restored.unlink(missing_ok=True)
        self.pending.unlink()
        for item in entries:
            if item["backup"] is not None:
                Path(item["backup"]).unlink(missing_ok=True)
            Path(item["staged"]).unlink(missing_ok=True)

    def __enter__(self) -> _OutputRun:
        self._recover()
        return self

    def path(self, target: Path) -> Path:
        target = self._checked_path(target)
        if target not in self.files:
            self.files[target] = _temporary_path(target.parent, self.prefix + "stage_")
        return self.files[target]

    def _publish(self) -> None:
        files = {str(path): _file_hash(temp) for path, temp in self.files.items()}
        self.path(self.manifest).write_text(
            json.dumps(
                {
                    "step": self.step,
                    "run_id": self.run_id,
                    "status": self.status,
                    "files": files,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        entries: list[dict[str, Any]] = []
        try:
            for target, temp in self.files.items():
                backup = None
                if target.exists():
                    backup = _temporary_path(target.parent, self.prefix + "backup_")
                    # Register before copying so failed copies are also cleaned up.
                    entries.append(
                        {
                            "target": str(target),
                            "backup": str(backup),
                            "staged": str(temp),
                        }
                    )
                    shutil.copy2(target, backup)
                else:
                    entries.append(
                        {"target": str(target), "backup": None, "staged": str(temp)}
                    )
            _atomic_json(self.pending, {"files": entries})
            for target, temp in self.files.items():
                os.replace(temp, target)
            self.pending.unlink()  # Commit point: all files and the marker are now published.
        except BaseException:
            if self.pending.exists():
                self._recover()
            raise
        finally:
            if not self.pending.exists():
                for item in entries:
                    if item["backup"] is not None:
                        Path(item["backup"]).unlink(missing_ok=True)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc_type is None and self.files:
                self.write_csv_tail_report()
                self._publish()
        finally:
            for temp in self.files.values():
                temp.unlink(missing_ok=True)


def _imaging_settings(imaging: dict[str, Any]) -> tuple[int, int, str]:
    workers, batch_size, layout = (
        imaging["dicom_workers"],
        imaging["dicom_batch_size"],
        imaging["dicom_path_layout"],
    )
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("dicom_workers must be an integer from 1 to 32")
    if type(batch_size) is not int or not workers <= batch_size <= 4096:
        raise ValueError("dicom_batch_size must be between workers and 4096")
    if layout not in {"auto", "standard", "flattened"}:
        raise ValueError("Unsupported dicom_path_layout")
    if imaging["time_priority"] != ["AcquisitionDateTime", "StudyDate+StudyTime"]:
        raise ValueError("Unsupported time_priority")
    return workers, batch_size, layout


def _progress(description: str, *, unit: str = "row", total: int | None = None) -> Any:
    return tqdm(
        desc=description,
        total=total,
        unit=unit,
        unit_scale=True,
        file=sys.stderr,
        disable=not sys.stderr.isatty(),
    )
