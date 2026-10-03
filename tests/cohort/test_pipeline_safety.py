"""Small synthetic regressions; no raw data or real decoder is required."""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow.parquet as pq

from src.cohort.audit_raw_data import run_audit
from src.cohort.build_cxr_index import build_cxr_index
from src.cohort.common import (
    _acquisition_time,
    _indexed_dicom_path,
    _json_file,
    _OutputRun,
    _protect_outputs,
    _run_paths,
    _verify_run,
)
from src.cohort.match_admissions import (
    _candidate_status,
    _load_relevant_admissions,
    match_admissions,
)
from src.cohort.select_frontal_image import _select_study, select_frontal_images

PROJECT = Path(__file__).resolve().parents[2]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class FakeDicom(dict):
    pixel_array = SimpleNamespace(size=4)


def fake_read(path: str, force: bool = False) -> FakeDicom:
    return FakeDicom(_json_file(Path(path)))


class PipelineSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="icu_pipeline_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def inputs(self, root: Path, records: int = 2, local: bool = True) -> tuple:
        root.mkdir(parents=True, exist_ok=True)
        cxr, iv = root / "cxr", root / "iv"
        (cxr / "dataset").mkdir(parents=True)
        rows = [
            dict(
                subject_id="10000000",
                study_id=str(100 + i),
                dicom_id=f"abc{i:x}",
                path=f"files/p10/p10000000/s{100 + i}/abc{i:x}.dcm",
            )
            for i in range(records)
        ]
        write_csv(
            cxr / "cxr-record-list.csv",
            ["subject_id", "study_id", "dicom_id", "path"],
            rows,
        )
        write_csv(
            cxr / "cxr-study-list.csv",
            ["subject_id", "study_id", "path"],
            [
                {key: row[key] for key in ("subject_id", "study_id", "path")}
                for row in rows
            ],
        )
        if local:
            image = cxr / "dataset/p10000000/s100/abc0.dcm"
            image.parent.mkdir(parents=True)
            write_json(
                image,
                dict(
                    ViewPosition="AP",
                    AcquisitionDateTime="20200101120000",
                    Rows=2,
                    Columns=2,
                    PixelData="synthetic",
                ),
            )
        ehr = _json_file(PROJECT / "configs/ehr.json")
        ehr["raw_audit"]["lab_sample_rows"] = 1
        for spec in ehr["raw_audit"]["required_tables"].values():
            row = {
                name: "2020-01-01 00:00:00"
                if name in ("charttime", "storetime")
                else "1"
                for name in spec["columns"]
            }
            write_csv(iv / spec["path"], spec["columns"], [row])
        write_csv(
            iv / "hosp/admissions.csv",
            ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime"],
            [
                dict(
                    subject_id=10000000,
                    hadm_id=10,
                    admittime="2020-01-01 00:00:00",
                    dischtime="2020-01-02 00:00:00",
                    deathtime="",
                )
            ],
        )
        paths = dict(
            mimic_cxr=str(cxr),
            mimic_iv=str(iv),
            outputs=str(root / "outputs"),
            processed=str(root / "processed"),
        )
        imaging = _json_file(PROJECT / "configs/imaging.json")
        imaging["raw_audit"].update(dicom_workers=1, dicom_batch_size=1)
        return (
            write_json(root / "paths.json", paths),
            write_json(root / "imaging.json", imaging),
            write_json(root / "ehr.json", ehr),
            write_json(
                root / "cohort.json", _json_file(PROJECT / "configs/cohort.json")
            ),
        )

    def test_paths_and_raw_protection(self) -> None:
        cxr, directory = self.root / "cxr", self.root / "cxr/dataset"
        row = dict(
            subject_id="10000000",
            study_id="100",
            dicom_id="abc",
            path="files/p10/p10000000/s100/abc.dcm",
        )
        standard, flat = directory / row["path"], directory / "p10000000/s100/abc.dcm"
        for layout, expected in (
            ("standard", standard),
            ("flattened", flat),
            ("auto", flat),
        ):
            result = _indexed_dicom_path(
                cxr, directory, row, layout, {expected}, probe_root=False
            )
            self.assertEqual((result[0], result[2]), (expected, True))
            for field, value in (
                ("subject_id", "20000000"),
                ("study_id", "200"),
                ("dicom_id", "def"),
                ("path", "../outside.dcm"),
                ("path", "C:/outside.dcm"),
            ):
                with (
                    self.subTest(layout=layout, field=field),
                    self.assertRaises(ValueError),
                ):
                    _indexed_dicom_path(
                        cxr,
                        directory,
                        dict(row, **{field: value}),
                        layout,
                        {standard, flat},
                        probe_root=False,
                    )
        paths = {"mimic_cxr": str(cxr), "mimic_iv": str(self.root / "iv")}
        with self.assertRaises(ValueError):
            _protect_outputs(paths, self.root, self.root / "iv/generated")

    def test_timezone_priority_and_validation(self) -> None:
        base = dict(StudyDate="20200101", StudyTime="120000")
        for first in ({}, {"AcquisitionDateTime": "20200101120000"}):
            for offset in ("+0800", "+1400", "-1200"):
                with self.subTest(first=first, offset=offset):
                    value, _, failure = _acquisition_time(
                        dict(base, **first, TimezoneOffsetFromUTC=offset)
                    )
                    self.assertTrue(value.endswith(offset[:3] + ":" + offset[3:]))
                    self.assertEqual(failure, "")
            for offset in ("+1401", "-1201", "+0860", "-0000", "UTC", "+08:00"):
                self.assertEqual(
                    _acquisition_time(
                        dict(base, **first, TimezoneOffsetFromUTC=offset)
                    ),
                    ("", "missing", "invalid_timezone_offset"),
                )
        self.assertEqual(
            _acquisition_time(
                dict(
                    base,
                    AcquisitionDateTime="20200101110000+0200",
                    TimezoneOffsetFromUTC="bad",
                )
            ),
            ("2020-01-01T11:00:00+02:00", "AcquisitionDateTime", ""),
        )
        self.assertEqual(
            _acquisition_time(dict(base, AcquisitionDateTime="bad"))[1],
            "StudyDate+StudyTime",
        )
        self.assertEqual(_acquisition_time(base)[0], "2020-01-01T12:00:00")
        self.assertEqual(_acquisition_time({})[1], "missing")

    def test_duplicate_admissions_and_boundaries(self) -> None:
        path = self.root / "admissions.csv"
        row = dict(
            subject_id=1,
            hadm_id=10,
            admittime="2020-01-01 00:00:00",
            dischtime="2020-01-02 00:00:00",
        )
        for second_subject in (1, 2, 999):
            write_csv(path, list(row), [row, dict(row, subject_id=second_subject)])
            with self.assertRaisesRegex(ValueError, "Duplicate admissions hadm_id"):
                _load_relevant_admissions(path, {1, 2})
        write_csv(path, list(row), [row])
        admissions, counts = _load_relevant_admissions(path, {1})
        self.assertEqual(counts["duplicate_hadm_id_rows"], 0)
        self.assertEqual(
            _candidate_status(datetime(2020, 1, 1), admissions[1][0])[0], "confirmed"
        )
        self.assertEqual(
            _candidate_status(datetime(2020, 1, 2), admissions[1][0])[0], "outside"
        )

    def test_empty_and_partial_audit(self) -> None:
        with patch.dict(sys.modules, {"pydicom": SimpleNamespace(dcmread=fake_read)}):
            for records, local, expected in (
                (0, False, 2),
                (2, False, 2),
                (2, True, 0),
            ):
                root = self.root / f"case_{records}_{local}"
                paths, imaging, ehr, _ = self.inputs(root, records, local)
                self.assertEqual(run_audit(root, paths, imaging, ehr), expected)
                outputs = root / "outputs"
                marker = _json_file(_run_paths(outputs, 1)[0])
                self.assertEqual(
                    marker["status"], "complete" if expected == 0 else "incomplete"
                )
                if expected:
                    with self.assertRaisesRegex(ValueError, "incomplete"):
                        _verify_run(outputs, 1)
                else:
                    _verify_run(outputs, 1)
                    log = _json_file(next((outputs / "logs").glob("*.json")))
                    self.assertEqual(log["counts"]["indexed_paths_existing"], 1)
                    self.assertIn("dicom_readability.csv", log["output_sha256"])

    def test_full_pipeline_rerun_and_failed_publication(self) -> None:
        paths, imaging, ehr, cohort = self.inputs(self.root)
        outputs, processed = self.root / "outputs", self.root / "processed"
        runners = [
            lambda: run_audit(self.root, paths, imaging, ehr),
            lambda: build_cxr_index(self.root, paths, imaging),
            lambda: select_frontal_images(self.root, paths, imaging),
            lambda: match_admissions(self.root, paths, cohort),
        ]
        fail_names = [
            "audit_report.md",
            "cxr_index_summary.json",
            "study_image_selection_summary.json",
            "admission_match_summary.csv",
        ]
        with (
            patch.dict(sys.modules, {"pydicom": SimpleNamespace(dcmread=fake_read)}),
            redirect_stdout(io.StringIO()),
        ):
            for _ in range(2):
                for step, runner in enumerate(runners, 1):
                    self.assertEqual(runner(), 0)
                    _verify_run(outputs, step)
            matches = pq.read_table(
                processed / "index/cxr_admission_matches.parquet"
            ).to_pylist()
            self.assertEqual(
                (len(matches), matches[0]["match_status"]), (1, "unique_match")
            )
            for step, (runner, fail_name) in enumerate(zip(runners, fail_names), 1):
                marker = _run_paths(outputs, step)[0]
                saved = {
                    Path(name): Path(name).read_bytes()
                    for name in _json_file(marker)["files"]
                }
                saved[marker] = marker.read_bytes()
                for error in (PermissionError, KeyboardInterrupt):
                    original, failed = os.replace, False

                    def failing_replace(source, target):
                        nonlocal failed
                        if Path(target).name == fail_name and not failed:
                            failed = True
                            raise error("synthetic publication failure")
                        return original(source, target)

                    with (
                        patch(
                            "src.cohort.common.os.replace", side_effect=failing_replace
                        ),
                        self.assertRaises(error),
                    ):
                        runner()
                    self.assertTrue(failed)
                    for path, value in saved.items():
                        self.assertEqual(path.read_bytes(), value)
                    _verify_run(outputs, step)
            index = processed / "index/cxr_index.parquet"
            previous = index.read_bytes()
            index.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                runners[2]()
            index.write_bytes(previous)
            admission_path = self.root / "iv/hosp/admissions.csv"
            with admission_path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                assert reader.fieldnames is not None
                fields, rows = list(reader.fieldnames), list(reader)
            write_csv(admission_path, fields, rows + rows)
            marker = _run_paths(outputs, 4)[0]
            old_marker = marker.read_bytes()
            with self.assertRaisesRegex(ValueError, "Duplicate admissions hadm_id"):
                runners[3]()
            self.assertEqual(marker.read_bytes(), old_marker)

    def test_selection_order_and_timezone_conflict(self) -> None:
        def image(dicom: str, hour: int, area: int) -> dict:
            return dict(
                subject_id=1,
                study_id=10,
                dicom_id=dicom,
                dicom_path=dicom,
                view_position="AP",
                is_frontal=True,
                is_readable=True,
                rows=area,
                columns=1,
                acquisition_datetime=f"2020-01-01T{hour:02d}:00:00",
                acquisition_time_source="AcquisitionDateTime",
                _parsed_time=datetime(2020, 1, 1, hour),
            )

        for images, expected, reason in (
            ([image("b", 10, 100), image("a", 11, 200)], "b", "earliest_frontal"),
            ([image("b", 10, 100), image("a", 10, 200)], "a", "largest_pixel_area"),
            ([image("b", 10, 100), image("a", 10, 100)], "a", "dicom_id_tiebreak"),
        ):
            chosen, actual = _select_study(images)
            assert chosen is not None
            self.assertEqual((chosen["selected_dicom_id"], actual), (expected, reason))
        conflict = image("a", 10, 100)
        conflict["_parsed_time"] = datetime.fromisoformat("2020-01-01T10:00:00+08:00")
        with self.assertRaises(ValueError):
            _select_study([image("b", 10, 100), conflict])

    def test_crash_recovery_and_missing_manifest(self) -> None:
        outputs = self.root / "outputs"
        first, second = outputs / "a.txt", outputs / "b.txt"
        with self.assertRaisesRegex(ValueError, "Missing step"):
            _verify_run(outputs, 2)
        with _OutputRun(outputs, 2) as run:
            run.path(first).write_text("old-a")
            run.path(second).write_text("old-b")
        marker = _run_paths(outputs, 2)[0]
        previous = marker.read_bytes()
        script = """
import os, sys
from pathlib import Path
from src.cohort.common import _OutputRun
outputs = Path(sys.argv[1])
original = os.replace
def crash(source, target):
    if Path(target).name == 'b.txt':
        os._exit(23)
    return original(source, target)
os.replace = crash
with _OutputRun(outputs, 2) as run:
    run.path(outputs / 'a.txt').write_text('new-a')
    run.path(outputs / 'b.txt').write_text('new-b')
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(outputs)],
            cwd=PROJECT,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 23, result.stderr.decode(errors="replace"))
        self.assertEqual(first.read_text(), "new-a")
        with self.assertRaisesRegex(ValueError, "interrupted publication"):
            _verify_run(outputs, 2)
        with _OutputRun(outputs, 2):
            self.assertEqual(
                (first.read_text(), second.read_text()), ("old-a", "old-b")
            )
        self.assertEqual(marker.read_bytes(), previous)
        _verify_run(outputs, 2)
        self.assertFalse(list(outputs.rglob(".step_02_*")))

    def test_first_publication_and_validation_failure(self) -> None:
        outputs = self.root / "outputs"
        first, second = outputs / "a.txt", outputs / "b.txt"
        original = os.replace
        failed = False

        def failing_replace(source, target):
            nonlocal failed
            if Path(target) == second and not failed:
                failed = True
                raise PermissionError("synthetic first publication failure")
            return original(source, target)

        with (
            self.assertRaises(PermissionError),
            patch("src.cohort.common.os.replace", side_effect=failing_replace),
            _OutputRun(outputs, 2) as run,
        ):
            run.path(first).write_text("new-a")
            run.path(second).write_text("new-b")
        self.assertFalse(first.exists() or second.exists())
        self.assertFalse(_run_paths(outputs, 2)[0].exists())
        with self.assertRaises(KeyboardInterrupt), _OutputRun(outputs, 2) as run:
            run.path(first).write_text("staged")
            raise KeyboardInterrupt("interrupted before publication")
        self.assertFalse(list(outputs.rglob(".step_02_*")))

        outside = self.root / "outside.txt"
        outside.write_text("keep")
        pending = _run_paths(outputs, 2)[1]
        write_json(
            pending,
            {
                "files": [
                    {
                        "target": str(outside),
                        "backup": None,
                        "staged": str(outputs / ".step_02_stage_invalid"),
                    }
                ]
            },
        )
        with self.assertRaisesRegex(ValueError, "outside configured output roots"):
            with _OutputRun(outputs, 2):
                pass
        self.assertEqual(outside.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
