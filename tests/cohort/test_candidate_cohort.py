"""Synthetic checks for configurable study-level candidate audits."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.apply_exclusion import (
    build_icu_timeline,
    cross_audit,
    icu_signature,
)
from src.cohort.build_cohort import ICU_CAREUNITS, build_candidate_cohort
from src.cohort.build_labels import evaluate_label

BASE = datetime(2020, 1, 1)


def at(hours: int) -> str:
    return (BASE + timedelta(hours=hours)).isoformat()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class LabelTests(TestCase):
    def test_icu_boundaries_and_death(self) -> None:
        for hours, expected in [(5, None), (6, None), (7, 1), (48, 1), (49, 0)]:
            result = evaluate_label(
                BASE,
                BASE + timedelta(hours=100),
                BASE + timedelta(hours=hours),
                None,
                0,
                6,
                48,
            )
            self.assertEqual(result.label, expected)
        dead = evaluate_label(
            BASE,
            BASE + timedelta(hours=20),
            None,
            BASE + timedelta(hours=10),
            1,
            6,
            48,
        )
        self.assertTrue(dead.death_within_H_without_icu)
        alive_discharge = evaluate_label(
            BASE, BASE + timedelta(hours=10), None, None, 0, 6, 48
        )
        self.assertEqual(alive_discharge.label, 0)
        self.assertEqual(
            evaluate_label(
                BASE,
                BASE + timedelta(hours=100),
                BASE + timedelta(hours=60),
                None,
                0,
                6,
                48,
            ).label,
            0,
        )
        self.assertEqual(
            evaluate_label(
                BASE,
                BASE + timedelta(hours=100),
                BASE + timedelta(hours=60),
                None,
                0,
                6,
                72,
            ).label,
            1,
        )
        for hours, expected in [(72, 1), (73, 0)]:
            self.assertEqual(
                evaluate_label(
                    BASE, BASE + timedelta(hours=100),
                    BASE + timedelta(hours=hours), None, 0, 6, 72,
                ).label,
                expected,
            )
        self.assertTrue(
            evaluate_label(
                BASE, BASE + timedelta(hours=100), None,
                BASE + timedelta(hours=60), 1, 6, 72,
            ).death_within_H_without_icu
        )

    def test_cross_audit_excludes_only_changed_decision(self) -> None:
        unit = "Medical Intensive Care Unit (MICU)"

        def timeline(hours: int):
            return build_icu_timeline(
                [
                    {
                        "careunit": unit,
                        "transfer_id": "1",
                        "intime": at(hours),
                        "outtime": at(hours + 10),
                        "first_careunit": unit,
                        "last_careunit": unit,
                        "stay_id": "1",
                    }
                ],
                ICU_CAREUNITS,
                "transfers",
            )

        transfer = timeline(7)
        self.assertEqual(
            icu_signature(transfer, BASE + timedelta(hours=7), 6, 48), "at_t0"
        )
        self.assertEqual(
            icu_signature(transfer, BASE + timedelta(hours=17), 6, 48), "prior"
        )
        similar_stay = build_icu_timeline(
            [
                {
                    "first_careunit": unit,
                    "last_careunit": unit,
                    "stay_id": "1",
                    "intime": at(8),
                    "outtime": at(18),
                }
            ],
            ICU_CAREUNITS,
            "icustays",
        )
        later_stay = build_icu_timeline(
            [
                {
                    "first_careunit": unit,
                    "last_careunit": unit,
                    "stay_id": "1",
                    "intime": at(60),
                    "outtime": at(70),
                }
            ],
            ICU_CAREUNITS,
            "icustays",
        )
        death_late_stay = build_icu_timeline(
            [
                {
                    "first_careunit": unit,
                    "last_careunit": unit,
                    "stay_id": "1",
                    "intime": at(20),
                    "outtime": at(30),
                }
            ],
            ICU_CAREUNITS,
            "icustays",
        )
        self.assertEqual(
            cross_audit(transfer, similar_stay, BASE, 6, 48),
            ("time_disagreement", False),
        )
        self.assertEqual(
            cross_audit(transfer, later_stay, BASE, 6, 48), ("time_disagreement", True)
        )
        self.assertEqual(
            cross_audit(
                transfer,
                death_late_stay,
                BASE,
                6,
                48,
                BASE + timedelta(hours=100),
                BASE + timedelta(hours=10),
                1,
            ),
            ("time_disagreement", True),
        )

    def test_invalid_interval_and_timezone_are_not_negative(self) -> None:
        unit = "Medical Intensive Care Unit (MICU)"
        invalid = build_icu_timeline(
            [{"careunit": unit, "transfer_id": "1", "intime": at(8), "outtime": at(7)}],
            ICU_CAREUNITS,
            "transfers",
        )
        self.assertEqual(icu_signature(invalid, BASE, 6, 48), "unknown")
        aware = build_icu_timeline(
            [
                {
                    "careunit": unit,
                    "transfer_id": "1",
                    "intime": at(8) + "+00:00",
                    "outtime": at(18) + "+00:00",
                }
            ],
            ICU_CAREUNITS,
            "transfers",
        )
        self.assertEqual(icu_signature(aware, BASE, 6, 48), "unknown")
        self.assertEqual(
            evaluate_label(
                BASE, BASE + timedelta(hours=100), None, None, None, 6, 48
            ).status,
            "unknown_death_status",
        )


class CandidateIntegrationTests(TestCase):
    def test_complete_audit_and_repeat_run(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            iv = root / "raw_iv"
            index = root / "processed" / "index"
            index.mkdir(parents=True)
            configs = root / "configs"
            configs.mkdir()
            paths_file = configs / "paths.local.json"
            cohort_file = configs / "cohort.json"
            paths_file.write_text(
                json.dumps(
                    {
                        "mimic_iv": str(iv),
                        "processed": "processed",
                        "outputs": "outputs",
                    }
                ),
                encoding="utf-8",
            )
            cohort_file.write_text(
                json.dumps(
                    {
                        "sample_unit": "study",
                        "ehr_history_hours": 24,
                        "positive_count_unit": "unique_subject_id",
                        "prediction": {"min_hours": 6, "max_hours": 48},
                        "icu_careunits": sorted(ICU_CAREUNITS),
                        "admission_matching": {
                            "admissions_path": "hosp/admissions.csv"
                        },
                        "cohort_sources": {
                            "patients_path": "hosp/patients.csv",
                            "transfers_path": "hosp/transfers.csv",
                            "icustays_path": "icu/icustays.csv",
                        },
                    }
                ),
                encoding="utf-8",
            )

            studies = [
                (1, 100, 1001, 0),
                (1, 100, 1002, 1),
                (1, 100, 1009, 8),
                (1, 100, 1010, 20),
                (2, 101, 1003, 0),
                (3, 102, 1004, 0),
                (4, None, 1005, 0),
                (5, 103, 1006, 0),
                (6, 104, 1007, 0),
                (7, 105, 1008, 0),
                (8, 106, 1011, 0),
            ]
            selections, matches, images = [], [], []
            for subject, hadm, study, hour in studies:
                dicom = f"dicom-{study}"
                selections.append(
                    {
                        "subject_id": subject,
                        "study_id": study,
                        "selected_dicom_id": dicom,
                        "dicom_path": f"{dicom}.dcm",
                        "view_position": "AP",
                        "acquisition_datetime": at(hour),
                        "acquisition_time_source": "AcquisitionDateTime",
                    }
                )
                images.append(
                    {
                        "subject_id": subject,
                        "study_id": study,
                        "dicom_id": dicom,
                        "is_readable": True,
                        "is_frontal": True,
                    }
                )
                matches.append(
                    {
                        "subject_id": subject,
                        "study_id": study,
                        "hadm_id": hadm,
                        "dicom_id": dicom,
                        "t0": at(hour),
                        "admittime": at(-10) if hadm else None,
                        "dischtime": at(100) if hadm else None,
                        "match_status": "unique_match" if hadm else "no_match",
                    }
                )
            pq.write_table(
                pa.Table.from_pylist(selections),
                index / "study_image_selection.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(matches), index / "cxr_admission_matches.parquet"
            )
            pq.write_table(pa.Table.from_pylist(images), index / "cxr_index.parquet")
            write_csv(
                iv / "hosp/patients.csv",
                ["subject_id", "gender", "anchor_age", "anchor_year"],
                [
                    {
                        "subject_id": i,
                        "gender": "F",
                        "anchor_age": 17 if i == 7 else 30,
                        "anchor_year": 2020,
                    }
                    for i in range(1, 9)
                ],
            )
            write_csv(
                iv / "hosp/admissions.csv",
                [
                    "subject_id",
                    "hadm_id",
                    "admittime",
                    "dischtime",
                    "deathtime",
                    "hospital_expire_flag",
                ],
                [
                    {
                        "subject_id": subject,
                        "hadm_id": hadm,
                        "admittime": at(-10),
                        "dischtime": at(100),
                        "deathtime": at(10) if hadm == 102 else "",
                        "hospital_expire_flag": 1 if hadm == 102 else 0,
                    }
                    for subject, hadm in [
                        (1, 100),
                        (2, 101),
                        (3, 102),
                        (5, 103),
                        (6, 104),
                        (7, 105),
                        (8, 106),
                    ]
                ],
            )
            unit = "Medical Intensive Care Unit (MICU)"
            transfer_rows = [
                {
                    "subject_id": subject,
                    "hadm_id": hadm,
                    "transfer_id": transfer_id,
                    "careunit": unit,
                    "intime": at(hour),
                    "outtime": at(hour + 10),
                }
                for subject, hadm, transfer_id, hour in [
                    (1, 100, 10001, 7),
                    (5, 103, 10002, 20),
                    (6, 104, 10003, 70),
                ]
            ]
            transfer_rows.extend(
                {
                    "subject_id": subject,
                    "hadm_id": hadm,
                    "transfer_id": 20000 + hadm,
                    "careunit": "Medicine",
                    "intime": at(-10),
                    "outtime": at(100),
                }
                for subject, hadm in [
                    (1, 100),
                    (2, 101),
                    (3, 102),
                    (5, 103),
                    (6, 104),
                    (7, 105),
                ]
            )
            write_csv(
                iv / "hosp/transfers.csv",
                [
                    "subject_id",
                    "hadm_id",
                    "transfer_id",
                    "careunit",
                    "intime",
                    "outtime",
                ],
                transfer_rows,
            )
            write_csv(
                iv / "icu/icustays.csv",
                [
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                    "first_careunit",
                    "last_careunit",
                    "intime",
                    "outtime",
                ],
                [
                    {
                        "subject_id": subject,
                        "hadm_id": hadm,
                        "stay_id": stay_id,
                        "first_careunit": unit,
                        "last_careunit": unit,
                        "intime": at(hour),
                        "outtime": at(hour + 10),
                    }
                    for subject, hadm, stay_id, hour in
                    [(1, 100, 10001, 7), (6, 104, 10003, 70)]
                ],
            )

            self.assertEqual(build_candidate_cohort(root, paths_file, cohort_file, 48), 0)
            audit_path = root / "outputs/audit/window_48h/cohort_audit.parquet"
            summary_path = root / "outputs/audit/window_48h/positive_case_counts.json"
            rows = {
                row["study_id"]: row for row in pq.read_table(audit_path).to_pylist()
            }
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 11)
            self.assertEqual(rows[1001]["label"], 1)
            self.assertEqual(rows[1002]["exclusion_reason"], "icu_within_6h")
            self.assertEqual(rows[1009]["exclusion_reason"], "icu_at_t0")
            self.assertEqual(rows[1010]["exclusion_reason"], "prior_icu")
            self.assertEqual(rows[1003]["label"], 0)
            self.assertEqual(
                rows[1004]["exclusion_reason"], "death_within_H_without_icu"
            )
            self.assertEqual(rows[1005]["exclusion_reason"], "admission_match_failed")
            self.assertEqual(rows[1006]["exclusion_reason"], "icu_source_conflict")
            self.assertEqual(rows[1007]["label"], 0)
            self.assertFalse(rows[1007]["icu_source_disagreement"])
            self.assertEqual(rows[1008]["exclusion_reason"], "underage_or_unknown_age")
            self.assertEqual(rows[1011]["exclusion_reason"], "invalid_icu_timeline")
            self.assertEqual(rows[1011]["icu_timeline_issue"], "missing_transfer_rows")
            self.assertEqual(summary["positive_studies"], 1)
            self.assertEqual(summary["positive_subjects"], 1)
            self.assertEqual(summary["positive_independent_icu_events"], 1)
            self.assertEqual(summary["eligible_studies"], 3)
            self.assertEqual(summary["stage_counts"]["final_eligible"]["studies"], 3)
            self.assertFalse((root / "processed/cohort/cohort.parquet").exists())
            first_hash = summary["audit_output_sha256"]
            self.assertEqual(build_candidate_cohort(root, paths_file, cohort_file, 48), 0)
            rerun = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(rerun["audit_output_sha256"], first_hash)

            cohort = json.loads(cohort_file.read_text(encoding="utf-8"))
            cohort["prediction"]["max_hours"] = 60
            cohort_file.write_text(json.dumps(cohort), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "--max-hours must equal"):
                build_candidate_cohort(root, paths_file, cohort_file, 72)
            self.assertEqual(build_candidate_cohort(root, paths_file, cohort_file, 60), 0)
            audit_60 = root / "outputs/audit/window_60h/cohort_audit.parquet"
            self.assertTrue(audit_60.is_file())
            self.assertEqual(
                next(
                    row["label"] for row in pq.read_table(audit_60).to_pylist()
                    if row["study_id"] == 1007
                ),
                0,
            )

            cohort["prediction"]["max_hours"] = 72
            cohort_file.write_text(json.dumps(cohort), encoding="utf-8")
            self.assertEqual(build_candidate_cohort(root, paths_file, cohort_file, 72), 0)
            audit_72 = root / "outputs/audit/window_72h/cohort_audit.parquet"
            summary_72 = json.loads(
                (root / "outputs/audit/window_72h/positive_case_counts.json")
                .read_text(encoding="utf-8")
            )
            rows_72 = {
                row["study_id"]: row for row in pq.read_table(audit_72).to_pylist()
            }
            self.assertEqual(rows_72[1007]["label"], 1)
            self.assertEqual(rows_72[1007]["prediction_window"], "6-72h")
            self.assertEqual(summary_72["mode"], "candidate_72h")
            self.assertEqual(summary_72["positive_studies"], 2)
            self.assertEqual(summary_72["positive_subjects"], 2)
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8"))["audit_output_sha256"],
                first_hash,
            )
