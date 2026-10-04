"""Synthetic candidate selection checks without clinical source scans."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa

from src.ehr import lab_audit as audit
from src.ehr.lab_selection import missingness_band, publish_selection, select_candidates


class SelectionTests(unittest.TestCase):
    def test_versioned_pair_reuse_recovery_and_changed_content(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_selection_") as temporary:
            output = Path(temporary) / "selection.csv"
            publish_selection(output, b"header\nrow\n", {"synthetic": True})
            original = output.stat().st_mtime_ns
            publish_selection(output, b"header\nrow\n", {"synthetic": True})
            self.assertEqual(output.stat().st_mtime_ns, original)
            output.with_suffix(".json").unlink()
            publish_selection(output, b"header\nrow\n", {"synthetic": True})
            self.assertTrue(output.with_suffix(".json").exists())
            with self.assertRaises(ValueError):
                publish_selection(output, b"header\nchanged\n", {"synthetic": True})
            with self.assertRaises(ValueError):
                publish_selection(output, b"header\nrow\n", {"synthetic": False})
            self.assertEqual(output.read_bytes(), b"header\nrow\n")

    def test_exact_boundaries_and_invalid_counts(self):
        for covered, expected in (
            (71, "below_30"),
            (70, "30_to_50"),
            (50, "30_to_50"),
            (49, "above_50_to_80"),
            (20, "above_50_to_80"),
            (19, "above_80"),
        ):
            self.assertEqual(missingness_band(covered, 100), expected)
        # 29.995% would round to 30.00%, but still qualifies for automatic selection.
        self.assertEqual(missingness_band(14001, 20000), "below_30")
        for covered, total in ((0, 0), (-1, 100), (101, 100), (True, 100)):
            with self.assertRaises(ValueError):
                missingness_band(covered, total)

    def test_training_coverage_decisions_and_branch_counts(self):
        with duckdb.connect(":memory:") as con:
            con.execute(
                "CREATE TABLE study_keys AS SELECT i AS sample_id, i AS subject_id FROM range(101) t(i)"
            )
            con.execute(
                "CREATE TABLE patient_splits AS SELECT i AS subject_id, CASE WHEN i=100 THEN 'test' ELSE 'train' END AS split FROM range(101) t(i)"
            )
            con.execute(
                "CREATE TABLE associations(itemid VARCHAR,sample_id BIGINT,numeric_value DOUBLE,availability_reason VARCHAR,ownership_reason VARCHAR)"
            )
            for itemid, n in (("1", 71), ("2", 70), ("3", 50), ("4", 20), ("5", 19)):
                con.execute(
                    "INSERT INTO associations SELECT ?,i,1.0,'available','owned' FROM range(?) t(i)",
                    [itemid, n],
                )
            con.execute(
                "INSERT INTO associations VALUES ('1',0,1,'available','owned'),('2',100,1,'available','owned'),('5',19,1,'future_storetime','owned'),('5',20,1,'available','different_admission'),('5',21,'Infinity','available','owned'),('5',22,NULL,'available','owned')"
            )
            inventory = [
                dict(
                    itemid=str(i),
                    label=f"synthetic_{i}",
                    fluid="Blood",
                    category="Synthetic",
                    source_unit="u",
                )
                for i in range(1, 7)
            ]
            inventory.append({**inventory[0], "source_unit": "other"})
            decisions = {
                "2": {"include": True, "rationale": "Explicit synthetic choice."},
                "4": {"include": False, "rationale": "Explicit synthetic deferral."},
            }
            rows = select_candidates(con, inventory, decisions)
            self.assertEqual(
                [r["covered_training_studies"] for r in rows], [71, 70, 50, 20, 19, 0]
            )
            self.assertTrue(rows[0]["pending_unit_review"])
            self.assertEqual(rows[1]["selection_status"], "selected")
            self.assertEqual(rows[2]["decision_basis"], "pending_individual_choice")
            self.assertEqual(rows[3]["decision_basis"], "explicit_user_choice")
            self.assertTrue(all(not r["clinical_rules_approved"] for r in rows))
            for invalid in (
                {"7": decisions["2"]},
                {"1": decisions["2"]},
                {"2": {"include": "yes", "rationale": "invalid"}},
            ):
                with self.assertRaises(ValueError):
                    select_candidates(con, inventory, invalid)
            con.execute("INSERT INTO patient_splits VALUES (0,'test')")
            with self.assertRaises(ValueError):
                select_candidates(con, inventory, decisions)

    def test_reused_audit_views_keep_inclusive_window_and_availability(self):
        with duckdb.connect(":memory:") as con:
            cohort = pa.table(
                {
                    "sample_id": ["a", "b", "test"],
                    "subject_id": [1, 1, 2],
                    "hadm_id": [10, 10, 20],
                    "t0": [
                        datetime(2020, 1, 2, 12),
                        datetime(2020, 1, 2, 13),
                        datetime(2020, 1, 2, 12),
                    ],
                }
            )
            con.register(
                "admissions",
                pa.table(
                    {
                        "subject_id": ["1", "2"],
                        "hadm_id": ["10", "20"],
                        "admittime": ["2020-01-01 00:00:00"] * 2,
                        "dischtime": ["2020-01-04 00:00:00"] * 2,
                    }
                ),
            )
            records = []
            for key, item, patient, admission, chart, store in (
                (
                    "shared",
                    "1",
                    "1",
                    "10",
                    "2020-01-02 12:00:00",
                    "2020-01-02 12:00:00",
                ),
                ("lower", "2", "1", "10", "2020-01-01 12:00:00", "2020-01-01 12:00:00"),
                (
                    "future",
                    "3",
                    "1",
                    "10",
                    "2020-01-02 11:00:00",
                    "2020-01-02 14:00:00",
                ),
                ("wrong", "3", "1", "20", "2020-01-02 11:00:00", "2020-01-02 11:00:00"),
                ("invalid", "3", "1", "10", "invalid", "2020-01-02 11:00:00"),
                ("test", "3", "2", "20", "2020-01-02 11:00:00", "2020-01-02 11:00:00"),
            ):
                records.append(
                    dict(
                        lab_record_key=key,
                        itemid=item,
                        subject_id=patient,
                        hadm_id=admission,
                        charttime=chart,
                        storetime=store,
                        valuenum="1",
                        value="1",
                        valueuom="u",
                        specimen_id="1",
                    )
                )
            con.register("labs", pa.Table.from_pylist(records))
            audit.build_views(con, cohort)
            con.register(
                "patient_splits",
                pa.table({"subject_id": [1, 2], "split": ["train", "test"]}),
            )
            inventory = [
                dict(
                    itemid=str(i),
                    label="synthetic",
                    fluid="Blood",
                    category="Synthetic",
                    source_unit="u",
                )
                for i in (1, 2, 3)
            ]
            rows = select_candidates(con, inventory, {})
            self.assertEqual([r["covered_training_studies"] for r in rows], [2, 1, 0])
            self.assertEqual(rows[0]["covered_training_patients"], 1)


if __name__ == "__main__":
    unittest.main()
