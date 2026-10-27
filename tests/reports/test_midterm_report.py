"""Synthetic checks for presentation denominators and patient disjointness."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/generate_midterm_report.py"
SPEC = importlib.util.spec_from_file_location("midterm_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class MidtermReportTest(unittest.TestCase):
    def setUp(self):
        self.samples = pd.DataFrame(
            dict(
                sample_id=["a", "b", "c", "d"],
                subject_id=[1, 1, 2, 3],
                hadm_id=[10, 10, 20, 30],
                split=["train", "train", "validation", "test"],
                burden_score=[3, 3, 0, 0],
                phys_death=[1, 1, 0, 0],
                high_burden=[1, 1, 0, 0],
            )
        )
        for name in report.COMPONENTS:
            self.samples[name] = (
                [1, 1, 0, 0] if name == "new_icu_ccu_transfer" else [0, 0, 0, 0]
            )
        self.samples["in_hospital_death"] = [1, 1, 0, 0]

    def test_overlapping_events_and_repeated_patient(self):
        scores, events = report.label_tables(self.samples)
        self.assertEqual(scores.studies.sum(), 4)
        self.assertEqual(scores.loc[scores.score == 8, "studies"].item(), 0)
        for name in ["new_icu_ccu_transfer", "in_hospital_death"]:
            row = events.loc[events.component == name].iloc[0]
            self.assertEqual((row.studies, row.patients), (2, 1))
        self.assertEqual(report.split_table(self.samples).patients.tolist(), [1, 1, 1])

    def test_cross_split_patient_and_duplicate_sample_rejected(self):
        crossed = self.samples.copy()
        crossed.loc[1, "split"] = "test"
        with self.assertRaises(ValueError):
            report.split_table(crossed)
        duplicated = self.samples.copy()
        duplicated.loc[1, "sample_id"] = "a"
        with self.assertRaises(ValueError):
            report.label_tables(duplicated)

    def test_missingness_uses_values_not_zero_measurement_counts(self):
        ehr = pd.DataFrame(
            dict(
                sample_id=["d", "c", "b", "a"],
                potassium_last_24h=[np.nan, np.nan, 4.0, np.nan],
                potassium_count_24h=[0, 0, 1, 0],
            )
        )
        table = report.missingness_table(self.samples, ehr)
        row = table.loc[table.split == "all"].iloc[0]
        self.assertEqual(
            (row.observed_studies, row.observed_patients, row.missing_rate),
            (1, 1, 0.75),
        )
        self.assertEqual(
            table.loc[table.split == "validation", "missing_rate"].item(), 1
        )
        ehr.loc[0, "sample_id"] = "unknown"
        with self.assertRaises(ValueError):
            report.missingness_table(self.samples, ehr)

    def test_joint_reasons_preserve_associations_and_do_not_sum_sources(self):
        window = pd.DataFrame(
            dict(
                itemid=[1, 2, 3],
                availability_reason=[
                    "future_storetime",
                    "future_storetime",
                    "available",
                ],
                ownership_reason=["owned", "owned", "no_admission_match"],
                study_associations=[2, 3, 4],
                source_records=[1, 2, 3],
            )
        )
        table = report.rejection_table(window)
        self.assertEqual(table.study_associations.sum(), 9)
        self.assertEqual(len(table), 2)
        self.assertNotIn("source_records", table)

    def test_empty_samples_and_out_of_range_score_rejected(self):
        with self.assertRaises(ValueError):
            report.label_tables(self.samples.iloc[:0])
        self.samples.loc[0, "burden_score"] = 9
        with self.assertRaises(ValueError):
            report.label_tables(self.samples)

    def test_supplementary_changes_count_patients_and_domain_caps(self):
        baseline = self.samples.copy()
        baseline.loc[0, "new_invasive_ventilation"] = 1
        baseline.loc[0, "burden_score"] = 5
        current = baseline.loc[baseline.sample_id != "d"].copy()
        current.loc[0, "new_severe_hypoxemia"] = 1
        current.loc[1, "in_hospital_death"] = 0
        current.loc[1, "burden_score"] = 1
        added = current.iloc[[0, 0]].copy()
        added["sample_id"] = ["e", "f"]
        added["subject_id"] = 4
        added["hadm_id"] = 40
        current = pd.concat([current, added], ignore_index=True)
        scope = pd.DataFrame(
            dict(
                sample_id=["e", "f", "g"],
                baseline_reason=["med", "med", "icu"],
                in_review_scope=[1, 1, 0],
                disposition=["restored", "restored", "protected_exclusion"],
            )
        )
        tables = report.supplementary_tables(baseline, current, scope)
        summary = tables["07_supplementary_summary"].set_index("metric")["count"]
        self.assertEqual(summary["added_studies"], 2)
        self.assertEqual(summary["added_independent_nonzero_patients"], 1)
        self.assertEqual(summary["removed_baseline_studies"], 1)
        self.assertEqual(summary["retained_baseline_score_decreases"], 1)
        events = tables["07_baseline_event_changes"].set_index("component")
        self.assertEqual(events.loc["new_severe_hypoxemia", "capped_additions"], 1)
        self.assertEqual(events.loc["in_hospital_death", "revoked_events"], 1)
        domains = tables["07_baseline_domain_points"].set_index("domain")
        self.assertEqual(domains.loc["R", "actual_points_gained"], 0)
        self.assertEqual(domains.loc["D", "actual_points_lost"], 3)
        scope.loc[2, "sample_id"] = "a"
        with self.assertRaisesRegex(ValueError, "Protected"):
            report.supplementary_tables(baseline, current, scope)


if __name__ == "__main__":
    unittest.main()
