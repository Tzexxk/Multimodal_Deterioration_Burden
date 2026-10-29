"""Synthetic allocation, frozen-source and patient-split publication checks."""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import test_burden_audit as clinical
import test_freeze_cohort as fixtures

from src.cohort.burden_io import COMPONENTS
from src.cohort.common import _file_hash, _json_file, _verify_run
from src.cohort.create_patient_split import (
    SCHEMA,
    aggregate_patients,
    allocate_patients,
    apportion,
    check_assignments,
    check_source_leakage,
    create_patient_split,
    load_settings,
    split_reports,
    CV_ALGORITHM,
    validate_cv_indices,
)

PROJECT = Path(__file__).parents[2]
VERSION = "burden_v2_h72_r01"


def synthetic_cohort(n=222, high=8, phys=10):
    schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("subject_id", pa.int64()),
            ("hadm_id", pa.int64()),
            ("study_id", pa.int64()),
            *[(c, pa.uint8()) for c in COMPONENTS],
            ("burden_score", pa.float32()),
            ("phys_death", pa.uint8()),
            ("high_burden", pa.uint8()),
            ("early_discharge", pa.uint8()),
            ("cohort_version", pa.string()),
        ]
    )
    rows = []
    for i in range(n):
        subject, hadm, study = 1000 + i, 2000 + i, 3000 + i
        flags = [0, 0, int(high <= i < phys), 0, 0, int(i < high)]
        rows.append(
            dict(
                sample_id=f"p{subject}_h{hadm}_s{study}",
                subject_id=subject,
                hadm_id=hadm,
                study_id=study,
                **dict(zip(COMPONENTS, flags)),
                burden_score=3.0 if i < high else 2.0 if i < phys else 0.0,
                phys_death=int(i < phys),
                high_burden=int(i < high),
                early_discharge=0,
                cohort_version=VERSION,
            )
        )
    return pa.Table.from_pylist(rows, schema=schema)


class PatientAllocationTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(PROJECT / "configs/patient_split_v1.json")

    def test_grouped_cv_indices_reproducible_group_safe_and_tamper_rejected(self):
        settings = {**self.settings, "algorithm_version": CV_ALGORITHM,
                    "weights": [68, 17, 15], "test_fraction": 0.15,
                    "n_folds": 5, "primary_fold": 1, "index_base": 0,
                    "primary_stratum": "phys_death_and_high_burden"}
        cohort = synthetic_cohort(n=24, high=12, phys=12)
        extra = cohort.to_pylist()[0].copy()
        extra.update(hadm_id=9999, study_id=9999, sample_id="p1000_h9999_s9999")
        cohort = pa.concat_tables([cohort, pa.Table.from_pylist([extra], schema=cohort.schema)])
        table, quotas = allocate_patients(cohort, settings)
        bundle = quotas["cross_validation"]
        validate_cv_indices(cohort, bundle, table)
        checks = check_assignments(cohort, table, settings, quotas)
        self.assertTrue(checks["patient_groups_disjoint"])
        other, other_quotas = allocate_patients(cohort.take(list(reversed(range(25)))), settings)
        self.assertTrue(table.equals(other))
        self.assertEqual(quotas, other_quotas)
        for mutation in ("duplicate", "patient", "label", "overlap", "missing_fold"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(bundle)
                if mutation == "duplicate":
                    bad["test_indices"].append(bad["test_indices"][0])
                elif mutation == "patient":
                    same = [r for r in bad["master"] if r["subject_id"] == 1000]
                    same[0]["fold_id"] = (same[0]["fold_id"] + 1) % 6
                elif mutation == "label":
                    bad["master"][0]["burden_score"] = 8
                elif mutation == "overlap":
                    bad["folds"][0]["train_indices"].append(bad["test_indices"][0])
                else:
                    bad["folds"].pop()
                with self.assertRaises(ValueError):
                    validate_cv_indices(cohort, bad, table)

    def test_grouped_cv_impossible_positive_coverage_fails(self):
        settings = {**self.settings, "algorithm_version": CV_ALGORITHM}
        with self.assertRaisesRegex(ValueError, "Grouped CV allocation failed"):
            allocate_patients(synthetic_cohort(n=12, high=5, phys=5), settings)

    def test_grouped_cv_synthetic_682_studies_with_rare_scores(self):
        base = synthetic_cohort(n=284, high=23, phys=24)
        rows = base.to_pylist()
        for i in (0, 1):
            rows[i].update(dict(zip(COMPONENTS, [1, 0, 1, 0, 1, 1])))
            rows[i]["burden_score"] = 8.0
        for i in range(2, 12):
            rows[i].update(dict(zip(COMPONENTS, [0, 0, 1, 0, 0, 1])))
            rows[i]["burden_score"] = 5.0
        for j in range(398):
            r = copy.deepcopy(rows[(j * 7) % 284])
            r.update(study_id=10000 + j,
                     sample_id=f"p{r['subject_id']}_h{r['hadm_id']}_s{10000+j}")
            r.update(dict.fromkeys(COMPONENTS, 0))
            r.update(burden_score=0.0, phys_death=0, high_burden=0)
            rows.append(r)
        cohort = pa.Table.from_pylist(rows, schema=base.schema)
        settings = {**self.settings, "algorithm_version": CV_ALGORITHM}
        patients, quotas = allocate_patients(cohort, settings)
        bundle = quotas["cross_validation"]
        validate_cv_indices(cohort, bundle, patients)
        self.assertEqual(len(bundle["test_indices"]), 102)
        self.assertTrue(all(len(f["validation_indices"]) == 116 for f in bundle["folds"]))

    def test_expected_quotas_order_invariance_and_multiple_admissions(self):
        cohort = synthetic_cohort()
        extra = cohort.to_pylist()[0].copy()
        extra.update(hadm_id=9999, study_id=9999, sample_id="p1000_h9999_s9999")
        cohort = pa.concat_tables(
            [cohort, pa.Table.from_pylist([extra], schema=cohort.schema)]
        )
        assigned, quotas = allocate_patients(cohort, self.settings)
        self.assertEqual(
            quotas,
            {
                "total": [156, 33, 33],
                "ever_phys_death": [7, 2, 1],
                "ever_high_burden": [6, 1, 1],
            },
        )
        self.assertTrue(
            check_assignments(cohort, assigned, self.settings, quotas)[
                "each_study_has_one_split"
            ]
        )
        self.assertEqual(assigned.schema, SCHEMA)
        self.assertEqual(assigned.to_pylist()[0]["study_count"], 2)
        reverse = cohort.take(pa.array(list(reversed(range(cohort.num_rows)))))
        repeated, repeated_quotas = allocate_patients(reverse, self.settings)
        self.assertTrue(assigned.equals(repeated))
        self.assertEqual(quotas, repeated_quotas)

    def test_rounding_ties_bounds_and_impossible_constraints(self):
        self.assertEqual(
            apportion(10, [70, 15, 15], [1, 1, 1], [156, 33, 33]), [7, 2, 1]
        )
        self.assertEqual(apportion(3, [70, 15, 15], [1, 1, 1], [3, 3, 3]), [1, 1, 1])
        self.assertEqual(apportion(8, [70, 15, 15], [1, 1, 1], [7, 2, 1]), [6, 1, 1])
        for n in range(3, 30):
            counts = apportion(n, [70, 15, 15], [1, 1, 1], [1, n, n])
            self.assertEqual(sum(counts), n)
            self.assertEqual(counts[0], 1)
            self.assertTrue(all(x >= 1 for x in counts))
        for n, lo, cap in (
            (2, [1, 1, 1], [2] * 3),
            (10, [1] * 3, [2] * 3),
            (3, [1] * 3, [0, 3, 3]),
        ):
            with self.assertRaises(ValueError):
                apportion(n, [70, 15, 15], lo, cap)
        with self.assertRaisesRegex(ValueError, "three high-burden"):
            allocate_patients(synthetic_cohort(n=8, high=2, phys=3), self.settings)

    def test_invalid_cohort_and_assignments_fail(self):
        cohort = synthetic_cohort(n=12, high=3, phys=5)
        assigned, quotas = allocate_patients(cohort, self.settings)
        for mutation in (
            lambda r: r.append(copy.deepcopy(r[0])),
            lambda r: r[0].update(sample_id="bad"),
            lambda r: r[0].update(burden_score=8),
            lambda r: r[0].update(phys_death=0),
            lambda r: r[0].update(high_burden=None),
            lambda r: r[0].update(cohort_version="other"),
            lambda r: r[0].update(in_hospital_death=2),
        ):
            rows = cohort.to_pylist()
            mutation(rows)
            with self.assertRaises(ValueError):
                aggregate_patients(
                    pa.Table.from_pylist(rows, schema=cohort.schema), VERSION
                )
        with self.assertRaises(ValueError):
            aggregate_patients(cohort.drop(["phys_death"]), VERSION)
        for mutation in (
            lambda r: r.pop(),
            lambda r: r.append(copy.deepcopy(r[0])),
            lambda r: r[0].update(split="invalid"),
            lambda r: r[0].update(seed=1),
            lambda r: r[0].update(study_count=99),
            lambda r: r[0].update(split=None),
        ):
            rows = assigned.to_pylist()
            mutation(rows)
            with self.assertRaises(ValueError):
                check_assignments(
                    cohort,
                    pa.Table.from_pylist(rows, schema=SCHEMA),
                    self.settings,
                    quotas,
                )

    def test_missing_classes_reported_and_shared_source_leakage(self):
        cohort = synthetic_cohort(n=3, high=3, phys=3)
        patients, quotas = allocate_patients(cohort, self.settings)
        summary, checks = split_reports(
            cohort, patients, check_assignments(cohort, patients, self.settings, quotas)
        )
        self.assertEqual(sum(r["category"] == "study_score" for r in summary), 27)
        self.assertTrue(
            all(
                not group["high_burden"]["discrimination_evaluable"]
                for group in checks["endpoint_evaluability"].values()
            )
        )
        repeated = cohort.to_pylist()[0].copy()
        repeated.update(study_id=8000, sample_id="p1000_h2000_s8000")
        cohort = pa.concat_tables(
            [cohort, pa.Table.from_pylist([repeated], schema=cohort.schema)]
        )
        patients, _ = allocate_patients(cohort, self.settings)
        evidence = [
            {
                "sample_id": r["sample_id"],
                "subject_id": r["subject_id"],
                "component": "in_hospital_death",
                "source_record_key": "same_patient"
                if r["subject_id"] == 1000
                else str(r["subject_id"]),
            }
            for r in cohort.to_pylist()
        ]
        with duckdb.connect() as connection:
            connection.register("common", cohort)
            connection.register("evidence", pa.Table.from_pylist(evidence))
            check_source_leakage(connection, patients)
            evidence[1]["source_record_key"] = "same_patient"
            connection.register("evidence", pa.Table.from_pylist(evidence))
            with self.assertRaisesRegex(ValueError, "crosses patient splits"):
                check_source_leakage(connection, patients)

    def test_configuration_confirmation_and_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for change in (
                {"confirmed": False},
                {"seed": 7},
                {"weights": [60, 20, 20]},
                {"require_high_burden_in_each_split": False},
                {"split_version": "../unsafe"},
                {"reviewer": ""},
                {"reviewed_on": "bad"},
            ):
                path.write_text(
                    json.dumps({**self.settings, **change}), encoding="utf-8"
                )
                with self.assertRaises(ValueError):
                    load_settings(path)


class PatientSplitPublicationTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.CommonCohortTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.root = self.f.root
        self.images = self.f.f
        self.data = self.images.f.processed / "splits/burden_v2"
        self.audit = self.f.audit
        self.split_file = self.root / "configs/patient_split_v1.json"
        self.split_file.write_bytes(
            (PROJECT / "configs/patient_split_v1.json").read_bytes()
        )

    def prepare(self):
        base = self.images.f
        for subject in (3, 4):
            base.tables["patients"].append(
                {**base.tables["patients"][0], "subject_id": str(subject)}
            )
            base.tables["admissions"].append(
                {
                    **base.tables["admissions"][0],
                    "subject_id": str(subject),
                    "hadm_id": str(subject * 10),
                }
            )
            base.tables["transfers"].append(
                dict(
                    subject_id=str(subject),
                    hadm_id=str(subject * 10),
                    transfer_id=str(subject * 100),
                    eventtype="admit",
                    careunit="Ward",
                    intime=clinical.dt(-10),
                    outtime=clinical.dt(70),
                )
            )
            base.matches.append(
                dict(
                    subject_id=subject,
                    hadm_id=subject * 10,
                    study_id=subject * 100,
                    dicom_id=f"d{subject * 100}",
                    t0=clinical.dt(0),
                    admittime=clinical.dt(-10),
                    dischtime=clinical.dt(70),
                    match_status="unique_match",
                )
            )
        self.f.prepare(qc_excluded=False)
        self.f.run_freeze()

    def run_split(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                create_patient_split(
                    self.root,
                    self.images.configs[0],
                    self.images.configs[1],
                    self.images.configs[2],
                    self.images.configs[3],
                    self.images.processing,
                    self.split_file,
                ),
                0,
            )
        _verify_run(self.images.f.outputs, 9)
        from src.cohort.create_patient_split import split_table_names
        return pq.read_table(self.data / split_table_names(_json_file(self.split_file))[0])

    def test_grouped_cv_publishes_and_reuses_all_five_index_lists(self):
        base = self.images.f
        for subject in range(5, 17):
            source = base.tables["admissions"][0 if subject < 10 else 1]
            base.tables["patients"].append({**base.tables["patients"][0], "subject_id": str(subject)})
            base.tables["admissions"].append({**source, "subject_id": str(subject),
                                               "hadm_id": str(subject * 10)})
            base.tables["transfers"].append(dict(
                subject_id=str(subject), hadm_id=str(subject * 10),
                transfer_id=str(subject * 100), eventtype="admit", careunit="Ward",
                intime=source["admittime"], outtime=source["dischtime"],
            ))
            base.matches.append(dict(
                subject_id=subject, hadm_id=subject * 10, study_id=subject * 100,
                dicom_id=f"d{subject * 100}", t0=clinical.dt(0),
                admittime=source["admittime"], dischtime=source["dischtime"],
                match_status="unique_match",
            ))
        self.prepare()
        settings = _json_file(self.split_file)
        settings.update(algorithm_version=CV_ALGORITHM, weights=[68, 17, 15],
                        test_fraction=0.15, n_folds=5, primary_fold=1, index_base=0,
                        primary_stratum="phys_death_and_high_burden")
        self.split_file.write_text(json.dumps(settings), encoding="utf-8")
        first = self.run_split()
        self.assertFalse((self.data / "patient_split.parquet").exists())
        self.assertFalse((self.data / f"releases/{settings['split_version']}/patient_split.parquet").exists())
        bundle = _json_file(self.audit / "cross_validation_indices.json")
        validate_cv_indices(pq.read_table(self.f.data / "cohort.parquet"), bundle, first)
        frozen = _json_file(self.audit / "split_freeze.json")
        self.assertNotIn("patient_split.parquet", frozen["artifact_sha256"])
        self.assertIn("cross_validation_indices.json", frozen["artifact_sha256"])
        from src.cohort.create_patient_split import select_fold
        test_patients = {r["subject_id"] for r in first.to_pylist() if r["split"] == "test"}
        for fold in range(1, 6):
            name = f"fold_{fold}/patient_split.parquet"
            table = pq.read_table(self.data / name)
            self.assertIn(name, frozen["artifact_sha256"])
            self.assertEqual(_file_hash(self.data / name), frozen["artifact_sha256"][name])
            self.assertTrue(table.equals(select_fold(first, bundle, fold)))
            self.assertEqual({r["subject_id"] for r in table.to_pylist() if r["split"] == "test"},
                             test_patients)
        for invalid in (0, 6, True, "1"):
            with self.assertRaises(ValueError):
                select_fold(first, bundle, invalid)
        self.assertTrue(first.equals(self.run_split()))
        self.assertEqual(bundle, _json_file(self.audit / "cross_validation_indices.json"))
        from src.ehr.build_ehr_features import verify_frozen_inputs
        verify_frozen_inputs(self.root, self.images.configs, self.images.processing,
                             self.split_file, settings)
        damaged = self.data / f"releases/{settings['split_version']}/fold_5/patient_split.parquet"
        damaged.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            verify_frozen_inputs(self.root, self.images.configs, self.images.processing,
                                 self.split_file, settings)
        with self.assertRaises(ValueError):
            self.run_split()

    def test_types_freeze_repeat_and_upstream_verification_rerun(self):
        self.prepare()
        overlapping = self.audit / "step_09_run_metadata.json"
        original_config = self.split_file.read_bytes()
        overlapping.write_bytes(original_config)
        with self.assertRaisesRegex(ValueError, "outside generated output roots"):
            create_patient_split(
                self.root,
                self.images.configs[0],
                self.images.configs[1],
                self.images.configs[2],
                self.images.configs[3],
                self.images.processing,
                overlapping,
            )
        self.assertEqual(overlapping.read_bytes(), original_config)
        overlapping.unlink()
        protected = {
            str(
                self.images.f.outputs / f"audit/step_{s:02d}_manifest.json"
            ): _file_hash(self.images.f.outputs / f"audit/step_{s:02d}_manifest.json")
            for s in range(2, 9)
        }
        table = self.run_split()
        self.assertEqual(table.schema, SCHEMA)
        self.assertEqual(table.num_rows, 4)
        frozen = _json_file(self.audit / "split_freeze.json")
        self.assertEqual(frozen["checks"]["quotas"]["total"], [2, 1, 1])
        self.assertTrue(frozen["checks"]["source_records_do_not_cross_splits"])
        manifest = _json_file(self.images.f.outputs / "audit/step_09_manifest.json")
        snapshots = {
            p: (_file_hash(Path(p)), Path(p).stat().st_mtime_ns)
            for p in manifest["files"]
            if "releases" in p
        }
        self.assertTrue(table.equals(self.run_split()))
        for p, (digest, mtime) in snapshots.items():
            self.assertEqual(
                (_file_hash(Path(p)), Path(p).stat().st_mtime_ns), (digest, mtime)
            )
        for p, digest in protected.items():
            self.assertEqual(_file_hash(Path(p)), digest)
        old_upstream = _json_file(self.audit / "step_08_run_metadata.json")["run_id"]
        self.f.run_freeze()
        self.assertNotEqual(
            old_upstream, _json_file(self.audit / "step_08_run_metadata.json")["run_id"]
        )
        self.assertTrue(table.equals(self.run_split()))
        self.assertEqual(_json_file(self.audit / "split_freeze.json"), frozen)
        with (self.images.f.outputs / "audit/step_09_csv_tail_checks.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            self.assertEqual(next(csv.DictReader(stream))["status"], "no_csv_inputs")
        with (self.audit / "split_summary.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            summary = list(csv.DictReader(stream))
        self.assertEqual(
            len([r for r in summary if r["category"] == "patient_max_score"]), 27
        )
        self.assertTrue(
            _json_file(self.audit / "step_09_run_metadata.json")["snapshot_reused"]
        )

    def test_version_protection_new_snapshot_and_rollback(self):
        self.prepare()
        self.run_split()
        first = _json_file(self.audit / "split_freeze.json")
        original_snapshots = {
            p: _file_hash(Path(p))
            for p in _json_file(self.images.f.outputs / "audit/step_09_manifest.json")[
                "files"
            ]
            if "releases" in p
        }
        with patch(
            "src.cohort.create_patient_split._code_hash", return_value="changed"
        ):
            with self.assertRaisesRegex(ValueError, "new split version"):
                self.run_split()
        settings = _json_file(self.split_file)
        settings["split_version"] = VERSION + "_split_v2"
        self.split_file.write_text(json.dumps(settings), encoding="utf-8")
        self.run_split()
        for path, digest in original_snapshots.items():
            self.assertEqual(_file_hash(Path(path)), digest)
        self.assertNotEqual(
            _json_file(self.audit / "split_freeze.json")["original_split_run_id"],
            first["original_split_run_id"],
        )
        old = _json_file(self.images.f.outputs / "audit/step_09_manifest.json")
        settings["split_version"] = VERSION + "_split_v3"
        self.split_file.write_text(json.dumps(settings), encoding="utf-8")
        original_replace = os.replace

        def fail_one(source, target):
            if (
                Path(target) == self.data / "patient_split.parquet"
                and ".step_09_stage_" in Path(source).name
            ):
                raise OSError("synthetic split publication failure")
            return original_replace(source, target)

        with patch("src.cohort.common.os.replace", side_effect=fail_one):
            with self.assertRaisesRegex(OSError, "synthetic split publication failure"):
                self.run_split()
        self.assertEqual(
            old, _json_file(self.images.f.outputs / "audit/step_09_manifest.json")
        )
        _verify_run(self.images.f.outputs, 9)
        self.assertFalse(
            (self.audit / f"releases/{VERSION}_split_v3/split_freeze.json").exists()
        )
        self.assertFalse(
            (self.data / f"releases/{VERSION}_split_v3/patient_split.parquet").exists()
        )
        self.run_split()

    def test_bad_sources_preserve_release_and_recovery(self):
        self.prepare()
        self.run_split()
        old = _json_file(self.images.f.outputs / "audit/step_09_manifest.json")
        source = self.f.data / "cohort.parquet"
        original = source.read_bytes()
        source.write_bytes(b"corrupt source")
        with self.assertRaises(ValueError):
            self.run_split()
        source.write_bytes(original)
        decision = self.audit / "cohort_freeze.json"
        saved = decision.read_bytes()
        frozen = _json_file(decision)
        frozen["common_cohort_frozen"] = False
        decision.write_text(json.dumps(frozen), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_split()
        decision.write_bytes(saved)
        self.assertEqual(
            old, _json_file(self.images.f.outputs / "audit/step_09_manifest.json")
        )
        target = self.data / "patient_split.parquet"
        content = target.read_bytes()
        backup = target.parent / ".step_09_backup_synthetic"
        staged = target.parent / ".step_09_stage_synthetic"
        backup.write_bytes(content)
        staged.write_bytes(b"unpublished")
        target.write_bytes(b"interrupted")
        pending = self.images.f.outputs / "audit/.step_09_pending.json"
        pending.write_text(
            json.dumps(
                {
                    "files": [
                        {
                            "target": str(target),
                            "backup": str(backup),
                            "staged": str(staged),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.run_split()
        self.assertEqual(target.read_bytes(), content)
        self.assertFalse(pending.exists())
        self.assertFalse(backup.exists())


if __name__ == "__main__":
    unittest.main()
