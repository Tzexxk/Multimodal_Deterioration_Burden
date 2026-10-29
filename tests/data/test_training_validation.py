"""Synthetic independent validation, leakage, relocation and report checks."""

from __future__ import annotations

import copy
import json
import stat
import tempfile
import unittest
import warnings
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.common import (
    _file_hash,
    _file_version,
    _OutputRun,
    _verify_run,
    _write_csv,
)
from src.data import validate_training_package as step

ROOT = Path(__file__).resolve().parents[2]


class TrainingValidationTests(unittest.TestCase):
    def test_independent_five_fold_index_verification_and_tamper_rejection(self):
        from test_patient_split import synthetic_cohort
        from src.cohort.create_patient_split import allocate_patients, CV_ALGORITHM, select_fold
        cohort = synthetic_cohort(n=24, high=12, phys=12)
        settings = dict(algorithm_version=CV_ALGORITHM, cohort_version="burden_v2_h72_r01",
                        split_version="burden_v2_h72_r01_split_v2", seed=42)
        patients, quotas = allocate_patients(cohort, settings)
        by_id = {r["subject_id"]: r["split"] for r in patients.to_pylist()}
        samples = pa.Table.from_pylist([
            {**r, "split": by_id[r["subject_id"]]}
            for r in sorted(cohort.to_pylist(), key=lambda r: r["sample_id"])
        ])
        bundle = quotas["cross_validation"]
        fold_tables = {fold: select_fold(patients, bundle, fold) for fold in range(1, 6)}
        step.validate_cross_validation(cohort, patients, bundle, samples, fold_tables)
        for kind in ("missing", "wrong_fold", "duplicate", "metadata"):
            with self.subTest(fold_table=kind):
                bad_tables = dict(fold_tables)
                if kind == "missing":
                    bad_tables.pop(5)
                elif kind == "wrong_fold":
                    bad_tables[5] = fold_tables[1]
                else:
                    rows = fold_tables[5].to_pylist()
                    if kind == "duplicate":
                        rows[-1] = rows[0]
                    else:
                        rows[0]["seed"] = 99
                    bad_tables[5] = pa.Table.from_pylist(rows, schema=patients.schema)
                with self.assertRaises(ValueError):
                    step.validate_cross_validation(cohort, patients, bundle, samples, bad_tables)
        for kind in ("test", "order", "fold", "label", "train"):
            with self.subTest(kind=kind):
                bad = copy.deepcopy(bundle)
                if kind == "test":
                    bad["test_indices"].append(bad["test_indices"][0])
                elif kind == "order":
                    bad["master"][0]["sample_id"] = "incorrect"
                elif kind == "fold":
                    bad["folds"].pop()
                elif kind == "label":
                    bad["master"][0]["high_burden"] ^= 1
                else:
                    bad["folds"][0]["train_indices"] = bad["folds"][0]["validation_indices"]
                with self.assertRaises(ValueError):
                    step.validate_cross_validation(cohort, patients, bad, samples)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic_step12_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.processed, self.outputs = (
            self.root / "data/processed",
            self.root / "outputs",
        )
        self.settings = step.load_settings(ROOT / "configs/training_validation_v1.json")
        self.settings.update(
            expected_counts=dict(studies=3, admissions=3, patients=3),
            expected_split_studies=dict.fromkeys(step.SPLITS, 1),
            expected_split_patients=dict.fromkeys(step.SPLITS, 1),
        )
        self.package = self.processed / "training" / self.settings["release_id"]
        self.package.mkdir(parents=True)
        self.cache = self.root / "data/cache/static"
        self.cache.mkdir(parents=True)
        self.lab_path = self.root / "data/cache/labs.parquet"
        self.lab_admissions_path = self.root / "data/cache/admissions.parquet"
        self.processing = self.root / "processing.json"
        step.write_json(self.processing, {"version": "synthetic"})
        self.t0 = datetime(2020, 1, 2, 12)
        (
            self.cohort_rows,
            sample_rows,
            patient_rows,
            static_patients,
            static_admissions,
            static_services,
        ) = [], [], [], [], [], []
        images, qc = [], []
        for i, group in enumerate(step.SPLITS, 1):
            patient, admission, study = 10000 + i, 20000 + i, 50000 + i
            key = f"p{patient}_h{admission}_s{study}"
            relative = f"images/p10/p{patient}/s{study}.npy"
            row: dict[str, Any] = dict.fromkeys(step.EVENTS, 0)
            row.update(
                sample_id=key,
                subject_id=patient,
                hadm_id=admission,
                study_id=study,
                dicom_id=f"dicom_{i}",
                t0=self.t0,
                admittime=self.t0 - timedelta(days=2),
                dischtime=self.t0 + timedelta(days=4),
                deathtime=None,
                hospital_expire_flag="0",
                exclusion_reason="",
                age_at_t0=50,
                careunit_at_t0="WARD",
                observation_end=self.t0 + timedelta(hours=72),
                observed_hours=72.0,
                early_discharge=0,
                burden_score=0.0,
                phys_death=0,
                high_burden=0,
                npy_relpath=relative,
                dicom_path=f"dataset/{i}.dcm",
                view_position="AP",
                qc_status="pass",
                processing_status="complete",
                processing_config_sha256=_file_hash(self.processing),
                processing_version="synthetic",
            )
            for event in step.EVENTS:
                row[event + "_first_event_time"] = None
            if i == 2:
                row.update(
                    burden_score=5.0,
                    phys_death=1,
                    high_burden=1,
                    hospital_expire_flag="1",
                    deathtime=self.t0 + timedelta(hours=3),
                )
                for event in (step.EVENTS[2], step.EVENTS[5]):
                    row[event] = 1
                    row[event + "_first_event_time"] = self.t0 + timedelta(hours=3)
            source_image = self.processed / "imaging/burden_v2" / relative
            source_image.parent.mkdir(parents=True, exist_ok=True)
            np.save(
                source_image,
                np.full((224, 224), i / 10, dtype=np.float32),
                allow_pickle=False,
            )
            target = self.package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source_image.read_bytes())
            dicom = self.root / "cxr" / row["dicom_path"]
            dicom.parent.mkdir(parents=True, exist_ok=True)
            dicom.write_bytes(f"synthetic dicom {i}".encode())
            row.update(
                npy_sha256=_file_hash(source_image),
                source_dicom_sha256=_file_hash(dicom),
            )
            self.cohort_rows.append(row)
            sample_rows.append(
                {
                    k: group
                    if k == "split"
                    else relative
                    if k == "image_relpath"
                    else row[k]
                    for k in step.SAMPLE_SCHEMA.names
                }
            )
            patient_rows.append(dict(subject_id=patient, split=group))
            static_patients.append(
                dict(
                    subject_id=str(patient),
                    gender="M",
                    anchor_age="50",
                    anchor_year="2020",
                )
            )
            static_admissions.append(
                dict(
                    subject_id=str(patient),
                    hadm_id=str(admission),
                    admittime=str(row["admittime"]),
                    dischtime=str(row["dischtime"]),
                    admission_type="EMERGENCY",
                    admission_location="ED",
                )
            )
            static_services.append(
                dict(
                    subject_id=str(patient),
                    hadm_id=str(admission),
                    transfertime=str(self.t0),
                    curr_service="MED",
                )
            )
            qc.append(dict(row))
            images.append(
                {
                    **row,
                    "split": group,
                    "dicom_path": str(dicom),
                    "npy_path": str(source_image),
                }
            )
        self.cohort = pa.Table.from_pylist(self.cohort_rows)
        self.samples = pa.Table.from_pylist(sample_rows, schema=step.SAMPLE_SCHEMA)
        self.split = pa.Table.from_pylist(patient_rows)
        self.images, self.qc = pa.Table.from_pylist(images), pa.Table.from_pylist(qc)
        self.rules = [
            dict(
                keep=True,
                itemid=1,
                feature_name="synthetic_lab",
                source_unit="mg/dL",
                canonical_unit="mg/dL",
                conversion_multiplier=1.0,
                conversion_offset=0.0,
                valid_min=None,
                valid_max=None,
                valid_range_basis="No reliable bounds",
                specimen_rule='{"kind":"item_identity"}',
                duplicate_rule='{"kind":"latest_available","conflicting_id":"error","missing_specimen":"error"}',
                numeric_result_rule='{"kind":"valuenum_only","comparator":"exclude","text_disagreement":"error"}',
                missing_unit_rule="exclude",
            )
        ]
        numeric = [f"synthetic_lab_{a}" for a in step.AGGREGATIONS] + [
            "age_at_t0",
            "hours_from_admission",
        ]
        self.schema = dict(
            numeric_features=numeric,
            categorical_features=step.CATEGORIES,
            features=dict.fromkeys(numeric + step.CATEGORIES, {}),
            key_column="sample_id",
            lookback_hours=24,
            categorical_missing_value="__MISSING__",
            preprocessing_applied=False,
        )
        rows = []
        for r in sample_rows:
            row: dict[str, Any] = dict(
                sample_id=r["sample_id"],
                age_at_t0=50,
                hours_from_admission=48,
                sex="M",
                admission_type="EMERGENCY",
                admission_location="ED",
                careunit_at_t0="WARD",
                current_service="MED",
            )
            row.update(
                {
                    f"synthetic_lab_{a}": 0 if a == "count_24h" else np.nan
                    for a in step.AGGREGATIONS
                }
            )
            rows.append(row)
        self.ehr = pa.Table.from_pylist(
            rows,
            schema=pa.schema(
                [
                    ("sample_id", pa.string()),
                    *[(n, pa.float32()) for n in numeric],
                    *[(n, pa.string()) for n in step.CATEGORIES],
                ]
            ),
        )
        self.static = dict(
            patients=pa.Table.from_pylist(static_patients),
            admissions=pa.Table.from_pylist(static_admissions),
            services=pa.Table.from_pylist(static_services),
        )
        for name, table in self.static.items():
            pq.write_table(table, self.cache / f"{name}.parquet")
        pq.write_table(self.static["admissions"], self.lab_admissions_path)
        self.write_labs([])
        self.decision = dict(
            target_formula="T + 2*R + 2*C + 3*D",
            hypoxemia_source="ICU",
            live_discharge_rule="truncate",
            outcome_version="burden_v2",
        )
        self.split_settings = dict(
            algorithm_version="synthetic",
            seed=42,
            weights=[70, 15, 15],
            primary_stratum="ever_phys_death",
            require_high_burden_in_each_split=True,
        )
        pq.write_table(self.samples, self.package / "samples.parquet")
        pq.write_table(self.ehr, self.package / "ehr_features.parquet")
        step.write_json(self.package / "feature_schema.json", self.schema)
        (self.package / "README.md").write_text(
            "Synthetic portable package. Change DATA_ROOT only."
        )
        self.summaries = step.expected_summary(self.samples)
        _write_csv(
            self.package / "summary.csv",
            ["split", "metric", "value", "unit", "denominator"],
            self.summaries,
        )
        self.freeze: dict[str, Any] = dict(
            package_file_sha256=self.hashes(),
            binding=dict(
                configuration_sha256={}, source_sha256={}, code_sha256="synthetic"
            ),
        )
        metadata = {
            k: self.settings[k]
            for k in (
                "interface_version",
                "release_id",
                "cohort_version",
                "split_version",
                "dictionary_version",
                "feature_set_version",
            )
        }
        metadata.update(
            self.decision,
            horizon_hours=72,
            ehr_lookback_hours=24,
            target_column="burden_score",
            target_range=[0, 8],
            split_column="split",
            split_values=list(step.SPLITS),
            split_unit="subject_id",
            ehr_preprocessing_applied=False,
            independent_validation_completed=False,
            image=dict(
                format="npy",
                dtype="float32",
                shape=[224, 224],
                value_range=[0.0, 1.0],
                model_normalization_applied=False,
            ),
            image_config={"version": "synthetic"},
            split_rules=self.split_settings,
            configuration_sha256={},
            upstream_file_sha256={},
            export_code_sha256="synthetic",
            file_sha256=self.hashes(),
            feature_counts=dict(numeric=8, categorical=5),
            split_counts={
                g: {
                    r["metric"]: r["value"]
                    for r in self.summaries
                    if r["split"] == g
                    and r["metric"] in ("studies", "admissions", "patients")
                }
                for g in ("all", *step.SPLITS)
            },
        )
        step.write_json(self.package / "dataset_metadata.json", metadata)
        self.freeze["package_file_sha256"] = self.hashes()
        self.archive = self.package.with_suffix(".zip")
        self.write_zip()
        self.freeze["zip_sha256"] = _file_hash(self.archive)
        self.manifest_path = self.processed / "imaging/burden_v2/image_manifest.parquet"
        pq.write_table(self.images, self.manifest_path)
        pq.write_table(self.qc, self.manifest_path.with_name("image_qc_status.parquet"))
        freeze_path = (
            self.outputs / "audit/burden_v2/training/training_package_freeze.json"
        )
        freeze_path.parent.mkdir(parents=True)
        step.write_json(freeze_path, self.freeze)
        self.context = dict(
            package=self.package,
            archive=self.archive,
            freeze=self.freeze,
            watched={str(self.lab_path): _file_hash(self.lab_path)},
            raw_versions={},
            cohort=self.cohort,
            split=self.split,
            samples=self.samples,
            ehr=self.ehr,
            schema=self.schema,
            rules=self.rules,
            evidence=self.cohort,
            static_cache=self.cache,
            image_manifest=self.manifest_path,
            processed=self.processed,
            outputs=self.outputs,
            lab_audit=dict(
                cache_paths=dict(
                    labs=str(self.lab_path), admissions=str(self.lab_admissions_path)
                )
            ),
            decision=self.decision,
            split_settings=self.split_settings,
            image_settings={"version": "synthetic"},
        )
        paths = self.root / "paths.json"
        step.write_json(
            paths,
            dict(
                outputs=str(self.outputs),
                processed=str(self.processed),
                mimic_cxr=str(self.root / "cxr"),
                mimic_iv=str(self.root / "iv"),
            ),
        )
        config = self.root / "validation.json"
        step.write_json(config, self.settings)
        self.args = SimpleNamespace(
            paths=paths, validation_config=config, processing=self.processing
        )

    def hashes(self):
        return {
            p.relative_to(self.package).as_posix(): _file_hash(p)
            for p in self.package.rglob("*")
            if p.is_file()
        }

    def write_zip(self, extra=None):
        with zipfile.ZipFile(
            self.archive, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name in self.hashes():
                archive.write(
                    self.package / name, f"{self.settings['release_id']}/{name}"
                )
            if extra:
                archive.writestr(*extra)

    def raw_lab(self, **changes):
        first = self.cohort_rows[0]
        row: dict[str, Any] = dict(
            lab_record_key="record1",
            labevent_id="1",
            subject_id=str(first["subject_id"]),
            hadm_id=str(first["hadm_id"]),
            specimen_id="1",
            itemid="1",
            charttime=str(self.t0),
            storetime=str(self.t0),
            value="7",
            valuenum="7",
            valueuom="mg/dL",
        )
        row.update(changes)
        return row

    def write_labs(self, rows):
        pq.write_table(
            pa.Table.from_pylist(
                rows, schema=pa.schema([(k, pa.string()) for k in self.raw_lab()])
            ),
            self.lab_path,
        )

    def labs(self, actual=None, admissions=None):
        return step.reconstruct_labs(
            self.lab_path,
            admissions or self.static["admissions"],
            self.cohort,
            self.rules,
            actual or self.ehr,
        )

    def with_feature(self, column, value):
        rows = self.ehr.to_pylist()
        rows[0][column] = value
        return pa.Table.from_pylist(rows, schema=self.ehr.schema)

    def run_synthetic(self):
        with patch.object(step, "verify_provenance", return_value=self.context):
            return step.run_validation(self.root, self.args)

    def test_normal_independent_checks_and_relocation(self):
        step.validate_samples(self.samples, self.cohort, self.split, self.settings)
        step.validate_ehr_contract(self.ehr, self.samples, self.schema, self.rules)
        self.labs()
        step.reconstruct_static(
            self.cohort, self.cohort, **self.static, actual=self.ehr
        )
        step.validate_metadata(
            self.package,
            self.samples,
            self.schema,
            self.freeze,
            self.settings,
            self.decision,
            self.split_settings,
            {"version": "synthetic"},
        )
        result = step.relocate_archive(
            self.archive, self.package, self.settings["release_id"], self.hashes()
        )
        self.assertEqual(result["relocated_images"], 3)
        self.assertTrue(result["temporary_directory_removed"])

    def test_wrong_formula_even_when_cohort_agrees(self):
        rows = self.cohort.to_pylist()
        rows[1]["burden_score"] = 4
        samples = self.samples.to_pylist()
        samples[1]["burden_score"] = 4
        with self.assertRaisesRegex(ValueError, "formula"):
            step.validate_samples(
                pa.Table.from_pylist(samples, schema=step.SAMPLE_SCHEMA),
                pa.Table.from_pylist(rows),
                self.split,
                self.settings,
            )

    def test_changed_split_key_and_types(self):
        for field, value in [
            ("split", "test"),
            ("sample_id", "wrong"),
            ("image_relpath", "../image.npy"),
        ]:
            rows = self.samples.to_pylist()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                step.validate_samples(
                    pa.Table.from_pylist(rows, schema=step.SAMPLE_SCHEMA),
                    self.cohort,
                    self.split,
                    self.settings,
                )
        with self.assertRaisesRegex(ValueError, "types"):
            step.validate_samples(
                self.samples.set_column(
                    7, "burden_score", self.samples["burden_score"].cast(pa.float64())
                ),
                self.cohort,
                self.split,
                self.settings,
            )

    def test_future_event_time_and_early_discharge(self):
        rows = self.cohort.to_pylist()
        rows[1][step.EVENTS[2] + "_first_event_time"] = self.t0 + timedelta(hours=73)
        with self.assertRaisesRegex(ValueError, "outside"):
            step.validate_samples(
                self.samples, pa.Table.from_pylist(rows), self.split, self.settings
            )
        rows = self.cohort.to_pylist()
        samples = self.samples.to_pylist()
        rows[0].update(
            dischtime=self.t0 + timedelta(hours=2),
            observation_end=self.t0 + timedelta(hours=2),
            observed_hours=2,
            early_discharge=1,
        )
        samples[0].update(
            observation_end=rows[0]["observation_end"],
            observed_hours=2,
            early_discharge=1,
        )
        step.validate_samples(
            pa.Table.from_pylist(samples, schema=step.SAMPLE_SCHEMA),
            pa.Table.from_pylist(rows),
            self.split,
            self.settings,
        )

    def test_inclusive_labs_unavailable_and_numeric_revisions(self):
        self.write_labs(
            [
                self.raw_lab(valuenum="", value="pending"),
                self.raw_lab(
                    lab_record_key="older",
                    labevent_id="2",
                    storetime=str(self.t0 - timedelta(hours=1)),
                    charttime=str(self.t0 - timedelta(hours=2)),
                ),
            ]
        )
        actual = self.ehr.to_pylist()
        actual[0].update(
            {
                f"synthetic_lab_{a}": v
                for a, v in zip(step.AGGREGATIONS, [7, 7, 7, 7, 1, 2], strict=True)
            }
        )
        self.labs(pa.Table.from_pylist(actual, schema=self.ehr.schema))
        self.write_labs(
            [
                self.raw_lab(
                    charttime=str(self.t0 - timedelta(hours=24)),
                    storetime=str(self.t0),
                    valuenum="1e10",
                    value="1e10",
                )
            ]
        )
        actual[0].update(
            {
                f"synthetic_lab_{a}": v
                for a, v in zip(
                    step.AGGREGATIONS, [1e10, 1e10, 1e10, 1e10, 1, 24], strict=True
                )
            }
        )
        self.labs(pa.Table.from_pylist(actual, schema=self.ehr.schema))
        for changes in [
            dict(storetime=str(self.t0 + timedelta(seconds=1))),
            dict(charttime="invalid"),
            dict(storetime=""),
            dict(hadm_id="99999"),
            dict(charttime=str(self.t0 + timedelta(seconds=1))),
            dict(charttime=str(self.t0 - timedelta(hours=24, seconds=1))),
        ]:
            self.write_labs([self.raw_lab(**changes)])
            self.labs()

    def test_new_nonnumeric_revision_prevents_numeric_backfill(self):
        self.write_labs(
            [
                self.raw_lab(
                    storetime=str(self.t0 - timedelta(hours=1)),
                    charttime=str(self.t0 - timedelta(hours=2)),
                ),
                self.raw_lab(
                    lab_record_key="revision",
                    labevent_id="2",
                    charttime=str(self.t0 - timedelta(hours=2)),
                    value="pending",
                    valuenum="",
                ),
            ]
        )
        self.labs()
        with self.assertRaisesRegex(ValueError, "Reconstruction"):
            self.labs(self.with_feature("synthetic_lab_count_24h", 1))

    def test_specimen_units_comparator_and_numeric_disagreement(self):
        for changes in [dict(value="<7"), dict(valueuom=""), dict(valueuom="other")]:
            self.write_labs([self.raw_lab(**changes)])
            self.labs()
        for changes in [
            dict(specimen_id=""),
            dict(specimen_id="invalid"),
            dict(value="8"),
            dict(labevent_id=""),
        ]:
            self.write_labs([self.raw_lab(**changes)])
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.labs()

    def test_admission_inference_uses_all_patient_admissions(self):
        self.write_labs([self.raw_lab(hadm_id="")])
        rows = self.static["admissions"].to_pylist()
        rows.append({**rows[0], "hadm_id": "99999"})
        self.labs(admissions=pa.Table.from_pylist(rows))
        rows[-1]["admittime"] = str(self.t0 + timedelta(hours=1))
        with self.assertRaisesRegex(ValueError, "Reconstruction"):
            self.labs(admissions=pa.Table.from_pylist(rows))

    def test_conflicting_ids_across_unretained_items(self):
        self.write_labs(
            [self.raw_lab(), self.raw_lab(lab_record_key="conflict", itemid="2")]
        )
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.labs()

    def test_overlapping_windows_legitimately_reuse_record(self):
        rows = self.cohort.to_pylist()
        rows[1].update(
            subject_id=rows[0]["subject_id"],
            hadm_id=rows[0]["hadm_id"],
            t0=self.t0 + timedelta(hours=1),
        )
        self.write_labs([self.raw_lab()])
        actual = self.ehr.to_pylist()
        for i in (0, 1):
            actual[i].update(
                {
                    f"synthetic_lab_{a}": v
                    for a, v in zip(step.AGGREGATIONS, [7, 7, 7, 7, 1, i], strict=True)
                }
            )
        result = step.reconstruct_labs(
            self.lab_path,
            self.static["admissions"],
            pa.Table.from_pylist(rows),
            self.rules,
            pa.Table.from_pylist(actual, schema=self.ehr.schema),
        )
        self.assertEqual(result["retained_associations"], 2)

    def test_contract_rejects_label_leak_and_nonfinite(self):
        with self.assertRaises(ValueError):
            step.validate_ehr_contract(
                self.ehr.append_column("burden_score", self.samples["burden_score"]),
                self.samples,
                self.schema,
                self.rules,
            )
        with self.assertRaises(ValueError):
            step.validate_ehr_contract(
                self.with_feature("age_at_t0", np.inf),
                self.samples,
                self.schema,
                self.rules,
            )
        with self.assertRaises(ValueError):
            step.validate_ehr_contract(
                self.with_feature("sex", ""), self.samples, self.schema, self.rules
            )

    def test_future_services_and_conflicting_latest_service(self):
        services = self.static["services"].to_pylist()
        services.append(
            {
                **services[0],
                "transfertime": str(self.t0 + timedelta(seconds=1)),
                "curr_service": "SURG",
            }
        )
        step.reconstruct_static(
            self.cohort,
            self.cohort,
            self.static["patients"],
            self.static["admissions"],
            pa.Table.from_pylist(services),
            self.ehr,
        )
        services[-1]["transfertime"] = str(self.t0)
        with self.assertRaisesRegex(ValueError, "latest service"):
            step.reconstruct_static(
                self.cohort,
                self.cohort,
                self.static["patients"],
                self.static["admissions"],
                pa.Table.from_pylist(services),
                self.ehr,
            )

    def test_latest_missing_service_does_not_backfill(self):
        services = self.static["services"].to_pylist()
        services[0]["curr_service"] = ""
        actual = self.with_feature("current_service", "__MISSING__")
        step.reconstruct_static(
            self.cohort,
            self.cohort,
            self.static["patients"],
            self.static["admissions"],
            pa.Table.from_pylist(services),
            actual,
        )
        with self.assertRaises(ValueError):
            step.reconstruct_static(
                self.cohort,
                self.cohort,
                self.static["patients"],
                self.static["admissions"],
                pa.Table.from_pylist(services),
                self.ehr,
            )

    def test_selector_and_conversion_rejected(self):
        for changes in [
            dict(conversion_multiplier=np.inf),
            dict(specimen_rule='{"kind":"expression"}'),
            dict(valid_min=0),
            dict(source_unit="", missing_unit_rule=""),
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                step.retained_rules(
                    pa.Table.from_pylist([{**self.rules[0], **changes}])
                )
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            step.retained_rules(pa.Table.from_pylist(self.rules * 2))

    def test_images_bad_pixels_qc_and_source_checksum(self):
        def check():
            return step.validate_images(
                self.package,
                self.samples,
                self.cohort,
                self.images,
                self.qc,
                self.root / "cxr",
                self.processed / "imaging/burden_v2",
                self.processing,
            )

        self.assertEqual(check()["images_checked"], 3)
        path = self.package / self.samples.to_pylist()[0]["image_relpath"]
        for image in [
            np.zeros((2, 2), dtype=np.float32),
            np.full((224, 224), 2, dtype=np.float32),
            np.zeros((224, 224), dtype=np.float64),
            np.full((224, 224), np.nan, dtype=np.float32),
        ]:
            np.save(path, image)
            with self.assertRaises(ValueError):
                step.validate_pixels(path)
        with self.assertRaises(ValueError):
            check()

    def test_summary_and_metadata_tampering(self):
        def check():
            step.validate_metadata(
                self.package,
                self.samples,
                self.schema,
                self.freeze,
                self.settings,
                self.decision,
                self.split_settings,
                {"version": "synthetic"},
            )

        path = self.package / "dataset_metadata.json"
        original = path.read_text()
        for changes in [
            dict(ehr_preprocessing_applied=True),
            dict(horizon_hours=24),
            dict(independent_validation_completed=True),
            dict(extra_path="C:/private/source"),
        ]:
            step.write_json(path, {**json.loads(original), **changes})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                check()
        path.write_text(original)
        (self.package / "summary.csv").write_text(
            "split,metric,value,unit,denominator\nall,studies,999,study,999\n"
        )
        with self.assertRaises(ValueError):
            check()

    def test_zip_duplicate_traversal_absolute_and_unknown(self):
        for name in [
            f"{self.settings['release_id']}/samples.parquet",
            "../outside",
            "/absolute",
            "C:/outside",
            f"{self.settings['release_id']}/unknown",
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.write_zip((name, b"malicious"))
            with self.subTest(name=name), self.assertRaises(ValueError):
                step.relocate_archive(
                    self.archive,
                    self.package,
                    self.settings["release_id"],
                    self.hashes(),
                )

    def test_zip_symlink_size_and_member_checksum(self):
        for mutation in ("symlink", "size", "checksum"):
            with zipfile.ZipFile(self.archive, "w") as archive:
                for name in self.hashes():
                    info = zipfile.ZipInfo(f"{self.settings['release_id']}/{name}")
                    data = (self.package / name).read_bytes()
                    if name == "README.md":
                        if mutation == "symlink":
                            info.external_attr = (stat.S_IFLNK | 0o777) << 16
                        elif mutation == "size":
                            data += b"x"
                        else:
                            data = b"X" + data[1:]
                    archive.writestr(info, data)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                step.relocate_archive(
                    self.archive,
                    self.package,
                    self.settings["release_id"],
                    self.hashes(),
                )

    def test_stale_hash_source_version_and_code(self):
        step.verify_stability(self.context, step.code_hash())
        self.context["raw_versions"][str(self.lab_path)] = _file_version(self.lab_path)
        self.write_labs([self.raw_lab()])
        with self.assertRaises(ValueError):
            step.verify_stability(self.context, step.code_hash())
        self.context["watched"] = {}
        with self.assertRaises(ValueError):
            step.verify_stability(self.context, step.code_hash())
        self.context["raw_versions"] = {}
        with self.assertRaisesRegex(ValueError, "code"):
            step.verify_stability(self.context, "wrong")

    def test_success_then_failure_supersedes_acceptance_without_package_changes(self):
        before = self.hashes()
        self.assertEqual(self.run_synthetic(), 0)
        path = self.outputs / "audit/burden_v2/training_package_acceptance.json"
        acceptance = json.loads(path.read_text())
        self.assertTrue(acceptance["training_package_accepted"])
        self.assertEqual(
            acceptance["checks"]["train_only_model_preprocessing"]["status"],
            "not_applicable",
        )
        self.context["ehr"] = self.with_feature("age_at_t0", 60)
        self.assertEqual(self.run_synthetic(), 1)
        self.assertFalse(json.loads(path.read_text())["training_package_accepted"])
        self.assertEqual(before, self.hashes())
        _verify_run(self.outputs, 12)
        self.assertEqual(
            len(
                list(
                    (self.outputs / "audit/burden_v2/validation").rglob(
                        "validation_report.json"
                    )
                )
            ),
            2,
        )

    def test_blocked_provenance_publishes_rejection(self):
        with patch.object(
            step, "verify_provenance", side_effect=FileNotFoundError("missing snapshot")
        ):
            self.assertEqual(step.run_validation(self.root, self.args), 1)
        report = json.loads(
            (self.outputs / "audit/burden_v2/leakage_checks.json").read_text()
        )
        self.assertFalse(report["training_package_accepted"])
        self.assertFalse(report["independent_validation_completed"])
        self.assertEqual(report["checks"]["samples"]["status"], "blocked")

    def test_report_publication_recovers_interruption(self):
        self.assertEqual(self.run_synthetic(), 0)
        from src.cohort import common

        original = common.os.replace
        calls = 0

        def interrupt(source, target):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("Synthetic interrupted publication")
            return original(source, target)

        with (
            patch.object(common.os, "replace", side_effect=interrupt),
            self.assertRaises(OSError),
        ):
            self.run_synthetic()
        self.assertEqual(self.run_synthetic(), 0)
        _verify_run(self.outputs, 12)

    def test_provenance_refuses_missing_step_manifest(self):
        with self.assertRaisesRegex(FileNotFoundError, "manifest"):
            step.verify_provenance(
                self.root, self.args, self.settings, self.processed, self.outputs
            )

    def prepare_provenance(
        self, supplementary=False, review_sources=False, pending_reviews=0, vital_scope=None
    ):
        from src.cohort import create_patient_split, freeze_cohort
        from src.data import export_training_package
        from src.ehr import build_ehr_features, step10b

        def save(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, pa.Table):
                pq.write_table(data, path)
            else:
                step.write_json(path, data)
            return path

        if supplementary:
            self.settings.update(
                release_id="burden_v2_h72_r02",
                cohort_version="burden_v2_h72_r02",
                split_version="burden_v2_h72_r02_split_v1",
                dictionary_version="lab_dictionary_r02",
                feature_set_version="ehr_features_r02",
            )
            target = self.package.with_name(self.settings["release_id"])
            self.package.rename(target)
            self.package, self.archive = target, target.with_suffix(".zip")
            save(self.args.validation_config, self.settings)
            metadata = json.loads((self.package / "dataset_metadata.json").read_text())
            metadata.update(
                {
                    k: self.settings[k]
                    for k in (
                        "release_id",
                        "cohort_version",
                        "split_version",
                        "dictionary_version",
                        "feature_set_version",
                    )
                }
            )
            save(self.package / "dataset_metadata.json", metadata)
        audit = self.outputs / "audit/burden_v2"
        data = self.processed / "ehr/burden_v2"
        cohort_version, split_version = (
            self.settings["cohort_version"],
            self.settings["split_version"],
        )
        export_settings = {
            k: self.settings[k]
            for k in (
                "interface_version",
                "release_id",
                "cohort_version",
                "split_version",
                "dictionary_version",
                "feature_set_version",
            )
        }
        lab_config={"audit_version": "lab_audit_v1"}
        if vital_scope is not None:
            lab_config.update(reviewer="ZE Xiaokang",vital_audit={"items":{"220045":"Heart Rate"}})
        for name, content in dict(
            cohort={},
            ehr={"burden_v2": {"tables": {}}},
            imaging={},
            split_config=self.split_settings,
            lab_config=lab_config,
            feature_config={"audit_version": "ehr_audit_v1"},
            export_config=export_settings,
        ).items():
            setattr(self.args, name, save(self.root / f"{name}.json", content))
        self.files = {
            "cohort": save(
                self.processed / "cohort/burden_v2/cohort.parquet", self.cohort
            ),
            "cohort_freeze": audit / "cohort_freeze.json",
            "split_freeze": audit / "split_freeze.json",
            "endpoint_decision": save(
                audit / "endpoint_decision.json",
                {
                    **self.decision,
                    "horizon_hours": 72,
                    "horizon_frozen": True,
                    "label_rules_frozen": True,
                    "source_audit": {"input_versions": {}},
                },
            ),
            "split": save(
                self.processed / "splits/burden_v2/patient_split.parquet", self.split
            ),
            "ehr": save(data / "ehr_features_raw.parquet", self.ehr),
            "schema": save(data / "feature_schema.json", self.schema),
            "feature_freeze": audit / "ehr/ehr_audit_v1/ehr_features_freeze.json",
            "dictionary": save(
                data / "lab_dictionary.parquet", pa.Table.from_pylist(self.rules)
            ),
            "dictionary_freeze": audit / "lab_dictionary_freeze.json",
            "lab_audit": audit / "labs/lab_audit_v1/lab_audit_manifest.json",
        }
        cohort_audit = save(
            self.processed / "cohort/burden_v2/cohort_audit.parquet", self.cohort
        )
        for version, tables, frozen_path, record in [
            (
                cohort_version,
                [self.files["cohort"], cohort_audit],
                self.files["cohort_freeze"],
                dict(
                    common_cohort_frozen=True,
                    release_version=cohort_version,
                    horizon_hours=72,
                    source_binding={"code_sha256": freeze_cohort._code_hash()},
                ),
            ),
            (
                split_version,
                [self.files["split"]],
                self.files["split_freeze"],
                dict(
                    patient_split_frozen=True,
                    split_version=split_version,
                    cohort_version=cohort_version,
                    source_binding={
                        "code_sha256": create_patient_split._code_hash(),
                        "split_config_sha256": _file_hash(self.args.split_config),
                    },
                ),
            ),
        ]:
            record["artifact_sha256"] = {p.name: _file_hash(p) for p in tables}
            save(frozen_path, record)
            save(audit / "releases" / version / frozen_path.name, record)
            for p in tables:
                target = p.parent / "releases" / version / p.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(p.read_bytes())
        cohort_identity, split_identity = (
            {"version": cohort_version},
            {"version": split_version},
        )
        raw = self.root / "iv/hosp"
        raw.mkdir(parents=True)
        raw_sources = {
            name: raw / f"{name}.csv"
            for name in ("d_labitems", "labevents", "admissions")
        }
        for p in raw_sources.values():
            p.write_text("synthetic source only\n")
        lab_binding = dict(
            cohort=cohort_identity,
            split=split_identity,
            code_sha256=build_ehr_features.code_hash(),
            configuration_sha256=_file_hash(self.args.lab_config),
            source_paths={k: str(v) for k, v in raw_sources.items()},
            source_versions={k: _file_version(v) for k, v in raw_sources.items()},
            dictionary_source_sha256=_file_hash(raw_sources["d_labitems"]),
        )
        save(
            self.files["lab_audit"],
            dict(
                source_binding=lab_binding,
                audit_fingerprint="synthetic",
                **({"vital_audit":{"features_approved":False}} if vital_scope is not None else {}),
                cache_paths=dict(
                    labs=str(self.lab_path), admissions=str(self.lab_admissions_path)
                ),
                artifact_sha256={
                    str(p): _file_hash(p)
                    for p in (self.lab_path, self.lab_admissions_path)
                },
            ),
        )
        dictionary_freeze = dict(
            dictionary_frozen=True,
            dictionary_version=self.settings["dictionary_version"],
            dictionary_sha256=_file_hash(self.files["dictionary"]),
            source_binding=dict(
                audit_manifest_sha256=_file_hash(self.files["lab_audit"]),
                audit_fingerprint="synthetic",
                review_files_sha256={},
                code_sha256=build_ehr_features.code_hash(),
            ),
        )
        if vital_scope is not None:
            self.scope_path=save(self.root / "feature_scope.json",dict(
                vital_feature_scope=vital_scope,audit_fingerprint="synthetic",
                audit_manifest_sha256=_file_hash(self.files["lab_audit"]),
                reviewer="ZE Xiaokang",reviewed_on="2026-10-29",review_version="synthetic_scope_v1",
                rationale="User excluded vital features",cohort_version=cohort_version,
                split_version=split_version,retained_lab_itemids=["1"],excluded_vital_itemids=["220045"],
                static_features_unchanged=True,no_study_excluded_for_missing_features=True,
                pending_raw_vital_issues_resolved=False))
            dictionary_freeze["source_binding"]["feature_scope_review"]={
                "path":str(self.scope_path),"sha256":_file_hash(self.scope_path),
                "vital_feature_scope":vital_scope}
            dictionary_freeze["source_binding"]["review_files_sha256"][str(self.scope_path)]=_file_hash(self.scope_path)
        save(self.files["dictionary_freeze"], dictionary_freeze)
        dictionary_snapshot = (
            data
            / "releases"
            / self.settings["dictionary_version"]
            / "lab_dictionary.parquet"
        )
        save(dictionary_snapshot, pa.Table.from_pylist(self.rules))
        dictionary_snapshot_freeze = save(
            audit
            / "labs/lab_audit_v1/dictionary_releases"
            / self.settings["dictionary_version"]
            / "lab_dictionary_freeze.json",
            dictionary_freeze,
        )
        save(
            audit / "lab_dictionary_releases.json",
            {
                self.settings["dictionary_version"]: dict(
                    dictionary_path=str(dictionary_snapshot),
                    dictionary_sha256=_file_hash(dictionary_snapshot),
                    freeze_path=str(dictionary_snapshot_freeze),
                    freeze_sha256=_file_hash(dictionary_snapshot_freeze),
                )
            },
        )
        binding = dict(
            cohort=cohort_identity,
            split=split_identity,
            code_sha256=step10b.code_hash(),
            configuration_sha256=_file_hash(self.args.feature_config),
            source_paths={},
            source_versions={},
            dictionary_sha256=_file_hash(self.files["dictionary"]),
            dictionary_freeze_sha256=_file_hash(self.files["dictionary_freeze"]),
            lab_audit_sha256=_file_hash(self.files["lab_audit"]),
        )
        candidate = save(self.cache / "candidate.parquet", self.ehr)
        feature_audit_path = save(
            self.files["feature_freeze"].parent / "ehr_audit_manifest.json",
            dict(
                binding=binding,
                audit_checks_passed=True,
                blocking_conflicts=0,
                cache_table=str(candidate),
                artifact_sha256={
                    str(p): _file_hash(p) for p in self.cache.glob("*.parquet")
                },
            ),
        )
        feature_freeze = dict(
            binding={
                **binding,
                "audit_manifest_sha256": _file_hash(feature_audit_path),
            },
            ehr_features_ready=True,
            feature_set_version=self.settings["feature_set_version"],
            artifact_sha256={
                p.name: _file_hash(p) for p in (self.files["ehr"], self.files["schema"])
            },
        )
        save(self.files["feature_freeze"], feature_freeze)
        feature_snapshot = data / "releases" / self.settings["feature_set_version"]
        save(feature_snapshot / "ehr_features_freeze.json", feature_freeze)
        for p in (self.files["ehr"], self.files["schema"]):
            (feature_snapshot / p.name).write_bytes(p.read_bytes())
        save(
            audit / "ehr/ehr_features_releases.json",
            {
                self.settings["feature_set_version"]: {
                    "freeze_sha256": _file_hash(self.files["feature_freeze"])
                }
            },
        )
        configs = [
            self.args.paths,
            self.args.cohort,
            self.args.ehr,
            self.args.imaging,
            self.args.processing,
            self.args.split_config,
            self.args.lab_config,
            self.args.feature_config,
            self.args.export_config,
        ]
        decisions = {}
        if supplementary:
            for name in ("cohort_decisions_path", "event_decisions_path"):
                path = self.root / (name + ".csv")
                path.write_text("sample_id,decision\n", encoding="utf-8")
                self.files[name] = path
                decisions[name.replace("_path", "_sha256")] = _file_hash(path)
            save(
                self.args.cohort,
                {
                    "burden_v2": {
                        name: str(self.files[name])
                        for name in ("cohort_decisions_path", "event_decisions_path")
                    }
                },
            )
        review_bindings, review_artifact = {}, None
        if review_sources:
            source_root = Path(json.loads(self.args.paths.read_text())["mimic_iv"])
            if not source_root.is_absolute():
                source_root = self.root / source_root
            source = source_root / "icu/review_inputevents.csv"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(
                "subject_id,hadm_id,starttime\n1,10,2120-01-01 12:00:00\n",
                encoding="utf-8",
            )
            self.review_source = source
            review_bindings = {
                "inputevents_full": {
                    "path": str(source),
                    "version": _file_version(source),
                    "columns": ["subject_id", "hadm_id", "starttime"],
                }
            }
            save(
                self.args.ehr,
                {
                    "burden_v2": {
                        "tables": {},
                        "review_evidence_sources": {
                            "inputevents_full": "icu/review_inputevents.csv"
                        },
                    }
                },
            )
            artifact = save(
                self.processed / "cohort/burden_v2/review_source_records.parquet",
                pa.Table.from_pylist(
                    [
                        dict(
                            subject_id=1,
                            hadm_id=10,
                            raw_evidence="synthetic reviewed source",
                        )
                    ]
                ),
            )
            self.files["review_source_artifact"] = artifact
            review_artifact = {
                "path": artifact.relative_to(self.root).as_posix(),
                "sha256": _file_hash(artifact),
            }
        for n in range(1, 12):
            with _OutputRun(self.outputs, n) as run:
                run.path(self.outputs / f"synthetic_step_{n}.txt").write_text(
                    "Complete synthetic provenance marker"
                )
                if supplementary and n == 5:
                    save(
                        run.path(audit / "run_metadata.json"),
                        dict(
                            run_id=run.run_id,
                            input_versions={},
                            review_evidence_sources=review_bindings,
                            review_source_artifact=review_artifact,
                            review_evidence_only=False,
                            pending_review_decisions={
                                "cohort": 0,
                                "event": pending_reviews,
                            },
                            **decisions,
                        ),
                    )
        if supplementary:
            from src.cohort.audit_burden_labels import save_audit_snapshot

            index = save_audit_snapshot(self.outputs)
            snapshot_bindings = {
                index.relative_to(self.root).as_posix(): _file_hash(index)
            }
            meta_path = audit / "run_metadata.json"
            metadata05 = json.loads(meta_path.read_text())
            # The snapshot is the original review source, not its own later audit.
            metadata05["review_snapshot_bindings"] = snapshot_bindings
            with _OutputRun(self.outputs, 5) as run:
                metadata05["run_id"] = run.run_id
                save(run.path(meta_path), metadata05)
            marker = self.outputs / "audit/step_05_manifest.json"
            review = save(
                self.root / "endpoint_review.json",
                {"reviewer": "Synthetic reviewer only"},
            )
            endpoint = json.loads(self.files["endpoint_decision"].read_text())
            endpoint.update(
                **decisions,
                review_evidence_sources=review_bindings,
                review_source_artifact=review_artifact,
                review_snapshot_bindings=snapshot_bindings,
                review_records_sha256={
                    review.relative_to(self.root).as_posix(): _file_hash(review)
                },
                source_audit=dict(
                    input_versions={},
                    run_id=metadata05["run_id"],
                    manifest_path=marker.relative_to(self.root).as_posix(),
                    manifest_sha256=_file_hash(marker),
                    metadata_path=meta_path.relative_to(self.root).as_posix(),
                    metadata_sha256=_file_hash(meta_path),
                ),
            )
            save(self.files["endpoint_decision"], endpoint)
        binding = dict(
            settings=export_settings,
            cohort=cohort_identity,
            split=split_identity,
            source_sha256={k: _file_hash(p) for k, p in self.files.items()},
            configuration_sha256={p.name: _file_hash(p) for p in configs},
            code_sha256=export_training_package.code_hash(),
        )
        metadata_path = self.package / "dataset_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(
            upstream_file_sha256=binding["source_sha256"],
            configuration_sha256=binding["configuration_sha256"],
            export_code_sha256=binding["code_sha256"],
        )
        save(metadata_path, metadata)
        self.write_zip()
        self.freeze.update(
            binding=binding,
            release_id=self.settings["release_id"],
            training_package_exported=True,
            package_file_sha256=self.hashes(),
            zip_sha256=_file_hash(self.archive),
            image_manifest_sha256=_file_hash(self.manifest_path),
        )
        save(audit / "training/training_package_freeze.json", self.freeze)
        package_snapshot = audit / "training/releases" / self.settings["release_id"]
        save(package_snapshot / "training_package_freeze.json", self.freeze)
        save(package_snapshot / "image_manifest.parquet", self.images)
        save(
            audit / "training/training_package_releases.json",
            {
                self.settings["release_id"]: {
                    "freeze_sha256": _file_hash(
                        package_snapshot / "training_package_freeze.json"
                    )
                }
            },
        )
        for mode in ("audit", "build"):
            namespace = self.outputs / "step_10b" / mode
            with _OutputRun(namespace, 10) as run:
                run.path(namespace / "synthetic.txt").write_text(
                    "Complete synthetic audit/build marker"
                )

    def test_complete_chain_reads_string_scope_path_and_rejects_tampering(self):
        self.prepare_provenance(vital_scope="exclude_all")
        self.assertEqual(step.run_validation(self.root,self.args),0)
        self.scope_path.write_text("{}")
        self.assertEqual(step.run_validation(self.root,self.args),1)
        report=json.loads((self.outputs / "audit/burden_v2/training_package_acceptance.json").read_text())
        self.assertEqual(report["checks"]["provenance"]["status"],"fail")

    def test_complete_chain_rejects_unapproved_vital_policy(self):
        self.prepare_provenance(vital_scope="include")
        self.assertEqual(step.run_validation(self.root,self.args),1)
        report=json.loads((self.outputs / "audit/burden_v2/training_package_acceptance.json").read_text())
        self.assertIn("Invalid feature exclusion scope",report["checks"]["provenance"]["detail"])

    def test_complete_synthetic_frozen_chain_without_mocked_provenance(self):
        self.prepare_provenance()
        self.assertEqual(step.run_validation(self.root, self.args), 0)
        before = self.hashes()
        self.processing.write_text('{"version":"changed"}')
        self.assertEqual(step.run_validation(self.root, self.args), 1)
        report = json.loads(
            (
                self.outputs / "audit/burden_v2/training_package_acceptance.json"
            ).read_text()
        )
        self.assertFalse(report["training_package_accepted"])
        self.assertEqual(report["checks"]["provenance"]["status"], "fail")
        self.assertEqual(before, self.hashes())

    def test_r02_synthetic_chain_checks_new_decisions_and_snapshots(self):
        self.prepare_provenance(supplementary=True)
        self.assertEqual(step.run_validation(self.root, self.args), 0)
        path = self.files["cohort_decisions_path"]
        path.write_text("sample_id,decision\nchanged,pending\n", encoding="utf-8")
        self.assertEqual(step.run_validation(self.root, self.args), 1)
        path.write_text("sample_id,decision\n", encoding="utf-8")
        self.assertEqual(step.run_validation(self.root, self.args), 0)
        endpoint = json.loads(self.files["endpoint_decision"].read_text())
        index = self.root / next(iter(endpoint["review_snapshot_bindings"]))
        snapshot = json.loads(index.read_text())
        Path(next(iter(snapshot["files"].values()))["path"]).write_text(
            "tampered snapshot"
        )
        self.assertEqual(step.run_validation(self.root, self.args), 1)

    def test_r02_supplemental_sources_pass_then_reject_raw_or_artifact_tampering(self):
        self.prepare_provenance(supplementary=True, review_sources=True)
        self.assertEqual(step.run_validation(self.root, self.args), 0)
        artifact = self.files["review_source_artifact"]
        original = artifact.read_bytes()
        artifact.write_text("tampered evidence\n", encoding="utf-8")
        self.assertEqual(step.run_validation(self.root, self.args), 1)
        artifact.write_bytes(original)
        self.assertEqual(step.run_validation(self.root, self.args), 0)
        self.review_source.write_text("tampered raw source\n", encoding="utf-8")
        self.assertEqual(step.run_validation(self.root, self.args), 1)

    def test_r02_pending_review_decisions_cannot_pass_acceptance(self):
        self.prepare_provenance(supplementary=True, pending_reviews=1)
        self.assertEqual(step.run_validation(self.root, self.args), 1)

    def test_configuration_and_empty_failure_headers(self):
        unfilled=copy.deepcopy(self.settings)
        unfilled["expected_counts"]={k:None for k in unfilled["expected_counts"]}
        step.write_json(self.args.validation_config,unfilled)
        with self.assertRaises(ValueError):
            step.load_settings(self.args.validation_config)
        settings = copy.deepcopy(self.settings)
        settings["relocation_policy"] = "skip"
        step.write_json(self.args.validation_config, settings)
        with self.assertRaises(ValueError):
            step.load_settings(self.args.validation_config)
        with _OutputRun(self.outputs, 12) as run:
            step.publish_reports(
                run,
                self.outputs,
                self.settings,
                {k: {"status": "pass"} for k in step.CHECKS},
                {},
            )
        files = list(
            (self.outputs / "audit/burden_v2/validation").rglob("failure_details.csv")
        )
        self.assertEqual(files[0].read_text().strip(), "check,status,detail")


if __name__ == "__main__":
    unittest.main()
