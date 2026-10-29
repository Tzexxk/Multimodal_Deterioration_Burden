"""Synthetic portable training-package export and recovery checks."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.burden_io import COMPONENTS
from src.cohort.common import _file_hash, _verify_run
from src.data import export_training_package as step
from src.ehr import step10b as ehr
from src.ehr.lab_dictionary_inputs import APPROVED_FEATURES

ROOT = Path(__file__).resolve().parents[2]


class TrainingExportTests(unittest.TestCase):
    def test_export_reconstructs_all_audit_sources_even_when_vital_features_are_excluded(self):
        lab_settings = dict(audit_version="synthetic_audit", tables={
            "d_labitems":"hosp/d_labitems.csv", "labevents":"hosp/labevents.csv",
            "admissions":"hosp/admissions.csv"})
        for include_vital_audit in (False,True):
            with self.subTest(include_vital_audit=include_vital_audit):
                settings=dict(lab_settings)
                if include_vital_audit:
                    settings["vital_audit"]={"tables":{k:f"icu/{k}.csv" for k in
                                                           ("chartevents","d_items","icustays")}}
                with (patch.object(step.upstream,"verify_frozen_inputs",return_value=self.context),
                      patch.object(step,"_verify_run"),
                      patch.object(step.ehr,"load_settings",return_value={"audit_version":"synthetic_ehr"}),
                      patch.object(step.upstream,"dictionary_releases"),
                      patch.object(step.upstream,"load_settings",return_value=settings),
                      patch.object(step,"_json_file",return_value={"mimic_iv":str(self.root / "iv")}),
                      patch.object(step.upstream,"source_binding",side_effect=RuntimeError("binding captured")) as capture):
                    with self.assertRaisesRegex(RuntimeError,"binding captured"):
                        step.verify_inputs(self.root,self.args,self.settings)
                sources=capture.call_args.args[1]
                expected=set(lab_settings["tables"])
                if include_vital_audit:
                    expected|={"chartevents","d_items","icustays"}
                self.assertEqual(set(sources),expected)
                self.assertTrue(all(path.is_absolute() for path in sources.values()))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic_training_export_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.processed, self.outputs = (
            self.root / "data/processed",
            self.root / "outputs",
        )
        self.image_dir = self.processed / "imaging/burden_v2"
        self.settings = json.loads(
            (ROOT / "configs/training_export_v1.json").read_text()
        )
        self.args = SimpleNamespace(
            **{
                k: self.root / f"{k}.json"
                for k in [
                    "paths",
                    "cohort",
                    "ehr",
                    "imaging",
                    "processing",
                    "split_config",
                    "lab_config",
                    "feature_config",
                    "export_config",
                ]
            }
        )
        for path in vars(self.args).values():
            path.write_text("{}")
        self.args.paths.write_text(
            json.dumps(
                {
                    "processed": str(self.processed),
                    "outputs": str(self.outputs),
                    "mimic_cxr": str(self.root / "cxr"),
                    "mimic_iv": str(self.root / "iv"),
                }
            )
        )
        self.args.export_config.write_text(json.dumps(self.settings))
        cohort, qc, patient_rows = [], [], []
        for index, patient in enumerate([10000001, 10000001, 10000002, 10000003]):
            study, admission = 50000001 + index, 20000000 + patient
            key = f"p{patient}_h{admission}_s{study}"
            row: dict[str, Any] = {k: 0 for k in COMPONENTS}
            row.update(
                sample_id=key,
                subject_id=patient,
                hadm_id=admission,
                study_id=study,
                dicom_id=f"synthetic_dicom_{index}",
                dicom_path=f"dataset/p{patient}/s{study}/synthetic.dcm",
                view_position="AP",
                t0=datetime(2020, 1, 1, 12),
                observation_end=datetime(2020, 1, 4, 12),
                observed_hours=72.0,
                early_discharge=0,
                burden_score=0.0,
                phys_death=0,
                high_burden=0,
                source_dicom_sha256="0" * 64,
                processing_config_sha256="1" * 64,
                processing_version="burden_v2_images_v2",
                processing_status="complete",
                qc_status="pass",
            )
            row["npy_relpath"] = step.expected_image_path(row)
            image = self.image_dir / row["npy_relpath"]
            image.parent.mkdir(parents=True, exist_ok=True)
            np.save(
                image,
                np.full((224, 224), 0.1 + index * 0.1, dtype=np.float32),
                allow_pickle=False,
            )
            row["npy_sha256"] = _file_hash(image)
            if index == 1:
                row[COMPONENTS[2]] = 1
                row[COMPONENTS[5]] = 1
                row["burden_score"] = 5.0
                row["phys_death"] = 1
                row["high_burden"] = 1
            cohort.append(row)
            qc.append(dict(row))
            if patient not in {r["subject_id"] for r in patient_rows}:
                patient_rows.append(
                    dict(
                        subject_id=patient,
                        split=["train", "validation", "test"][len(patient_rows)],
                    )
                )
        self.cohort = pa.Table.from_pylist(cohort)
        self.rules = [
            {"itemid": int(k), "feature_name": n, "canonical_unit": unit}
            for k, (n, unit) in APPROVED_FEATURES.items()
        ]
        schema = ehr.schema_for(self.rules, "ehr_features_v1")
        table = ehr.float32_table(
            [
                {
                    "sample_id": r["sample_id"],
                    "age_at_t0": 50.0,
                    "sex": "F",
                    **{
                        k: 0
                        for k in schema["numeric_features"]
                        if k.endswith("_count_24h")
                    },
                }
                for r in cohort
            ],
            schema,
        )
        source = self.root / "frozen"
        source.mkdir()
        files = {
            "cohort": source / "cohort.parquet",
            "split": source / "split.parquet",
            "ehr": source / "ehr.parquet",
            "schema": source / "schema.json",
            "feature_freeze": source / "feature_freeze.json",
        }
        pq.write_table(self.cohort, files["cohort"])
        pq.write_table(pa.Table.from_pylist(patient_rows), files["split"])
        pq.write_table(table, files["ehr"])
        files["schema"].write_text(json.dumps(schema))
        files["feature_freeze"].write_text("{}")
        self.context = {
            "cohort": self.cohort,
            "qc_records": qc,
            "image_dir": self.image_dir,
            "processed": self.processed,
            "outputs": self.outputs,
            "schema": schema,
            "cohort_identity": {"synthetic": True},
            "split_identity": {"synthetic": True},
            "feature_freeze": {"binding": {"source_paths": {}, "source_versions": {}}},
            "source_files": files,
            "source_hashes_export": {k: _file_hash(v) for k, v in files.items()},
            "export_config_hashes": {
                p.name: _file_hash(p) for p in vars(self.args).values()
            },
            "lab_source_paths": {},
            "lab_source_versions": {},
            "decision": {
                "outcome_version": "burden_v2",
                "target_formula": "T + 2*R + 2*C + 3*D",
                "hypoxemia_source": "MIMIC-IV ICU chartevents only",
                "live_discharge_rule": "Observe until discharge or H; absent recorded events are operational zeros.",
            },
            "split_settings": {
                "algorithm_version": "patient_stratified_pcg64_v1",
                "seed": 42,
                "weights": [70, 15, 15],
                "primary_stratum": "ever_phys_death",
                "require_high_burden_in_each_split": True,
            },
            "image_settings": json.loads(
                (ROOT / "configs/image_processing_v1.json").read_text()
            ),
        }

    def run_export(self):
        with patch.object(step, "verify_inputs", return_value=self.context):
            return step.run_export(self.root, self.args)

    @property
    def package(self):
        return self.processed / "training" / self.settings["release_id"]

    @property
    def archive(self):
        return self.package.with_suffix(".zip")

    def test_five_fold_tables_export_once_with_shared_images_and_features(self):
        sources = self.context["source_files"]
        sources["cross_validation"] = self.root / "cross_validation_indices.json"
        sources["cross_validation"].write_text(json.dumps({"synthetic": True}))
        for fold in range(1, 6):
            path = self.root / f"patient_split_fold_{fold}.parquet"
            shutil.copyfile(sources["split"], path)
            sources[f"cv_fold_{fold}"] = path
        self.context["source_hashes_export"] = {k: _file_hash(v) for k, v in sources.items()}
        self.run_export()
        meta = json.loads((self.package / "dataset_metadata.json").read_text())
        for fold in range(1, 6):
            relative = f"splits/fold_{fold}/patient_split.parquet"
            self.assertEqual(_file_hash(self.package / relative), _file_hash(sources[f"cv_fold_{fold}"]))
            self.assertIn(relative, meta["file_sha256"])
            self.assertIn(relative, meta["cross_validation"]["patient_split_files"])
            with zipfile.ZipFile(self.archive) as archive:
                self.assertIn(f"{self.settings['release_id']}/{relative}", archive.namelist())
        self.assertEqual(len(list(self.package.rglob("ehr_features.parquet"))), 1)
        self.assertEqual(len(list(self.package.rglob("*.npy"))), self.cohort.num_rows)
        self.run_export()
        (self.package / "splits/fold_5/patient_split.parquet").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            self.run_export()

    def test_export_reuse_and_relocated_loading(self):
        self.run_export()
        _verify_run(self.outputs, 11)
        samples = pq.read_table(self.package / "samples.parquet")
        self.assertEqual(samples.schema, step.SAMPLE_SCHEMA)
        self.assertEqual(samples.num_rows, 4)
        self.assertEqual(
            _file_hash(self.package / "ehr_features.parquet"),
            _file_hash(self.context["source_files"]["ehr"]),
        )
        self.assertEqual(
            _file_hash(self.package / "feature_schema.json"),
            _file_hash(self.context["source_files"]["schema"]),
        )
        metadata = json.loads((self.package / "dataset_metadata.json").read_text())
        step.assert_portable(metadata)
        self.assertEqual(metadata["feature_counts"], {"numeric": 152, "categorical": 5})
        self.assertFalse(metadata["independent_validation_completed"])
        self.assertFalse(metadata["image"]["model_normalization_applied"])
        frozen = (
            self.outputs
            / "audit/burden_v2/training/releases"
            / self.settings["release_id"]
            / "training_package_freeze.json"
        )
        record = json.loads(frozen.read_text())
        digest = _file_hash(frozen)
        zip_digest = _file_hash(self.archive)
        self.run_export()
        self.assertEqual(digest, _file_hash(frozen))
        self.assertEqual(zip_digest, _file_hash(self.archive))
        latest = json.loads((self.outputs / "audit/step_11_manifest.json").read_text())
        self.assertIn(
            str(
                self.outputs / "audit/burden_v2/training/training_package_releases.json"
            ),
            latest["files"],
        )
        relocated = self.root / "moved"
        relocated.mkdir()
        with zipfile.ZipFile(self.archive) as archive:
            archive.extractall(relocated)
        data_root = relocated / self.settings["release_id"]
        shutil.rmtree(self.package)
        moved_samples = pq.read_table(data_root / "samples.parquet")
        moved_ehr = pq.read_table(data_root / "ehr_features.parquet")
        self.assertEqual(
            set(moved_samples["sample_id"].to_pylist()),
            set(moved_ehr["sample_id"].to_pylist()),
        )
        for row in moved_samples.to_pylist():
            pixels = np.load(data_root / row["image_relpath"], allow_pickle=False)
            step.validate_pixels(pixels)
        for name, h in record["package_file_sha256"].items():
            self.assertEqual(_file_hash(data_root / name), h)
        self.assertFalse((data_root / "image_manifest.parquet").exists())

    def test_summary_patient_and_study_denominators(self):
        patients = pq.read_table(self.context["source_files"]["split"])
        samples = step.build_samples(self.cohort, patients)
        rows = step.summary_rows(samples)
        lookup = {(r["split"], r["metric"]): r for r in rows}
        self.assertEqual(lookup["train", "studies"]["value"], 2)
        self.assertEqual(lookup["train", "patients"]["value"], 1)
        self.assertEqual(lookup["train", "score_0_studies"]["value"], 1)
        self.assertEqual(lookup["train", "maximum_score_5_patients"]["value"], 1)
        self.assertEqual(lookup["train", "high_burden_patients"]["denominator"], 1)
        self.assertEqual(lookup["all", "patients"]["value"], 3)

    def test_invalid_split_and_extra_samples(self):
        patients = pq.read_table(self.context["source_files"]["split"]).to_pylist()
        for records in [
            patients + [patients[0]],
            patients[:-1],
            [{**patients[0], "split": "unknown"}, *patients[1:]],
        ]:
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    step.build_samples(self.cohort, pa.Table.from_pylist(records))
        samples = step.build_samples(self.cohort, pa.Table.from_pylist(patients))
        rows = samples.to_pylist()
        rows[1]["split"] = "test"
        with self.assertRaises(ValueError):
            step.validate_samples(pa.Table.from_pylist(rows, schema=step.SAMPLE_SCHEMA))
        rows = samples.to_pylist()
        rows[1]["sample_id"] = rows[0]["sample_id"]
        with self.assertRaises(ValueError):
            step.validate_samples(pa.Table.from_pylist(rows, schema=step.SAMPLE_SCHEMA))

    def test_label_type_formula_time_and_paths(self):
        samples = step.build_samples(
            self.cohort, pq.read_table(self.context["source_files"]["split"])
        )
        for changes in [
            {"burden_score": 2.0},
            {"phys_death": 1},
            {"early_discharge": 2},
            {"observed_hours": 71.0},
            {"observation_end": datetime(2020, 1, 5, 12)},
            {"image_relpath": "../image.npy"},
            {"image_relpath": "G:/image.npy"},
        ]:
            with self.subTest(changes=changes):
                rows = samples.to_pylist()
                rows[0].update(changes)
                with self.assertRaises(ValueError):
                    step.validate_samples(
                        pa.Table.from_pylist(rows, schema=step.SAMPLE_SCHEMA)
                    )
        with self.assertRaises(ValueError):
            step.validate_samples(
                samples.set_column(
                    0, "sample_id", pa.array([None] * 4, type=pa.string())
                )
            )
        with self.assertRaises(ValueError):
            step.validate_samples(samples.append_column("index", pa.array(range(4))))

    def test_ehr_whitelist_and_identity_mismatch(self):
        table = pq.read_table(self.context["source_files"]["ehr"])
        pq.write_table(
            table.append_column("burden_score", pa.array([0.0] * 4)),
            self.context["source_files"]["ehr"],
        )
        with self.assertRaises(ValueError):
            self.run_export()
        self.assertFalse(self.package.exists())

    def test_qc_identity_or_incomplete_image(self):
        self.context["qc_records"][0]["dicom_id"] = "other"
        with self.assertRaisesRegex(ValueError, "disagrees with QC"):
            self.run_export()
        self.context["qc_records"][0]["dicom_id"] = self.cohort.to_pylist()[0][
            "dicom_id"
        ]
        self.context["qc_records"][0]["qc_status"] = "fail"
        with self.assertRaises(ValueError):
            self.run_export()

    def test_bad_pixels_missing_image_and_source_race(self):
        row = self.cohort.to_pylist()[0]
        source = self.image_dir / row["npy_relpath"]
        target = self.root / "copied.npy"
        for pixels in [
            np.zeros((224, 224), dtype=np.float64),
            np.zeros((8, 8), dtype=np.float32),
            np.full((224, 224), np.nan, dtype=np.float32),
            np.full((224, 224), 2, dtype=np.float32),
        ]:
            np.save(source, pixels)
            with self.assertRaises(ValueError):
                step.copy_image(source, target, _file_hash(source))
        np.save(source, np.zeros((224, 224), dtype=np.float32))
        digest = _file_hash(source)
        original = shutil.copyfile

        def mutate(path, destination):
            result = original(path, destination)
            source.write_bytes(b"changed")
            return result

        with patch.object(step.shutil, "copyfile", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "during export"):
                step.copy_image(source, target, digest)
        source.unlink()
        with self.assertRaises(FileNotFoundError):
            step.copy_image(source, target, digest)

    def test_zip_determinism_and_corruption(self):
        file = self.root / "member.txt"
        file.write_text("portable fixture")
        files = {"README.md": file}
        first = self.root / "first.zip"
        second = self.root / "second.zip"
        step.archive_files(first, files, "synthetic")
        step.archive_files(second, files, "synthetic")
        self.assertEqual(_file_hash(first), _file_hash(second))
        step.verify_archive(first, {"README.md": _file_hash(file)}, "synthetic")
        with zipfile.ZipFile(first, "a") as archive:
            archive.writestr("synthetic/extra.txt", "bad")
        with self.assertRaises(ValueError):
            step.verify_archive(first, {"README.md": _file_hash(file)}, "synthetic")
        with self.assertRaises(ValueError):
            step.archive_files(second, {"../escape.txt": file}, "synthetic")
        with self.assertRaises(ValueError):
            step.assert_portable({"source": "G:/private/path"})

    def test_frozen_binding_and_tamper_block_overwrite(self):
        self.run_export()
        before = _file_hash(self.archive)
        self.context["cohort_identity"] = {"changed": True}
        with self.assertRaisesRegex(ValueError, "cannot overwrite"):
            self.run_export()
        self.assertEqual(before, _file_hash(self.archive))
        self.context["cohort_identity"] = {"synthetic": True}
        (self.package / "README.md").write_text("changed")
        with self.assertRaises(ValueError):
            self.run_export()

    def test_configuration_changes_and_unregistered_content(self):
        self.package.mkdir(parents=True)
        (self.package / "existing.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            self.run_export()
        self.assertEqual((self.package / "existing.txt").read_text(), "keep")
        settings = dict(self.settings, release_id="../escape")
        self.args.export_config.write_text(json.dumps(settings))
        with self.assertRaises(ValueError):
            self.run_export()

    def test_interrupted_first_publication_retries(self):
        original = os.replace
        attempts = 0

        def fail_once(source, destination):
            nonlocal attempts
            if "stage_" in str(source):
                attempts += 1
                if attempts == 3:
                    raise OSError("synthetic publication interruption")
            return original(source, destination)

        with patch("src.cohort.common.os.replace", side_effect=fail_once):
            with self.assertRaises(OSError):
                self.run_export()
        self.assertFalse(self.archive.exists())
        self.assertFalse((self.package / "samples.parquet").exists())
        self.run_export()
        _verify_run(self.outputs, 11)

    def test_actual_process_crash_recovers_before_retry(self):
        # Only this test's self-created synthetic fixture is deserialized.
        fixture = self.root / "fixture.pkl"
        fixture.write_bytes(pickle.dumps((self.root, self.args, self.context)))
        script = """
import os,pickle,sys
from pathlib import Path
from typing import Any
from src.data import export_training_package as step
root,args,context=pickle.loads(Path(sys.argv[1]).read_bytes())
step.verify_inputs=lambda *args: context
original=os.replace
def crash(source,destination):
    if Path(destination).name=='ehr_features.parquet':
        os._exit(23)
    return original(source,destination)
os.replace=crash
step.run_export(root,args)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(fixture)],
            cwd=ROOT,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 23, result.stderr.decode(errors="replace"))
        self.assertTrue((self.outputs / "audit/.step_11_pending.json").exists())
        self.assertTrue((self.package / "samples.parquet").exists())
        self.run_export()
        _verify_run(self.outputs, 11)
        self.assertFalse((self.outputs / "audit/.step_11_pending.json").exists())

    def test_source_change_aborts_before_commit(self):
        with patch.object(
            step, "check_sources", side_effect=ValueError("synthetic source race")
        ):
            with self.assertRaisesRegex(ValueError, "source race"):
                self.run_export()
        self.assertFalse(self.archive.exists())
        self.assertFalse((self.package / "samples.parquet").exists())
        self.run_export()

    def test_new_release_preserves_previous_package(self):
        self.run_export()
        previous = self.package
        previous_zip = self.archive
        digest = _file_hash(previous_zip)
        self.settings["release_id"] = "burden_v2_h72_r02"
        self.args.export_config.write_text(json.dumps(self.settings))
        self.context["export_config_hashes"][self.args.export_config.name] = _file_hash(
            self.args.export_config
        )
        self.run_export()
        self.assertTrue(previous.exists())
        self.assertTrue(self.package.exists())
        self.assertEqual(digest, _file_hash(previous_zip))


if __name__ == "__main__":
    unittest.main()
