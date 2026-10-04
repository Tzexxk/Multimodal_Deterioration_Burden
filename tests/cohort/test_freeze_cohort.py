"""Synthetic common-cohort, provenance, release and recovery regressions."""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import test_image_qc as fixtures

from src.cohort.burden_io import COMPONENTS, open_store
from src.cohort.common import _file_hash, _json_file, _verify_run
from src.cohort.freeze_cohort import (
    freeze_common_cohort,
    validate_cohort,
    verify_inputs,
)


class CommonCohortTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.ImagePublicationTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.root = self.f.root
        self.data = self.f.f.processed / "cohort/burden_v2"
        self.audit = self.f.f.outputs / "audit/burden_v2"

    def prepare(self, upstream_excluded=False, qc_excluded=True):
        if upstream_excluded:
            self.f.f.tables["patients"][1]["anchor_age"] = "10"
        self.f.prepare()
        rows = self.f.run_images()
        decisions = (
            {r["sample_id"]: "exclude" for r in rows if r["study_id"] == 101}
            if qc_excluded
            else {}
        )
        reviews = self.f.review_file(decisions)
        self.f.run_images(reviews)
        self.reviews = reviews

    def run_freeze(self, version=None):
        with redirect_stdout(io.StringIO()):
            result = freeze_common_cohort(
                self.root, *self.f.configs, self.f.processing, version
            )
        self.assertEqual(result, 0)
        _verify_run(self.f.f.outputs, 8)
        return pq.read_table(self.data / "cohort.parquet")

    def context(self):
        value = verify_inputs(self.root, self.f.configs, self.f.processing)
        value["release_version"] = "burden_v2_h72_r01"
        return value

    def refresh07(self):
        path = self.f.f.outputs / "audit/step_07_manifest.json"
        manifest = _json_file(path)
        manifest["files"] = {p: _file_hash(Path(p)) for p in manifest["files"]}
        self.f.labels_fixture.write_json(path, manifest)

    def test_intersection_full_audit_types_counts_and_repeat(self):
        self.prepare(upstream_excluded=True)
        protected = {
            str(self.f.f.outputs / f"audit/step_{s:02d}_manifest.json"): _file_hash(
                self.f.f.outputs / f"audit/step_{s:02d}_manifest.json"
            )
            for s in range(2, 8)
        }
        labels_schema = pq.read_schema(self.data / "study_burden_labels.parquet")
        cohort = self.run_freeze()
        self.assertEqual(cohort.num_rows, 1)
        for field in labels_schema:
            self.assertEqual(cohort.schema.field(field.name), field)
        self.assertEqual(cohort["burden_score"].to_pylist(), [8.0])
        self.assertEqual(
            cohort["new_severe_hypoxemia_first_event_time"].to_pylist(),
            [fixtures.fixtures.fixtures.BASE + timedelta(hours=3)],
        )
        candidates = {
            r["study_id"]: r
            for r in pq.read_table(self.data / "cohort_audit.parquet").to_pylist()
        }
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[200]["qc_performed"], 0)
        self.assertIsNone(candidates[200]["qc_status"])
        self.assertIsNone(candidates[200]["burden_score"])
        self.assertEqual(candidates[101]["cohort_exclusion_stage"], "image_qc")
        self.assertIsNotNone(candidates[101]["burden_score"])
        frozen = _json_file(self.audit / "cohort_freeze.json")
        self.assertEqual(frozen["summary"]["qc_loss"]["wholly_removed_patients"], 0)
        self.assertEqual(frozen["summary"]["qc_loss"]["wholly_removed_admissions"], 0)
        with (self.audit / "training_feasibility.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            scores = [r for r in csv.DictReader(stream) if r["category"] == "score"]
        self.assertEqual([int(r["value"]) for r in scores], list(range(9)))
        self.assertEqual(sum(int(r["studies"]) for r in scores), 1)
        with (self.f.f.outputs / "audit/step_08_csv_tail_checks.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            self.assertEqual(next(csv.DictReader(stream))["status"], "no_csv_inputs")
        hashes = {
            p: _file_hash(Path(p))
            for p in _json_file(self.f.f.outputs / "audit/step_08_manifest.json")[
                "files"
            ]
            if "releases" in p or p.endswith("cohort.parquet")
        }
        snapshot_times = {
            p: Path(p).stat().st_mtime_ns for p in hashes if "releases" in p
        }
        self.assertTrue(cohort.equals(self.run_freeze()))
        self.assertEqual(
            snapshot_times, {p: Path(p).stat().st_mtime_ns for p in snapshot_times}
        )
        self.assertEqual(_json_file(self.audit / "cohort_freeze.json"), frozen)
        metadata = _json_file(self.audit / "step_08_run_metadata.json")
        self.assertTrue(metadata["snapshot_reused"])
        self.assertNotEqual(metadata["run_id"], metadata["original_freeze_run_id"])
        for path, digest in (protected | hashes).items():
            self.assertEqual(_file_hash(Path(path)), digest)

    def test_shared_sources_early_events_and_alive_discharge(self):
        self.prepare(qc_excluded=False)
        cohort = self.run_freeze()
        self.assertEqual(cohort.num_rows, 3)
        self.assertEqual(cohort["early_discharge"].to_pylist(), [0, 0, 1])
        summary = _json_file(self.audit / "cohort_freeze.json")["summary"]
        self.assertGreater(summary["shared_component_source_records"], 0)
        self.assertEqual(
            next(r for r in summary["feasibility"] if r["category"] == "early_3h")[
                "studies"
            ],
            1,
        )
        with (self.audit / "source_overlap.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            records = list(csv.DictReader(stream))
        shared = [r for r in records if int(r["linked_studies"]) > 1]
        self.assertTrue(shared)
        self.assertTrue(all(int(r["linked_patients"]) == 1 for r in shared))

    def test_label_and_evidence_semantics_fail(self):
        self.prepare()
        context = self.context()
        labels = pq.read_table(context["labels_path"])
        evidence = pq.read_table(context["evidence_path"])
        mutations = (
            ("labels", lambda rows: rows.append(copy.deepcopy(rows[0]))),
            ("labels", lambda rows: rows.pop()),
            ("labels", lambda rows: rows[0].update(burden_score=7)),
            ("labels", lambda rows: rows[0].update(subject_id=999)),
            ("labels", lambda rows: rows[0].update(new_invasive_ventilation=None)),
            ("evidence", lambda rows: rows[0].update(qualifies=None)),
            ("evidence", lambda rows: rows[0].update(review_status="pending")),
            ("evidence", lambda rows: rows[0].update(subject_id=999)),
            (
                "evidence",
                lambda rows: next(r for r in rows if r["qualifies"] == 1).update(
                    event_time=labels.to_pylist()[0]["t0"]
                ),
            ),
            (
                "evidence",
                lambda rows: next(r for r in rows if r["qualifies"] == 1).update(
                    first_event_time=labels.to_pylist()[0]["horizon_end"]
                    + timedelta(seconds=1)
                ),
            ),
        )
        for number, (kind, mutate) in enumerate(mutations):
            with self.subTest(number=number):
                table = labels if kind == "labels" else evidence
                rows = table.to_pylist()
                mutate(rows)
                path = self.root / f"bad{number}.parquet"
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
                changed = {**context, f"{kind}_path": path}
                with open_store(self.root, "128MB") as connection:
                    with self.assertRaises(ValueError):
                        validate_cohort(connection, changed)
                # Remove the test-only disk store before the next independent case.
                (self.root / "events.duckdb").unlink()
        missing = self.root / "missing.parquet"
        pq.write_table(labels.drop([COMPONENTS[0]]), missing)
        with open_store(self.root, "128MB") as connection:
            with self.assertRaises(ValueError):
                validate_cohort(connection, {**context, "labels_path": missing})

    def test_qc_keys_pending_identity_and_stale_reviews_fail(self):
        self.prepare()
        files = [
            self.f.image_dir / name
            for name in ("image_processing.parquet", "image_qc_status.parquet")
        ]
        original = pq.read_table(files[0])
        for change in (
            lambda rows: rows.append(copy.deepcopy(rows[0])),
            lambda rows: rows.pop(),
            lambda rows: rows[0].update(sample_id="extra"),
            lambda rows: rows[0].update(qc_status="pending_review"),
            lambda rows: rows[0].update(review_complete=False),
            lambda rows: rows[0].update(t0=rows[0]["t0"] + timedelta(seconds=1)),
            lambda rows: rows[0].update(npy_relpath="../unsafe.npy"),
        ):
            rows = original.to_pylist()
            change(rows)
            for path in files:
                pq.write_table(pa.Table.from_pylist(rows, schema=original.schema), path)
            self.refresh07()
            with self.assertRaises(ValueError):
                self.context()
        for path in files:
            pq.write_table(original, path)
        self.refresh07()
        self.reviews.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "review input changed"):
            self.context()

    def test_npy_missing_corrupt_dtype_range_and_hash_fail(self):
        self.prepare()
        row = next(
            r for r in self.f.run_images(self.reviews) if r["qc_status"] == "pass"
        )
        path = self.f.image_dir / row["npy_relpath"]
        original = path.read_bytes()
        table_files = [
            self.f.image_dir / name
            for name in ("image_processing.parquet", "image_qc_status.parquet")
        ]
        original_table = pq.read_table(table_files[0])
        for value in (
            None,
            b"broken",
            np.zeros((224, 224), np.float64),
            np.full((224, 224), 2, np.float32),
            np.full((224, 224), np.nan, np.float32),
            np.zeros((2, 2), np.float32),
        ):
            with self.subTest(value=type(value)):
                if value is None:
                    path.unlink()
                    with self.assertRaises((FileNotFoundError, ValueError)):
                        self.context()
                else:
                    if isinstance(value, bytes):
                        path.write_bytes(value)
                    else:
                        np.save(path, value, allow_pickle=False)
                    changed = original_table.to_pylist()
                    for record in changed:
                        if record["sample_id"] == row["sample_id"]:
                            record["npy_sha256"] = _file_hash(path)
                    for file in table_files:
                        pq.write_table(
                            pa.Table.from_pylist(changed, schema=original_table.schema),
                            file,
                        )
                    self.refresh07()
                    with self.assertRaises(ValueError):
                        self.context()
                path.write_bytes(original)
                for file in table_files:
                    pq.write_table(original_table, file)
                self.refresh07()

    def test_release_protection_new_version_and_rollback(self):
        self.prepare()
        self.run_freeze()
        old_snapshot = {
            p: _file_hash(Path(p))
            for p in _json_file(self.f.f.outputs / "audit/step_08_manifest.json")[
                "files"
            ]
            if "releases" in p
        }
        meta_path = self.f.qc / "run_metadata.json"
        metadata = _json_file(meta_path)
        metadata["finished_utc"] = "2026-10-05T00:00:00+00:00"
        self.f.labels_fixture.write_json(meta_path, metadata)
        self.refresh07()
        with self.assertRaisesRegex(ValueError, "new release"):
            self.run_freeze()
        self.run_freeze("burden_v2_h72_r02")
        for path, digest in old_snapshot.items():
            self.assertEqual(_file_hash(Path(path)), digest)
        old = _json_file(self.f.f.outputs / "audit/step_08_manifest.json")
        original = os.replace

        def fail_one(source, target):
            if (
                Path(target) == self.data / "cohort_audit.parquet"
                and ".step_08_stage_" in Path(source).name
            ):
                raise OSError("synthetic freeze failure")
            return original(source, target)

        with patch("src.cohort.common.os.replace", side_effect=fail_one):
            with self.assertRaisesRegex(OSError, "synthetic freeze failure"):
                self.run_freeze("burden_v2_h72_r03")
        self.assertEqual(
            old, _json_file(self.f.f.outputs / "audit/step_08_manifest.json")
        )
        _verify_run(self.f.f.outputs, 8)
        self.assertFalse(
            (self.audit / "releases/burden_v2_h72_r03/cohort_freeze.json").exists()
        )
        self.assertFalse(
            (self.data / "releases/burden_v2_h72_r03/cohort.parquet").exists()
        )
        self.run_freeze("burden_v2_h72_r03")

    def test_invalid_version_unfrozen_decision_and_changed_config_fail(self):
        self.prepare()
        for version in ("burden_v2_h48_r01", "../escape", "burden_v2_h72_r00"):
            with self.assertRaises(ValueError):
                self.run_freeze(version)
        original_processing = self.f.processing.read_bytes()
        settings = _json_file(self.f.processing)
        settings["processing_version"] = "changed"
        self.f.labels_fixture.write_json(self.f.processing, settings)
        with self.assertRaisesRegex(ValueError, "configuration, code"):
            self.run_freeze()
        self.f.processing.write_bytes(original_processing)
        decision_path = self.audit / "endpoint_decision.json"
        decision = _json_file(decision_path)
        decision["horizon_frozen"] = False
        self.f.labels_fixture.write_json(decision_path, decision)
        with self.assertRaisesRegex(ValueError, "not frozen"):
            self.run_freeze()

    def test_snapshot_corruption_and_step08_code_changes_fail(self):
        self.prepare()
        self.run_freeze()
        self.run_freeze()
        with patch("src.cohort.freeze_cohort._code_hash", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "new release"):
                self.run_freeze()
            self.run_freeze("burden_v2_h72_r02")
        path = self.data / "releases/burden_v2_h72_r01/cohort.parquet"
        path.write_bytes(b"corrupt frozen snapshot")
        with self.assertRaises(ValueError):
            self.run_freeze()

    def test_interrupted_step08_recovery(self):
        self.prepare()
        self.run_freeze()
        target = self.data / "cohort.parquet"
        original = target.read_bytes()
        backup = target.parent / ".step_08_backup_synthetic"
        staged = target.parent / ".step_08_stage_synthetic"
        backup.write_bytes(original)
        staged.write_bytes(b"unpublished")
        target.write_bytes(b"interrupted replacement")
        pending = self.f.f.outputs / "audit/.step_08_pending.json"
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
        self.run_freeze()
        self.assertEqual(target.read_bytes(), original)
        self.assertFalse(pending.exists())
        self.assertFalse(backup.exists())


if __name__ == "__main__":
    unittest.main()
