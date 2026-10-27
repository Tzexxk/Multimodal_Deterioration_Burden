"""Synthetic Step 10B time, cleaning, publication and interface checks."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.common import _file_hash, _OutputRun, _verify_run, _write_csv
from src.ehr import lab_audit as lab
from src.ehr import lab_review as review
from src.ehr import step10b as step
from src.ehr.lab_dictionary_inputs import (
    APPROVED_FEATURES,
    DUPLICATE_RULE,
    NUMERIC_RULE,
    RANGE_BASIS,
)

ROOT = Path(__file__).resolve().parents[2]


def approved_dictionary():
    rows = []
    for item, (name, canonical) in APPROVED_FEATURES.items():
        units = (
            ["%", "g/dL"]
            if item == "51249"
            else [""]
            if item == "51237"
            else [canonical]
        )
        for unit in units:
            row: dict[str, Any] = dict.fromkeys(review.STRING_FIELDS, "")
            row.update(
                itemid=int(item),
                keep=True,
                feature_name=name,
                source_unit=unit,
                review_key=lab.review_key(item, unit),
                branch_id="item_identity",
                canonical_unit=canonical,
                dictionary_version="lab_dictionary_v1",
                conversion_multiplier=1.0,
                conversion_offset=0.0,
                valid_min=None,
                valid_max=None,
                specimen_rule='{"kind":"item_identity"}',
                duplicate_rule=DUPLICATE_RULE,
                numeric_result_rule=NUMERIC_RULE,
                valid_range_basis=RANGE_BASIS,
                missing_unit_rule="explicit_item_unit" if item == "51237" else "",
            )
            rows.append(row)
    return pa.Table.from_pylist(rows, schema=review.DICTIONARY_SCHEMA)


class Step10BTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic_step10b_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output, self.processed = (
            self.root / "outputs",
            self.root / "data/processed",
        )
        self.raw = self.root / "raw"
        (self.raw / "hosp").mkdir(parents=True)
        self.cohort = pa.table(
            {
                "sample_id": ["a", "b", "c"],
                "subject_id": [1, 1, 2],
                "hadm_id": [10, 10, 20],
                "t0": pa.array(
                    [
                        datetime(2020, 1, 2, 12),
                        datetime(2020, 1, 2, 13),
                        datetime(2020, 1, 2, 12),
                    ],
                    type=pa.timestamp("us"),
                ),
            }
        )
        self.evidence = pa.table(
            {
                **self.cohort.to_pydict(),
                "admittime": [datetime(2020, 1, 1)] * 3,
                "age_at_t0": [50, 50, 60],
                "careunit_at_t0": ["Ward", "Ward", "Ward"],
            }
        )
        self.context = {
            "cohort": self.cohort,
            "cohort_audit": self.evidence,
            "outputs": self.output,
            "processed": self.processed,
            "cohort_identity": {"synthetic": True},
            "split_identity": {"synthetic": True},
        }
        self.dictionary = approved_dictionary()
        self.rules = step.validate_dictionary(self.dictionary, "lab_dictionary_v1")
        self.labs = [
            self.lab(
                "1", charttime="2020-01-01 12:00:00", storetime="2020-01-01 12:00:00"
            ),
            self.lab("2", specimen_id="21", valueuom="g/dL", value="34", valuenum="34"),
            self.lab(
                "3",
                itemid="51237",
                specimen_id="22",
                valueuom="",
                value="1.2",
                valuenum="1.2",
            ),
        ]
        self.admissions = [
            dict(
                subject_id=str(p),
                hadm_id=str(a),
                admittime="2020-01-01 00:00:00",
                dischtime="2020-01-05 00:00:00",
                admission_type="URGENT",
                admission_location="ED",
            )
            for p, a in [(1, 10), (2, 20)]
        ]
        self.patients = [
            dict(subject_id="1", gender="F", anchor_age="50", anchor_year="2020"),
            dict(subject_id="2", gender="M", anchor_age="60", anchor_year="2020"),
        ]
        self.services = [
            dict(
                subject_id="1",
                hadm_id="10",
                transfertime="2020-01-02 11:00:00",
                curr_service="MED",
            ),
            dict(
                subject_id="1",
                hadm_id="10",
                transfertime="2020-01-02 14:00:00",
                curr_service="SURG",
            ),
        ]
        self.settings = json.loads((ROOT / "configs/ehr_features_v1.json").read_text())
        self.settings["memory_limit"] = "256MB"
        self.feature_config = self.root / "features.json"
        self.feature_config.write_text(json.dumps(self.settings))
        self.lab_config = self.root / "lab.json"
        self.lab_config.write_text(
            (ROOT / "configs/lab_processing_v1.json").read_text()
        )
        self.paths = self.root / "paths.json"
        self.paths.write_text(
            json.dumps(
                {
                    "mimic_iv": str(self.raw),
                    "mimic_cxr": str(self.root / "cxr"),
                    "processed": str(self.processed),
                    "outputs": str(self.output),
                }
            )
        )
        self.args = SimpleNamespace(
            mode="audit",
            dictionary_version="lab_dictionary_v1",
            feature_set_version=None,
            paths=self.paths,
            feature_config=self.feature_config,
            lab_config=self.lab_config,
            **{
                k: self.root / f"{k}.json"
                for k in ["cohort", "ehr", "imaging", "processing", "split_config"]
            },
        )
        for name in ["cohort", "ehr", "imaging", "processing", "split_config"]:
            getattr(self.args, name).write_text("{}")
        data = self.processed / "ehr/burden_v2"
        data.mkdir(parents=True)
        pq.write_table(self.dictionary, data / "lab_dictionary.parquet")
        split = self.processed / "splits/burden_v2"
        split.mkdir(parents=True)
        pq.write_table(
            pa.table({"subject_id": [1, 2], "split": ["train", "validation"]}),
            split / "patient_split.parquet",
        )
        self.cache = self.root / "lab_cache"
        self.cache.mkdir()
        self.lab_directory = self.output / "audit/burden_v2/labs/lab_audit_v1"
        self.lab_directory.mkdir(parents=True)
        (self.lab_directory / "lab_audit_manifest.json").write_text("{}")
        frozen = {
            "dictionary_frozen": True,
            "dictionary_version": "lab_dictionary_v1",
            "dictionary_sha256": _file_hash(data / "lab_dictionary.parquet"),
            "source_binding": {
                "code_sha256": step.upstream.code_hash(),
                "audit_fingerprint": "synthetic",
                "audit_manifest_sha256": _file_hash(
                    self.lab_directory / "lab_audit_manifest.json"
                ),
            },
        }
        (self.output / "audit/burden_v2/lab_dictionary_freeze.json").write_text(
            json.dumps(frozen)
        )
        self.write_sources()

    def lab(self, identifier, **changes):
        row: dict[str, Any] = dict.fromkeys([*lab.LAB_REQUIRED, *lab.LAB_OPTIONAL], "")
        row.update(
            labevent_id=identifier,
            subject_id="1",
            hadm_id="10",
            itemid="51249",
            specimen_id="20",
            charttime="2020-01-02 11:00:00",
            storetime="2020-01-02 11:30:00",
            value="32",
            valuenum="32",
            valueuom="%",
        )
        row.update(changes)
        return row

    def write_sources(self):
        for name, rows in [
            ("patients", self.patients),
            ("admissions", self.admissions),
            ("services", self.services),
        ]:
            _write_csv(
                self.raw / f"hosp/{name}.csv",
                list(rows[0]) if rows else step.SOURCE_COLUMNS[name].split(),
                rows,
            )
        _write_csv(
            self.raw / "hosp/labevents.csv",
            [*lab.LAB_REQUIRED, *lab.LAB_OPTIONAL],
            self.labs,
        )
        _write_csv(
            self.raw / "hosp/d_labitems.csv",
            ["itemid", "label", "fluid", "category"],
            [],
        )
        rows = [
            {
                **r,
                "lab_record_key": lab.json_hash(r),
                "raw_row_count": 1,
                "comments_truncated": False,
            }
            for r in self.labs
        ]
        pq.write_table(pa.Table.from_pylist(rows), self.cache / "labs.parquet")
        pq.write_table(
            pa.Table.from_pylist(self.admissions), self.cache / "admissions.parquet"
        )
        pq.write_table(
            pa.table({"itemid": pa.array([], type=pa.string())}),
            self.cache / "dictionary.parquet",
        )

    def connection(self):
        connection = duckdb.connect(":memory:")
        self.addCleanup(connection.close)
        lab.load_cache(
            connection,
            {
                "labs": self.cache / "labs.parquet",
                "admissions": self.cache / "admissions.parquet",
                "dictionary": self.cache / "dictionary.parquet",
            },
        )
        lab.build_views(connection, self.cohort)
        return connection

    def run_mode(self, mode="audit", version=None):
        self.args.mode, self.args.feature_set_version = mode, version
        record = {
            "audit_fingerprint": "synthetic",
            "cache_paths": {
                "labs": str(self.cache / "labs.parquet"),
                "admissions": str(self.cache / "admissions.parquet"),
                "dictionary": str(self.cache / "dictionary.parquet"),
            },
        }

        def verify(namespace, number):
            if namespace != self.output:
                _verify_run(namespace, number)

        with (
            patch.object(
                step.upstream, "verify_frozen_inputs", return_value=self.context
            ),
            patch.object(step.upstream, "dictionary_releases"),
            patch.object(step.upstream, "load_audit", return_value=record),
            patch.object(step, "_verify_run", side_effect=verify),
        ):
            return step.run_step10b(self.root, self.args)

    def test_explicit_cache_root_is_not_parent_cache(self):
        paths = json.loads(self.paths.read_text())
        cache = self.processed / "supplementary_r02/cache"
        paths["cache"] = str(cache)
        self.paths.write_text(json.dumps(paths))
        self.run_mode()
        record = json.loads(
            (
                self.output / "audit/burden_v2/ehr/ehr_audit_v1/ehr_audit_manifest.json"
            ).read_text()
        )
        self.assertTrue(Path(record["cache_table"]).is_relative_to(cache))
        self.assertFalse((self.processed.parent / "cache").exists())

    def test_boundaries_units_and_overlapping_windows(self):
        self.labs.extend(
            [
                self.lab(
                    "4",
                    charttime="2020-01-02 12:00:00",
                    storetime="2020-01-02 12:00:00",
                    specimen_id="24",
                    value="36",
                    valuenum="36",
                ),
                self.lab("5", specimen_id="25", storetime="2020-01-02 14:00:00"),
                self.lab("6", specimen_id="26", storetime=""),
                self.lab("7", specimen_id="27", hadm_id="20"),
                self.lab("8", specimen_id="28", charttime="invalid"),
            ]
        )
        self.write_sources()
        connection = self.connection()
        step.clean_labs(connection, self.rules)
        rows = step.aggregate_labs(connection, self.cohort, self.rules)
        self.assertEqual(rows[0]["mchc_count_24h"], 3)
        self.assertEqual(rows[0]["mchc_last_24h"], 36)
        self.assertEqual(rows[0]["mchc_min_24h"], 32)
        self.assertEqual(rows[0]["mchc_hours_since_last"], 0)
        self.assertEqual(rows[1]["mchc_count_24h"], 2)
        self.assertEqual(rows[2]["mchc_count_24h"], 0)
        self.assertIsNone(rows[2]["mchc_last_24h"])
        self.assertEqual(
            connection.execute(
                "SELECT count(*) FROM cleaned WHERE labevent_id='2'"
            ).fetchall()[0][0],
            2,
        )

    def test_revision_precedes_numerical_filter(self):
        self.labs = [
            self.lab("1"),
            self.lab("2", storetime="2020-01-02 12:30:00", value="<32", valuenum="32"),
        ]
        self.write_sources()
        connection = self.connection()
        step.clean_labs(connection, self.rules)
        rows = step.aggregate_labs(connection, self.cohort, self.rules)
        self.assertEqual(rows[0]["mchc_count_24h"], 1)
        self.assertEqual(rows[1]["mchc_count_24h"], 0)
        self.assertEqual(
            connection.execute(
                "SELECT count(*) FROM lab_trace WHERE disposition='superseded_revision'"
            ).fetchall()[0][0],
            1,
        )

    def test_nonnumeric_latest_extremes_and_ties(self):
        self.labs = [
            self.lab("1", value="1e20", valuenum="1e20"),
            self.lab("2", value="not numeric", valuenum=""),
            self.lab("3", specimen_id="21", value="33", valuenum="33"),
            self.lab("4", specimen_id="22", value="34", valuenum="34"),
        ]
        self.write_sources()
        connection = self.connection()
        step.clean_labs(connection, self.rules)
        rows = step.aggregate_labs(connection, self.cohort, self.rules)
        self.assertEqual(rows[0]["mchc_count_24h"], 2)
        self.assertEqual(rows[0]["mchc_last_24h"], 34)
        self.assertEqual(rows[0]["mchc_max_24h"], 34)
        self.labs = [self.lab("1", value="1e20", valuenum="1e20")]
        self.write_sources()
        connection = self.connection()
        step.clean_labs(connection, self.rules)
        self.assertEqual(
            step.aggregate_labs(connection, self.cohort, self.rules)[0]["mchc_max_24h"],
            1e20,
        )

    def test_lab_conflicts_and_dictionary_validation(self):
        for changes in [
            {"specimen_id": ""},
            {"value": "31"},
            {"labevent_id": "invalid"},
        ]:
            with self.subTest(changes=changes):
                self.labs = [self.lab("1", **changes)]
                self.write_sources()
                with self.assertRaises(ValueError):
                    step.clean_labs(self.connection(), self.rules)
        rows = self.dictionary.to_pylist()
        rows[0]["source_unit"] = "unsupported"
        with self.assertRaises(ValueError):
            step.validate_dictionary(
                pa.Table.from_pylist(rows, schema=review.DICTIONARY_SCHEMA),
                "lab_dictionary_v1",
            )
        with self.assertRaises(ValueError):
            step.validate_dictionary(self.dictionary, "other_version")

    def test_static_future_missing_conflicts_and_age(self):
        connection = self.connection()
        for name, rows in [
            ("patients", self.patients),
            ("admissions", self.admissions),
            ("services", self.services),
        ]:
            connection.register(f"static_{name}", pa.Table.from_pylist(rows))
        static, issues = step.static_features(connection, self.cohort, self.evidence)
        self.assertEqual(issues, [])
        self.assertEqual(static["a"]["current_service"], "MED")
        self.assertEqual(static["a"]["hours_from_admission"], 36)
        self.assertIsNone(static["c"]["current_service"])
        self.services.append({**self.services[0], "curr_service": "SURG"})
        connection.unregister("static_services")
        connection.register("static_services", pa.Table.from_pylist(self.services))
        _, issues = step.static_features(connection, self.cohort, self.evidence)
        self.assertTrue(
            any(i["reason"] == "conflicting_latest_service" for i in issues)
        )
        self.patients[0]["anchor_age"] = "51"
        connection.unregister("static_patients")
        connection.register("static_patients", pa.Table.from_pylist(self.patients))
        _, issues = step.static_features(connection, self.cohort, self.evidence)
        self.assertTrue(any(i["field"] == "age_at_t0" for i in issues))

    def test_float32_schema_overflow_and_whitelist(self):
        schema = step.schema_for(self.rules, "ehr_features_v1")
        self.assertEqual(len(schema["numeric_features"]), 152)
        self.assertEqual(len(schema["categorical_features"]), 5)
        rows: list[dict[str, Any]] = [{"sample_id": k} for k in ["a", "b", "c"]]
        table = step.float32_table(rows, schema)
        step.validate_wide(table, self.cohort, schema)
        self.assertTrue(math.isnan(table["age_at_t0"][0].as_py()))
        self.assertEqual(table["sex"][0].as_py(), "__MISSING__")
        with self.assertRaises(ValueError):
            step.validate_wide(
                table.append_column("burden_score", pa.array([0, 0, 0])),
                self.cohort,
                schema,
            )
        rows[0]["age_at_t0"] = 1e100
        with self.assertRaises(ValueError):
            step.float32_table(rows, schema)

    def test_audit_build_reuse_and_version_protection(self):
        self.run_mode()
        candidate = (
            self.root
            / "data/cache/burden_v2/ehr/ehr_audit_v1/ehr_features_candidate.parquet"
        )
        before = _file_hash(candidate)
        self.run_mode()
        self.assertEqual(before, _file_hash(candidate))
        directory = self.output / "audit/burden_v2/ehr/ehr_audit_v1"
        self.assertTrue(
            json.loads((directory / "ehr_audit_manifest.json").read_text())[
                "audit_checks_passed"
            ]
        )
        self.run_mode("build", "ehr_features_v1")
        table_path = self.processed / "ehr/burden_v2/ehr_features_raw.parquet"
        snapshot = (
            table_path.parent / "releases/ehr_features_v1/ehr_features_raw.parquet"
        )
        self.assertEqual(_file_hash(table_path), _file_hash(snapshot))
        self.assertEqual(pq.read_table(table_path).num_rows, 3)
        freeze_path = snapshot.parent / "ehr_features_freeze.json"
        frozen_hash = _file_hash(freeze_path)
        self.run_mode("build", "ehr_features_v1")
        self.assertEqual(frozen_hash, _file_hash(freeze_path))
        record = json.loads(freeze_path.read_text())
        record["binding"] = {"changed": True}
        with _OutputRun(self.root / "separate", 10, self.processed, self.output) as run:
            with self.assertRaises(ValueError):
                step.publish_features(
                    run,
                    table_path.parent,
                    directory,
                    pq.read_table(table_path),
                    step.schema_for(self.rules, "ehr_features_v1"),
                    record["binding"],
                    "ehr_features_v1",
                )

    def test_static_conflicts_publish_audit_but_block_build(self):
        self.services.append({**self.services[0], "curr_service": "SURG"})
        self.write_sources()
        self.run_mode()
        with self.assertRaisesRegex(ValueError, "Unresolved audit conflicts"):
            self.run_mode("build", "ehr_features_v1")
        self.assertFalse(
            (self.processed / "ehr/burden_v2/ehr_features_raw.parquet").exists()
        )

    def test_laboratory_conflicts_have_bound_blocking_report(self):
        self.labs[0]["specimen_id"] = ""
        self.write_sources()
        self.run_mode()
        record = json.loads(
            (
                self.output / "audit/burden_v2/ehr/ehr_audit_v1/ehr_audit_manifest.json"
            ).read_text()
        )
        self.assertFalse(record["audit_checks_passed"])
        self.assertIsNone(record["cache_table"])
        with self.assertRaises(ValueError):
            self.run_mode("build", "ehr_features_v1")

    def test_stale_cache_source_changes_and_missing_audit(self):
        with self.assertRaisesRegex(ValueError, "audit before build"):
            self.run_mode("build", "ehr_features_v1")
        self.run_mode()
        self.patients[0]["gender"] = "M"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "Stale EHR audit"):
            self.run_mode()

    def test_csv_tail_and_middle_record_rejection(self):
        services_path = self.raw / "hosp/services.csv"
        services_path.write_text(services_path.read_text().rstrip("\n\r"))
        self.run_mode()
        self.assertTrue(
            (self.output / "step_10b/audit/audit/step_10_csv_tail_checks.csv").exists()
        )
        self.settings["audit_version"] = "bad_middle"
        self.feature_config.write_text(json.dumps(self.settings))
        services_path.write_text(
            "subject_id,hadm_id,transfertime,curr_service\n1,10,broken\n1,10,2020-01-02 11:00:00,MED\n"
        )
        with self.assertRaises(ValueError):
            self.run_mode()

    def test_publication_failure_restores_previous_files(self):
        target = self.root / "publication/value.txt"
        target.parent.mkdir()
        target.write_text("old")
        import os

        original = os.replace
        attempts = 0

        def fail_once(source, destination):
            nonlocal attempts
            if "stage_" in str(source):
                attempts += 1
                if attempts == 2:
                    raise OSError("synthetic publication interruption")
            return original(source, destination)

        with self.assertRaises(OSError):
            with _OutputRun(self.root / "publication", 10) as run:
                run.path(target).write_text("new")
                with patch("src.cohort.common.os.replace", side_effect=fail_once):
                    run._publish()
        self.assertEqual(target.read_text(), "old")

    def test_first_feature_publication_failure_can_retry(self):
        self.run_mode()
        import os

        original = os.replace
        attempts = 0

        def fail_once(source, destination):
            nonlocal attempts
            if "stage_" in str(source):
                attempts += 1
                if attempts == 2:
                    raise OSError("synthetic first freeze interruption")
            return original(source, destination)

        with patch("src.cohort.common.os.replace", side_effect=fail_once):
            with self.assertRaises(OSError):
                self.run_mode("build", "ehr_features_v1")
        self.assertFalse(
            (self.processed / "ehr/burden_v2/ehr_features_raw.parquet").exists()
        )
        self.run_mode("build", "ehr_features_v1")
        self.assertTrue(
            (self.processed / "ehr/burden_v2/ehr_features_raw.parquet").exists()
        )

    def test_checksum_source_race_and_historical_tamper(self):
        self.run_mode()
        with self.assertRaisesRegex(ValueError, "Source versions changed"):
            step.verify_sources(
                {"patients": self.raw / "hosp/patients.csv"},
                {"patients": {"exists": True}},
            )
        candidate = (
            self.root
            / "data/cache/burden_v2/ehr/ehr_audit_v1/ehr_features_candidate.parquet"
        )
        saved = candidate.read_bytes()
        candidate.write_bytes(saved + b"changed")
        with self.assertRaises(ValueError):
            self.run_mode("build", "ehr_features_v1")
        candidate.write_bytes(saved)
        self.run_mode("build", "ehr_features_v1")
        snapshot = (
            self.processed
            / "ehr/burden_v2/releases/ehr_features_v1/feature_schema.json"
        )
        snapshot.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Historical EHR snapshot"):
            self.run_mode("build", "ehr_features_v1")

    def test_ordinary_static_missing_and_split_reports(self):
        self.services = []
        self.patients[1]["gender"] = ""
        self.admissions[1]["admission_location"] = ""
        self.write_sources()
        self.run_mode()
        directory = self.output / "audit/burden_v2/ehr/ehr_audit_v1"
        self.assertTrue(
            json.loads((directory / "ehr_audit_manifest.json").read_text())[
                "audit_checks_passed"
            ]
        )
        import csv

        with (directory / "lab_missingness.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(
            {r["split"] for r in rows}, {"all", "train", "validation", "test"}
        )
        self.assertEqual(
            next(
                r for r in rows if r["split"] == "validation" and r["feature"] == "mchc"
            )["after_studies"],
            "0",
        )
        self.assertEqual(
            next(r for r in rows if r["split"] == "train" and r["feature"] == "mchc")[
                "patients"
            ],
            "1",
        )
        self.run_mode("build", "ehr_features_v1")
        table = pq.read_table(self.processed / "ehr/burden_v2/ehr_features_raw.parquet")
        self.assertEqual(table["sex"][2].as_py(), "__MISSING__")
        self.assertEqual(table["current_service"].to_pylist(), ["__MISSING__"] * 3)


if __name__ == "__main__":
    unittest.main()
