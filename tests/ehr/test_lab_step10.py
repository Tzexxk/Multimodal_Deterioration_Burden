"""Synthetic laboratory pipeline checks; never scan clinical source files."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.common import _file_hash, _OutputRun, _verify_run, _write_csv
from src.ehr import build_ehr_features as step
from src.ehr import lab_audit as audit
from src.ehr import lab_review as review

PROJECT = Path(__file__).resolve().parents[2]


class LaboratoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic_step10_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.output = self.root / "outputs"
        self.processed = self.root / "data/processed"
        self.settings = json.loads(
            (PROJECT / "configs/lab_processing_v1.json").read_text()
        )
        self.settings.update(memory_limit="256MB", comment_max_chars=32)
        self.config = self.root / "lab.json"
        self.config.write_text(json.dumps(self.settings))
        self.configs = [
            self.root / f"{k}.json" for k in ("paths", "cohort", "ehr", "imaging")
        ]
        self.configs[0].write_text(
            json.dumps(
                {
                    "mimic_iv": str(self.raw),
                    "mimic_cxr": str(self.root / "cxr"),
                    "processed": str(self.processed),
                    "outputs": str(self.output),
                }
            )
        )
        for config in self.configs[1:]:
            config.write_text("{}")
        self.context = {
            "outputs": self.output,
            "processed": self.processed,
            "cohort_identity": {
                "cohort_version": self.settings["cohort_version"],
                "synthetic": True,
            },
            "split_identity": {
                "split_version": self.settings["split_version"],
                "synthetic": True,
            },
            "cohort": pa.table(
                {
                    "sample_id": ["synthetic_a", "synthetic_b"],
                    "subject_id": pa.array([1, 1], type=pa.int64()),
                    "hadm_id": pa.array([10, 10], type=pa.int64()),
                    "t0": pa.array(
                        [datetime(2020, 1, 2, 12), datetime(2020, 1, 2, 13)],
                        type=pa.timestamp("us"),
                    ),
                }
            ),
        }
        self.dictionary = [
            dict(
                itemid="100", label="Hemoglobin", fluid="Blood", category="Hematology"
            ),
            dict(
                itemid="101", label="Specimen Type", fluid="Blood", category="Blood Gas"
            ),
            dict(itemid="102", label="pH", fluid="Blood", category="Blood Gas"),
            dict(
                itemid="900",
                label="Unrelated synthetic analyte",
                fluid="Blood",
                category="Other",
            ),
        ]
        self.admissions = [
            dict(
                subject_id="1",
                hadm_id="10",
                admittime="2020-01-01 00:00:00",
                dischtime="2020-01-05 00:00:00",
            )
        ]
        self.labs = [
            self.lab("1"),
            self.lab("2", itemid="900", valueuom="other"),
            self.lab("3", itemid="102", valueuom="", value="7.2", valuenum="7.2"),
        ]
        self.write_sources()
        silence = redirect_stdout(io.StringIO())
        silence.__enter__()
        self.addCleanup(silence.__exit__, None, None, None)

    def lab(self, identifier: str, **changes) -> dict:
        row = dict.fromkeys([*audit.LAB_REQUIRED, *audit.LAB_OPTIONAL], "")
        row.update(
            labevent_id=identifier,
            subject_id="1",
            hadm_id="10",
            specimen_id="20",
            itemid="100",
            charttime="2020-01-02 11:00:00",
            storetime="2020-01-02 11:30:00",
            value="12",
            valuenum="12",
            valueuom="g/dL",
            comments="Synthetic evidence only",
        )
        row.update(changes)
        return row

    def write_sources(self):
        hosp = self.raw / "hosp"
        hosp.mkdir(exist_ok=True)
        _write_csv(
            hosp / "d_labitems.csv",
            ["itemid", "label", "fluid", "category"],
            self.dictionary,
        )
        _write_csv(
            hosp / "admissions.csv",
            ["subject_id", "hadm_id", "admittime", "dischtime"],
            self.admissions,
        )
        _write_csv(
            hosp / "labevents.csv",
            [*audit.LAB_REQUIRED, *audit.LAB_OPTIONAL],
            self.labs,
        )

    @property
    def directory(self):
        return self.output / "audit/burden_v2/labs" / self.settings["audit_version"]

    def run_mode(self, mode="audit", paths=None, version=None):
        with patch.object(step, "verify_frozen_inputs", return_value=self.context):
            return step.run_step10(
                self.root,
                mode,
                self.configs,
                self.root / "processing.json",
                self.root / "split.json",
                self.config,
                paths or [],
                version,
            )

    def record(self):
        return json.loads((self.directory / "lab_audit_manifest.json").read_text())

    def decisions(self):
        record = self.record()
        rows = []
        for item in record["inventory"]:
            if item["suggested_group"] == "out_of_scope":
                continue
            row = dict.fromkeys(review.REVIEW_COLUMNS, "")
            row.update(
                {
                    k: item[k]
                    for k in (
                        "review_key",
                        "itemid",
                        "label",
                        "fluid",
                        "category",
                        "source_unit",
                    )
                }
            )
            row.update(
                record_type="decision",
                branch_id="default",
                audit_fingerprint=record["audit_fingerprint"],
                reviewer="ZE Xiaokang",
                reviewed_on="2026-10-05",
                review_version="synthetic_review_v1",
                rationale="Synthetic test decision",
                specimen_rule='{"kind":"item_identity"}',
            )
            if row["source_unit"]:
                row.update(
                    keep="1",
                    feature_name="synthetic_hemoglobin",
                    canonical_unit="g/dL",
                    conversion_multiplier="1",
                    conversion_offset="0",
                    valid_range_basis="No reliable range set in this synthetic test",
                    duplicate_rule='{"kind":"latest_available","conflicting_id":"exclude","missing_specimen":"record_id"}',
                    numeric_result_rule='{"kind":"valuenum_only","comparator":"exclude","text_disagreement":"exclude"}',
                )
            else:
                row.update(
                    keep="0",
                    missing_unit_rule="exclude",
                    exclusion_reason="Synthetic missing-unit exclusion",
                )
            rows.append(row)
        ack = dict.fromkeys(review.REVIEW_COLUMNS, "")
        ack.update(
            record_type="scope_ack",
            review_key="scope_" + record["scope_inventory_sha256"],
            scope_inventory_sha256=record["scope_inventory_sha256"],
            audit_fingerprint=record["audit_fingerprint"],
            reviewer="ZE Xiaokang",
            reviewed_on="2026-10-05",
            review_version="synthetic_review_v1",
            rationale="Signed synthetic scope inventory",
        )
        return rows, ack

    def write_reviews(self, rows, name="decisions.csv"):
        path = self.root / name
        _write_csv(path, review.REVIEW_COLUMNS, rows)
        return path

    def cache_connection(self):
        connection = duckdb.connect(":memory:")
        audit.load_cache(
            connection, {k: Path(v) for k, v in self.record()["cache_paths"].items()}
        )
        audit.build_views(connection, self.context["cohort"])
        self.addCleanup(connection.close)
        return connection

    def test_inventory_cache_and_full_freeze_reuse(self):
        original = {p: p.read_bytes() for p in (self.raw / "hosp").iterdir()}
        self.run_mode()
        counts = self.record()["counts"]
        self.assertEqual(counts["patient_scan_distinct_records"], 3)
        self.assertEqual(counts["study_window_associations"], 6)
        self.assertEqual(len(self.record()["inventory"]), 3)
        rows, ack = self.decisions()
        path = self.write_reviews(rows + [ack])
        with patch.object(
            audit, "scan_csv", side_effect=AssertionError("Clinical CSV rescan")
        ):
            self.run_mode("review", [path])
            self.run_mode("freeze", [path], "synthetic_dictionary_v1")
        table = self.processed / "ehr/burden_v2/lab_dictionary.parquet"
        snapshot = table.parent / "releases/synthetic_dictionary_v1" / table.name
        digest = _file_hash(table)
        self.assertEqual(pq.read_table(table).num_rows, 3)
        self.assertEqual(_file_hash(snapshot), digest)
        frozen = json.loads(
            (self.output / "audit/burden_v2/lab_dictionary_freeze.json").read_text()
        )
        self.run_mode("freeze", [path], "synthetic_dictionary_v1")
        self.assertEqual(_file_hash(snapshot), digest)
        self.assertEqual(
            json.loads(
                (self.output / "audit/burden_v2/lab_dictionary_freeze.json").read_text()
            ),
            frozen,
        )
        self.assertFalse(
            json.loads(
                (self.output / "audit/burden_v2/step_10_run_metadata.json").read_text()
            )["ehr_features_ready"]
        )
        _verify_run(self.output, 10)
        for source, content in original.items():
            self.assertEqual(source.read_bytes(), content)

    def test_time_boundaries_invalid_times_and_availability(self):
        self.labs = [
            self.lab(
                "1", charttime="2020-01-01 12:00:00", storetime="2020-01-01 12:00:00"
            ),
            self.lab(
                "2", charttime="2020-01-02 12:00:00", storetime="2020-01-02 12:00:00"
            ),
            self.lab("3", storetime=""),
            self.lab("4", storetime="2020-01-02 10:59:59"),
            self.lab("5", storetime="2020-01-02 12:30:00"),
            self.lab("6", charttime="2020-01-02"),
            self.lab("7", charttime="2020-01-02 11:00:00+08:00"),
            self.lab("8", charttime=""),
            self.lab("9", storetime="bad"),
            self.lab("10", charttime="2020-01-01 11:59:59"),
        ]
        self.write_sources()
        self.run_mode()
        con = self.cache_connection()
        first = {
            r["labevent_id"]: r
            for r in audit.query_rows(
                con,
                "SELECT labevent_id,availability_reason FROM associations WHERE sample_id='synthetic_a'",
            )
        }
        self.assertEqual(first["1"]["availability_reason"], "available")
        self.assertEqual(first["2"]["availability_reason"], "available")
        self.assertEqual(first["3"]["availability_reason"], "missing_storetime")
        self.assertEqual(
            first["4"]["availability_reason"], "storetime_before_charttime"
        )
        self.assertEqual(first["5"]["availability_reason"], "future_storetime")
        self.assertEqual(first["9"]["availability_reason"], "invalid_storetime")
        self.assertNotIn("10", first)
        self.assertEqual(
            self.record()["counts"]["patient_scan_invalid_charttime_records"], 3
        )

    def test_missing_admission_infers_all_patient_admissions_and_conflicts(self):
        self.admissions += [
            dict(
                subject_id="1",
                hadm_id="11",
                admittime="2020-01-02 10:00:00",
                dischtime="2020-01-02 11:10:00",
            )
        ]
        self.labs = [
            self.lab("1", hadm_id=""),
            self.lab("2", hadm_id="", charttime="2020-01-02 11:20:00"),
            self.lab("3", hadm_id="11"),
            self.lab("4", hadm_id="999"),
            self.lab("5", hadm_id="bad"),
        ]
        self.write_sources()
        self.run_mode()
        records = {
            r["labevent_id"]: r
            for r in audit.query_rows(
                self.cache_connection(),
                "SELECT labevent_id,ownership_reason,hadm_assignment_source FROM associations WHERE sample_id='synthetic_a'",
            )
        }
        self.assertEqual(records["1"]["ownership_reason"], "ambiguous_admission")
        self.assertEqual(
            records["2"]["hadm_assignment_source"], "unique_patient_charttime"
        )
        self.assertEqual(records["3"]["ownership_reason"], "different_admission")
        self.assertEqual(records["4"]["ownership_reason"], "provided_owner_conflict")
        self.assertEqual(records["5"]["ownership_reason"], "invalid_hadm_id")

    def test_duplicates_extremes_and_html_escaping(self):
        self.labs = [
            self.lab("1", valuenum="1000000", value="1000000"),
            self.lab("1", valuenum="1000000", value="1000000"),
            self.lab("1", valuenum="9", value="8"),
            self.lab(
                "2",
                valuenum="5",
                value="<5",
                comments="<script>alert('synthetic')</script>" + "x" * 100,
            ),
        ]
        self.write_sources()
        self.run_mode()
        counts = self.record()["counts"]
        self.assertEqual(counts["patient_scan_raw_rows"], 4)
        self.assertEqual(counts["patient_scan_distinct_records"], 3)
        entry = self.record()["inventory"][0]
        self.assertEqual(entry["raw_max"], 1000000)
        self.assertEqual(entry["numeric_text_disagreements"], 1)
        self.assertEqual(entry["comparator_records"], 1)
        page = (self.directory / "lab_review.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>alert", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn("burden_score", page)
        self.assertNotIn("synthetic_a", page)

    def test_batches_all_units_together_and_stable(self):
        self.dictionary = [
            dict(
                itemid=str(100 + i),
                label="Hemoglobin",
                fluid="Blood",
                category="Hematology",
            )
            for i in range(21)
        ]
        self.labs = [
            self.lab(str(i), itemid=str(100 + i), valueuom=u)
            for i in range(21)
            for u in ("unit_a", "unit_b")
        ]
        self.write_sources()
        self.run_mode()
        batches = self.record()["batches"]
        self.assertEqual(len(set(b["batch_id"] for b in batches)), 3)
        self.assertEqual(len(self.record()["inventory"]), 42)
        self.run_mode()
        self.assertEqual(self.record()["batches"], batches)

    def test_partial_review_and_freeze_gate(self):
        self.run_mode()
        rows, ack = self.decisions()
        partial = self.write_reviews(rows[:1])
        self.run_mode("review", [partial])
        state = json.loads((self.directory / "review_state.json").read_text())
        self.assertFalse(state["review_complete"])
        manifest_hash = _file_hash(self.output / "audit/step_10_manifest.json")
        with self.assertRaisesRegex(ValueError, "incomplete clinical"):
            self.run_mode("freeze", [partial], "synthetic_dictionary_v1")
        self.assertEqual(
            _file_hash(self.output / "audit/step_10_manifest.json"), manifest_hash
        )
        self.assertFalse(
            (self.processed / "ehr/burden_v2/lab_dictionary.parquet").exists()
        )
        second = self.write_reviews(rows[1:] + [ack], "second.csv")
        self.run_mode("review", [partial, second])
        self.assertTrue(
            json.loads((self.directory / "review_state.json").read_text())[
                "review_complete"
            ]
        )

    def test_review_identity_scope_and_coefficients_are_validated(self):
        self.run_mode()
        rows, ack = self.decisions()
        for changes in (
            {"audit_fingerprint": "stale"},
            {"reviewer": "Other"},
            {"source_unit": "changed"},
            {"conversion_multiplier": "inf"},
            {"valid_min": "5", "valid_max": "4"},
            {"specimen_rule": "__import__('os')"},
            {"duplicate_rule": "{}"},
            {"valid_range_basis": ""},
        ):
            with self.subTest(changes=changes):
                path = self.write_reviews([dict(rows[0], **changes), *rows[1:], ack])
                with self.assertRaises(ValueError):
                    self.run_mode("review", [path])
        path = self.write_reviews(rows + [dict(ack, scope_inventory_sha256="stale")])
        with self.assertRaisesRegex(ValueError, "Scope inventory"):
            self.run_mode("review", [path])
        missing_unit = dict(
            rows[1],
            keep="1",
            feature_name="ph",
            canonical_unit="unitless",
            conversion_multiplier="1",
            conversion_offset="0",
            valid_range_basis="No range",
            exclusion_reason="",
            missing_unit_rule="",
            duplicate_rule=rows[0]["duplicate_rule"],
            numeric_result_rule=rows[0]["numeric_result_rule"],
        )
        path = self.write_reviews([rows[0], missing_unit, ack])
        with self.assertRaisesRegex(ValueError, "Missing units"):
            self.run_mode("review", [path])

    def test_specimen_evidence_available_at_each_t0_and_unmatched_policy(self):
        self.labs = [
            self.lab("1"),
            self.lab(
                "2",
                itemid="101",
                value="ART",
                valuenum="",
                valueuom="",
                storetime="2020-01-02 12:30:00",
            ),
        ]
        self.write_sources()
        self.run_mode()
        con = self.cache_connection()
        sql, args = review.selector_sql(
            {"kind": "same_specimen_value", "itemid": 101, "values": ["ART"]}
        )
        selected = audit.query_rows(
            con,
            "SELECT sample_id, "
            + sql
            + " AS matched FROM associations a WHERE a.itemid='100' ORDER BY sample_id",
            args,
        )
        self.assertEqual([r["matched"] for r in selected], [False, True])
        rows, ack = self.decisions()
        target = rows[0]
        branch = dict(
            target,
            branch_id="arterial",
            specimen_rule='{"kind":"same_specimen_value","itemid":101,"values":["ART"]}',
        )
        self.assertFalse(review.check_branches(con, [branch])[0])
        remaining = dict(
            target, branch_id="remaining", specimen_rule='{"kind":"unmatched"}'
        )
        self.assertTrue(review.check_branches(con, [branch, remaining])[0])
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            review.check_branches(
                con, [branch, dict(branch, branch_id="overlap"), remaining]
            )

    def test_conflicting_specimen_evidence_blocks_review(self):
        self.labs = [
            self.lab("1"),
            self.lab("2", itemid="101", value="ART", valuenum=""),
            self.lab("3", itemid="101", value="VEN", valuenum=""),
        ]
        self.write_sources()
        self.run_mode()
        rows, _ = self.decisions()
        base = rows[0]
        branches = [
            dict(
                base,
                branch_id="a",
                specimen_rule='{"kind":"same_specimen_value","itemid":101,"values":["ART"]}',
            ),
            dict(
                base,
                branch_id="v",
                specimen_rule='{"kind":"same_specimen_value","itemid":101,"values":["VEN"]}',
            ),
            dict(base, branch_id="other", specimen_rule='{"kind":"unmatched"}'),
        ]
        with self.assertRaisesRegex(ValueError, "Conflicting same-specimen"):
            review.check_branches(self.cache_connection(), branches)

    def test_cache_source_changes_and_corruption_block_reuse(self):
        self.run_mode()
        cache = Path(self.record()["cache_paths"]["labs"])
        original = cache.read_bytes()
        cache.write_bytes(original + b"corrupt")
        with self.assertRaisesRegex(
            ValueError, "artifact hash changed|checksum mismatch"
        ):
            self.run_mode()
        cache.write_bytes(original)
        with (self.raw / "hosp/labevents.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Stale laboratory cache"):
            self.run_mode()

    def test_frozen_version_changed_reviews_block_overwrite(self):
        self.run_mode()
        rows, ack = self.decisions()
        path = self.write_reviews(rows + [ack])
        self.run_mode("freeze", [path], "synthetic_dictionary_v1")
        table = self.processed / "ehr/burden_v2/lab_dictionary.parquet"
        digest = _file_hash(table)
        rows[0]["conversion_offset"] = "1"
        self.write_reviews(rows + [ack])
        with self.assertRaisesRegex(ValueError, "cannot accept changed content"):
            self.run_mode("freeze", [path], "synthetic_dictionary_v1")
        self.assertEqual(_file_hash(table), digest)

    def test_failed_publication_restores_dictionary_and_manifest(self):
        self.run_mode()
        rows, ack = self.decisions()
        path = self.write_reviews(rows + [ack])
        self.run_mode("freeze", [path], "synthetic_dictionary_v1")
        table = self.processed / "ehr/burden_v2/lab_dictionary.parquet"
        manifest = self.output / "audit/step_10_manifest.json"
        originals = {_file_hash(table), _file_hash(manifest)}
        replace = os.replace

        def fail_current(source, target):
            if Path(target) == table:
                raise OSError("Synthetic publication failure")
            return replace(source, target)

        # Recovery also replaces the current table; fail only the first attempted publication.
        failed = False

        def fail_once(source, target):
            nonlocal failed
            if Path(target) == table and not failed:
                failed = True
                return fail_current(source, target)
            return replace(source, target)

        with (
            patch("src.cohort.common.os.replace", side_effect=fail_once),
            self.assertRaises(OSError),
        ):
            self.run_mode("freeze", [path], "synthetic_dictionary_v2")
        self.assertEqual({_file_hash(table), _file_hash(manifest)}, originals)
        self.assertFalse(
            (
                table.parent / "releases/synthetic_dictionary_v2/lab_dictionary.parquet"
            ).exists()
        )
        _verify_run(self.output, 10)

    def test_csv_incomplete_eof_only_and_required_headers(self):
        path = self.raw / "hosp/labevents.csv"
        with path.open("a") as stream:
            stream.write("99,1")
        self.run_mode()
        self.assertEqual(
            self.record()["source_csv_tail_checks"]["skipped_record_count"], 1
        )
        self.settings["audit_version"] = "lab_audit_v2"
        self.config.write_text(json.dumps(self.settings))
        content = path.read_text()
        first_end = content.index("\n")
        path.write_text(content[: first_end + 1] + "99,1\n" + content[first_end + 1 :])
        with self.assertRaisesRegex(ValueError, "Multiple malformed|not at EOF"):
            self.run_mode()

    def test_interrupted_journal_recovery_and_affine_overflow(self):
        output = self.output / "audit/test.txt"
        with _OutputRun(self.output, 10, self.processed) as run:
            run.path(output).write_text("original")
        journal = self.output / "audit/.step_10_pending.json"
        backup = output.parent / ".step_10_backup_synthetic"
        staged = output.parent / ".step_10_stage_synthetic"
        backup.write_text("original")
        staged.write_text("new")
        output.write_text("partial")
        journal.write_text(
            json.dumps(
                {
                    "files": [
                        {
                            "target": str(output),
                            "backup": str(backup),
                            "staged": str(staged),
                        }
                    ]
                }
            )
        )
        with _OutputRun(self.output, 10, self.processed):
            pass
        self.assertEqual(output.read_text(), "original")
        self.run_mode()
        rows, ack = self.decisions()
        rows[0]["conversion_multiplier"] = "1e308"
        path = self.write_reviews(rows + [ack])
        with self.assertRaisesRegex(ValueError, "non-finite values"):
            self.run_mode("freeze", [path], "synthetic_dictionary_v1")

    def test_scope_promotion_requires_all_units_and_header_only_reports(self):
        self.labs.append(self.lab("4", itemid="900", valueuom="second_unit"))
        self.write_sources()
        self.run_mode()
        rows, ack = self.decisions()
        entry = next(r for r in self.record()["inventory"] if r["itemid"] == "900")
        promoted = dict(
            rows[0],
            **{
                k: entry[k]
                for k in (
                    "review_key",
                    "itemid",
                    "label",
                    "fluid",
                    "category",
                    "source_unit",
                )
            },
        )
        path = self.write_reviews(rows + [promoted, ack])
        self.run_mode("review", [path])
        self.assertFalse(
            json.loads((self.directory / "review_state.json").read_text())[
                "review_complete"
            ]
        )
        with self.assertRaisesRegex(ValueError, "incomplete clinical"):
            self.run_mode("freeze", [path], "synthetic_dictionary_v1")
        for name in (
            "lab_specimen_failures.csv",
            "lab_unit_conflicts.csv",
            "lab_duplicate_reports.csv",
        ):
            with (self.directory / name).open(encoding="utf-8", newline="") as stream:
                self.assertTrue(next(csv.reader(stream)))

    def test_future_revisions_do_not_create_retained_conflicts(self):
        self.context["cohort"] = self.context["cohort"].slice(0, 1)
        self.labs = [
            self.lab("1"),
            self.lab("1", value="99", valuenum="99", storetime="2020-01-02 12:30:00"),
        ]
        self.write_sources()
        self.run_mode()
        rows, ack = self.decisions()
        rows[0]["duplicate_rule"] = (
            '{"kind":"latest_available","conflicting_id":"error","missing_specimen":"record_id"}'
        )
        path = self.write_reviews(rows + [ack])
        self.run_mode("freeze", [path], "synthetic_dictionary_v1")

    def test_review_eof_skip_stays_pending_and_headers_fail(self):
        self.run_mode()
        rows, _ = self.decisions()
        path = self.write_reviews(rows[:1])
        with path.open("a", encoding="utf-8") as stream:
            stream.write("decision,missing_final_fields")
        self.run_mode("review", [path])
        state = json.loads((self.directory / "review_state.json").read_text())
        metadata = json.loads(
            (self.output / "audit/burden_v2/step_10_run_metadata.json").read_text()
        )
        self.assertFalse(state["review_complete"])
        self.assertEqual(metadata["csv_tail_checks"]["skipped_record_count"], 1)
        path = self.write_reviews(rows[:1])
        path.write_text(
            path.read_text().replace("branch_id", "unexpected_column"), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "required columns"):
            self.run_mode("review", [path])

    def test_snapshot_index_survives_modes_and_detects_historical_corruption(self):
        self.run_mode()
        rows, ack = self.decisions()
        path = self.write_reviews(rows + [ack])
        self.run_mode("freeze", [path], "synthetic_dictionary_v1")
        self.run_mode()
        metadata = json.loads(
            (self.output / "audit/burden_v2/step_10_run_metadata.json").read_text()
        )
        self.assertTrue(metadata["dictionary_frozen"])
        self.run_mode("freeze", [path], "synthetic_dictionary_v2")
        self.run_mode("review", [path])
        snapshot = (
            self.directory
            / "dictionary_releases/synthetic_dictionary_v1/lab_dictionary_freeze.json"
        )
        snapshot.write_text(snapshot.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "Historical dictionary snapshot"):
            self.run_mode()

    def test_full_source_hash_includes_unprojected_columns(self):
        path = self.raw / "hosp/labevents.csv"
        first = dict(self.lab("1"), order_provider_id="synthetic_provider_a")
        second = dict(self.lab("1"), order_provider_id="synthetic_provider_b")
        _write_csv(
            path,
            [*audit.LAB_REQUIRED, *audit.LAB_OPTIONAL, "order_provider_id"],
            [first, second],
        )
        self.run_mode()
        self.assertEqual(self.record()["counts"]["patient_scan_distinct_records"], 2)
        self.assertIn(
            "conflicting_labevent_id",
            (self.directory / "lab_duplicate_reports.csv").read_text(),
        )

    def test_partial_unmatched_specimen_branch_remains_pending(self):
        self.run_mode()
        rows, _ = self.decisions()
        rows[0]["specimen_rule"] = '{"kind":"unmatched"}'
        path = self.write_reviews(rows[:1])
        self.run_mode("review", [path])
        self.assertFalse(
            json.loads((self.directory / "review_state.json").read_text())[
                "review_complete"
            ]
        )


if __name__ == "__main__":
    unittest.main()
