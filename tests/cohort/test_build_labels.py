"""Synthetic frozen-input, formal-label and transaction regressions."""

from __future__ import annotations

import csv
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import test_burden_audit as fixtures

from src.cohort.audit_burden_labels import freeze_endpoint
from src.cohort.build_labels import build_burden_labels, verify_frozen_inputs
from src.cohort.burden_io import COMPONENTS
from src.cohort.common import _file_hash, _json_file, _verify_run


class FormalLabelTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BurdenAuditTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.data = self.fixture.processed / "cohort/burden_v2"
        self.audit = self.fixture.outputs / "audit/burden_v2"
        self.configs = [
            self.fixture.config / f"{name}.json"
            for name in ("paths.local", "cohort", "ehr", "imaging")
        ]

    def write_json(self, path, value):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def freeze(self):
        manifest_path = self.fixture.outputs / "audit/step_05_manifest.json"
        manifest = _json_file(manifest_path)
        manifest["files"] = {p: _file_hash(Path(p)) for p in manifest["files"]}
        self.write_json(manifest_path, manifest)
        metadata = _json_file(self.audit / "run_metadata.json")
        self.assertTrue(metadata["audit_ready_for_window_decision"])
        wide = pq.read_table(
            self.data / "window_72h/study_event_audit.parquet"
        ).to_pylist()
        eligible = [r for r in wide if r["candidate_status"] == "eligible"]
        counts = dict(
            eligible_studies=len(eligible),
            eligible_admissions=len({r["hadm_id"] for r in eligible}),
            eligible_patients=len({r["subject_id"] for r in eligible}),
            nonzero_studies=sum(
                r["burden_score"] is not None and r["burden_score"] > 0
                for r in eligible
            ),
            phys_death_studies=sum(r["phys_death"] == 1 for r in eligible),
            high_burden_studies=sum(r["high_burden"] == 1 for r in eligible),
            unresolved_studies=0,
        )
        selection = self.root / "configs/selection.md"
        selection.write_text("Synthetic reviewed selection.\n", encoding="utf-8")
        window = self.audit / "window_decision.md"
        window.write_text("Synthetic reviewed 72-hour decision.\n", encoding="utf-8")
        source = dict(
            run_id=metadata["run_id"],
            manifest_path=manifest_path.relative_to(self.root).as_posix(),
            manifest_sha256=_file_hash(manifest_path),
            metadata_path=(self.audit / "run_metadata.json")
            .relative_to(self.root)
            .as_posix(),
            metadata_sha256=_file_hash(self.audit / "run_metadata.json"),
            **{
                k: metadata[k]
                for k in (
                    "config_sha256",
                    "code_sha256",
                    "input_versions",
                    "upstream",
                    "audit_checks_passed",
                    "dictionary_handling_review_complete",
                    "audit_ready_for_window_decision",
                    "csv_tail_checks",
                )
            },
        )
        artifact_paths = [
            self.data / "candidate_studies.parquet",
            self.data / "candidate_evidence.parquet",
            *(
                self.data / f"window_72h/{name}.parquet"
                for name in (
                    "candidate_observation",
                    "study_event_audit",
                    "event_evidence",
                )
            ),
        ]
        decision = dict(
            decision_version="synthetic_endpoint_v1",
            decision_status="frozen",
            reviewer="synthetic_reviewer",
            reviewed_on="2026-10-04",
            rationale="Synthetic three-day question",
            horizon_frozen=True,
            label_rules_frozen=True,
            outcome_version="burden_v2",
            sample_unit="study",
            horizon_hours=72,
            ehr_lookback_hours=24,
            event_window="(t0, min(t0+H, dischtime)]",
            transfer_weight=1,
            respiratory_weight=2,
            circulatory_weight=2,
            death_weight=3,
            domain_capping=True,
            target_column="burden_score",
            target_range=[0, 8],
            high_burden_threshold=3,
            early_3h_events_included=True,
            hypoxemia_source="MIMIC-IV ICU chartevents only",
            hypoxemia_itemid=220277,
            hypoxemia_threshold_percent=85,
            target_formula="T + 2*R + 2*C + 3*D; T=max(icu,stepdown), R=max(imv,hypoxemia), C=infusion, D=death",
            phys_death_rule="int(R == 1 or C == 1 or D == 1)",
            live_discharge_rule="Retain events in (t0, dischtime] when alive discharge precedes t0+H; absent recorded events are operational zeros.",
            source_audit=source,
            **{
                k: metadata[k]
                for k in (
                    "rules_version",
                    "dictionary_version",
                    "rules_config_sha256",
                    "dictionary_sha256",
                    "dictionary_handling_reviews_sha256",
                    "override_sha256",
                    "candidate_exclusions_sha256",
                )
            },
            selection_record=dict(
                path=selection.relative_to(self.root).as_posix(),
                sha256=_file_hash(selection),
            ),
            window_decision_path=window.relative_to(self.root).as_posix(),
            window_decision_sha256=_file_hash(window),
            selected_window_artifact_sha256={
                p.relative_to(self.root).as_posix(): _file_hash(p)
                for p in artifact_paths
            },
            window_comparison={"72": counts},
        )
        self.write_json(self.audit / "endpoint_decision.json", decision)
        return decision

    def prepare(self):
        self.fixture.prepare()
        self.fixture.run_audit()
        return self.freeze()

    def run_labels(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(build_burden_labels(self.root, *self.configs), 0)
        _verify_run(self.fixture.outputs, 6)
        return pq.read_table(self.data / "study_burden_labels.parquet")

    def endpoint_review(self):
        metadata = _json_file(self.audit / "run_metadata.json")
        path = self.root / "configs/endpoint_review.json"
        self.write_json(
            path,
            dict(
                confirmed=True,
                decision_version="synthetic_endpoint_r02",
                reviewer="synthetic_reviewer",
                reviewed_on="2026-10-08",
                rationale="Synthetic explicit 72-hour review",
                rules_revision="Synthetic supplementary revision",
                horizon_hours=72,
                source_run_id=metadata["run_id"],
                source_manifest_sha256=_file_hash(
                    self.fixture.outputs / "audit/step_05_manifest.json"
                ),
                review_records=[self.configs[1].relative_to(self.root).as_posix()],
            ),
        )
        return path

    def test_step05_freeze_transaction_and_immutable_binding(self):
        self.fixture.prepare()
        self.fixture.run_audit()
        review = self.endpoint_review()
        marker = self.fixture.outputs / "audit/step_05_manifest.json"
        before = marker.read_bytes()
        original = os.replace

        def fail_window(source, target):
            if (
                Path(target).name == "window_decision.md"
                and ".step_05_stage_" in Path(source).name
            ):
                raise OSError("synthetic freeze interruption")
            return original(source, target)

        with patch("src.cohort.common.os.replace", side_effect=fail_window):
            with self.assertRaisesRegex(OSError, "freeze interruption"):
                freeze_endpoint(self.root, *self.configs, review)
        self.assertFalse((self.audit / "endpoint_decision.json").exists())
        self.assertFalse((self.audit / "window_decision.md").exists())
        self.assertEqual(marker.read_bytes(), before)
        self.assertEqual(freeze_endpoint(self.root, *self.configs, review), 0)
        frozen = (self.audit / "endpoint_decision.json").read_bytes()
        self.assertEqual(freeze_endpoint(self.root, *self.configs, review), 0)
        self.assertEqual(marker.read_bytes(), before)
        self.run_labels()
        value = _json_file(review)
        value["rationale"] = "Changed review requires a new isolated revision"
        self.write_json(review, value)
        with self.assertRaisesRegex(ValueError, "already frozen"):
            freeze_endpoint(self.root, *self.configs, review)
        self.assertEqual((self.audit / "endpoint_decision.json").read_bytes(), frozen)

    def test_supplementary_nonblocking_evidence_survives_formal_labels(self):
        self.fixture.prepare()
        self.fixture.run_audit()
        old_outputs = self.fixture.outputs
        self.fixture.cohort["burden_v2"]["supplementary_audit"] = dict(
            baseline_outputs=str(old_outputs),
            baseline_exclusions_path=str(
                old_outputs / "audit/burden_v2/excluded_studies.csv"
            ),
            expected_scope_counts=[0, 0],
        )
        self.fixture.tables["chartevents"].append(
            dict(
                self.fixture.tables["chartevents"][0],
                charttime=fixtures.dt(5),
                warning="",
            )
        )
        self.fixture.processed = self.root / "r02/processed"
        self.fixture.outputs = self.root / "r02/outputs"
        self.fixture.paths.update(
            processed=str(self.fixture.processed), outputs=str(self.fixture.outputs)
        )
        self.data = self.fixture.processed / "cohort/burden_v2"
        self.audit = self.fixture.outputs / "audit/burden_v2"
        self.fixture.prepare()
        self.fixture.run_audit()
        review = self.endpoint_review()
        freeze_endpoint(self.root, *self.configs, review)
        self.run_labels()
        records = pq.read_table(self.data / "event_evidence.parquet").to_pylist()
        unknown = [r for r in records if r["qualifies"] is None]
        self.assertTrue(unknown)
        self.assertTrue(all(r["blocks_event"] == 0 for r in unknown))
        value = _json_file(review)
        value["source_manifest_sha256"] = "0" * 64
        self.write_json(review, value)
        with self.assertRaisesRegex(ValueError, "latest Step 05"):
            freeze_endpoint(self.root, *self.configs, review)

    def rewrite(self, path, transform):
        table = pq.read_table(path)
        rows = table.to_pylist()
        transform(rows)
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)

    def test_types_evidence_reports_repeat_and_rollback(self):
        self.prepare()
        protected = _json_file(self.fixture.outputs / "audit/step_05_manifest.json")[
            "files"
        ]
        protected[str(self.fixture.outputs / "audit/step_05_manifest.json")] = (
            _file_hash(self.fixture.outputs / "audit/step_05_manifest.json")
        )
        protected[str(self.audit / "endpoint_decision.json")] = _file_hash(
            self.audit / "endpoint_decision.json"
        )
        labels = self.run_labels()
        rows = labels.to_pylist()
        self.assertEqual(len(rows), 3)
        for field in COMPONENTS + ("phys_death", "high_burden", "early_discharge"):
            self.assertEqual(labels.schema.field(field).type, pa.uint8())
            self.assertEqual(labels[field].null_count, 0)
        for field in ("burden_score", "observed_hours"):
            self.assertEqual(labels.schema.field(field).type, pa.float32())
        self.assertEqual(labels.schema.field("t0").type, pa.timestamp("us"))
        self.assertEqual(rows[0]["burden_score"], 8)
        self.assertEqual(
            rows[0]["new_severe_hypoxemia_first_event_time"],
            fixtures.BASE + timedelta(hours=3),
        )
        with (self.fixture.outputs / "audit/step_06_csv_tail_checks.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            self.assertEqual(next(csv.DictReader(stream))["status"], "no_csv_inputs")
        with (self.audit / "early_3h_events.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            early = list(csv.DictReader(stream))
        self.assertEqual(early[0]["early_3h_new_severe_hypoxemia"], "1")
        self.assertEqual(early[0]["early_3h_score"], "3")
        evidence = pq.read_table(self.data / "event_evidence.parquet").to_pylist()
        self.assertTrue(
            all(r["sample_id"] in {w["sample_id"] for w in rows} for r in evidence)
        )
        self.assertEqual(labels.to_pylist(), self.run_labels().to_pylist())
        old = _json_file(self.fixture.outputs / "audit/step_06_manifest.json")
        original_replace = os.replace

        def fail_one(source, target):
            if (
                str(target).endswith("event_evidence.parquet")
                and ".step_06_stage_" in Path(source).name
            ):
                raise OSError("synthetic failed publication")
            return original_replace(source, target)

        with patch("src.cohort.common.os.replace", side_effect=fail_one):
            with self.assertRaisesRegex(OSError, "synthetic failed publication"):
                self.run_labels()
        self.assertEqual(
            old, _json_file(self.fixture.outputs / "audit/step_06_manifest.json")
        )
        _verify_run(self.fixture.outputs, 6)
        for path, digest in protected.items():
            self.assertEqual(_file_hash(Path(path)), digest)
        self.assertFalse((self.data / "cohort.parquet").exists())
        self.assertFalse((self.data / "cohort_audit.parquet").exists())

    def test_frozen_guards_preserve_existing_release(self):
        decision = self.prepare()
        self.run_labels()
        old = _json_file(self.fixture.outputs / "audit/step_06_manifest.json")
        for key, value in (
            ("horizon_frozen", False),
            ("death_weight", 2),
            ("horizon_hours", 48),
        ):
            with self.subTest(key=key):
                changed = {**decision, key: value}
                self.write_json(self.audit / "endpoint_decision.json", changed)
                with self.assertRaises(ValueError):
                    self.run_labels()
                self.assertEqual(
                    old,
                    _json_file(self.fixture.outputs / "audit/step_06_manifest.json"),
                )
        self.write_json(self.audit / "endpoint_decision.json", decision)
        config = self.configs[1]
        original = config.read_bytes()
        config.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_labels()
        config.write_bytes(original)
        raw = self.fixture.iv / "hosp/patients.csv"
        raw.write_bytes(raw.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "versions changed"):
            self.run_labels()
        _verify_run(self.fixture.outputs, 6)

    def test_invalid_evidence_even_when_hashes_are_rebound(self):
        self.prepare()
        path = self.data / "window_72h/event_evidence.parquet"
        original = path.read_bytes()
        for reason, change in (
            ("unresolved", lambda r: r.update(qualifies=None)),
            ("wrong admission", lambda r: r.update(hadm_id=999)),
            ("at t0", lambda r: r.update(event_time=fixtures.BASE)),
            (
                "after cutoff",
                lambda r: r.update(
                    first_event_time=fixtures.BASE + timedelta(hours=73)
                ),
            ),
            ("wrong component", lambda r: r.update(component="unknown")),
        ):
            with self.subTest(reason=reason):
                path.write_bytes(original)
                self.rewrite(
                    path,
                    lambda rows: change(next(r for r in rows if r["qualifies"] == 1)),
                )
                self.freeze()
                with self.assertRaises(ValueError):
                    self.run_labels()
                self.assertFalse(
                    (self.fixture.outputs / "audit/step_06_manifest.json").exists()
                )

    def test_invalid_keys_schema_and_score(self):
        self.prepare()
        path = self.data / "window_72h/study_event_audit.parquet"
        original = path.read_bytes()
        for reason, transform in (
            ("duplicate", lambda rows: rows.append(rows[0].copy())),
            ("score", lambda rows: rows[0].update(burden_score=7.0)),
            ("unresolved", lambda rows: rows[0].update(label_status="unresolved")),
            ("discharge", lambda rows: rows[0].update(early_discharge=1)),
        ):
            with self.subTest(reason=reason):
                path.write_bytes(original)
                self.rewrite(path, transform)
                self.freeze()
                with self.assertRaises(ValueError):
                    self.run_labels()
        path.write_bytes(original)
        table = pq.read_table(path).drop([COMPONENTS[0]])
        pq.write_table(table, path)
        self.freeze()
        with self.assertRaisesRegex(ValueError, "Missing wide fields"):
            self.run_labels()

    def test_discharge_event_and_death_without_icu(self):
        f = self.fixture
        # A death within H does not require an ICU event.
        f.tables["admissions"][1].update(
            hospital_expire_flag="1",
            deathtime=fixtures.dt(20),
            discharge_location="DIED",
        )
        self.prepare()
        rows = self.run_labels().to_pylist()
        row = next(r for r in rows if r["study_id"] == 200)
        self.assertEqual(row["in_hospital_death"], 1)
        self.assertEqual(row["new_icu_ccu_transfer"], 0)
        self.assertEqual(row["burden_score"], 3)
        self.assertEqual(row["early_discharge"], 0)
        with (self.audit / "early_discharge.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            self.assertIn("burden_score", (reader.fieldnames or []))
            self.assertEqual(list(reader), [])

    def test_empty_early_report_and_alive_discharge_positive(self):
        f = self.fixture
        f.tables["procedureevents"] = []
        f.tables["chartevents"] = []
        f.tables["emar"] = []
        f.tables["emar_detail"] = []
        f.tables["icustays"] = []
        f.tables["transfers"] = [
            dict(
                subject_id=str(s),
                hadm_id=str(h),
                transfer_id=str(h),
                eventtype="admit",
                careunit="Ward",
                intime=fixtures.dt(-10),
                outtime=fixtures.dt(end),
            )
            for s, h, end in ((1, 10, 70), (2, 20, 20))
        ]
        f.tables["transfers"][-1]["outtime"] = fixtures.dt(5)
        f.tables["transfers"].append(
            dict(
                subject_id="2",
                hadm_id="20",
                transfer_id="21",
                eventtype="transfer",
                careunit="Stepdown",
                intime=fixtures.dt(5),
                outtime=fixtures.dt(20),
            )
        )
        self.prepare()
        rows = self.run_labels().to_pylist()
        row = next(r for r in rows if r["study_id"] == 200)
        self.assertEqual(row["burden_score"], 1)
        self.assertEqual(row["early_discharge"], 1)
        with (self.audit / "early_3h_events.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            self.assertIn("early_3h_score", (reader.fieldnames or []))
            self.assertEqual(list(reader), [])

    def test_unknown_classification_and_excluded_study(self):
        self.fixture.matches[1]["t0"] = fixtures.dt(3)
        self.prepare()
        meta_path = self.audit / "run_metadata.json"
        metadata = _json_file(meta_path)
        metadata["dictionary_review_complete"] = False
        self.write_json(meta_path, metadata)
        self.freeze()
        rows = self.run_labels().to_pylist()
        self.assertEqual({r["study_id"] for r in rows}, {100, 200})
        evidence = pq.read_table(self.data / "event_evidence.parquet").to_pylist()
        self.assertFalse(any(r["study_id"] == 101 for r in evidence))

    def test_cutoff_and_manual_record_exclusion(self):
        self.prepare()
        evidence_path = self.data / "window_72h/event_evidence.parquet"
        wide_path = self.data / "window_72h/study_event_audit.parquet"

        # Keep a direct positive and add a reviewed, excluded indirect record.
        def extra(rows):
            positive = next(
                r
                for r in rows
                if r["component"] == COMPONENTS[2] and r["qualifies"] == 1
            )
            rows.append(
                {
                    **positive,
                    "source_record_key": "synthetic_reviewed_record",
                    "qualifies": 0,
                    "first_event_time": None,
                    "event_status": "manual_source_exclusion",
                    "review_status": "reviewed",
                    "override_version": "synthetic_v1",
                    "automated_qualifies": None,
                }
            )

        self.rewrite(evidence_path, extra)
        # Exercise an inclusive 72-hour cutoff by extending study 100's admission.
        end = fixtures.BASE + timedelta(hours=72)
        for name in (
            "candidate_studies",
            "window_72h/candidate_observation",
            "window_72h/study_event_audit",
        ):

            def extend(rows):
                for row in rows:
                    if row["study_id"] == 100:
                        row.update(dischtime=end, deathtime=end)
                        if "observation_end" in row:
                            row.update(observation_end=end, observed_hours=72.0)
                        if "in_hospital_death_first_event_time" in row:
                            row["in_hospital_death_first_event_time"] = end

            self.rewrite(self.data / f"{name}.parquet", extend)

        def death_at_end(rows):
            for r in rows:
                if (
                    r["study_id"] == 100
                    and r["component"] == COMPONENTS[-1]
                    and r["qualifies"] == 1
                ):
                    r.update(event_time=end, first_event_time=end)

        self.rewrite(evidence_path, death_at_end)
        self.freeze()
        labels = self.run_labels().to_pylist()
        row = next(r for r in labels if r["study_id"] == 100)
        self.assertEqual(
            row["new_invasive_ventilation_first_event_time"],
            fixtures.BASE + timedelta(hours=4),
        )
        self.assertEqual(row["in_hospital_death_first_event_time"], end)
        self.assertEqual(row["burden_score"], 8)
        self.assertTrue(wide_path.exists())

    def test_changed_input_during_staging_preserves_old_output(self):
        self.prepare()
        self.run_labels()
        old = _json_file(self.fixture.outputs / "audit/step_06_manifest.json")
        calls = 0

        def change_on_second(root, configs):
            nonlocal calls
            calls += 1
            if calls == 2:
                path = self.audit / "window_decision.md"
                path.write_text("Changed during staging.\n", encoding="utf-8")
            return verify_frozen_inputs(root, configs)

        with patch(
            "src.cohort.build_labels.verify_frozen_inputs", side_effect=change_on_second
        ):
            with self.assertRaisesRegex(ValueError, "Window decision changed"):
                self.run_labels()
        self.assertEqual(
            old, _json_file(self.fixture.outputs / "audit/step_06_manifest.json")
        )
        _verify_run(self.fixture.outputs, 6)


if __name__ == "__main__":
    unittest.main()
