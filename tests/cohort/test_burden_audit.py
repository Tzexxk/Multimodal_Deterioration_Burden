"""V2 synthetic clinical boundaries and real disk/manifest integration."""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.audit_burden_labels import (
    CONFLICT_POLICY,
    IDENTITY_POLICY,
    audit_burden_labels,
    candidate_conflict_exclusion,
    load_candidate_exclusions,
    load_review_decisions,
    medication_identity_exclusion,
    save_audit_snapshot,
    validate_rules,
    verify_snapshot,
)
from src.cohort.build_candidates import (
    build_candidate,
    observation,
    ownership,
    prepare_intervals,
)
from src.cohort.burden_events import (
    aggregate_medications,
    apply_override,
    evaluate,
    hypoxemia_evidence,
    infusion_evidence,
    ventilation_episodes,
    ventilation_evidence,
)
from src.cohort.burden_io import (
    COMPONENTS,
    SUPPLEMENTARY_COLUMNS,
    TABLE_COLUMNS,
    ParquetSink,
    clinical_time,
    record_key,
    score,
)
from src.cohort.common import _file_hash, _json_file, _OutputRun, _verify_run

PROJECT = Path(__file__).resolve().parents[2]
BASE = datetime(2120, 1, 1, 10)


def dt(hours: float) -> str:
    return (BASE + timedelta(hours=hours)).isoformat()


def reviewed(**fields) -> dict:
    return dict(
        confirmed=True,
        reviewer="synthetic_reviewer",
        mapping_reason="synthetic fixture rule",
        **fields,
    )


class BurdenAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="burden_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.iv, self.cxr = self.root / "iv", self.root / "cxr"
        self.config = self.root / "configs"
        self.config.mkdir()
        self.processed, self.outputs = self.root / "processed", self.root / "outputs"
        self.tables = {name: [] for name in TABLE_COLUMNS}
        self.paths = dict(
            mimic_iv=str(self.iv),
            mimic_cxr=str(self.cxr),
            processed=str(self.processed),
            outputs=str(self.outputs),
        )
        self.cohort = copy.deepcopy(_json_file(PROJECT / "configs/cohort.json"))
        self.ehr = copy.deepcopy(_json_file(PROJECT / "configs/ehr.json"))
        self.imaging = copy.deepcopy(_json_file(PROJECT / "configs/imaging.json"))
        self.ehr["burden_v2"]["memory_limit"] = "128MB"
        rules = self.cohort["burden_v2"]
        rules["dictionary_handling_reviews"] = {}
        rules["overrides_path"] = None
        rules["candidate_exclusions_path"] = None
        rules["candidate_conflict_policies"] = {}
        rules[IDENTITY_POLICY]["confirmed"] = False
        rules["dictionary_version"] = "synthetic_reviewed_v1"
        rules["transfer_event_actions"] = {
            "admit": reviewed(action="location"),
            "transfer": reviewed(action="location"),
            "discharge": reviewed(action="exclude"),
        }
        rules["careunits"] = {
            name: reviewed(care_level=level)
            for name, level in (
                ("Ward", "ward"),
                ("Stepdown", "stepdown_intermediate"),
                ("ICU", "icu_ccu"),
            )
        }
        rules["medications"] = {
            "norepinephrine": reviewed(canonical_drug="norepinephrine"),
            "acetaminophen": reviewed(canonical_drug=None),
        }
        rules["administration_types"] = {
            "IV Infusion": reviewed(action="infusion"),
            "IV Bolus": reviewed(action="exclude"),
        }
        rules["medication_event_actions"] = {
            name: reviewed(action=action)
            for name, action in (
                ("Started", "start"),
                ("Restarted", "restart"),
                ("Continued", "continue"),
                ("Stopped", "stop"),
                ("Given", "given"),
                ("Cancelled", "exclude"),
            )
        }
        rules["procedure_status_actions"] = {
            "Finished": reviewed(action="include"),
            "Cancelled": reviewed(action="exclude"),
        }
        rules["iv_routes"] = {"IV": reviewed(action="iv")}
        rules["spo2_units"] = {"%": reviewed()}
        for value in rules["items"].values():
            value.update(reviewed())
        self.tables["d_items"] = [
            dict(itemid=key, label=value["label"], linksto=value["linksto"])
            for key, value in rules["items"].items()
        ]
        self.tables["patients"] = [
            dict(
                subject_id=str(subject),
                gender="F",
                anchor_age="40",
                anchor_year="2120",
                anchor_year_group="2017 - 2019",
                dod="",
            )
            for subject in (1, 2)
        ]
        self.tables["admissions"] = [
            dict(
                subject_id="1",
                hadm_id="10",
                admittime=dt(-10),
                dischtime=dt(70),
                deathtime=dt(60),
                hospital_expire_flag="1",
                discharge_location="DIED",
            ),
            dict(
                subject_id="2",
                hadm_id="20",
                admittime=dt(-10),
                dischtime=dt(20),
                deathtime="",
                hospital_expire_flag="0",
                discharge_location="HOME",
            ),
        ]
        for index, (unit, start, end) in enumerate(
            (
                ("ICU", -9, -8),
                ("Ward", -8, 1),
                ("Stepdown", 1, 2),
                ("ICU", 2, 12),
                ("Ward", 12, 70),
            )
        ):
            self.tables["transfers"].append(
                dict(
                    subject_id="1",
                    hadm_id="10",
                    transfer_id=str(index),
                    eventtype="transfer",
                    careunit=unit,
                    intime=dt(start),
                    outtime=dt(end),
                )
            )
        self.tables["transfers"].append(
            dict(
                subject_id="2",
                hadm_id="20",
                transfer_id="20",
                eventtype="admit",
                careunit="Ward",
                intime=dt(-10),
                outtime=dt(20),
            )
        )
        self.tables["icustays"] = [
            dict(
                subject_id="1",
                hadm_id="10",
                stay_id=str(stay),
                intime=dt(start),
                outtime=dt(end),
            )
            for stay, start, end in ((11, -9, -8), (12, 2, 12))
        ]
        self.tables["procedureevents"] = [
            dict(
                subject_id="1",
                hadm_id="10",
                stay_id="12",
                itemid="225792",
                starttime=dt(4),
                endtime=dt(6),
                storetime=dt(8),
                orderid="1",
                linkorderid="1",
                statusdescription="Finished",
                continueinnextdept="0",
            )
        ]
        self.tables["chartevents"] = [
            dict(
                subject_id="1",
                hadm_id="10",
                stay_id="12",
                itemid="220277",
                charttime=dt(3),
                storetime=dt(4),
                value="85",
                valuenum="85",
                valueuom="%",
                warning="0",
            )
        ]
        self.add_medication("m1", 5, "Started")
        self.add_medication("m2", 14, "Continued")
        self.matches = [
            dict(
                subject_id=subject,
                hadm_id=hadm,
                study_id=study,
                dicom_id=f"d{study}",
                t0=dt(time),
                admittime=dt(-10),
                dischtime=dt(end),
                match_status="unique_match",
            )
            for subject, hadm, study, time, end in (
                (1, 10, 100, 0, 70),
                (1, 10, 101, 13, 70),
                (2, 20, 200, 0, 20),
            )
        ]

    def add_medication(self, key: str, time: float, event: str, **detail) -> None:
        self.tables["emar"].append(
            dict(
                subject_id="1",
                hadm_id="10",
                emar_id=key,
                emar_seq=key[1:],
                pharmacy_id="1",
                charttime=dt(time),
                medication="Norepinephrine",
                event_txt=event,
                scheduletime=dt(time - 1),
                storetime=dt(time + 1),
            )
        )
        value = dict(
            subject_id="1",
            emar_id=key,
            emar_seq=key[1:],
            parent_field_ordinal="",
            administration_type="IV Infusion",
            complete_dose_not_given="0",
            dose_given="1",
            route="IV",
            prior_infusion_rate="",
            infusion_rate="2",
            infusion_complete="0",
            new_iv_bag_hung="0",
            continued_infusion_in_other_location="0",
        )
        value.update(detail)
        self.tables["emar_detail"].append(value)

    def prepare(self) -> None:
        for name, fields in TABLE_COLUMNS.items():
            if self.cohort["burden_v2"].get("supplementary_audit"):
                fields += " " + SUPPLEMENTARY_COLUMNS.get(name, "")
            path = self.iv / self.ehr["burden_v2"]["tables"][name]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields.split())
                writer.writeheader()
                writer.writerows(self.tables[name])
        for name, value in (
            ("paths.local", self.paths),
            ("cohort", self.cohort),
            ("ehr", self.ehr),
            ("imaging", self.imaging),
        ):
            (self.config / f"{name}.json").write_text(
                json.dumps(value), encoding="utf-8"
            )
        selections = []
        for row in self.matches:
            path = self.cxr / f"dataset/{row['dicom_id']}.dcm"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"already decoded upstream")
            selections.append(
                dict(
                    subject_id=row["subject_id"],
                    study_id=row["study_id"],
                    selected_dicom_id=row["dicom_id"],
                    acquisition_datetime=row["t0"],
                    dicom_path=path.relative_to(self.cxr).as_posix(),
                    view_position="AP",
                    selection_reason="single_frontal",
                )
            )
        selected = self.processed / "index/study_image_selection.parquet"
        with _OutputRun(self.outputs, 3, self.processed) as run:
            pq.write_table(pa.Table.from_pylist(selections), run.path(selected))
        with _OutputRun(self.outputs, 4, self.processed) as run:
            pq.write_table(
                pa.Table.from_pylist(self.matches),
                run.path(self.processed / "index/cxr_admission_matches.parquet"),
            )
            with run.path(self.outputs / "audit/admission_match_summary.csv").open(
                "w", newline=""
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(["metric", "value"])
                writer.writerow(["selection_input_sha256", _file_hash(selected)])
                writer.writerow(
                    [
                        "admissions_input_sha256",
                        _file_hash(self.iv / "hosp/admissions.csv"),
                    ]
                )

    def run_audit(self) -> dict:
        with redirect_stdout(io.StringIO()):
            result = audit_burden_labels(
                self.root,
                *(
                    self.config / f"{name}.json"
                    for name in ("paths.local", "cohort", "ehr", "imaging")
                ),
            )
        self.assertEqual(result, 0)
        _verify_run(self.outputs, 5)
        return _json_file(self.outputs / "audit/burden_v2/run_metadata.json")

    def wide(self, horizon=48) -> dict:
        values = pq.read_table(
            self.processed
            / f"cohort/burden_v2/window_{horizon}h/study_event_audit.parquet"
        ).to_pylist()
        return {row["study_id"]: row for row in values}

    def test_truncated_tail_keeps_scores_and_middle_errors_preserve_outputs(
        self,
    ) -> None:
        self.prepare()
        self.run_audit()
        original_wide = self.wide()
        manifest = self.outputs / "audit/step_05_manifest.json"
        for name in ("chartevents", "emar_detail"):
            with self.subTest(source=name):
                path = self.iv / self.ehr["burden_v2"]["tables"][name]
                original = path.read_bytes()
                last_record = original.rstrip(b"\r\n").rsplit(b"\n", 1)[-1]
                shortened = b",".join(last_record.split(b",")[:-1])
                path.write_bytes(original + shortened)
                truncated_hash = _file_hash(path)
                try:
                    metadata = self.run_audit()
                    self.assertEqual(self.wide(), original_wide)
                    self.assertTrue(metadata["audit_ready_for_window_decision"])
                    self.assertEqual(
                        metadata["csv_tail_checks"]["skipped_record_count"], 1
                    )
                    self.assertEqual(_file_hash(path), truncated_hash)
                    with (
                        self.outputs / "audit/step_05_csv_tail_checks.csv"
                    ).open() as stream:
                        checks = list(csv.DictReader(stream))
                    found = next(
                        row for row in checks if row["file"] == str(path.resolve())
                    )
                    self.assertEqual(found["status"], "incomplete")
                    self.assertEqual(found["skipped_rows"], "1")
                    old_manifest = manifest.read_bytes()
                    # The same bad record in the middle must stop the run.
                    header, body = original.split(b"\n", 1)
                    # Validate raw rejects even when their patient is outside the cohort.
                    unrelated = b"999999," + shortened.split(b",", 1)[1]
                    path.write_bytes(header + b"\n" + unrelated + b"\n" + body)
                    with self.assertRaisesRegex(ValueError, "not.*(final|EOF)"):
                        self.run_audit()
                    self.assertEqual(manifest.read_bytes(), old_manifest)
                    _verify_run(self.outputs, 5)
                finally:
                    path.write_bytes(original)

    def test_both_windows_caps_prior_icu_and_live_discharge(self) -> None:
        self.tables["chartevents"].append(
            {
                **self.tables["chartevents"][0],
                "charttime": dt(4),
                "valuenum": "83",
                "value": "83",
            }
        )
        duplicate = {**self.tables["emar_detail"][0], "parent_field_ordinal": "1"}
        self.tables["emar_detail"].append(duplicate)
        self.prepare()
        metadata = self.run_audit()
        a, b = self.wide(), self.wide(72)
        self.assertEqual(a[100]["burden_score"], 5)
        self.assertEqual(b[100]["burden_score"], 8)
        self.assertEqual([b[100][key] for key in COMPONENTS], [1] * 6)
        self.assertEqual(a[100]["prior_icu_before_t0"], 1)
        self.assertEqual(a[100]["candidate_status"], "eligible")
        self.assertEqual(a[100][f"{COMPONENTS[3]}_baseline_status"], "not_observed")
        self.assertEqual(
            a[100][f"{COMPONENTS[0]}_baseline_status"], "inactive_documented"
        )
        self.assertEqual(a[101][COMPONENTS[4]], 0)
        self.assertEqual(a[101]["burden_score"], 3)
        self.assertEqual(a[200]["burden_score"], 0)
        self.assertEqual(a[200]["early_discharge"], 1)
        self.assertTrue(metadata["audit_ready_for_window_decision"])
        self.assertFalse(metadata["ready_for_formal_labels"])
        administrations = pq.read_table(
            self.processed / "cohort/burden_v2/medication_administrations.parquet"
        )
        self.assertEqual(administrations.num_rows, 2)
        before = _file_hash(
            self.processed / "cohort/burden_v2/window_48h/study_event_audit.parquet"
        )
        self.run_audit()
        self.assertEqual(
            before,
            _file_hash(
                self.processed / "cohort/burden_v2/window_48h/study_event_audit.parquet"
            ),
        )
        with (self.outputs / "audit/burden_v2/window_48h/early_3h_events.csv").open(
            newline=""
        ) as f:
            early = list(csv.DictReader(f))
        self.assertEqual(float(early[0]["early_3h_score"]), 3)

    def test_pending_dictionaries_and_conflicts_never_become_zero(self) -> None:
        self.cohort["burden_v2"]["medications"]["norepinephrine"]["confirmed"] = False
        self.tables["chartevents"].append(
            {**self.tables["chartevents"][0], "value": "99", "valuenum": "99"}
        )
        self.prepare()
        metadata = self.run_audit()
        self.assertIsNone(self.wide()[100][COMPONENTS[4]])
        self.assertIsNone(self.wide()[100][COMPONENTS[3]])
        self.assertIsNone(self.wide()[100]["burden_score"])
        self.assertFalse(metadata["audit_ready_for_window_decision"])

    def test_source_change_and_failed_publication_preserve_previous_outputs(
        self,
    ) -> None:
        self.prepare()
        self.run_audit()
        path = self.outputs / "audit/step_05_manifest.json"
        old = path.read_bytes()
        original_replace, failed = os.replace, False

        def fail_once(source, destination):
            nonlocal failed
            if not failed and str(destination).endswith("study_event_audit.parquet"):
                failed = True
                raise OSError("publication failure")
            return original_replace(source, destination)

        with patch("src.cohort.common.os.replace", side_effect=fail_once):
            with self.assertRaises(OSError):
                self.run_audit()
        self.assertEqual(path.read_bytes(), old)
        _verify_run(self.outputs, 5)
        with (self.iv / "hosp/admissions.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Admissions changed"):
            self.run_audit()
        self.assertEqual(path.read_bytes(), old)

    def test_supplementary_positive_and_history_uncertainty(self):
        rules = self.cohort["burden_v2"]
        rules["supplementary_audit"] = {"enabled": True}
        chart = {**self.tables["chartevents"][0], "charttime": dt(5), "warning": ""}
        self.tables["chartevents"].append(chart)
        values, records = evaluate(self.study(), self.context(), rules, {})
        self.assertEqual(values[COMPONENTS[3]], 1)
        unknown = next(r for r in records if r["qualifies"] is None)
        self.assertEqual(unknown["blocks_event"], 0)
        self.assertEqual(
            unknown["uncertainty_relevance"], "independent_after_confirmed_positive"
        )
        self.tables["chartevents"][-1]["charttime"] = dt(3)
        values, _ = evaluate(self.study(), self.context(), rules, {})
        self.assertIsNone(values[COMPONENTS[3]])
        self.assertIsNone(values["burden_score"])
        self.tables["chartevents"].pop()
        self.tables["emar"] = [
            dict(self.tables["emar"][0], charttime=dt(-11), medication="unknown")
        ]
        context = self.context()
        values, records = evaluate(self.study(), context, rules, {})
        self.assertEqual(values[COMPONENTS[4]], 0)
        self.assertTrue(
            any(
                r["uncertainty_relevance"] == "history_without_window_infusion"
                for r in records
            )
        )
        self.tables["emar"][0]["charttime"] = dt(-1)
        rules[IDENTITY_POLICY]["confirmed"] = True
        self.assertIsNone(
            medication_identity_exclusion(self.study(), self.context(), rules)[0]
        )
        self.tables["emar"].append(
            dict(
                self.tables["emar"][0],
                emar_id="new",
                charttime=dt(5),
                medication="norepinephrine",
            )
        )
        values, _ = evaluate(self.study(), self.context(), rules, {})
        self.assertIsNone(values[COMPONENTS[4]])
        self.assertIsNotNone(
            medication_identity_exclusion(self.study(), self.context(), rules)[0]
        )

    def test_supplementary_mixed_components_and_missing_support_fields(self):
        rules = self.cohort["burden_v2"]
        rules["supplementary_audit"] = {"enabled": True}
        self.tables["emar"][0].update(pharmacy_id="9", medication="")
        self.tables["prescriptions"] = [
            dict(
                subject_id="1",
                hadm_id="10",
                pharmacy_id="9",
                poe_id="9",
                poe_seq="1",
                starttime=dt(0),
                stoptime=dt(10),
                drug_type="BASE",
                drug="saline",
                route="IV",
            )
        ]
        rules["medications"]["saline"] = reviewed(canonical_drug=None)
        context = self.context()
        self.assertEqual(
            context["medication_events"][0]["component_identity_status"],
            "all_confirmed_non_target",
        )
        self.assertEqual(context["medication_events"][0]["action"], "exclude")
        self.tables["prescriptions"].append(
            dict(
                self.tables["prescriptions"][0],
                drug_type="ADDITIVE",
                drug="norepinephrine",
            )
        )
        context = self.context()
        self.assertEqual(
            context["medication_events"][0]["reason"],
            "unresolved_medication_components",
        )
        self.assertIsNone(evaluate(self.study(), context, rules, {})[0][COMPONENTS[4]])
        self.tables["prescriptions"] = [self.tables["prescriptions"][-1]]
        self.tables["emar"][0]["medication"] = "saline"
        context = self.context()
        self.assertEqual(
            context["medication_events"][0]["reason"],
            "unresolved_medication_components",
        )
        self.assertIsNone(evaluate(self.study(), context, rules, {})[0][COMPONENTS[4]])
        self.tables["procedureevents"][0]["endtime"] = ""
        context = self.context()
        self.assertTrue(context["ventilation_rejected"])
        self.assertIsNone(evaluate(self.study(), context, rules, {})[0][COMPONENTS[2]])

    def test_versioned_decision_validation_and_supersession(self):
        legacy = dict(
            sample_id="p1_h10_s100",
            reason="blocker",
            source_record_keys='["admissions:key"]',
            reviewer="synthetic",
            rationale="old review",
            version="v1",
        )
        row = {
            **legacy,
            "version": "v2",
            "decision": "resolve_blocker",
            "reviewed_on": "2026-10-08",
            "source_run_id": "a" * 32,
            "source_manifest_sha256": "b" * 64,
            "supersedes": json.dumps(
                [
                    {
                        "sample_id": legacy["sample_id"],
                        "reason": "blocker",
                        "version": "v1",
                    }
                ]
            ),
        }
        path = self.config / "decisions.csv"

        def write(rows):
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)

        write([row])
        active = load_review_decisions(path, {legacy["sample_id"]}, False, [legacy])
        self.assertEqual([r["decision"] for r in active], ["resolve_blocker"])
        write([{**row, "supersedes": ""}])
        with self.assertRaisesRegex(ValueError, "Conflicting active"):
            load_review_decisions(path, {legacy["sample_id"]}, False, [legacy])
        write([{**row, "decision": "pending"}])
        with self.assertRaisesRegex(ValueError, "Pending"):
            load_review_decisions(path, {legacy["sample_id"]}, False, [legacy])
        write([{**row, "source_manifest_sha256": "bad"}])
        with self.assertRaisesRegex(ValueError, "binding"):
            load_review_decisions(path, {legacy["sample_id"]}, False, [legacy])

    def test_versioned_event_decisions_zero_one_pending_and_source_scope(self):
        row = dict(
            sample_id="p1_h10_s100",
            horizon_hours="72",
            component=COMPONENTS[3],
            event_flag="0",
            first_event_time="",
            source_record_keys='["chartevents:key"]',
            reviewer="Synthetic only",
            rationale="Synthetic event review",
            version="v1",
            override_scope="component",
            decision="set_event",
            reviewed_on="2026-10-08",
            source_run_id="a" * 32,
            source_manifest_sha256="b" * 64,
            supersedes="",
        )
        path = self.config / "event_decisions.csv"

        def load(value):
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(value)
            return load_review_decisions(path, {row["sample_id"]}, True, [])

        self.assertEqual(load(row)[0]["event_flag"], "0")
        self.assertEqual(
            load({**row, "event_flag": "1", "first_event_time": dt(4)})[0][
                "event_flag"
            ],
            "1",
        )
        self.assertEqual(load({**row, "decision": "pending", "event_flag": ""}), [])
        source = load(
            {
                **row,
                "decision": "exclude_source",
                "override_scope": "source_record",
                "event_flag": "",
            }
        )[0]
        self.assertEqual(source["event_flag"], "0")
        for change in (
            {"event_flag": "1"},
            {"first_event_time": dt(4)},
            {"decision": "pending"},
            {"component": "not_an_event"},
            {"horizon_hours": "24"},
            {"source_record_keys": '["key", "key"]'},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                load({**row, **change})

    def test_supplementary_isolation_shadow_audit_and_bound_resolution(self):
        keys = json.dumps([record_key("admissions", self.tables["admissions"][0])])
        legacy = dict(
            sample_id="p1_h10_s100",
            reason="Invasive ventilation evidence or onset cannot be verified",
            source_record_keys=keys,
            reviewer="synthetic",
            rationale="synthetic exclusion",
            version="old_v1",
        )
        exclusion_path = self.config / "legacy.csv"
        with exclusion_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(legacy))
            writer.writeheader()
            writer.writerow(legacy)
        self.cohort["burden_v2"]["candidate_exclusions_path"] = str(exclusion_path)
        self.prepare()
        self.run_audit()
        baseline_outputs = self.outputs
        old_manifest = (baseline_outputs / "audit/step_05_manifest.json").read_bytes()
        self.cohort["burden_v2"]["supplementary_audit"] = dict(
            baseline_outputs=str(baseline_outputs),
            baseline_exclusions_path=str(
                baseline_outputs / "audit/burden_v2/excluded_studies.csv"
            ),
            expected_scope_counts=[1, 0],
        )
        self.processed, self.outputs = (
            self.root / "r02/processed",
            self.root / "r02/outputs",
        )
        self.paths.update(
            processed=str(self.processed),
            outputs=str(self.outputs),
            cache=str(self.root / "r02/cache"),
        )
        self.prepare()
        metadata = self.run_audit()
        self.assertEqual(self.wide(72)[100]["label_status"], "excluded")
        preview = pq.read_table(
            self.processed / "cohort/burden_v2/supplementary_event_audit.parquet"
        ).to_pylist()
        self.assertTrue(
            any(r["study_id"] == 100 and r["burden_score"] == 8 for r in preview)
        )
        index = save_audit_snapshot(self.outputs)
        snapshot = verify_snapshot(index)
        row = {
            **legacy,
            "version": "new_v1",
            "decision": "resolve_blocker",
            "reviewed_on": "2026-10-08",
            "source_run_id": metadata["run_id"],
            "source_manifest_sha256": snapshot["source_manifest_sha256"],
            "supersedes": json.dumps(
                [
                    {
                        "sample_id": legacy["sample_id"],
                        "reason": legacy["reason"],
                        "version": legacy["version"],
                    }
                ]
            ),
        }
        review_path = self.config / "cohort_decisions.csv"
        with review_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        self.cohort["burden_v2"]["cohort_decisions_path"] = str(review_path)
        preview_evidence = pq.read_table(
            self.processed / "cohort/burden_v2/supplementary_event_evidence.parquet"
        ).to_pylist()
        episode = next(
            r
            for r in preview_evidence
            if r["sample_id"] == legacy["sample_id"]
            and r["horizon_hours"] == 72
            and r["component"] == COMPONENTS[2]
            and r["qualifies"] == 1
        )
        event_row = {
            k: row[k]
            for k in (
                "sample_id",
                "reviewer",
                "rationale",
                "version",
                "reviewed_on",
                "source_run_id",
                "source_manifest_sha256",
            )
        }
        event_row.update(
            horizon_hours="72",
            component=COMPONENTS[2],
            event_flag="",
            first_event_time="",
            source_record_keys=json.dumps([episode["source_record_key"]]),
            decision="exclude_source",
            override_scope="source_record",
            supersedes="",
        )
        event_path = self.config / "event_decisions.csv"
        with event_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(event_row))
            writer.writeheader()
            writer.writerow(event_row)
        self.cohort["burden_v2"]["event_decisions_path"] = str(event_path)
        (self.config / "cohort.json").write_text(
            json.dumps(self.cohort), encoding="utf-8"
        )
        reviewed_metadata = self.run_audit()
        self.assertEqual(self.wide(72)[100]["burden_score"], 8)
        self.assertEqual(self.wide(72)[100][COMPONENTS[2]], 0)
        self.assertEqual(self.wide(72)[100][COMPONENTS[3]], 1)
        self.assertTrue(reviewed_metadata["review_snapshot_bindings"])
        self.assertEqual(
            (baseline_outputs / "audit/step_05_manifest.json").read_bytes(),
            old_manifest,
        )
        retained = (self.outputs / "audit/step_05_manifest.json").read_bytes()
        original_config = (self.config / "cohort.json").read_bytes()
        changed_rules = copy.deepcopy(self.cohort)
        changed_rules["burden_v2"]["rules_version"] += "_changed"
        (self.config / "cohort.json").write_text(
            json.dumps(changed_rules), encoding="utf-8"
        )
        with self.assertRaisesRegex(
            ValueError, "rules, dictionary or audit code changed"
        ):
            self.run_audit()
        self.assertEqual(
            (self.outputs / "audit/step_05_manifest.json").read_bytes(), retained
        )
        (self.config / "cohort.json").write_bytes(original_config)
        self.tables["chartevents"].append(
            dict(self.tables["chartevents"][0], charttime=dt(6))
        )
        with (self.iv / self.ehr["burden_v2"]["tables"]["chartevents"]).open(
            "a", newline="", encoding="utf-8"
        ) as stream:
            csv.writer(stream).writerow(
                [
                    self.tables["chartevents"][-1].get(f, "")
                    for f in TABLE_COLUMNS["chartevents"].split()
                ]
            )
        with self.assertRaisesRegex(ValueError, "raw source versions changed"):
            self.run_audit()
        self.assertEqual(
            (self.outputs / "audit/step_05_manifest.json").read_bytes(), retained
        )
        entry = next(iter(snapshot["files"].values()))
        Path(entry["path"]).write_bytes(b"tampered synthetic snapshot")
        with self.assertRaisesRegex(ValueError, "snapshot evidence changed"):
            self.run_audit()
        self.assertEqual(
            (self.outputs / "audit/step_05_manifest.json").read_bytes(), retained
        )

    def context(self) -> dict:
        admission = self.tables["admissions"][0]
        rules = self.cohort["burden_v2"]
        context = dict(
            admission=admission,
            admissions=self.tables["admissions"],
            source_names=self.ehr["burden_v2"]["tables"],
            transfers=prepare_intervals(
                "transfers",
                self.tables["transfers"][:-1],
                admission,
                self.tables["admissions"],
                rules,
            ),
            stays=prepare_intervals(
                "icustays",
                self.tables["icustays"],
                admission,
                self.tables["admissions"],
                rules,
            ),
            **{
                table: self.tables[table]
                for table in (
                    "procedureevents",
                    "chartevents",
                    "emar",
                    "pharmacy",
                    "prescriptions",
                )
            },
        )
        context["ventilation_episodes"], context["ventilation_rejected"] = (
            ventilation_episodes(context, rules)
        )
        context["medication_events"] = aggregate_medications(
            context, self.tables["emar_detail"], rules
        )
        return context

    def study(self, time=0, hours=48) -> dict:
        candidate = dict(
            sample_id="p1_h10_s100",
            subject_id=1,
            hadm_id=10,
            study_id=100,
            t0=clinical_time(dt(time)),
            dischtime=clinical_time(dt(70)),
            candidate_status="eligible",
            hospital_expire_flag="1",
            exclusion_reason="",
        )
        return observation(candidate, hours)

    def test_onset_boundaries_merged_support_and_restart(self) -> None:
        self.tables["icustays"][1].update(intime=dt(-2), outtime=dt(65))
        self.tables["procedureevents"][0].update(starttime=dt(0), endtime=dt(2))
        self.tables["procedureevents"].append(
            {
                **self.tables["procedureevents"][0],
                "orderid": "2",
                "starttime": dt(2),
                "endtime": dt(4),
            }
        )
        context = self.context()
        self.assertEqual(len(context["ventilation_episodes"]), 1)
        events = ventilation_evidence(self.study(), context)
        self.assertFalse(any(row["qualifies"] == 1 for row in events))
        self.assertEqual(events[0]["baseline_status"], "active_documented")
        for start, expected48, expected72 in (
            (1, 1, 1),
            (3, 1, 1),
            (48, 1, 1),
            (49, 0, 1),
            (72, 0, 1),
        ):
            with self.subTest(start=start):
                self.tables["admissions"][0]["dischtime"] = dt(80)
                self.tables["icustays"][1]["outtime"] = dt(80)
                self.tables["procedureevents"] = [
                    {
                        **self.tables["procedureevents"][0],
                        "starttime": dt(start),
                        "endtime": dt(start + 1),
                    }
                ]
                context = self.context()
                study = {**self.study(), "dischtime": clinical_time(dt(80))}
                for horizon, expected in ((48, expected48), (72, expected72)):
                    records = ventilation_evidence(observation(study, horizon), context)
                    self.assertEqual(
                        int(any(row["qualifies"] == 1 for row in records)), expected
                    )
        self.tables["emar"], self.tables["emar_detail"] = [], []
        self.add_medication("m1", -1, "Started")
        self.add_medication("m2", 1, "Continued", new_iv_bag_hung="1")
        self.add_medication("m3", 2, "Stopped")
        self.add_medication("m4", 3, "Restarted")
        records = infusion_evidence(self.study(), self.context())
        positive = [row for row in records if row["qualifies"] == 1]
        self.assertEqual(
            [row["event_time"] for row in positive], [clinical_time(dt(3))]
        )
        self.assertTrue(
            any(row["exclusion_reason"] == "preexisting_support" for row in records)
        )

    def test_spo2_recovery_artifacts_stale_stay_and_missing_records(self) -> None:
        rules = self.cohort["burden_v2"]
        self.tables["chartevents"] = []
        self.assertEqual(hypoxemia_evidence(self.study(), self.context(), rules), [])
        template = dict(
            subject_id="1",
            hadm_id="10",
            stay_id="11",
            itemid="220277",
            charttime=dt(-8.5),
            storetime=dt(-8),
            value="80",
            valuenum="80",
            valueuom="%",
            warning="0",
        )
        self.tables["chartevents"] = [
            template,
            {**template, "stay_id": "12", "charttime": dt(3)},
        ]
        records = hypoxemia_evidence(self.study(), self.context(), rules)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["qualifies"], 1)
        self.assertEqual(records[0]["baseline_status"], "not_observed")
        self.tables["icustays"][1].update(intime=dt(-1), outtime=dt(12))
        self.tables["chartevents"] = [
            {
                **template,
                "stay_id": "12",
                "charttime": dt(time),
                "valuenum": str(value),
                "value": str(value),
            }
            for time, value in ((-0.5, 80), (1, 82), (2, 95), (3, 85))
        ]
        records = hypoxemia_evidence(self.study(), self.context(), rules)
        self.assertEqual(
            [row["event_time"] for row in records if row["qualifies"] == 1],
            [clinical_time(dt(3))],
        )
        self.tables["chartevents"][-1]["warning"] = "1"
        records = hypoxemia_evidence(self.study(), self.context(), rules)
        self.assertIsNone(records[-1]["qualifies"])

    def test_bolus_cancelled_unknown_onset_and_admission_ownership(self) -> None:
        self.tables["emar"][0]["event_txt"] = "Given"
        records = infusion_evidence(self.study(), self.context())
        self.assertTrue(any(row["qualifies"] is None for row in records))
        self.tables["emar_detail"][0]["administration_type"] = "IV Bolus"
        records = infusion_evidence(self.study(), self.context())
        self.assertEqual(records[0]["qualifies"], 0)
        self.tables["emar_detail"][0]["administration_type"] = "IV Infusion"
        self.tables["emar_detail"][0]["complete_dose_not_given"] = "1"
        records = infusion_evidence(self.study(), self.context())
        self.assertEqual(records[0]["qualifies"], 0)
        self.tables["chartevents"][0]["hadm_id"] = ""
        records = hypoxemia_evidence(
            self.study(), self.context(), self.cohort["burden_v2"]
        )
        self.assertEqual(records[0]["qualifies"], 1)
        self.tables["admissions"].append(
            dict(
                subject_id="1",
                hadm_id="11",
                admittime=dt(2),
                dischtime=dt(4),
                deathtime="",
                hospital_expire_flag="0",
                discharge_location="HOME",
            )
        )
        records = hypoxemia_evidence(
            self.study(), self.context(), self.cohort["burden_v2"]
        )
        self.assertIsNone(records[0]["qualifies"])

    def test_excluded_administration_types_preserve_history_and_conflicts(self) -> None:
        rules = self.cohort["burden_v2"]
        excluded_types = (
            "IV Flush",
            "Transdermal Patch",
            "Insulin SC Sliding Scale",
            "Insulin SC Fixed Dose",
            "Epidural",
            "Heparin IV Bolus",
            "Heparin IV Infusion",
            "Insulin Infusion (IV Drip)",
            "TPN",
            "IV Fluid Repletion",
        )
        rules["administration_types"].update(
            {name: reviewed(action="exclude") for name in excluded_types}
        )
        for name in excluded_types:
            with self.subTest(administration_type=name):
                self.tables["emar"], self.tables["emar_detail"] = [], []
                self.add_medication("m1", 1, "Started")
                self.add_medication("m2", 2, "Stopped", administration_type=name)
                self.add_medication("m3", 3, "Continued")
                context = self.context()
                records = infusion_evidence(self.study(), context)
                self.assertEqual([row["qualifies"] for row in records], [1, 0, 0])
                self.assertTrue(context["medication_events"][-1]["active_before"])

                self.tables["emar"][1]["medication"] = ""
                records = infusion_evidence(self.study(), self.context())
                self.assertEqual([row["qualifies"] for row in records], [1, 0, 0])

                self.tables["emar"][1]["medication"] = "Norepinephrine"
                self.tables["emar_detail"].append(
                    {
                        **self.tables["emar_detail"][1],
                        "parent_field_ordinal": "1",
                        "administration_type": "IV Infusion",
                    }
                )
                context = self.context()
                records = infusion_evidence(self.study(), context)
                self.assertEqual([row["qualifies"] for row in records], [1, None, 0])
                self.assertEqual(
                    records[1]["exclusion_reason"], "unresolved_administration_type"
                )
                self.assertTrue(context["medication_events"][-1]["active_before"])

        self.tables["emar_detail"].pop()
        self.tables["emar"][1]["medication"] = ""
        self.tables["emar_detail"][1]["subject_id"] = "2"
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[1]["qualifies"])
        self.assertEqual(records[1]["exclusion_reason"], "emar_detail_key_mismatch")

        self.tables["emar_detail"][1]["subject_id"] = "1"
        self.tables["emar_detail"][1]["administration_type"] = "IV Infusion"
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[1]["qualifies"])
        self.assertEqual(records[1]["exclusion_reason"], "unconfirmed_medication_name")

    def test_review_type_policies_and_blank_detail_aggregation(self) -> None:
        ambiguous_types = (
            "Standard Maintenance Medication",
            "PCA",
            "Sliding Scale",
            "OVR",
            "PACU-only order",
            "",
        )
        rules = self.cohort["burden_v2"]
        rules["administration_types"].update(
            {name: reviewed(action="review") for name in ambiguous_types}
        )
        validate_rules(self.cohort, self.ehr, self.imaging)
        for name in ambiguous_types:
            with self.subTest(administration_type=name):
                self.tables["emar"], self.tables["emar_detail"] = [], []
                self.add_medication("m1", 1, "Started")
                self.add_medication("m2", 2, "Stopped", administration_type=name)
                self.add_medication("m3", 3, "Continued")
                context = self.context()
                records = infusion_evidence(self.study(), context)
                self.assertEqual([row["qualifies"] for row in records], [1, None, 0])
                self.assertEqual(
                    records[1]["exclusion_reason"], "unresolved_administration_type"
                )
                self.assertTrue(context["medication_events"][-1]["active_before"])

                self.tables["emar"][1]["medication"] = "Acetaminophen"
                records = infusion_evidence(self.study(), self.context())
                self.assertEqual([row["qualifies"] for row in records], [1, 0, 0])

                self.tables["emar"][1]["medication"] = ""
                records = infusion_evidence(self.study(), self.context())
                self.assertIsNone(records[1]["qualifies"])
                self.assertEqual(
                    records[1]["exclusion_reason"], "unconfirmed_medication_name"
                )

        self.tables["emar"], self.tables["emar_detail"] = [], []
        self.add_medication("m1", 1, "Started", administration_type="")
        self.tables["emar_detail"].append(
            {
                **self.tables["emar_detail"][0],
                "parent_field_ordinal": "1",
                "administration_type": "IV Infusion",
            }
        )
        records = infusion_evidence(self.study(), self.context())
        self.assertEqual([row["qualifies"] for row in records], [1])

        self.tables["emar_detail"][0]["administration_type"] = "PCA"
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[0]["qualifies"])
        self.assertEqual(records[0]["review_status"], "pending")

        self.tables["emar_detail"][1]["administration_type"] = "IV Bolus"
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[0]["qualifies"])

        self.tables["emar_detail"][0]["administration_type"] = ""
        records = infusion_evidence(self.study(), self.context())
        self.assertEqual([row["qualifies"] for row in records], [0])

    def test_excluded_preadmission_medications_do_not_reopen_infusion_review(
        self,
    ) -> None:
        for medication, administration_type in (
            ("Acetaminophen", "IV Infusion"),
            ("Norepinephrine", "IV Bolus"),
            ("", "IV Bolus"),
        ):
            with self.subTest(
                medication=medication, administration_type=administration_type
            ):
                self.tables["emar"], self.tables["emar_detail"] = [], []
                self.add_medication(
                    "m0", -11, "Stopped", administration_type=administration_type
                )
                self.tables["emar"][0]["medication"] = medication
                self.add_medication("m1", -2, "Started")
                self.add_medication("m2", 1, "Continued")
                context = self.context()
                records = infusion_evidence(self.study(), context)
                self.assertEqual(records[0]["qualifies"], 0)
                self.assertEqual(
                    records[0]["exclusion_reason"], "event_outside_admission"
                )
                self.assertEqual(records[0]["review_status"], "not_required")
                self.assertTrue(context["medication_events"][-1]["active_before"])
                result, _ = evaluate(
                    self.study(), context, self.cohort["burden_v2"], {}
                )
                self.assertEqual(result[COMPONENTS[4]], 0)
                self.assertEqual(result["label_status"], "complete")

                self.tables["emar_detail"][0]["subject_id"] = "2"
                records = infusion_evidence(self.study(), self.context())
                self.assertIsNone(records[0]["qualifies"])
                self.assertEqual(
                    records[0]["exclusion_reason"], "emar_detail_key_mismatch"
                )

        self.tables["emar_detail"][0]["subject_id"] = "1"
        self.tables["emar_detail"][0]["administration_type"] = "IV Infusion"
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[0]["qualifies"])
        self.assertEqual(records[0]["review_status"], "pending")

    def test_review_event_states_preserve_history_and_baseline(self) -> None:
        states = (
            "",
            " in Other Location",
            "Not Confirmed",
            "Delayed ",
            "Infusion Reconciliation Not Done",
            "Documented in O.R. Holding",
        )
        rules = self.cohort["burden_v2"]
        rules["medication_event_actions"].update(
            {state: reviewed(action="review") for state in states}
        )
        validate_rules(self.cohort, self.ehr, self.imaging)
        for state in states:
            with self.subTest(event_state=state):
                self.tables["emar"], self.tables["emar_detail"] = [], []
                self.add_medication("m1", -2, "Started")
                self.add_medication("m2", -1, state, infusion_complete="1")
                self.add_medication("m3", 1, "Continued")
                context = self.context()
                records = infusion_evidence(self.study(), context)
                self.assertEqual([row["qualifies"] for row in records], [0, None, 0])
                self.assertEqual(
                    records[1]["exclusion_reason"], "unresolved_medication_event"
                )
                self.assertTrue(context["medication_events"][-1]["active_before"])
                self.assertEqual(records[-1]["baseline_status"], "active_documented")
                result, _ = evaluate(self.study(), context, rules, {})
                self.assertIsNone(result[COMPONENTS[4]])
                self.assertIsNone(result["burden_score"])
                self.assertEqual(result["label_status"], "unresolved")

                self.tables["emar"][1]["medication"] = "Acetaminophen"
                context = self.context()
                self.assertEqual(context["medication_events"][1]["action"], "exclude")
                records = infusion_evidence(self.study(), context)
                self.assertEqual([row["qualifies"] for row in records], [0, 0])
                self.assertEqual(records[-1]["baseline_status"], "active_documented")

                self.tables["emar"][1]["medication"] = "Norepinephrine"
                self.tables["emar_detail"][1]["administration_type"] = "IV Bolus"
                records = infusion_evidence(self.study(), self.context())
                self.assertEqual([row["qualifies"] for row in records], [0, 0, 0])

                self.tables["emar"][1]["medication"] = ""
                self.tables["emar_detail"][1]["administration_type"] = "IV Infusion"
                records = infusion_evidence(self.study(), self.context())
                self.assertIsNone(records[1]["qualifies"])
                self.assertEqual(
                    records[1]["exclusion_reason"], "unconfirmed_medication_name"
                )

                self.tables["emar"], self.tables["emar_detail"] = [], []
                self.add_medication("m1", -3, "Started")
                self.add_medication("m2", -2, "Stopped")
                self.add_medication("m3", -1, state)
                self.add_medication("m4", 1, "Restarted")
                context = self.context()
                records = infusion_evidence(self.study(), context)
                self.assertEqual(records[-1]["qualifies"], 1)
                self.assertEqual(records[-1]["baseline_status"], "inactive_documented")

    def test_route_consistency_and_missing_route_policy(self) -> None:
        rules = self.cohort["burden_v2"]
        non_iv_routes = ("PO", "NG", "SC", "IM", "PR")
        rules["iv_routes"].update(
            {route: reviewed(action="non_iv") for route in non_iv_routes}
        )
        rules["iv_routes"][""] = reviewed(action="missing")
        validate_rules(self.cohort, self.ehr, self.imaging)
        for route in non_iv_routes:
            with self.subTest(route=route):
                self.tables["emar"], self.tables["emar_detail"] = [], []
                self.add_medication("m1", 1, "Started", route=route)
                records = infusion_evidence(self.study(), self.context())
                self.assertIsNone(records[0]["qualifies"])
                self.assertEqual(
                    records[0]["exclusion_reason"], "route_conflict_or_unconfirmed"
                )

                self.tables["emar"][0]["medication"] = "Acetaminophen"
                records = infusion_evidence(self.study(), self.context())
                self.assertEqual(records[0]["qualifies"], 0)

                self.tables["emar"][0]["medication"] = "Norepinephrine"
                self.tables["emar_detail"][0]["administration_type"] = "IV Bolus"
                records = infusion_evidence(self.study(), self.context())
                self.assertEqual(records[0]["qualifies"], 0)

        self.tables["emar"], self.tables["emar_detail"] = [], []
        self.add_medication("m1", 1, "Started", route="")
        records = infusion_evidence(self.study(), self.context())
        self.assertEqual(records[0]["qualifies"], 1)
        self.tables["emar_detail"].append(
            {
                **self.tables["emar_detail"][0],
                "parent_field_ordinal": "1",
                "route": "PO",
            }
        )
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[0]["qualifies"])
        self.tables["emar_detail"][1]["route"] = "IV"
        records = infusion_evidence(self.study(), self.context())
        self.assertEqual(records[0]["qualifies"], 1)

        self.tables["emar_detail"] = self.tables["emar_detail"][:1]
        self.tables["emar_detail"][0]["administration_type"] = ""
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[0]["qualifies"])
        self.assertEqual(
            records[0]["exclusion_reason"], "unresolved_administration_type"
        )

        self.tables["emar"], self.tables["emar_detail"] = [], []
        self.add_medication("m1", -2, "Started", route="")
        self.add_medication("m2", -1, "Stopped", route="PO", infusion_complete="1")
        self.add_medication("m3", 1, "Continued", route="")
        context = self.context()
        records = infusion_evidence(self.study(), context)
        self.assertEqual([row["qualifies"] for row in records], [0, None, 0])
        self.assertTrue(context["medication_events"][-1]["active_before"])
        self.assertEqual(records[-1]["baseline_status"], "active_documented")

        rules["iv_routes"]["PO"]["action"] = "missing"
        with self.assertRaisesRegex(ValueError, "blank route"):
            validate_rules(self.cohort, self.ehr, self.imaging)
        rules["iv_routes"]["PO"]["action"] = "non_iv"
        rules["iv_routes"][""]["action"] = "iv"
        with self.assertRaisesRegex(ValueError, "blank route"):
            validate_rules(self.cohort, self.ehr, self.imaging)

    def test_partial_administration_and_not_stopped_need_history(self) -> None:
        rules = self.cohort["burden_v2"]
        rules["medication_event_actions"].update(
            {
                "Partial Administered": reviewed(action="given"),
                "Not Stopped": reviewed(action="continue"),
            }
        )
        for state in ("Partial Administered", "Not Stopped"):
            with self.subTest(state=state):
                self.tables["emar"][0]["event_txt"] = "Started"
                self.tables["emar"][1]["event_txt"] = state
                records = infusion_evidence(self.study(), self.context())
                self.assertEqual([row["qualifies"] for row in records], [1, 0])
                records = infusion_evidence(self.study(time=6), self.context())
                self.assertEqual([row["qualifies"] for row in records], [0, 0])
                self.tables["emar"][0]["event_txt"] = state
                self.tables["emar"][1]["event_txt"] = "Stopped"
                records = infusion_evidence(self.study(), self.context())
                self.assertIsNone(records[0]["qualifies"])
                self.assertEqual(records[0]["review_status"], "pending")
        self.tables["emar"][0]["event_txt"] = "Partial Administered"
        self.tables["emar_detail"][0]["complete_dose_not_given"] = "1"
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[0]["qualifies"])
        self.assertEqual(
            records[0]["exclusion_reason"], "unresolved_partial_administration"
        )

    def test_live_discharge_keeps_events_candidate_conflicts_and_age(self) -> None:
        self.tables["admissions"][0].update(
            dischtime=dt(20),
            deathtime="",
            hospital_expire_flag="0",
            discharge_location="HOME",
        )
        self.tables["transfers"][4]["outtime"] = dt(20)
        for row in self.matches[:2]:
            row["dischtime"] = dt(20)
        self.tables["transfers"].append(
            dict(
                subject_id="1",
                hadm_id="10",
                transfer_id="99",
                eventtype="discharge",
                careunit="",
                intime=dt(20),
                outtime="",
            )
        )
        self.cohort["burden_v2"]["careunits"][""] = reviewed(care_level="unknown")
        self.prepare()
        self.run_audit()
        self.assertEqual(self.wide()[100]["burden_score"], 5)
        self.assertEqual(self.wide()[100]["early_discharge"], 1)
        self.tables["patients"][0]["anchor_age"] = "17"
        self.tables["transfers"].append(
            dict(
                subject_id="2",
                hadm_id="20",
                transfer_id="21",
                eventtype="transfer",
                careunit="Stepdown",
                intime=dt(-1),
                outtime=dt(1),
            )
        )
        self.prepare()
        self.run_audit()
        self.assertEqual(self.wide()[100]["candidate_status"], "excluded")
        self.assertEqual(self.wide()[200]["candidate_status"], "needs_review")
        self.assertIsNone(self.wide()[200]["burden_score"])

    def test_crossing_intervals_preserve_location_history_and_identity(self) -> None:
        admission = self.tables["admissions"][0]
        rules = self.cohort["burden_v2"]
        crossing = dict(
            subject_id="1",
            hadm_id="10",
            transfer_id="90",
            eventtype="admit",
            careunit="Ward",
            intime=dt(-11),
            outtime=dt(71),
        )
        discharge = dict(
            crossing,
            transfer_id="91",
            eventtype="discharge",
            careunit="UNKNOWN",
            intime=dt(70 + 44 / 3600),
            outtime="",
        )
        values = prepare_intervals(
            "transfers",
            [crossing, discharge],
            admission,
            self.tables["admissions"],
            rules,
        )
        ward, administrative = values
        self.assertEqual(ward["reason"], "")
        self.assertEqual(ward["intime"], crossing["intime"])
        self.assertEqual(ward["outtime"], crossing["outtime"])
        self.assertEqual(ward["start"], clinical_time(dt(-11)))
        self.assertEqual(ward["end"], clinical_time(dt(71)))
        self.assertEqual(ward["admission_overlap_start"], clinical_time(dt(-10)))
        self.assertEqual(ward["admission_overlap_end"], clinical_time(dt(70)))
        self.assertEqual(ward["starts_before_admission"], 1)
        self.assertEqual(ward["ends_after_discharge"], 1)
        self.assertEqual(administrative["reason"], "excluded_nonlocation_record")
        self.assertIsNone(administrative["end"])
        self.assertEqual(administrative["outtime_imputed"], 0)

        # Point events still require admission containment.
        self.assertEqual(
            ownership(crossing, admission, self.tables["admissions"], ward["start"]),
            "event_outside_admission",
        )
        for changes, expected in (
            ({"subject_id": "2"}, "subject_mismatch"),
            ({"hadm_id": "20"}, "other_admission"),
            ({"intime": dt(72), "outtime": dt(73)}, "interval_outside_admission"),
            ({"intime": dt(-12), "outtime": dt(-10)}, "interval_outside_admission"),
            ({"intime": dt(2), "outtime": dt(1)}, "invalid_care_interval"),
            ({"intime": dt(1), "outtime": dt(1)}, "invalid_care_interval"),
            (
                {"intime": dt(1), "outtime": "2120-01-01T12:00:00+08:00"},
                "invalid_care_interval",
            ),
        ):
            with self.subTest(changes=changes):
                row = prepare_intervals(
                    "transfers",
                    [{**crossing, **changes}],
                    admission,
                    self.tables["admissions"],
                    rules,
                )[0]
                self.assertEqual(row["reason"], expected)
        row = prepare_intervals(
            "transfers",
            [{**discharge, "subject_id": "2"}],
            admission,
            self.tables["admissions"],
            rules,
        )[0]
        self.assertEqual(row["reason"], "subject_mismatch")

        # Missing IDs remain tied to a unique containing admission, not overlap.
        missing = {**crossing, "hadm_id": "", "intime": dt(0)}
        row = prepare_intervals(
            "transfers", [missing], admission, self.tables["admissions"], rules
        )[0]
        self.assertEqual(row["reason"], "")
        row = prepare_intervals(
            "transfers",
            [missing],
            admission,
            [admission, {**admission, "hadm_id": "11"}],
            rules,
        )[0]
        self.assertEqual(row["reason"], "ambiguous_missing_hadm")
        row = prepare_intervals(
            "transfers",
            [{**missing, "intime": dt(-11)}],
            admission,
            self.tables["admissions"],
            rules,
        )[0]
        self.assertEqual(row["reason"], "other_admission")

    def test_boundary_stays_keep_ventilation_and_hypoxemia(self) -> None:
        self.tables["icustays"][1]["outtime"] = dt(71)
        context = self.context()
        self.assertEqual(context["stays"][1]["reason"], "")
        self.assertEqual(context["ventilation_episodes"][0]["reason"], "")
        result, _ = evaluate(self.study(), context, self.cohort["burden_v2"], {})
        self.assertEqual(result[COMPONENTS[2]], 1)
        self.assertEqual(result[COMPONENTS[3]], 1)

        # A linked ICU interval starting before admission still identifies ICU at t0.
        self.tables["transfers"] = [
            dict(
                subject_id="1",
                hadm_id="10",
                transfer_id="90",
                eventtype="admit",
                careunit="ICU",
                intime=dt(-11),
                outtime=dt(71),
            ),
            self.tables["transfers"][-1],
        ]
        self.tables["icustays"] = [
            dict(
                subject_id="1",
                hadm_id="10",
                stay_id="12",
                intime=dt(-11),
                outtime=dt(71),
            )
        ]
        context = self.context()
        candidate = build_candidate(
            self.matches[0],
            dict(
                dicom_path="d100.dcm",
                view_position="AP",
                selection_reason="single_frontal",
            ),
            context["admission"],
            self.tables["patients"][0],
            context["transfers"],
            context["stays"],
            self.cohort["burden_v2"],
            True,
        )
        self.assertEqual(candidate["candidate_status"], "excluded")
        self.assertEqual(candidate["exclusion_reason"], "icu_at_t0")

    def test_confirmed_careunit_exclusion_only_applies_at_t0(self) -> None:
        rules = self.cohort["burden_v2"]
        rules["careunits"]["Discharge Lounge"] = reviewed(
            care_level="other", exclude_at_t0=True
        )
        self.tables["transfers"][1]["careunit"] = "Discharge Lounge"
        self.prepare()
        self.run_audit()
        for horizon in (48, 72):
            rows = self.wide(horizon)
            self.assertEqual(rows[100]["careunit_at_t0"], "Discharge Lounge")
            self.assertEqual(rows[100]["care_level_at_t0"], "other")
            self.assertEqual(rows[100]["candidate_status"], "excluded")
            self.assertEqual(rows[100]["exclusion_reason"], "excluded_careunit_at_t0")
            self.assertEqual(rows[100]["review_status"], "not_required")
            self.assertEqual(rows[100]["label_status"], "excluded")
            self.assertIsNone(rows[100]["burden_score"])
            self.assertTrue(json.loads(rows[100]["candidate_evidence_keys"]))
            self.assertEqual(rows[101]["candidate_status"], "eligible")
            self.assertEqual(rows[200]["candidate_status"], "eligible")
        with (self.outputs / "audit/burden_v2/careunit_dictionary.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            mapping = next(
                row
                for row in csv.DictReader(stream)
                if row["careunit_raw"] == "Discharge Lounge"
            )
        self.assertEqual(mapping["care_level"], "other")
        self.assertEqual(mapping["exclude_at_t0"], "1")

        def current_candidate() -> dict:
            context = self.context()
            return build_candidate(
                self.matches[0],
                dict(
                    dicom_path="d100.dcm",
                    view_position="AP",
                    selection_reason="single_frontal",
                ),
                context["admission"],
                self.tables["patients"][0],
                context["transfers"],
                context["stays"],
                rules,
                True,
            )

        self.matches[0]["t0"] = dt(1)  # The old location ends exactly at t0.
        self.assertEqual(current_candidate()["candidate_status"], "eligible")
        self.matches[0]["t0"] = dt(0)
        self.tables["transfers"][1]["careunit"] = "Ward"
        self.tables["transfers"][2]["careunit"] = "Discharge Lounge"
        self.assertEqual(current_candidate()["candidate_status"], "eligible")
        self.tables["transfers"][1]["careunit"] = "Discharge Lounge"
        self.tables["transfers"][2]["careunit"] = "Stepdown"

        # Classification as other does not itself authorize candidate exclusion.
        rules["careunits"]["Discharge Lounge"]["exclude_at_t0"] = False
        self.assertEqual(current_candidate()["candidate_status"], "needs_review")
        rules["careunits"]["Discharge Lounge"]["exclude_at_t0"] = True
        self.tables["transfers"].insert(
            0,
            dict(
                subject_id="1",
                hadm_id="10",
                transfer_id="90",
                eventtype="transfer",
                careunit="Ward",
                intime=dt(-1),
                outtime=dt(1),
            ),
        )
        candidate = current_candidate()
        self.assertEqual(candidate["candidate_status"], "needs_review")
        self.assertNotIn("excluded_careunit_at_t0", candidate["exclusion_reason"])
        self.tables["transfers"].pop(0)
        rules["careunits"]["Discharge Lounge"]["confirmed"] = False
        with self.assertRaisesRegex(ValueError, "exclude_at_t0"):
            validate_rules(self.cohort, self.ehr, self.imaging)
        rules["careunits"]["Discharge Lounge"]["confirmed"] = True
        rules["careunits"]["Discharge Lounge"]["exclude_at_t0"] = "true"
        with self.assertRaisesRegex(ValueError, "exclude_at_t0"):
            validate_rules(self.cohort, self.ehr, self.imaging)

    def test_interval_conflicts_are_scoped_to_current_location_and_events(self) -> None:
        selection = dict(
            dicom_path="d100.dcm", view_position="AP", selection_reason="single_frontal"
        )
        self.tables["transfers"][3]["outtime"] = dt(1)  # Future reversed interval.
        self.tables["transfers"].insert(
            0,
            dict(
                subject_id="1",
                hadm_id="10",
                transfer_id="91",
                eventtype="transfer",
                careunit="Ward",
                intime=dt(-2),
                outtime=dt(-3),
            ),
        )
        context = self.context()
        candidate = build_candidate(
            self.matches[0],
            selection,
            context["admission"],
            self.tables["patients"][0],
            context["transfers"],
            context["stays"],
            self.cohort["burden_v2"],
            True,
        )
        self.assertEqual(candidate["candidate_status"], "eligible")
        result, records = evaluate(self.study(), context, self.cohort["burden_v2"], {})
        self.assertIsNone(result[COMPONENTS[0]])
        self.assertEqual(result["label_status"], "unresolved")
        self.assertTrue(
            any(row["exclusion_reason"] == "invalid_care_interval" for row in records)
        )

        self.tables["transfers"][0].update(intime=dt(-1), outtime=dt(1))
        self.tables["transfers"][0]["eventtype"] = "unconfirmed"
        context = self.context()
        candidate = build_candidate(
            self.matches[0],
            selection,
            context["admission"],
            self.tables["patients"][0],
            context["transfers"],
            context["stays"],
            self.cohort["burden_v2"],
            True,
        )
        self.assertEqual(candidate["candidate_status"], "needs_review")
        self.assertIn("care_interval_conflict", candidate["exclusion_reason"])

    def test_boundary_report_discharge_endpoint_and_rerun(self) -> None:
        endpoint = dt(70 + 44 / 3600)
        self.tables["transfers"].extend(
            [
                dict(
                    subject_id="1",
                    hadm_id="10",
                    transfer_id="90",
                    eventtype="transfer",
                    careunit="ICU",
                    intime=dt(70),
                    outtime=endpoint,
                ),
                dict(
                    subject_id="1",
                    hadm_id="10",
                    transfer_id="91",
                    eventtype="discharge",
                    careunit="UNKNOWN",
                    intime=endpoint,
                    outtime="",
                ),
            ]
        )
        self.tables["icustays"].append(
            dict(
                subject_id="1",
                hadm_id="10",
                stay_id="13",
                intime=dt(70),
                outtime=endpoint,
            )
        )
        self.tables["transfers"][0]["intime"] = dt(-11)
        self.cohort["burden_v2"]["careunits"]["UNKNOWN"] = reviewed(
            care_level="unknown"
        )
        self.prepare()
        metadata = self.run_audit()
        self.assertEqual(self.wide()[100]["candidate_status"], "eligible")
        self.assertEqual(self.wide()[100]["burden_score"], 5)
        self.assertEqual(self.wide(72)[100]["burden_score"], 8)
        audit = self.outputs / "audit/burden_v2/care_interval_audit.csv"
        with audit.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(
            metadata["care_interval_audit"]["admission_interval_count"], len(rows)
        )
        self.assertEqual(
            metadata["care_interval_audit"]["accepted_boundary_crossing_count"], 3
        )
        self.assertEqual(
            len(rows), len(self.tables["transfers"]) + len(self.tables["icustays"])
        )
        self.assertTrue(
            all(
                row["reason"] == ""
                for row in rows
                if row["ends_after_discharge"] == "1"
            )
        )
        discharge = next(
            row
            for row in rows
            if json.loads(row["raw_evidence"]).get("transfer_id") == "91"
        )
        self.assertEqual(discharge["reason"], "excluded_nonlocation_record")
        self.assertEqual(discharge["outtime_imputed"], "0")
        evidence_path = (
            self.processed / "cohort/burden_v2/window_72h/event_evidence.parquet"
        )
        records = pq.read_table(evidence_path).to_pylist()
        endpoint_event = next(
            row
            for row in records
            if row["study_id"] == 100
            and row["component"] == COMPONENTS[0]
            and row["event_time"] == clinical_time(dt(70))
        )
        self.assertEqual(endpoint_event["qualifies"], 1)
        raw = json.loads(endpoint_event["raw_evidence"])
        self.assertEqual(raw["admission_overlap_start"], raw["admission_overlap_end"])
        before = audit.read_bytes()
        self.run_audit()
        self.assertEqual(audit.read_bytes(), before)
        manifest = _json_file(self.outputs / "audit/step_05_manifest.json")
        self.assertIn(str(audit.resolve()), manifest["files"])

    def test_outtime_imputation_ignores_nonlocation_and_unconfirmed_records(
        self,
    ) -> None:
        admission = self.tables["admissions"][0]
        rows = [
            dict(
                subject_id="1",
                hadm_id="10",
                transfer_id=str(index),
                eventtype=event,
                careunit="Ward",
                intime=dt(start),
                outtime=end,
            )
            for index, (event, start, end) in enumerate(
                (
                    ("admit", -10, ""),
                    ("discharge", -2, ""),
                    ("unconfirmed", -1, dt(1)),
                    ("transfer", 2, dt(70)),
                )
            )
        ]
        values = prepare_intervals(
            "transfers",
            rows,
            admission,
            self.tables["admissions"],
            self.cohort["burden_v2"],
        )
        self.assertEqual(values[0]["end"], clinical_time(dt(2)))
        self.assertEqual(values[0]["outtime_imputed"], 1)
        self.assertEqual(values[0]["imputation_basis"], "next_intime_or_dischtime")
        self.assertIsNone(values[1]["end"])
        self.assertEqual(values[1]["outtime_imputed"], 0)

    def test_pharmacy_identity_requires_one_exact_match_and_confirmed_name(
        self,
    ) -> None:
        original = copy.deepcopy(self.tables["emar"][0])
        order = dict.fromkeys(TABLE_COLUMNS["pharmacy"].split(), "")
        order.update(
            subject_id="1", hadm_id="10", pharmacy_id="1", medication="Norepinephrine"
        )
        cases = (
            ([order], "resolved", "norepinephrine"),
            ([], "no_match", None),
            ([{**order, "subject_id": "2"}], "no_match", None),
            ([{**order, "hadm_id": "20"}], "no_match", None),
            ([{**order, "pharmacy_id": "2"}], "no_match", None),
            ([order, order], "ambiguous_match", None),
            (
                [order, {**order, "medication": "Acetaminophen"}],
                "ambiguous_match",
                None,
            ),
            ([{**order, "medication": ""}], "blank_pharmacy_medication", None),
            ([{**order, "medication": "Unreviewed exact alias"}], "resolved", None),
            ([{**order, "medication": "Acetaminophen"}], "resolved", None),
        )
        for orders, status, canonical in cases:
            with self.subTest(status=status, orders=orders):
                self.tables["emar"] = [{**original, "medication": ""}]
                self.tables["pharmacy"] = orders
                context = self.context()
                event = context["medication_events"][0]
                self.assertEqual(event["medication"], "")
                self.assertEqual(event["pharmacy_lookup_status"], status)
                self.assertEqual(event["canonical_drug"], canonical)
                result, _ = evaluate(
                    self.study(), context, self.cohort["burden_v2"], {}
                )
                expected = (
                    1 if canonical else 0 if event["action"] == "exclude" else None
                )
                self.assertEqual(result[COMPONENTS[4]], expected)
                if status == "ambiguous_match":
                    self.assertEqual(event["pharmacy_match_count"], 2)
        for field in ("subject_id", "hadm_id", "pharmacy_id"):
            self.tables["emar"] = [{**original, "medication": "", field: ""}]
            self.tables["pharmacy"] = [order]
            self.assertEqual(
                self.context()["medication_events"][0]["pharmacy_lookup_status"],
                "missing_lookup_key",
            )
        self.tables["emar"] = [original]
        self.tables["pharmacy"] = [{**order, "medication": "Acetaminophen"}]
        event = self.context()["medication_events"][0]
        self.assertEqual(event["pharmacy_lookup_status"], "not_needed")
        self.assertEqual(event["canonical_drug"], "norepinephrine")
        self.assertFalse(
            any(key.startswith("pharmacy:") for key in event["source_record_keys"])
        )
        self.tables["emar"] = []
        self.assertEqual(self.context()["medication_events"], [])
        self.tables["emar"] = [{**original, "medication": ""}]
        self.cohort["burden_v2"]["pharmacy_identity_lookup"]["confirmed"] = False
        self.assertEqual(
            self.context()["medication_events"][0]["pharmacy_lookup_status"], "disabled"
        )

    def test_reviewed_prescription_fluids_preserve_target_infusion_history(
        self,
    ) -> None:
        names = (
            "Sodium Chloride 0.9%",
            "0.9% Sodium Chloride",
            "0.45% Sodium Chloride",
            "Dextrose 5%",
            "5% Dextrose",
            "Lactated Ringers",
            "D5 1/2NS",
            "D5NS",
            "D5LR",
            "Potassium Chl 20 mEq / 1000 mL D5 1/2 NS",
            "Potassium Chl 20 mEq / 1000 mL D5NS",
            "Potassium Chl 40 mEq / 1000 mL D5 1/2 NS",
            "Potassium Chl 20 mEq / 1000 mL NS",
            "Potassium Chl 20 mEq / 1000 mL D5LR",
            "Potassium Chl 40 mEq / 1000 mL D5NS",
            "Amino Acids 5%-Dextrose 15%",
            "Multivitamin IV",
            "Trace Elements",
            "Prismasate (B22 K4)",
            "Prismasate (B32 K2)",
            "Sodium Chloride 3% (Hypertonic)",
        )
        project_rules = _json_file(PROJECT / "configs/cohort.json")["burden_v2"]
        rules = self.cohort["burden_v2"]
        rules["medications"].update(
            {
                name.casefold(): project_rules["medications"][name.casefold()]
                for name in names
            }
        )
        for start in (-1, 1):
            with self.subTest(target_start=start):
                self.tables["emar"], self.tables["emar_detail"] = [], []
                self.tables["prescriptions"] = []
                self.add_medication("m1", start, "Started")
                for index, name in enumerate(names):
                    self.add_medication(
                        f"m{index + 2}",
                        2 + index / 10,
                        "Started" if index % 2 else "Stopped",
                        infusion_complete="1",
                        new_iv_bag_hung="1",
                    )
                    self.tables["emar"][-1].update(
                        medication="", pharmacy_id=str(100 + index)
                    )
                    prescription = dict.fromkeys(
                        TABLE_COLUMNS["prescriptions"].split(), ""
                    )
                    prescription.update(
                        subject_id="1",
                        hadm_id="10",
                        pharmacy_id=str(100 + index),
                        drug=name,
                        drug_type="MAIN",
                        route="IV",
                    )
                    self.tables["prescriptions"].append(prescription)
                self.add_medication(f"m{len(names) + 2}", 4.5, "Continued")
                context = self.context()
                events = context["medication_events"]
                self.assertEqual(len(events), len(names) + 2)
                for name, event in zip(names, events[1:-1], strict=True):
                    self.assertEqual(event["resolved_medication"], name)
                    self.assertEqual(event["medication_name_source"], "prescriptions")
                    self.assertTrue(event["medication_mapping_confirmed"])
                    self.assertIsNone(event["canonical_drug"])
                    self.assertEqual(event["action"], "exclude")
                    self.assertIs(event["onset"], False)
                continuation = events[-1]
                self.assertTrue(continuation["active_before"])
                self.assertEqual(
                    continuation["episode_start_before"], clinical_time(dt(start))
                )
                self.assertIs(continuation["onset"], False)
                self.assertFalse(continuation["reason"])
                for horizon in (48, 72):
                    study = self.study(hours=horizon)
                    records = infusion_evidence(study, context)
                    self.assertEqual(
                        sum(row["qualifies"] for row in records), int(start > 0)
                    )
                    self.assertTrue(all(row["qualifies"] == 0 for row in records[1:]))
                    result, _ = evaluate(study, context, rules, {})
                    self.assertEqual(result[COMPONENTS[4]], int(start > 0))
                    self.assertEqual(
                        result[COMPONENTS[4] + "_first_event_time"],
                        clinical_time(dt(start)) if start > 0 else None,
                    )
                    self.assertEqual(result["label_status"], "complete")
                    self.assertEqual(
                        result["burden_score"],
                        (5 if horizon == 48 else 8) - (2 if start < 0 else 0),
                    )
        self.add_medication(f"m{len(names) + 3}", 5, "Started")
        self.tables["emar"][-1].update(medication="", pharmacy_id="")
        result, _ = evaluate(self.study(), self.context(), rules, {})
        self.assertIsNone(result[COMPONENTS[4]])
        self.assertIsNone(result["burden_score"])
        self.assertEqual(result["label_status"], "unresolved")

    def test_prescriptions_identity_requires_unique_evidence_and_preserves_conflicts(
        self,
    ) -> None:
        original = copy.deepcopy(self.tables["emar"][0])
        order = dict.fromkeys(TABLE_COLUMNS["prescriptions"].split(), "")
        order.update(
            subject_id="1",
            hadm_id="10",
            pharmacy_id="1",
            drug="Norepinephrine",
            drug_type="MAIN",
            route="PO",
            starttime=dt(-100),
            stoptime=dt(-99),
        )
        cases = (
            ([order], "resolved", "norepinephrine"),
            ([], "no_match", None),
            ([{**order, "subject_id": "2"}], "no_match", None),
            ([{**order, "hadm_id": "20"}], "no_match", None),
            ([{**order, "pharmacy_id": "2"}], "no_match", None),
            ([order, order], "ambiguous_match", None),
            (
                [order, {**order, "drug_type": "BASE", "drug": "Acetaminophen"}],
                "ambiguous_match",
                None,
            ),
            ([{**order, "drug": ""}], "blank_prescriptions_drug", None),
            ([{**order, "drug": "Unreviewed alias"}], "resolved", None),
            ([{**order, "drug": "Acetaminophen"}], "resolved", None),
        )
        for orders, status, canonical in cases:
            with self.subTest(status=status, orders=orders):
                self.tables["emar"] = [{**original, "medication": ""}]
                self.tables["prescriptions"] = orders
                context = self.context()
                event = context["medication_events"][0]
                self.assertEqual(event["medication"], "")
                self.assertEqual(event["pharmacy_lookup_status"], "no_match")
                self.assertEqual(event["prescriptions_lookup_status"], status)
                self.assertEqual(event["canonical_drug"], canonical)
                result, _ = evaluate(
                    self.study(), context, self.cohort["burden_v2"], {}
                )
                expected = (
                    1 if canonical else 0 if event["action"] == "exclude" else None
                )
                self.assertEqual(result[COMPONENTS[4]], expected)
                if expected is None:
                    self.assertIsNone(result["burden_score"])
                if status == "ambiguous_match":
                    self.assertEqual(event["prescriptions_match_count"], 2)
        self.tables["prescriptions"] = [order]
        for field in ("subject_id", "hadm_id", "pharmacy_id"):
            self.tables["emar"] = [{**original, "medication": "", field: ""}]
            self.assertEqual(
                self.context()["medication_events"][0]["prescriptions_lookup_status"],
                "missing_lookup_key",
            )
        self.tables["emar"] = [{**original, "medication": ""}]
        pharmacy = dict.fromkeys(TABLE_COLUMNS["pharmacy"].split(), "")
        pharmacy.update(subject_id="1", hadm_id="10", pharmacy_id="1")
        self.tables["pharmacy"] = [pharmacy]
        event = self.context()["medication_events"][0]
        self.assertEqual(event["pharmacy_lookup_status"], "blank_pharmacy_medication")
        self.assertEqual(event["medication_name_source"], "prescriptions")
        self.assertTrue(
            any(key.startswith("prescriptions:") for key in event["source_record_keys"])
        )
        self.tables["pharmacy"] = [pharmacy, pharmacy]
        event = self.context()["medication_events"][0]
        self.assertEqual(
            event["prescriptions_lookup_status"], "blocked_pharmacy_conflict"
        )
        self.assertIsNone(event["canonical_drug"])
        self.tables["pharmacy"] = [{**pharmacy, "medication": "Acetaminophen"}]
        event = self.context()["medication_events"][0]
        self.assertEqual(event["prescriptions_lookup_status"], "not_needed")
        self.assertEqual(event["resolved_medication"], "Acetaminophen")
        self.tables["emar"] = [original]
        event = self.context()["medication_events"][0]
        self.assertEqual(event["medication_name_source"], "emar")
        self.assertEqual(event["prescriptions_lookup_status"], "not_needed")
        self.tables["emar"] = []
        self.assertEqual(self.context()["medication_events"], [])
        self.tables["emar"] = [{**original, "medication": ""}]
        self.tables["pharmacy"] = []
        policy = self.cohort["burden_v2"]["prescriptions_identity_lookup"]
        policy["confirmed"] = False
        self.assertEqual(
            self.context()["medication_events"][0]["prescriptions_lookup_status"],
            "disabled",
        )
        policy["confirmed"] = True
        policy["reviewer"] = ""
        with self.assertRaisesRegex(
            ValueError, "prescriptions identity lookup requires review provenance"
        ):
            validate_rules(self.cohort, self.ehr, self.imaging)

    def test_prescriptions_publication_evidence_tail_and_failure_recovery(self) -> None:
        self.tables["emar"][0]["medication"] = ""
        self.tables["emar"][1]["pharmacy_id"] = "2"
        pharmacy = dict.fromkeys(TABLE_COLUMNS["pharmacy"].split(), "")
        pharmacy.update(subject_id="1", hadm_id="10", pharmacy_id="1")
        self.tables["pharmacy"] = [pharmacy]
        order = dict.fromkeys(TABLE_COLUMNS["prescriptions"].split(), "")
        order.update(
            subject_id="1",
            hadm_id="10",
            pharmacy_id="1",
            drug="Norepinephrine",
            drug_type="MAIN",
            route="PO",
            starttime=dt(-100),
            stoptime=dt(-99),
        )
        self.tables["prescriptions"] = [
            order,
            {**order, "subject_id": "2", "hadm_id": "20", "drug": "Unrelated drug"},
            {**order, "pharmacy_id": "2", "drug": "Named eMAR must not be overwritten"},
        ]
        self.prepare()
        metadata = self.run_audit()
        self.assertEqual(metadata["source_retained_row_counts"]["prescriptions"], 1)
        self.assertIn("prescriptions", metadata["input_versions"])
        self.assertEqual(
            metadata["csv_tail_checks"]["checked_file_count"], len(TABLE_COLUMNS) + 1
        )
        self.assertEqual(
            metadata["prescriptions_identity_lookup"]["policy"],
            self.cohort["burden_v2"]["prescriptions_identity_lookup"],
        )
        self.assertFalse(metadata["ready_for_formal_labels"])
        self.assertFalse(metadata["horizon_frozen"])
        for horizon, burden in ((48, 5), (72, 8)):
            self.assertEqual(self.wide(horizon)[100]["burden_score"], burden)
            self.assertEqual(
                self.wide(horizon)[100][COMPONENTS[4] + "_first_event_time"],
                clinical_time(dt(5)),
            )
            evidence = pq.read_table(
                self.processed
                / f"cohort/burden_v2/window_{horizon}h/event_evidence.parquet"
            ).to_pylist()
            event = next(
                row
                for row in evidence
                if row["study_id"] == 100
                and row["component"] == COMPONENTS[4]
                and row["qualifies"] == 1
            )
            self.assertIn(
                record_key(
                    "prescriptions",
                    {key: value or None for key, value in order.items()},
                ),
                json.loads(event["source_record_keys"]),
            )
            raw = json.loads(event["raw_evidence"])
            self.assertIsNone(raw["medication"])
            self.assertEqual(raw["resolved_medication"], "Norepinephrine")
            self.assertEqual(raw["prescriptions_records"][0]["route"], "PO")
            self.assertEqual(raw["details"][0]["route"], "IV")
        report = self.outputs / "audit/burden_v2/prescriptions_identity_audit.csv"
        with report.open(encoding="utf-8", newline="") as stream:
            identities = list(csv.DictReader(stream))
        self.assertEqual(len(identities), 1)
        self.assertEqual(identities[0]["original_medication"], "")
        self.assertEqual(
            identities[0]["pharmacy_lookup_status"], "blank_pharmacy_medication"
        )
        self.assertEqual(identities[0]["prescriptions_lookup_status"], "resolved")
        self.assertEqual(identities[0]["medication_name_source"], "prescriptions")
        with (self.outputs / "audit/burden_v2/dictionary_inventory.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            names = [
                row
                for row in csv.DictReader(stream)
                if row["family"] == "medications"
                and row["value_raw"] == "Norepinephrine"
            ]
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0]["record_count"], "2")
        self.assertTrue(
            (
                self.outputs / "audit/burden_v2/prescriptions_identity_summary.csv"
            ).is_file()
        )
        before = self.wide()
        self.run_audit()
        self.assertEqual(self.wide(), before)
        path = self.iv / "hosp/prescriptions.csv"
        original = path.read_bytes()
        shortened = original.rstrip(b"\r\n").rsplit(b"\n", 1)[-1].rsplit(b",", 1)[0]
        path.write_bytes(original + shortened)
        metadata = self.run_audit()
        self.assertEqual(metadata["csv_tail_checks"]["skipped_record_count"], 1)
        self.assertEqual(self.wide(), before)
        manifest = self.outputs / "audit/step_05_manifest.json"
        previous_manifest = manifest.read_bytes()
        header, body = original.split(b"\n", 1)
        path.write_bytes(header + b"\n" + shortened + b"\n" + body)
        with self.assertRaisesRegex(ValueError, "not.*(final|EOF)"):
            self.run_audit()
        self.assertEqual(manifest.read_bytes(), previous_manifest)
        self.assertEqual(self.wide(), before)
        path.write_bytes(b"subject_id,hadm_id,pharmacy_id\n1,10,1\n")
        with self.assertRaisesRegex(
            ValueError, "Missing or duplicate V2 columns in prescriptions"
        ):
            self.run_audit()
        self.assertEqual(manifest.read_bytes(), previous_manifest)
        missing = path.with_suffix(".missing")
        path.rename(missing)
        try:
            with self.assertRaisesRegex(FileNotFoundError, "Missing V2 source"):
                self.run_audit()
        finally:
            missing.rename(path)
        self.assertEqual(manifest.read_bytes(), previous_manifest)
        _verify_run(self.outputs, 5)

    def test_pharmacy_lookup_publication_evidence_tail_and_failure_recovery(
        self,
    ) -> None:
        self.tables["emar"][0]["medication"] = ""
        self.tables["emar"][1]["pharmacy_id"] = "2"
        order = dict.fromkeys(TABLE_COLUMNS["pharmacy"].split(), "")
        order.update(
            subject_id="1",
            hadm_id="10",
            pharmacy_id="1",
            medication="Norepinephrine",
            starttime=dt(-100),
            stoptime=dt(-99),
            route="PO",
            status="Inactive",
        )
        self.tables["pharmacy"] = [
            order,
            {
                **order,
                "subject_id": "2",
                "hadm_id": "20",
                "medication": "Unrelated drug",
            },
            {
                **order,
                "pharmacy_id": "2",
                "medication": "Named eMAR must not be overwritten",
            },
        ]
        self.prepare()
        metadata = self.run_audit()
        self.assertEqual(metadata["source_retained_row_counts"]["pharmacy"], 1)
        self.assertIn("pharmacy", metadata["input_versions"])
        with (self.outputs / "audit/burden_v2/dictionary_inventory.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            names = [
                row
                for row in csv.DictReader(stream)
                if row["family"] == "medications"
                and row["value_raw"] == "Norepinephrine"
            ]
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0]["record_count"], "2")
        self.assertEqual(
            metadata["csv_tail_checks"]["checked_file_count"], len(TABLE_COLUMNS) + 1
        )
        self.assertEqual(self.wide()[100]["burden_score"], 5)
        self.assertEqual(self.wide(72)[100]["burden_score"], 8)
        for horizon in (48, 72):
            self.assertEqual(
                self.wide(horizon)[100][COMPONENTS[4] + "_first_event_time"],
                clinical_time(dt(5)),
            )
            evidence = pq.read_table(
                self.processed
                / f"cohort/burden_v2/window_{horizon}h/event_evidence.parquet"
            ).to_pylist()
            event = next(
                row
                for row in evidence
                if row["study_id"] == 100
                and row["component"] == COMPONENTS[4]
                and row["qualifies"] == 1
            )
            self.assertIn(
                record_key(
                    "pharmacy", {key: value or None for key, value in order.items()}
                ),
                json.loads(event["source_record_keys"]),
            )
            raw = json.loads(event["raw_evidence"])
            self.assertIsNone(raw["medication"])
            self.assertEqual(raw["resolved_medication"], "Norepinephrine")
            self.assertEqual(raw["pharmacy_records"][0]["route"], "PO")
        with (self.outputs / "audit/burden_v2/pharmacy_identity_audit.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["original_medication"], "")
        self.assertEqual(rows[0]["pharmacy_lookup_status"], "resolved")
        self.assertEqual(rows[0]["medication_name_source"], "pharmacy")
        self.assertTrue(
            (self.outputs / "audit/burden_v2/pharmacy_identity_summary.csv").is_file()
        )
        before = self.wide()
        self.run_audit()
        self.assertEqual(self.wide(), before)
        path = self.iv / "hosp/pharmacy.csv"
        original = path.read_bytes()
        shortened = original.rstrip(b"\r\n").rsplit(b"\n", 1)[-1].rsplit(b",", 1)[0]
        path.write_bytes(original + shortened)
        metadata = self.run_audit()
        self.assertEqual(metadata["csv_tail_checks"]["skipped_record_count"], 1)
        self.assertEqual(self.wide(), before)
        manifest = self.outputs / "audit/step_05_manifest.json"
        previous_manifest = manifest.read_bytes()
        header, body = original.split(b"\n", 1)
        path.write_bytes(header + b"\n" + shortened + b"\n" + body)
        with self.assertRaisesRegex(ValueError, "not.*(final|EOF)"):
            self.run_audit()
        self.assertEqual(manifest.read_bytes(), previous_manifest)
        self.assertEqual(self.wide(), before)
        path.write_bytes(b"subject_id,hadm_id,pharmacy_id\n1,10,1\n")
        with self.assertRaisesRegex(
            ValueError, "Missing or duplicate V2 columns in pharmacy"
        ):
            self.run_audit()
        self.assertEqual(manifest.read_bytes(), previous_manifest)
        _verify_run(self.outputs, 5)

        missing = path.with_suffix(".missing")
        path.rename(missing)
        try:
            with self.assertRaisesRegex(FileNotFoundError, "Missing V2 source"):
                self.run_audit()
        finally:
            missing.rename(path)
        self.assertEqual(manifest.read_bytes(), previous_manifest)
        _verify_run(self.outputs, 5)

    def test_unknown_identity_nonadministration_preserves_history_and_conflicts(
        self,
    ) -> None:
        rules = self.cohort["burden_v2"]
        rules[IDENTITY_POLICY]["confirmed"] = True
        for name in ("Not Given", "Not Started"):
            rules["medication_event_actions"][name] = reviewed(action="exclude")
        self.tables["emar"], self.tables["emar_detail"] = [], []
        self.add_medication("m1", 1, "Started")
        for index, event in enumerate(("Not Given", "Not Started", "Started"), 2):
            self.add_medication(
                f"m{index}",
                index,
                event,
                complete_dose_not_given="1" if event == "Started" else "0",
            )
            self.tables["emar"][-1].update(medication="", pharmacy_id="")
        self.add_medication("m5", 5, "Continued")
        context = self.context()
        self.assertTrue(all(not row["reason"] for row in context["medication_events"]))
        self.assertIsNone(
            medication_identity_exclusion(self.study(), context, rules)[0]
        )
        result, _ = evaluate(self.study(), context, rules, {})
        self.assertEqual(result[COMPONENTS[4]], 1)
        self.assertEqual(
            result[f"{COMPONENTS[4]}_first_event_time"], clinical_time(dt(1))
        )
        self.assertFalse(context["medication_events"][-1]["onset"])

        self.tables["emar_detail"].append(
            {
                **self.tables["emar_detail"][1],
                "parent_field_ordinal": "1",
                "complete_dose_not_given": "1",
            }
        )
        context = self.context()
        self.assertEqual(
            context["medication_events"][1]["reason"], "conflicting_given_status"
        )
        self.assertIsNone(evaluate(self.study(), context, rules, {})[0][COMPONENTS[4]])
        self.assertIsNone(
            medication_identity_exclusion(self.study(), context, rules)[0]
        )
        self.tables["emar"][1]["charttime"] = ""
        self.assertTrue(self.context()["medication_events"][0]["reason"])

    def test_confirmed_nonadministration_review_type_preserves_support_and_conflicts(
        self,
    ) -> None:
        rules = self.cohort["burden_v2"]
        rules["administration_types"].update(
            {
                "Standard Maintenance Medication": reviewed(action="review"),
                "Other Review": reviewed(action="review"),
                "Unconfirmed Review": {**reviewed(action="review"), "confirmed": False},
            }
        )
        for status in ("Not Given", "Not Started"):
            rules["medication_event_actions"][status] = reviewed(action="exclude")
        for start in (-1, 1):
            for medication in ("", "Norepinephrine"):
                with self.subTest(start=start, medication=medication):
                    self.tables["emar"], self.tables["emar_detail"] = [], []
                    self.add_medication("m1", start, "Started")
                    for index, status in enumerate(("Not Given", "Not Started"), 2):
                        self.add_medication(
                            f"m{index}",
                            index,
                            status,
                            administration_type="Standard Maintenance Medication",
                            dose_given="0",
                            infusion_complete="1",
                            new_iv_bag_hung="1",
                        )
                        self.tables["emar"][-1].update(
                            medication=medication, pharmacy_id=""
                        )
                    self.add_medication("m4", 4, "Continued")
                    context = self.context()
                    for row in context["medication_events"][1:3]:
                        self.assertFalse(row["reason"])
                        self.assertEqual(row["action"], "exclude")
                        self.assertFalse(row["onset"])
                        self.assertEqual(row["medication"], medication)
                        self.assertTrue(row["source_record_keys"])
                    continuation = context["medication_events"][-1]
                    self.assertTrue(continuation["active_before"])
                    self.assertFalse(continuation["onset"])
                    self.assertEqual(
                        continuation["episode_start_before"], clinical_time(dt(start))
                    )
                    for horizon in (48, 72):
                        result, evidence = evaluate(
                            self.study(hours=horizon), context, rules, {}
                        )
                        self.assertEqual(result[COMPONENTS[4]], int(start > 0))
                        self.assertEqual(result["label_status"], "complete")
                        self.assertEqual(
                            result["burden_score"],
                            (5 if horizon == 48 else 8) - (2 if start < 0 else 0),
                        )
                        self.assertTrue(
                            all(
                                row["qualifies"] == 0
                                for row in evidence
                                if row["source_record_key"]
                                in {
                                    value["source_record_key"]
                                    for value in context["medication_events"][1:3]
                                }
                            )
                        )

        for status in ("Not Given", "Not Started"):
            for details, changes, extra, expected in (
                ({"dose_given": "1"}, {}, {}, "conflicting_not_given_dose"),
                ({"dose_given": "unknown"}, {}, {}, "invalid_not_given_dose"),
                ({"dose_given": "-1"}, {}, {}, "invalid_not_given_dose"),
                ({"dose_given": "NaN"}, {}, {}, "invalid_not_given_dose"),
                ({"administration_type": ""}, {}, {}, "unresolved_administration_type"),
                (
                    {"administration_type": "Unconfirmed Review"},
                    {},
                    {},
                    "unresolved_administration_type",
                ),
                (
                    {},
                    {},
                    {"administration_type": "Other Review"},
                    "unresolved_administration_type",
                ),
                (
                    {},
                    {},
                    {"administration_type": "IV Infusion"},
                    "unresolved_administration_type",
                ),
                ({}, {}, {"complete_dose_not_given": "1"}, "conflicting_given_status"),
                ({"route": "PO"}, {}, {}, "route_conflict_or_unconfirmed"),
                (
                    {"prior_infusion_rate": "invalid"},
                    {},
                    {},
                    "invalid_prior_infusion_rate",
                ),
                ({}, {"charttime": ""}, {}, "invalid_medication_time"),
                ({"emar_seq": "wrong"}, {}, {}, "emar_detail_key_mismatch"),
            ):
                with self.subTest(
                    status=status, details=details, changes=changes, extra=extra
                ):
                    self.tables["emar"], self.tables["emar_detail"] = [], []
                    base_detail = {
                        "administration_type": "Standard Maintenance Medication",
                        "dose_given": "0",
                        **details,
                    }
                    self.add_medication("m1", 1, status, **base_detail)
                    self.tables["emar"][-1].update(changes)
                    if extra:
                        self.tables["emar_detail"].append(
                            {
                                **self.tables["emar_detail"][0],
                                "parent_field_ordinal": "1",
                                **extra,
                            }
                        )
                    context = self.context()
                    self.assertEqual(
                        context["medication_events"][0]["reason"], expected
                    )
                    result, evidence = evaluate(self.study(), context, rules, {})
                    self.assertIsNone(result[COMPONENTS[4]])
                    self.assertIsNone(result["burden_score"])
                    self.assertTrue(any(row["qualifies"] is None for row in evidence))
        self.tables["emar"][0].update(charttime=dt(-11))
        self.tables["emar_detail"][0].update(emar_seq="1")
        self.assertEqual(
            self.context()["medication_events"][0]["reason"], "event_outside_admission"
        )

    def test_nonadministration_review_type_publication_and_failure_recovery(
        self,
    ) -> None:
        rules = self.cohort["burden_v2"]
        rules["administration_types"]["Standard Maintenance Medication"] = reviewed(
            action="review"
        )
        rules["medication_event_actions"]["Not Given"] = reviewed(action="exclude")
        self.add_medication(
            "m3",
            0.5,
            "Not Given",
            administration_type="Standard Maintenance Medication",
            dose_given="",
        )
        self.tables["emar"][-1].update(medication="", pharmacy_id="")
        self.prepare()
        metadata = self.run_audit()
        self.assertTrue(metadata["audit_checks_passed"])
        self.assertEqual(metadata["unresolved_evidence_count"], 0)
        self.assertFalse(metadata["audit_ready_for_window_decision"])
        self.assertFalse(metadata["ready_for_formal_labels"])
        for horizon in (48, 72):
            wide = self.wide(horizon)
            self.assertTrue(
                all(row["candidate_status"] == "eligible" for row in wide.values())
            )
            self.assertTrue(
                all(row["label_status"] == "complete" for row in wide.values())
            )
            self.assertEqual(wide[100]["burden_score"], 5 if horizon == 48 else 8)
            self.assertEqual(wide[101][COMPONENTS[4]], 0)
            records = pq.read_table(
                self.processed
                / f"cohort/burden_v2/window_{horizon}h/event_evidence.parquet"
            ).to_pylist()
            record = next(
                row
                for row in records
                if row["study_id"] == 100
                and json.loads(row["raw_evidence"]).get("emar_id") == "m3"
            )
            self.assertEqual(record["qualifies"], 0)
            self.assertEqual(
                record["exclusion_reason"], "excluded_medication_or_administration"
            )
            raw = json.loads(record["raw_evidence"])
            self.assertEqual(raw["resolved_medication"], "")
            self.assertEqual(
                raw["details"][0]["administration_type"],
                "Standard Maintenance Medication",
            )
        before = self.wide()
        self.run_audit()
        self.assertEqual(self.wide(), before)
        manifest = self.outputs / "audit/step_05_manifest.json"
        previous_manifest = manifest.read_bytes()
        previous_hashes = {
            file: _file_hash(Path(file)) for file in _json_file(manifest)["files"]
        }
        original_replace, failed = os.replace, False

        def fail_once(source, destination):
            nonlocal failed
            if not failed and str(destination).endswith("study_event_audit.parquet"):
                failed = True
                raise OSError("nonadministration publication failure")
            return original_replace(source, destination)

        with patch("src.cohort.common.os.replace", side_effect=fail_once):
            with self.assertRaisesRegex(
                OSError, "nonadministration publication failure"
            ):
                self.run_audit()
        self.assertTrue(failed)
        self.assertEqual(manifest.read_bytes(), previous_manifest)
        self.assertEqual(
            {file: _file_hash(Path(file)) for file in previous_hashes}, previous_hashes
        )
        _verify_run(self.outputs, 5)

    def test_identity_exclusion_scopes_existing_baseline_and_window_evidence(
        self,
    ) -> None:
        rules = self.cohort["burden_v2"]
        rules[IDENTITY_POLICY]["confirmed"] = True
        original_emar = copy.deepcopy(self.tables["emar"])
        original_details = copy.deepcopy(self.tables["emar_detail"])
        for time, changes, detail, expected in (
            (-1, {}, {}, True),
            (0, {}, {}, True),
            (60, {}, {}, True),
            (70, {}, {}, True),
            (71, {}, {}, False),
            (60, {"hadm_id": "20"}, {}, False),
            (3, {}, {"administration_type": "IV Bolus"}, False),
            (3, {"medication": "Acetaminophen"}, {}, False),
            (3, {"medication": "Unreviewed exact name"}, {}, True),
        ):
            with self.subTest(time=time, changes=changes, detail=detail):
                self.tables["emar"] = copy.deepcopy(original_emar)
                self.tables["emar_detail"] = copy.deepcopy(original_details)
                self.add_medication("m3", time, "Started", **detail)
                self.tables["emar"][-1].update(
                    {"medication": "", "pharmacy_id": "", **changes}
                )
                exclusion, triggers = medication_identity_exclusion(
                    self.study(), self.context(), rules
                )
                self.assertEqual(exclusion is not None, expected)
                if exclusion:
                    self.assertTrue(triggers)
                    self.assertTrue(all(row["qualifies"] is None for row in triggers))
                    self.assertEqual(
                        exclusion["reviewer"], rules[IDENTITY_POLICY]["reviewer"]
                    )
        for state in ("excluded", "needs_review"):
            self.assertIsNone(
                medication_identity_exclusion(
                    {**self.study(), "candidate_status": state}, self.context(), rules
                )[0]
            )
        rules[IDENTITY_POLICY]["confirmed"] = False
        self.assertIsNone(
            medication_identity_exclusion(self.study(), self.context(), rules)[0]
        )
        rules[IDENTITY_POLICY]["confirmed"] = True
        rules[IDENTITY_POLICY]["reviewer"] = " "
        with self.assertRaisesRegex(ValueError, "review provenance"):
            validate_rules(self.cohort, self.ehr, self.imaging)

    def test_identity_exclusion_publication_union_provenance_and_recovery(self) -> None:
        rules = self.cohort["burden_v2"]
        rules[IDENTITY_POLICY]["confirmed"] = True
        self.add_medication("m3", 60, "Started")
        self.tables["emar"][-1].update(medication="", pharmacy_id="40")
        order = dict.fromkeys(TABLE_COLUMNS["prescriptions"].split(), "")
        for name in ("Multivitamin IV", "Trace Elements"):
            rules["medications"][name.casefold()] = reviewed(canonical_drug=None)
            self.tables["prescriptions"].append(
                {
                    **order,
                    "subject_id": "1",
                    "hadm_id": "10",
                    "pharmacy_id": "40",
                    "drug": name,
                }
            )
        self.add_medication("m4", 1, "Continued")
        self.tables["emar"][-1].update(subject_id="2", hadm_id="20")
        self.tables["emar_detail"][-1]["subject_id"] = "2"
        self.prepare()
        metadata = self.run_audit()
        self.assertEqual(metadata["candidate_exclusion_total_count"], 2)
        policy_metadata = metadata["candidate_medication_identity_exclusions"]
        self.assertEqual(policy_metadata["policy"], rules[IDENTITY_POLICY])
        self.assertEqual(
            policy_metadata["summary"],
            dict(
                study_count=2, admission_count=1, patient_count=1, emar_record_count=1
            ),
        )
        for horizon in (48, 72):
            wide = self.wide(horizon)
            for study in (100, 101):
                self.assertEqual(wide[study]["candidate_status"], "excluded")
                self.assertEqual(
                    wide[study]["candidate_exclusion_version"],
                    rules[IDENTITY_POLICY]["version"],
                )
                self.assertEqual(wide[study]["label_status"], "excluded")
                self.assertTrue(all(wide[study][name] is None for name in COMPONENTS))
                self.assertTrue(
                    all(
                        wide[study][f"{name}_first_event_time"] is None
                        for name in COMPONENTS
                    )
                )
                for field in ("burden_score", "phys_death", "high_burden"):
                    self.assertIsNone(wide[study][field])
            self.assertEqual(wide[200]["candidate_status"], "eligible")
            self.assertEqual(wide[200]["label_status"], "unresolved")
            self.assertIsNone(wide[200]["burden_score"])
        report = self.outputs / "audit/burden_v2/medication_identity_exclusions.csv"
        with report.open(encoding="utf-8", newline="") as stream:
            triggers = list(csv.DictReader(stream))
        self.assertEqual(
            [(int(row["study_id"]), int(row["horizon_hours"])) for row in triggers],
            [(100, 72), (101, 48), (101, 72)],
        )
        self.assertTrue(
            all(
                row["prescriptions_lookup_status"] == "ambiguous_match"
                for row in triggers
            )
        )
        self.assertTrue(all(row["qualifies"] == "" for row in triggers))
        candidate_evidence = pq.read_table(
            self.processed / "cohort/burden_v2/candidate_evidence.parquet"
        ).to_pylist()
        for trigger in triggers:
            keys = {
                row["source_record_key"]
                for row in candidate_evidence
                if row["sample_id"] == trigger["sample_id"]
            }
            self.assertTrue(set(json.loads(trigger["source_record_keys"])) <= keys)
        self.assertFalse(metadata["ready_for_formal_labels"])
        previous_report = report.read_bytes()
        self.run_audit()
        self.assertEqual(report.read_bytes(), previous_report)
        manifest = self.outputs / "audit/step_05_manifest.json"
        previous_manifest = manifest.read_bytes()
        hashes = {
            path: _file_hash(Path(path)) for path in _json_file(manifest)["files"]
        }
        self.assertIn(str(report), hashes)
        original_replace, failed = os.replace, False

        def fail_once(source, destination):
            nonlocal failed
            if not failed and str(destination) == str(report):
                failed = True
                raise OSError("identity exclusion publication failure")
            return original_replace(source, destination)

        with patch("src.cohort.common.os.replace", side_effect=fail_once):
            with self.assertRaisesRegex(
                OSError, "identity exclusion publication failure"
            ):
                self.run_audit()
        self.assertTrue(failed)
        self.assertEqual(manifest.read_bytes(), previous_manifest)
        self.assertEqual({path: _file_hash(Path(path)) for path in hashes}, hashes)
        _verify_run(self.outputs, 5)

    def test_reviewed_conflict_policy_excludes_only_matching_t0_studies(self) -> None:
        self.tables["icustays"][0].update(intime=dt(-1), outtime=dt(0.25))
        rules = self.cohort["burden_v2"]
        policy = dict(
            action="exclude",
            confirmed=True,
            reviewer="synthetic_reviewer",
            reason="Conflicting care-unit sources at t0",
            rationale="Reviewed non-ICU transfer versus active icustay exclusion.",
            version="conflict_policy_v1",
        )
        rules["candidate_conflict_policies"] = {CONFLICT_POLICY: policy}
        self.prepare()
        metadata = self.run_audit()
        self.assertEqual(metadata["candidate_policy_exclusion_count"], 1)
        self.assertEqual(metadata["candidate_exclusion_count"], 0)
        self.assertEqual(metadata["candidate_exclusion_total_count"], 1)
        self.assertEqual(
            metadata["candidate_conflict_policies"][CONFLICT_POLICY], policy
        )
        for horizon in (48, 72):
            records = self.wide(horizon)
            self.assertEqual(records[100]["candidate_status"], "excluded")
            self.assertEqual(records[100]["exclusion_reason"], policy["reason"])
            self.assertEqual(
                records[100]["candidate_exclusion_reviewer"], policy["reviewer"]
            )
            self.assertEqual(
                records[100]["candidate_exclusion_version"], policy["version"]
            )
            self.assertTrue(all(records[100][name] is None for name in COMPONENTS))
            self.assertIsNone(records[100]["burden_score"])
            self.assertEqual(records[101]["candidate_status"], "eligible")
            self.assertEqual(records[200]["candidate_status"], "eligible")
        evidence = pq.read_table(
            self.processed / "cohort/burden_v2/candidate_evidence.parquet"
        ).to_pylist()
        raw = json.loads(
            next(
                row["raw_evidence"]
                for row in evidence
                if row["source_file"] == str(self.config / "cohort.json")
            )
        )
        self.assertEqual(raw["policy_name"], CONFLICT_POLICY)
        self.assertEqual(raw["automated_candidate_status"], "needs_review")
        self.assertEqual(
            raw["automated_exclusion_reason"], "transfers_icustays_conflict"
        )
        cited = set(json.loads(raw["source_record_keys"]))
        self.assertEqual(len(cited), 2)
        self.assertTrue(
            cited
            <= {row["source_record_key"] for row in evidence if row["study_id"] == 100}
        )

        original = {
            **self.wide()[100],
            "candidate_status": "needs_review",
            "exclusion_reason": "transfers_icustays_conflict",
        }
        context = self.context()
        policy["confirmed"] = False
        self.assertIsNone(candidate_conflict_exclusion(original, context, rules))
        policy["confirmed"] = True
        for fields in (
            {"care_level_at_t0": "unknown"},
            {"care_level_at_t0": "icu_ccu"},
            {"careunit_at_t0": ""},
            {"exclusion_reason": "care_interval_conflict"},
            {"t0": clinical_time(dt(13))},
        ):
            self.assertIsNone(
                candidate_conflict_exclusion({**original, **fields}, context, rules)
            )
        policy["reviewer"] = " "
        with self.assertRaisesRegex(ValueError, "review provenance"):
            validate_rules(self.cohort, self.ehr, self.imaging)
        policy["reviewer"] = "synthetic_reviewer"
        policy["action"] = "include"
        with self.assertRaisesRegex(ValueError, "action=exclude"):
            validate_rules(self.cohort, self.ehr, self.imaging)

    def test_reviewed_preadmission_exclusion_cites_medication_context_only(
        self,
    ) -> None:
        self.add_medication("m3", -11, "Administered")
        self.cohort["burden_v2"]["medication_event_actions"]["Administered"] = reviewed(
            action="given"
        )
        self.tables["emar"][-1].update(medication="", pharmacy_id="40")
        for table, name_field in (
            ("pharmacy", "medication"),
            ("prescriptions", "drug"),
        ):
            value = dict.fromkeys(TABLE_COLUMNS[table].split(), "")
            value.update(subject_id="1", hadm_id="10", pharmacy_id="40")
            value[name_field] = ""
            self.tables[table].append(value)
        self.add_medication("m4", 1, "Started")
        self.tables["emar"][-1].update(subject_id="2", hadm_id="20")
        self.tables["emar_detail"][-1]["subject_id"] = "2"
        self.prepare()
        self.run_audit()
        evidence = pq.read_table(
            self.processed / "cohort/burden_v2/window_48h/event_evidence.parquet"
        ).to_pylist()
        source = next(
            row
            for row in evidence
            if row["study_id"] == 100
            and row["exclusion_reason"] == "event_outside_admission"
        )
        self.assertIsNone(source["qualifies"])
        candidate_evidence = pq.read_table(
            self.processed / "cohort/burden_v2/candidate_evidence.parquet"
        ).to_pylist()
        admission_key = next(
            row["source_record_key"]
            for row in candidate_evidence
            if row["study_id"] == 100
            and row["source_record_key"].startswith("admissions:")
        )
        keys = sorted({admission_key, *json.loads(source["source_record_keys"])})
        self.assertTrue(
            all(
                any(key.startswith(table + ":") for key in keys)
                for table in ("emar", "emar_detail", "pharmacy", "prescriptions")
            )
        )
        exclusion = dict(
            sample_id=source["sample_id"],
            reason="Pre-admission medication record with unresolved identity",
            source_record_keys=json.dumps(keys),
            reviewer="synthetic_reviewer",
            rationale="Reviewed exclusion of this named study only; preserve timestamp and identity uncertainties.",
            version="candidate_exclusions_v4",
        )
        path = self.root / "candidate_exclusions_v4.csv"

        def write_decision(decision):
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(exclusion))
                writer.writeheader()
                writer.writerow(decision)

        write_decision(exclusion)
        self.cohort["burden_v2"]["candidate_exclusions_path"] = str(path)
        (self.config / "cohort.json").write_text(
            json.dumps(self.cohort), encoding="utf-8"
        )
        metadata = self.run_audit()
        self.assertEqual(metadata["candidate_exclusion_count"], 1)
        self.assertEqual(metadata["candidate_exclusions_sha256"], _file_hash(path))
        for horizon in (48, 72):
            records = self.wide(horizon)
            self.assertEqual(records[100]["candidate_status"], "excluded")
            self.assertEqual(records[100]["exclusion_reason"], exclusion["reason"])
            self.assertTrue(all(records[100][field] is None for field in COMPONENTS))
            self.assertIsNone(records[100]["burden_score"])
            self.assertEqual(records[101]["candidate_status"], "eligible")
            self.assertEqual(records[101]["label_status"], "unresolved")
            self.assertEqual(records[200]["candidate_status"], "eligible")
        preserved = pq.read_table(
            self.processed / "cohort/burden_v2/candidate_evidence.parquet"
        ).to_pylist()
        kept = {
            row["source_record_key"]: row for row in preserved if row["study_id"] == 100
        }
        self.assertTrue(set(keys) <= set(kept))
        raw = json.loads(kept[source["source_record_key"]]["raw_evidence"])
        self.assertEqual(raw["reason"], "event_outside_admission")
        self.assertEqual(raw["charttime"], dt(-11))
        self.assertFalse(raw["medication_mapping_confirmed"])
        manual = next(row for row in preserved if row["source_file"] == str(path))
        self.assertEqual(
            json.loads(manual["raw_evidence"])["automated_candidate_status"], "eligible"
        )
        previous = self.wide()
        self.run_audit()
        self.assertEqual(self.wide(), previous)
        manifest = self.outputs / "audit/step_05_manifest.json"
        previous_hashes = {
            file: _file_hash(Path(file)) for file in _json_file(manifest)["files"]
        }
        previous_manifest = manifest.read_bytes()
        foreign = next(
            row["source_record_key"]
            for row in evidence
            if row["study_id"] == 200 and row["component"] == COMPONENTS[4]
        )
        for unavailable in (foreign, "emar:missing"):
            write_decision(
                {**exclusion, "source_record_keys": json.dumps(keys + [unavailable])}
            )
            with self.assertRaisesRegex(ValueError, "unavailable source keys"):
                self.run_audit()
            self.assertEqual(manifest.read_bytes(), previous_manifest)
            self.assertEqual(
                {file: _file_hash(Path(file)) for file in previous_hashes},
                previous_hashes,
            )
            _verify_run(self.outputs, 5)

    def test_candidate_exclusion_is_study_specific_and_preserves_provenance(
        self,
    ) -> None:
        self.tables["transfers"][1]["intime"] = dt(0.5)
        source = record_key("transfers", self.tables["transfers"][1])
        exclusion = dict(
            sample_id="p1_h10_s100",
            reason="Unable to verify care unit at t0",
            source_record_keys=json.dumps([source]),
            reviewer="synthetic_reviewer",
            rationale="The first linked ward interval starts after this study.",
            version="candidate_exclusions_v1",
        )
        path = self.root / "candidate_exclusions_v1.csv"

        def write_rows(values):
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(exclusion))
                writer.writeheader()
                writer.writerows(values)

        write_rows([exclusion])
        for invalid in (
            {**exclusion, "sample_id": "unknown"},
            {**exclusion, "reviewer": " "},
            {**exclusion, "source_record_keys": "[]"},
            {**exclusion, "source_record_keys": "[null]"},
            {**exclusion, "source_record_keys": "{}"},
        ):
            write_rows([invalid])
            with self.assertRaises(ValueError):
                load_candidate_exclusions(path, {exclusion["sample_id"]})
        write_rows([exclusion, exclusion])
        with self.assertRaises(ValueError):
            load_candidate_exclusions(path, {exclusion["sample_id"]})
        write_rows([exclusion])
        self.cohort["burden_v2"]["candidate_exclusions_path"] = str(path)
        self.prepare()
        metadata = self.run_audit()
        self.assertEqual(metadata["candidate_exclusion_count"], 1)
        self.assertEqual(metadata["candidate_exclusions_sha256"], _file_hash(path))
        for horizon in (48, 72):
            records = self.wide(horizon)
            excluded = records[100]
            self.assertEqual(excluded["candidate_status"], "excluded")
            self.assertEqual(excluded["exclusion_reason"], exclusion["reason"])
            self.assertEqual(excluded["review_status"], "reviewed")
            self.assertEqual(
                excluded["candidate_exclusion_reviewer"], exclusion["reviewer"]
            )
            self.assertEqual(
                excluded["candidate_exclusion_version"], exclusion["version"]
            )
            self.assertIsNone(excluded["burden_score"])
            self.assertTrue(all(excluded[name] is None for name in COMPONENTS))
            self.assertEqual(records[101]["candidate_status"], "eligible")
            self.assertEqual(records[200]["candidate_status"], "eligible")
        with (self.outputs / "audit/burden_v2/excluded_studies.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            reported = next(
                row for row in csv.DictReader(stream) if row["study_id"] == "100"
            )
            self.assertEqual(reported["exclusion_reason"], exclusion["reason"])
            self.assertEqual(
                reported["candidate_exclusion_rationale"], exclusion["rationale"]
            )
        evidence = pq.read_table(
            self.processed / "cohort/burden_v2/candidate_evidence.parquet"
        ).to_pylist()
        manual = next(row for row in evidence if row["source_file"] == str(path))
        raw = json.loads(manual["raw_evidence"])
        self.assertEqual(raw["automated_candidate_status"], "needs_review")
        self.assertIn("unresolved_careunit_at_t0", raw["automated_exclusion_reason"])
        self.assertEqual(raw["automated_review_status"], "pending")
        self.assertEqual(raw["source_record_keys"], exclusion["source_record_keys"])
        self.assertTrue(
            any(
                row["study_id"] == 100 and row["source_record_key"] == source
                for row in evidence
            )
        )
        with (self.outputs / "audit/step_05_csv_tail_checks.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            check = next(
                row for row in csv.DictReader(stream) if row["file"] == str(path)
            )
            self.assertEqual(check["status"], "complete")
            self.assertEqual(check["skipped_rows"], "0")
        manifest = self.outputs / "audit/step_05_manifest.json"
        before = self.wide()
        self.run_audit()
        self.assertEqual(self.wide(), before)
        prior_manifest = _file_hash(manifest)
        write_rows([{**exclusion, "source_record_keys": '["transfers:missing"]'}])
        with self.assertRaisesRegex(ValueError, "unavailable source keys"):
            self.run_audit()
        self.assertEqual(_file_hash(manifest), prior_manifest)
        self.assertEqual(self.wide(), before)
        _verify_run(self.outputs, 5)

    def test_versioned_override_preserves_automated_evidence(self) -> None:
        self.tables["chartevents"][0]["warning"] = "1"
        self.prepare()
        self.run_audit()
        evidence_path = (
            self.processed / "cohort/burden_v2/window_48h/event_evidence.parquet"
        )
        source = next(
            row
            for row in pq.read_table(evidence_path).to_pylist()
            if row["study_id"] == 100 and row["component"] == COMPONENTS[3]
        )
        override = dict(
            sample_id="p1_h10_s100",
            horizon_hours="48",
            component=COMPONENTS[3],
            event_flag="1",
            first_event_time=dt(3),
            source_record_keys=json.dumps([source["source_record_key"]]),
            reviewer="synthetic_reviewer",
            rationale="artifact reviewed with original evidence",
            version="review_v1",
        )
        path = self.root / "overrides.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(override))
            writer.writeheader()
            writer.writerow(override)
        self.cohort["burden_v2"]["overrides_path"] = str(path)
        (self.config / "cohort.json").write_text(json.dumps(self.cohort))
        metadata = self.run_audit()
        self.assertEqual(metadata["override_count"], 1)
        self.assertEqual(self.wide()[100]["burden_score"], 5)
        self.assertIsNone(self.wide(72)[100]["burden_score"])
        original = next(
            row
            for row in pq.read_table(evidence_path).to_pylist()
            if row["source_record_key"] == source["source_record_key"]
        )
        self.assertIsNone(original["automated_qualifies"])
        self.assertEqual(original["review_status"], "reviewed")

    def test_source_record_exclusions_keep_positives_and_future_conflicts(self) -> None:
        template = self.tables["chartevents"][0]
        self.tables["chartevents"] += [
            {**template, "charttime": dt(1.99), "value": "94", "valuenum": "94"},
            {
                **template,
                "charttime": dt(5),
                "value": "90",
                "valuenum": "90",
                "warning": "1",
            },
            {**template, "charttime": dt(7), "itemid": "223848", "value": "BiPAP/CPAP"},
        ]
        rules = self.cohort["burden_v2"]
        context = self.context()
        _, original = evaluate(self.study(), context, rules, {})
        overrides = {}
        for horizon in (48, 72):
            _, horizon_original = evaluate(
                self.study(hours=horizon), context, rules, {}
            )
            for component in COMPONENTS[2:4]:
                keys = [
                    r["source_record_key"]
                    for r in original
                    if r["component"] == component and r["qualifies"] is None
                ]
                overrides["p1_h10_s100", horizon, component] = dict(
                    event_flag="0",
                    first_event_time="",
                    source_record_keys=json.dumps(keys),
                    reviewer="synthetic_reviewer",
                    rationale="Exclude only reviewed source records.",
                    version="record_review_v1",
                    override_scope="source_record",
                )
            wide, records = evaluate(
                self.study(hours=horizon), context, rules, overrides
            )
            self.assertEqual(wide["burden_score"], 5 if horizon == 48 else 8)
            for component, onset in ((COMPONENTS[2], 4), (COMPONENTS[3], 3)):
                self.assertEqual(wide[component], 1)
                self.assertEqual(
                    wide[f"{component}_first_event_time"], clinical_time(dt(onset))
                )
            originals = {
                (r["component"], r["source_record_key"]): r for r in horizon_original
            }
            for row in records:
                prior = originals.get((row["component"], row["source_record_key"]))
                if prior:
                    self.assertEqual(row["raw_evidence"], prior["raw_evidence"])
                    self.assertEqual(
                        row["automated_qualifies"], prior["automated_qualifies"]
                    )
                    self.assertEqual(
                        row["automated_exclusion_reason"],
                        prior["automated_exclusion_reason"],
                    )
                    if prior["qualifies"] == 1:
                        self.assertEqual(row, {**prior, "horizon_hours": horizon})
        override = overrides["p1_h10_s100", 48, COMPONENTS[3]]
        for invalid in (
            {**override, "override_scope": "unknown"},
            {**override, "event_flag": "1"},
            {**override, "first_event_time": dt(5)},
            {**override, "baseline_status": "active_documented"},
            {**override, "source_record_keys": '["chartevents:missing"]'},
            {
                **override,
                "source_record_keys": json.dumps(
                    [
                        next(
                            r["source_record_key"]
                            for r in original
                            if r["component"] == COMPONENTS[2]
                        )
                    ]
                ),
            },
        ):
            with self.assertRaises(ValueError):
                evaluate(
                    self.study(),
                    context,
                    rules,
                    {("p1_h10_s100", 48, COMPONENTS[3]): invalid},
                )
        targeted = [
            copy.deepcopy(r) for r in original if r["component"] == COMPONENTS[3]
        ]
        for field, foreign in (("subject_id", "2"), ("hadm_id", "20")):
            values = copy.deepcopy(targeted)
            row = next(r for r in values if r["qualifies"] is None)
            raw = json.loads(row["raw_evidence"])
            raw[field] = foreign
            row["raw_evidence"] = json.dumps(raw)
            with self.assertRaisesRegex(ValueError, "another patient or admission"):
                apply_override(self.study(), COMPONENTS[3], values, override)
        episode = next(
            r
            for r in original
            if r["component"] == COMPONENTS[2] and r["qualifies"] == 1
        )
        with self.assertRaisesRegex(ValueError, "exact evidence row keys"):
            apply_override(
                self.study(),
                COMPONENTS[2],
                [copy.deepcopy(episode)],
                {**override, "source_record_keys": episode["source_record_keys"]},
            )
        self.tables["chartevents"].append(
            {**template, "charttime": dt(9), "warning": "1"}
        )
        unresolved, evidence_rows = evaluate(
            self.study(), self.context(), rules, overrides
        )
        self.assertIsNone(unresolved[COMPONENTS[3]])
        self.assertIsNone(unresolved["burden_score"])
        self.assertTrue(
            any(
                r["qualifies"] is None and r["event_time"] == clinical_time(dt(9))
                for r in evidence_rows
            )
        )
        self.tables["chartevents"].pop()
        self.tables["chartevents"][0].update(value="96", valuenum="96")
        no_hypoxemia, _ = evaluate(self.study(), self.context(), rules, overrides)
        self.assertEqual(no_hypoxemia[COMPONENTS[3]], 0)
        self.assertIsNone(no_hypoxemia[f"{COMPONENTS[3]}_first_event_time"])

    def test_record_review_publication_and_chart_exclusion_recovery(self) -> None:
        warning = {**self.tables["chartevents"][0], "charttime": dt(5), "warning": "1"}
        self.tables["chartevents"].append(warning)
        key = record_key("chartevents", warning)
        review = dict(
            sample_id="p1_h10_s100",
            horizon_hours="48",
            component=COMPONENTS[3],
            event_flag="0",
            first_event_time="",
            source_record_keys=json.dumps([key]),
            reviewer="synthetic_reviewer",
            rationale="Exclude only this warning record.",
            version="record_review_v1",
            override_scope="source_record",
        )
        override_path = self.root / "record_reviews.csv"
        exclusion = dict(
            sample_id="p1_h10_s101",
            reason="Reviewed chart-source study exclusion",
            source_record_keys=json.dumps([key]),
            reviewer="synthetic_reviewer",
            rationale="Exclude this study only; retain the original chart record.",
            version="exclusions_v5",
        )
        exclusion_path = self.root / "exclusions_v5.csv"

        def write_review(path, value):
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(value))
                writer.writeheader()
                writer.writerow(value)

        write_review(override_path, review)
        write_review(exclusion_path, exclusion)
        self.cohort["burden_v2"].update(
            overrides_path=str(override_path),
            candidate_exclusions_path=str(exclusion_path),
        )
        self.prepare()
        metadata = self.run_audit()
        self.assertEqual(metadata["override_sha256"], _file_hash(override_path))
        self.assertEqual(
            metadata["candidate_exclusions_sha256"], _file_hash(exclusion_path)
        )
        self.assertEqual(metadata["override_count"], 1)
        self.assertEqual(self.wide()[100]["burden_score"], 5)
        self.assertIsNone(self.wide(72)[100]["burden_score"])
        for horizon in (48, 72):
            excluded = self.wide(horizon)[101]
            self.assertEqual(excluded["candidate_status"], "excluded")
            self.assertTrue(all(excluded[field] is None for field in COMPONENTS))
            self.assertIsNone(excluded["burden_score"])
        preserved = pq.read_table(
            self.processed / "cohort/burden_v2/candidate_evidence.parquet"
        ).to_pylist()
        raw = next(
            json.loads(r["raw_evidence"])
            for r in preserved
            if r["study_id"] == 101 and r["source_record_key"] == key
        )
        self.assertEqual(raw["warning"], "1")
        self.assertEqual(raw["charttime"], dt(5))
        evidence = pq.read_table(
            self.processed / "cohort/burden_v2/window_48h/event_evidence.parquet"
        ).to_pylist()
        reviewed_record = next(
            r
            for r in evidence
            if r["study_id"] == 100 and r["source_record_key"] == key
        )
        self.assertEqual(reviewed_record["qualifies"], 0)
        self.assertIsNone(reviewed_record["automated_qualifies"])
        self.assertEqual(
            reviewed_record["automated_exclusion_reason"], "spo2_artifact_review"
        )
        self.run_audit()
        expected = self.wide()
        manifest = self.outputs / "audit/step_05_manifest.json"
        hashes = {f: _file_hash(Path(f)) for f in _json_file(manifest)["files"]}
        manifest_bytes = manifest.read_bytes()
        for path, invalid in (
            (override_path, {**review, "override_scope": "unknown"}),
            (
                override_path,
                {**review, "source_record_keys": '["chartevents:missing"]'},
            ),
            (
                exclusion_path,
                {**exclusion, "source_record_keys": '["chartevents:missing"]'},
            ),
            (exclusion_path, {**exclusion, "sample_id": "p2_h20_s200"}),
        ):
            write_review(path, invalid)
            with self.assertRaises(ValueError):
                self.run_audit()
            self.assertEqual(manifest.read_bytes(), manifest_bytes)
            self.assertEqual({f: _file_hash(Path(f)) for f in hashes}, hashes)
            self.assertEqual(self.wide(), expected)
            _verify_run(self.outputs, 5)
            write_review(override_path, review)
            write_review(exclusion_path, exclusion)
        original_replace, failed = os.replace, False

        def fail_once(source, destination):
            nonlocal failed
            if not failed and str(destination).endswith("study_event_audit.parquet"):
                failed = True
                raise OSError("synthetic publish failure")
            return original_replace(source, destination)

        with patch("src.cohort.common.os.replace", side_effect=fail_once):
            with self.assertRaises(OSError):
                self.run_audit()
        self.assertTrue(failed)
        self.assertEqual(manifest.read_bytes(), manifest_bytes)
        self.assertEqual({f: _file_hash(Path(f)) for f in hashes}, hashes)
        _verify_run(self.outputs, 5)

    def test_helpers_reject_timezone_and_score_invariants(self) -> None:
        self.assertIsNone(clinical_time("2120-01-01"))
        self.assertIsNone(clinical_time("2120-01-01T10:00:00+08:00"))
        self.assertEqual(score([1] * 6)["burden_score"], 8)
        self.assertEqual(score([1, 1, 1, 1, 0, 0])["burden_score"], 3)
        self.assertIsNone(score([1, 0, None, 0, 0, 0])["burden_score"])
        with self.assertRaises(ValueError):
            score([0, 0, 0, 0, 0, 2])

    def test_score_all_binary_combinations_and_pending_components(self) -> None:
        for flags in product((0, 1), repeat=6):
            result = score(list(flags))
            respiratory = max(flags[2:4])
            expected = max(flags[:2]) + 2 * respiratory + 2 * flags[4] + 3 * flags[5]
            self.assertEqual(result["burden_score"], expected)
            self.assertEqual(
                result["phys_death"], int(bool(respiratory or flags[4] or flags[5]))
            )
            self.assertEqual(result["high_burden"], int(expected >= 3))
        for index in range(6):
            pending: list[int | None] = [0] * 6
            pending[index] = None
            self.assertIsNone(score(pending)["burden_score"])

    def test_closed_parquet_sink_rejects_nonempty_writes(self) -> None:
        path = self.root / "closed.parquet"
        sink = ParquetSink(path, ["sample_id"])
        sink.close()
        with self.assertRaisesRegex(ValueError, "closed Parquet sink"):
            sink.write([{"sample_id": "synthetic"}])
        self.assertEqual(pq.read_table(path).num_rows, 0)

    def test_death_without_icu_missing_outtime_and_current_icu(self) -> None:
        self.tables["admissions"][1].update(
            dischtime=dt(70),
            deathtime=dt(60),
            hospital_expire_flag="1",
            discharge_location="DIED",
        )
        self.tables["transfers"][-1]["outtime"] = dt(70)
        self.tables["transfers"][1]["outtime"] = ""
        self.matches[-1]["dischtime"] = dt(70)
        self.prepare()
        self.run_audit()
        self.assertEqual(self.wide()[200]["burden_score"], 0)
        self.assertEqual(self.wide(72)[200]["burden_score"], 3)
        self.assertEqual(self.wide()[100]["outtime_imputed"], 1)
        self.matches[0]["t0"] = dt(3)
        self.prepare()
        self.run_audit()
        self.assertEqual(self.wide()[100]["candidate_status"], "excluded")
        self.assertIn("icu_at_t0", self.wide()[100]["exclusion_reason"])

    def test_dictionary_handling_review_preserves_labels_and_blocks_new_issues(
        self,
    ) -> None:
        rules = self.cohort["burden_v2"]
        for index, unit in enumerate(("UNKNOWN", "Unknown")):
            rules["careunits"][unit] = dict(care_level="unknown", confirmed=False)
            self.tables["transfers"].append(
                dict(
                    subject_id="1",
                    hadm_id="10",
                    transfer_id=f"unknown{index}",
                    eventtype="discharge",
                    careunit=unit,
                    intime=dt(70),
                    outtime=dt(70),
                )
            )
        for index, name in enumerate(("", "Bag", "SW"), start=3):
            self.add_medication(f"m{index}", 1, "Cancelled", dose_given="0")
            self.tables["emar"][-1]["medication"] = name
        self.prepare()
        before = self.run_audit()
        previous_labels = {h: self.wide(h) for h in (48, 72)}
        classifications = copy.deepcopy(
            {f: rules[f] for f in ("careunits", "medications")}
        )
        self.assertFalse(before["dictionary_review_complete"])
        self.assertFalse(before["dictionary_handling_review_complete"])
        self.assertEqual(before["unresolved_evidence_count"], 0)
        rules["dictionary_handling_reviews"] = copy.deepcopy(
            _json_file(PROJECT / "configs/cohort.json")["burden_v2"][
                "dictionary_handling_reviews"
            ]
        )
        self.prepare()
        metadata = self.run_audit()
        self.assertFalse(metadata["dictionary_review_complete"])
        self.assertTrue(metadata["dictionary_handling_review_complete"])
        self.assertTrue(metadata["audit_ready_for_window_decision"])
        self.assertFalse(metadata["ready_for_formal_labels"])
        self.assertFalse(metadata["horizon_frozen"])
        self.assertEqual(metadata["dictionary_sha256"], before["dictionary_sha256"])
        self.assertEqual(
            metadata["dictionary_handling_reviews"],
            rules["dictionary_handling_reviews"],
        )
        self.assertNotEqual(
            metadata["dictionary_handling_reviews_sha256"],
            before["dictionary_handling_reviews_sha256"],
        )
        self.assertEqual({f: rules[f] for f in classifications}, classifications)
        for h in (48, 72):
            self.assertEqual(self.wide(h), previous_labels[h])
            report = self.outputs / f"audit/burden_v2/window_{h}h/audit_report.md"
            self.assertIn(
                "Dictionary handling review complete: True", report.read_text()
            )
        with (self.outputs / "audit/burden_v2/dictionary_inventory.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            inventory = list(csv.DictReader(stream))
        unknown = [r for r in inventory if r["confirmed"] == "False"]
        self.assertEqual(len(unknown), 5)
        self.assertTrue(
            all(
                r["handling_confirmed"] == "True"
                and r["handling_policy"] == "preserve_unknown"
                and r["handling_reviewer"] == "ZE Xiaokang"
                for r in unknown
            )
        )
        manifest = self.outputs / "audit/step_05_manifest.json"
        self.assertIn(
            str(self.outputs / "audit/burden_v2/dictionary_inventory.csv"),
            _json_file(manifest)["files"],
        )
        self.run_audit()
        self.assertEqual(self.wide(), previous_labels[48])
        original_replace, failed = os.replace, False

        def fail_once(source, destination):
            nonlocal failed
            if not failed and str(destination).endswith("dictionary_inventory.csv"):
                failed = True
                raise OSError("synthetic dictionary publication failure")
            return original_replace(source, destination)

        hashes = {f: _file_hash(Path(f)) for f in _json_file(manifest)["files"]}
        manifest_bytes = manifest.read_bytes()
        with patch("src.cohort.common.os.replace", side_effect=fail_once):
            with self.assertRaises(OSError):
                self.run_audit()
        self.assertTrue(failed)
        self.assertEqual(manifest.read_bytes(), manifest_bytes)
        self.assertEqual({f: _file_hash(Path(f)) for f in hashes}, hashes)
        _verify_run(self.outputs, 5)
        self.add_medication("m6", 1, "Cancelled", dose_given="0")
        self.tables["emar"][-1]["medication"] = "NewUnknownName"
        self.prepare()
        unreviewed = self.run_audit()
        self.assertFalse(unreviewed["dictionary_handling_review_complete"])
        self.assertFalse(unreviewed["audit_ready_for_window_decision"])
        self.assertEqual(unreviewed["unresolved_evidence_count"], 0)
        rules["medications"]["newunknownname"] = reviewed(canonical_drug=None)
        self.tables["emar"][2]["event_txt"] = "Started"
        self.prepare()
        conflict = self.run_audit()
        self.assertTrue(conflict["dictionary_handling_review_complete"])
        self.assertFalse(conflict["audit_ready_for_window_decision"])
        self.assertGreater(conflict["unresolved_evidence_count"], 0)
        self.assertIsNone(self.wide()[100][COMPONENTS[4]])
        self.assertIsNone(self.wide()[100]["burden_score"])
        self.tables["emar"][2]["event_txt"] = "Cancelled"
        self.tables["patients"][0]["anchor_age"] = ""
        self.prepare()
        candidate = self.run_audit()
        self.assertTrue(candidate["dictionary_handling_review_complete"])
        self.assertFalse(candidate["audit_ready_for_window_decision"])
        self.assertEqual(candidate["unresolved_evidence_count"], 0)
        self.assertEqual(self.wide()[100]["candidate_status"], "needs_review")

    def test_dictionary_handling_validation_and_legacy_configuration(self) -> None:
        rules = self.cohort["burden_v2"]
        rules.pop("dictionary_handling_reviews")
        validate_rules(self.cohort, self.ehr, self.imaging)
        self.prepare()
        legacy = self.run_audit()
        self.assertTrue(legacy["dictionary_review_complete"])
        self.assertTrue(legacy["dictionary_handling_review_complete"])
        self.assertTrue(legacy["audit_ready_for_window_decision"])
        review = dict(
            handling_confirmed=True,
            handling="preserve_unknown",
            reviewer="synthetic_reviewer",
            reviewed_on="2026-10-04",
            rationale="Keep unknown identity under existing rules.",
            version="handling_v1",
        )
        for invalid in (
            None,
            {"items": {"220277": review}},
            {"medications": {"bag": {**review, "reviewer": " "}}},
            {"medications": {"bag": {**review, "reviewed_on": "not-a-date"}}},
            {"medications": {"bag": {**review, "handling_confirmed": "true"}}},
            {"medications": {"bag": {**review, "handling": "exclude"}}},
            {"medications": {"Bag": review}},
            {"medications": {"norepinephrine": review}},
            {"careunits": {"Ward": review}},
        ):
            rules["dictionary_handling_reviews"] = invalid
            with self.assertRaises(ValueError):
                validate_rules(self.cohort, self.ehr, self.imaging)

    def test_unreviewed_first_run_missing_source_and_duplicate_keys(self) -> None:
        for family in (
            "careunits",
            "transfer_event_actions",
            "medications",
            "administration_types",
            "medication_event_actions",
            "procedure_status_actions",
            "iv_routes",
            "spo2_units",
            "items",
        ):
            for value in self.cohort["burden_v2"][family].values():
                value["confirmed"] = False
        self.prepare()
        metadata = self.run_audit()
        self.assertFalse(metadata["dictionary_review_complete"])
        self.assertFalse(metadata["audit_ready_for_window_decision"])
        self.assertTrue(
            all(row["burden_score"] is None for row in self.wide().values())
        )
        manifest = self.outputs / "audit/step_05_manifest.json"
        old = manifest.read_bytes()
        (self.iv / "icu/chartevents.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_audit()
        self.assertEqual(manifest.read_bytes(), old)
        self.tables["admissions"].append(self.tables["admissions"][0].copy())
        self.prepare()
        with self.assertRaisesRegex(ValueError, "Duplicate source key"):
            self.run_audit()
        self.assertEqual(manifest.read_bytes(), old)


if __name__ == "__main__":
    unittest.main()
