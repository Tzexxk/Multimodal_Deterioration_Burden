"""Synthetic tests for the explicitly approved dictionary input rules."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pyarrow as pa

from src.cohort.common import _check_csv_tail, _write_csv
from src.ehr import lab_dictionary_inputs as inputs
from src.ehr import lab_review as review
from src.ehr.lab_audit import review_key


class DictionaryInputTests(unittest.TestCase):
    def setUp(self):
        inventory = []
        for itemid, (_, unit) in inputs.APPROVED_FEATURES.items():
            units = (
                ["%", "g/dL"]
                if itemid == "51249"
                else [""]
                if itemid == "51237"
                else [unit]
            )
            for raw_unit in units:
                inventory.append(
                    dict(
                        itemid=itemid,
                        source_unit=raw_unit,
                        label=f"Synthetic {itemid}",
                        fluid="Blood",
                        category="Synthetic",
                        dictionary_item_present=True,
                        suggested_group="out_of_scope",
                        review_key=review_key(itemid, raw_unit),
                    )
                )
        inventory.append(
            dict(
                itemid="99999",
                source_unit="",
                label="Synthetic excluded",
                fluid="Other",
                category="Synthetic",
                dictionary_item_present=True,
                suggested_group="out_of_scope",
                review_key=review_key("99999", ""),
            )
        )
        self.record = dict(
            inventory=inventory,
            audit_fingerprint="synthetic",
            scope_inventory_sha256="synthetic_scope",
        )
        self.selection = list(
            {
                r["itemid"]: dict(
                    itemid=r["itemid"],
                    label=r["label"],
                    fluid=r["fluid"],
                    category=r["category"],
                    audit_fingerprint="synthetic",
                    selection_status="deferred"
                    if r["itemid"] == "99999"
                    else "selected",
                    rationale="Explicit synthetic decision.",
                )
                for r in inventory
            }.values()
        )

    def parse(self, record=None):
        record = record or self.record
        rows, ack = inputs.build_approved_reviews(record, self.selection)
        with tempfile.TemporaryDirectory(
            prefix="synthetic_dictionary_input_"
        ) as temporary:
            decisions = Path(temporary) / "decisions.csv"
            scope = Path(temporary) / "scope.csv"
            _write_csv(decisions, review.REVIEW_COLUMNS, rows)
            _write_csv(scope, review.REVIEW_COLUMNS, ack)
            dictionary = list(
                {
                    r["itemid"]: {
                        k: r[k] for k in ("itemid", "label", "fluid", "category")
                    }
                    for r in record["inventory"]
                }.values()
            )
            return review.parse_reviews(
                [decisions, scope],
                record["inventory"],
                dictionary,
                record,
                {"reviewer": "ZE Xiaokang"},
                SimpleNamespace(check_csv=_check_csv_tail),
            )

    def test_complete_decisions_and_explicit_special_units(self):
        decisions, ack = self.parse()
        retained = [r for r in decisions if r["keep"]]
        self.assertEqual(len(retained), 26)
        self.assertEqual(len({r["feature_name"] for r in retained}), 25)
        self.assertTrue(
            all(r["valid_min"] is None and r["valid_max"] is None for r in retained)
        )
        for r in retained:
            self.assertEqual(
                (r["conversion_multiplier"], r["conversion_offset"]), (1, 0)
            )
            if r["itemid"] == "51249":
                self.assertEqual(r["canonical_unit"], "g/dL")
            if r["itemid"] == "51237":
                self.assertEqual(
                    (r["canonical_unit"], r["missing_unit_rule"]),
                    ("1", "explicit_item_unit"),
                )
        excluded = next(r for r in decisions if not r["keep"])
        self.assertEqual(excluded["missing_unit_rule"], "exclude")
        self.assertEqual(excluded["feature_name"], "")
        with duckdb.connect(":memory:") as con:
            state, progress = review.review_progress(
                con, self.record["inventory"], decisions, ack
            )
            self.assertTrue(state["review_complete"])
            self.assertTrue(all(r["status"] == "reviewed" for r in progress))
            partial = [
                r
                for r in decisions
                if not (r["itemid"] == "51249" and r["source_unit"] == "g/dL")
            ]
            state, _ = review.review_progress(
                con, self.record["inventory"], partial, ack
            )
            self.assertFalse(state["review_complete"])
            state, _ = review.review_progress(
                con, self.record["inventory"], decisions, None
            )
            self.assertFalse(state["review_complete"])

    def test_stale_scope_identity_and_units_rejected(self):
        for field, value in (
            ("audit_fingerprint", "stale"),
            ("selection_status", "pending"),
        ):
            selection = copy.deepcopy(self.selection)
            selection[0][field] = value
            with self.assertRaises(ValueError):
                inputs.build_approved_reviews(self.record, selection)
        record = copy.deepcopy(self.record)
        record["inventory"][0]["source_unit"] = "unreviewed"
        with self.assertRaises(ValueError):
            inputs.build_approved_reviews(record, self.selection)
        rows, ack = inputs.build_approved_reviews(self.record, self.selection)
        with tempfile.TemporaryDirectory(prefix="synthetic_scope_hash_") as temporary:
            path = Path(temporary) / "scope.csv"
            _write_csv(path, review.REVIEW_COLUMNS, ack)
            record = {**self.record, "scope_inventory_sha256": "changed"}
            with self.assertRaises(ValueError):
                review.parse_reviews(
                    [path],
                    record["inventory"],
                    [],
                    record,
                    {"reviewer": "ZE Xiaokang"},
                    SimpleNamespace(check_csv=_check_csv_tail),
                )

    def test_extremes_future_revision_and_conflict_blocking(self):
        decisions, _ = self.parse()
        mchc = [
            r for r in decisions if r["itemid"] == "51249" and r["source_unit"] == "%"
        ]
        rows = [
            dict(
                sample_id="synthetic",
                lab_record_key="a",
                labevent_id="1",
                specimen_id="1",
                itemid="51249",
                valueuom="%",
                value="1000",
                numeric_value=1000.0,
                text_numeric=1000.0,
                availability_reason="available",
                ownership_reason="owned",
            )
        ]
        rows.append(
            {
                **rows[0],
                "lab_record_key": "future",
                "numeric_value": 1.0,
                "text_numeric": 1.0,
                "availability_reason": "future_storetime",
            }
        )
        with duckdb.connect(":memory:") as con:
            con.register("associations", pa.Table.from_pylist(rows))
            result = review.validate_retained_conflicts(con, mchc)
            self.assertEqual(result[0]["converted_min"], 1000)
            self.assertEqual(result[0]["converted_max"], 1000)
            self.assertEqual(result[0]["numeric_records"], 1)
            for changes in ({"specimen_id": ""}, {"text_numeric": 999.0}):
                con.register(
                    "associations", pa.Table.from_pylist([{**rows[0], **changes}])
                )
                with self.assertRaises(ValueError):
                    review.validate_retained_conflicts(con, mchc)
            con.register(
                "associations",
                pa.Table.from_pylist(
                    [rows[0], {**rows[0], "lab_record_key": "conflict"}]
                ),
            )
            with self.assertRaises(ValueError):
                review.validate_retained_conflicts(con, mchc)


if __name__ == "__main__":
    unittest.main()
