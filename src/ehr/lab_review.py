"""Explicit laboratory decisions, safe selectors, and review completion gates."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa

from src.ehr.lab_audit import json_hash, query_rows, read_small_csv

REVIEW_COLUMNS: Sequence[str] = (
    "record_type review_key branch_id audit_fingerprint scope_inventory_sha256 "
    "feature_name itemid label fluid category source_unit canonical_unit "
    "conversion_multiplier conversion_offset valid_min valid_max valid_range_basis "
    "specimen_rule duplicate_rule numeric_result_rule missing_unit_rule keep "
    "exclusion_reason dictionary_version rationale reviewer reviewed_on review_version"
).split()
STRING_FIELDS: Sequence[str] = (
    "feature_name label fluid category source_unit canonical_unit valid_range_basis "
    "specimen_rule duplicate_rule numeric_result_rule missing_unit_rule exclusion_reason "
    "dictionary_version review_key branch_id audit_fingerprint rationale reviewer "
    "reviewed_on review_version"
).split()
DICTIONARY_SCHEMA = pa.schema(
    [("itemid", pa.int64()), ("keep", pa.bool_())]
    + [
        (k, pa.float64())
        for k in (
            "conversion_multiplier",
            "conversion_offset",
            "valid_min",
            "valid_max",
        )
    ]
    + [(k, pa.string()) for k in STRING_FIELDS]
)


def write_templates(
    run,
    directory: Path,
    inventory: list[dict],
    fingerprint: str,
    scope_hash: str,
    batches: list[dict],
) -> None:
    from src.cohort.common import _write_csv

    records = []
    for entry in inventory:
        row: dict[str, Any] = dict.fromkeys(REVIEW_COLUMNS, "")
        row.update(
            record_type="decision", branch_id="default", audit_fingerprint=fingerprint
        )
        row.update(
            {
                k: entry[k]
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
        records.append(row)
    candidate_keys = {
        r["review_key"] for r in inventory if r["suggested_group"] != "out_of_scope"
    }
    _write_csv(
        run.path(directory / "review_template.csv"),
        REVIEW_COLUMNS,
        [r for r in records if r["review_key"] in candidate_keys],
    )
    for batch_id in dict.fromkeys(b["batch_id"] for b in batches):
        ids = {b["itemid"] for b in batches if b["batch_id"] == batch_id}
        _write_csv(
            run.path(directory / f"batch_{batch_id}_review.csv"),
            REVIEW_COLUMNS,
            [r for r in records if r["itemid"] in ids],
        )
    acknowledgment: dict[str, Any] = dict.fromkeys(REVIEW_COLUMNS, "")
    acknowledgment.update(
        record_type="scope_ack",
        review_key="scope_" + scope_hash,
        audit_fingerprint=fingerprint,
        scope_inventory_sha256=scope_hash,
    )
    _write_csv(
        run.path(directory / "scope_ack_template.csv"), REVIEW_COLUMNS, [acknowledgment]
    )


def finite_number(value: str, name: str, required: bool = False) -> float | None:
    if not value.strip():
        if required:
            raise ValueError(f"Missing {name}")
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {name}")
    return parsed


def selector(value: str, dictionary_ids: set[str]) -> dict:
    try:
        rule = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Specimen rule must be a JSON object") from exc
    if not isinstance(rule, dict):
        raise ValueError("Specimen rule must be a JSON object")
    if rule == {"kind": "item_identity"} or rule == {"kind": "unmatched"}:
        return rule
    if (
        set(rule) == {"kind", "itemid", "values"}
        and rule["kind"] == "same_specimen_value"
        and type(rule["itemid"]) is int
        and str(rule["itemid"]) in dictionary_ids
        and isinstance(rule["values"], list)
        and rule["values"]
        and all(isinstance(v, str) and v.strip() for v in rule["values"])
        and len(set(rule["values"])) == len(rule["values"])
    ):
        return rule
    raise ValueError("Unsupported specimen selector or unknown evidence itemid")


def duplicate_rule(value: str) -> dict:
    try:
        rule = json.loads(value)
    except ValueError as exc:
        raise ValueError("Duplicate rule must be JSON") from exc
    if (
        not isinstance(rule, dict)
        or set(rule) != {"kind", "conflicting_id", "missing_specimen"}
        or rule.get("kind") != "latest_available"
        or rule.get("conflicting_id") not in ("exclude", "error")
        or rule.get("missing_specimen") not in ("record_id", "exclude", "error")
    ):
        raise ValueError("Unsupported duplicate rule")
    return rule


def numeric_rule(value: str) -> dict:
    try:
        rule = json.loads(value)
    except ValueError as exc:
        raise ValueError("Numeric result rule must be JSON") from exc
    if (
        not isinstance(rule, dict)
        or set(rule) != {"kind", "comparator", "text_disagreement"}
        or rule.get("kind") != "valuenum_only"
        or rule.get("comparator") != "exclude"
        or rule.get("text_disagreement") not in ("exclude", "error")
    ):
        raise ValueError("Unsupported numeric result rule")
    return rule


def parse_reviews(
    paths: list[Path],
    inventory: list[dict],
    dictionary: list[dict],
    audit: dict,
    settings: dict,
    run,
) -> tuple[list[dict], dict | None]:
    by_key = {r["review_key"]: r for r in inventory}
    dictionary_ids = {r["itemid"] for r in dictionary}
    decisions, acknowledgments = {}, []
    for path in paths:
        versions = set()
        for raw in read_small_csv(path, REVIEW_COLUMNS, run, exact=True):
            row = {k: raw[k] for k in REVIEW_COLUMNS}
            # Unfilled templates are intentionally rejected: pass completed rows only.
            if row["audit_fingerprint"] != audit["audit_fingerprint"]:
                raise ValueError("Review audit fingerprint changed")
            if row["reviewer"] != settings["reviewer"] or not all(
                row[k].strip() for k in ("rationale", "reviewed_on", "review_version")
            ):
                raise ValueError("Missing or unauthorized review provenance")
            date.fromisoformat(row["reviewed_on"])
            versions.add(row["review_version"])
            if row["record_type"] == "scope_ack":
                if (
                    row["scope_inventory_sha256"] != audit["scope_inventory_sha256"]
                    or row["review_key"] != "scope_" + audit["scope_inventory_sha256"]
                ):
                    raise ValueError("Scope inventory hash changed")
                if any(
                    row[k]
                    for k in REVIEW_COLUMNS
                    if k
                    not in (
                        "record_type",
                        "review_key",
                        "audit_fingerprint",
                        "scope_inventory_sha256",
                        "rationale",
                        "reviewer",
                        "reviewed_on",
                        "review_version",
                    )
                ):
                    raise ValueError(
                        "Scope acknowledgment contains laboratory decisions"
                    )
                acknowledgments.append(row)
                continue
            if row["record_type"] != "decision" or row["review_key"] not in by_key:
                raise ValueError("Unknown review record or laboratory reference")
            entry = by_key[row["review_key"]]
            if any(
                row[k] != entry[k]
                for k in ("itemid", "source_unit", "label", "fluid", "category")
            ):
                raise ValueError("Review laboratory identity changed")
            if row["scope_inventory_sha256"] or not re.fullmatch(
                r"[a-z][a-z0-9_]{0,63}", row["branch_id"]
            ):
                raise ValueError("Invalid review branch identity")
            if row["dictionary_version"]:
                raise ValueError(
                    "dictionary_version is assigned by freeze, not review inputs"
                )
            if row["keep"] not in ("0", "1"):
                raise ValueError("Explicit keep=0 or keep=1 required")
            row["keep"] = row["keep"] == "1"
            if not entry["dictionary_item_present"] and row["keep"]:
                raise ValueError("Cannot retain an item absent from d_labitems")
            rule = selector(row["specimen_rule"], dictionary_ids)
            row["specimen_rule"] = json.dumps(
                rule, sort_keys=True, separators=(",", ":")
            )
            if not row["source_unit"].strip():
                expected = "explicit_item_unit" if row["keep"] else "exclude"
                if row["missing_unit_rule"] != expected:
                    raise ValueError(
                        "Missing units require an explicit reviewed handling rule"
                    )
            elif row["missing_unit_rule"]:
                raise ValueError("missing_unit_rule applies only to missing units")
            for name in (
                "conversion_multiplier",
                "conversion_offset",
                "valid_min",
                "valid_max",
            ):
                row[name] = finite_number(
                    row[name], name, row["keep"] and name.startswith("conversion_")
                )
            if row["keep"]:
                if (
                    not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", row["feature_name"])
                    or not row["canonical_unit"].strip()
                    or not row["valid_range_basis"].strip()
                    or row["exclusion_reason"]
                ):
                    raise ValueError(
                        "Retained rule needs feature, unit, and explicit validity-range basis"
                    )
                lo, hi = row["valid_min"], row["valid_max"]
                if lo is not None and hi is not None and lo > hi:
                    raise ValueError("Reversed valid range")
                row["duplicate_rule"] = json.dumps(
                    duplicate_rule(row["duplicate_rule"]),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                row["numeric_result_rule"] = json.dumps(
                    numeric_rule(row["numeric_result_rule"]),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            else:
                if not row["exclusion_reason"].strip():
                    raise ValueError("Excluded rule requires a reason")
                if any(
                    row[k]
                    for k in (
                        "feature_name",
                        "canonical_unit",
                        "duplicate_rule",
                        "numeric_result_rule",
                        "valid_range_basis",
                    )
                ) or any(
                    row[k] is not None
                    for k in (
                        "conversion_multiplier",
                        "conversion_offset",
                        "valid_min",
                        "valid_max",
                    )
                ):
                    raise ValueError("Excluded rule contains retained-feature settings")
            key = (row["review_key"], row["branch_id"])
            if key in decisions and decisions[key] != row:
                raise ValueError(
                    "Conflicting review decisions; provide a replacement cumulative input"
                )
            decisions[key] = row
        if len(versions) > 1:
            raise ValueError("Each review file must have one review_version")
    if acknowledgments and any(a != acknowledgments[0] for a in acknowledgments):
        raise ValueError("Conflicting scope acknowledgments")
    return sorted(
        decisions.values(),
        key=lambda r: (int(r["itemid"]), r["source_unit"], r["branch_id"]),
    ), acknowledgments[0] if acknowledgments else None


def selector_sql(rule: dict) -> tuple[str, list]:
    if rule["kind"] == "item_identity":
        return "true", []
    if rule["kind"] != "same_specimen_value":
        raise ValueError(
            "Unmatched selectors must be evaluated relative to other branches"
        )
    placeholders = ",".join("?" for _ in rule["values"])
    sql = (
        "EXISTS(SELECT 1 FROM assigned_labs e WHERE e.patient_id=a.patient_id "
        "AND regexp_full_match(a.specimen_id,'[1-9][0-9]*') "
        "AND e.specimen_id=a.specimen_id AND e.assigned_hadm_id=a.study_hadm_id "
        "AND (trim(e.hadm_id)='' OR e.provided_owner_valid) "
        "AND e.chart_at<=a.t0 AND e.stored_at BETWEEN e.chart_at AND a.t0 "
        f"AND e.itemid=? AND e.value IN ({placeholders}))"
    )
    return sql, [str(rule["itemid"]), *rule["values"]]


def check_branches(connection, rows: list[dict]) -> tuple[bool, list[dict]]:
    """Require a disjoint, exhaustive policy, including unavailable specimen evidence."""
    selectors = [json.loads(r["specimen_rule"]) for r in rows]
    if len(selectors) == 1 and selectors[0]["kind"] == "item_identity":
        return True, []
    if (
        any(s["kind"] == "item_identity" for s in selectors)
        or sum(s["kind"] == "unmatched" for s in selectors) > 1
    ):
        raise ValueError("Overlapping item or unmatched selectors")
    exact = [s for s in selectors if s["kind"] == "same_specimen_value"]
    if len({s["itemid"] for s in exact}) > 1:
        raise ValueError(
            "Branches must use one evidence itemid to establish disjointness"
        )
    values = [v for s in exact for v in s["values"]]
    if len(set(values)) != len(values):
        raise ValueError("Overlapping specimen values")
    checks = [selector_sql(s) for s in exact]
    if checks:
        total = " + ".join(f"CAST(({sql}) AS INTEGER)" for sql, _ in checks)
        parameters = [v for _, args in checks for v in args]
        overlaps = query_rows(
            connection,
            "SELECT a.sample_id,a.lab_record_key,a.specimen_id FROM associations a "
            f"WHERE a.itemid=? AND a.valueuom=? AND ({total})>1 AND a.availability_reason='available' AND a.ownership_reason='owned' ORDER BY a.sample_id,a.lab_record_key",
            [rows[0]["itemid"], rows[0]["source_unit"], *parameters],
        )
        if overlaps:
            raise ValueError(
                "Conflicting same-specimen evidence matches several branches"
            )
    return bool(exact) and any(s["kind"] == "unmatched" for s in selectors), []


def review_progress(
    connection,
    inventory: list[dict],
    decisions: list[dict],
    acknowledgment: dict | None,
) -> tuple[dict, list[dict]]:
    by_key = {}
    for row in decisions:
        by_key.setdefault(row["review_key"], []).append(row)
    promoted_itemids = {r["itemid"] for r in decisions}
    progress = []
    for entry in inventory:
        rows = by_key.get(entry["review_key"], [])
        complete = check_branches(connection, rows)[0] if rows else False
        scoped = (
            entry["suggested_group"] == "out_of_scope"
            and not rows
            and entry["itemid"] not in promoted_itemids
            and acknowledgment is not None
        )
        progress.append(
            {
                "review_key": entry["review_key"],
                "itemid": entry["itemid"],
                "source_unit": entry["source_unit"],
                "status": "reviewed"
                if complete
                else "scope_excluded"
                if scoped
                else "pending_review",
            }
        )
    state = {
        "candidate_unit_branches": len(inventory),
        "completed_unit_branches": sum(
            r["status"] != "pending_review" for r in progress
        ),
        "pending_unit_branches": sum(r["status"] == "pending_review" for r in progress),
        "scope_acknowledgment_complete": acknowledgment is not None,
        "review_complete": acknowledgment is not None
        and all(r["status"] != "pending_review" for r in progress),
    }
    return state, progress


def expanded_decisions(
    inventory: list[dict], decisions: list[dict], acknowledgment: dict | None
) -> list[dict]:
    result = list(decisions)
    reviewed = {r["review_key"] for r in decisions}
    promoted_itemids = {r["itemid"] for r in decisions}
    if acknowledgment:
        for entry in inventory:
            if (
                entry["suggested_group"] != "out_of_scope"
                or entry["review_key"] in reviewed
                or entry["itemid"] in promoted_itemids
            ):
                continue
            row: dict[str, Any] = dict.fromkeys(REVIEW_COLUMNS, "")
            row.update(
                {
                    k: entry[k]
                    for k in (
                        "itemid",
                        "label",
                        "fluid",
                        "category",
                        "source_unit",
                        "review_key",
                    )
                }
            )
            row.update(
                {
                    k: acknowledgment[k]
                    for k in (
                        "audit_fingerprint",
                        "rationale",
                        "reviewer",
                        "reviewed_on",
                        "review_version",
                    )
                }
            )
            row.update(
                record_type="decision",
                branch_id="scope_excluded",
                keep=False,
                exclusion_reason="Explicit signed out-of-scope inventory acknowledgment",
                specimen_rule='{"kind":"item_identity"}',
                missing_unit_rule="exclude" if not row["source_unit"].strip() else "",
            )
            for k in (
                "conversion_multiplier",
                "conversion_offset",
                "valid_min",
                "valid_max",
            ):
                row[k] = None
            result.append(row)
    return sorted(
        result, key=lambda r: (int(r["itemid"]), r["source_unit"], r["branch_id"])
    )


def validate_retained_conflicts(connection, decisions: list[dict]) -> list[dict]:
    """Check reviewed policies and affine arithmetic without producing features."""
    reports = []
    by_key = {}
    for r in decisions:
        by_key.setdefault(r["review_key"], []).append(r)
    complete_keys = {
        k for k, rows in by_key.items() if check_branches(connection, rows)[0]
    }
    for row in decisions:
        if not row["keep"] or row["review_key"] not in complete_keys:
            continue
        rule = json.loads(row["specimen_rule"])
        if rule["kind"] == "unmatched":
            other = [
                selector_sql(json.loads(r["specimen_rule"]))
                for r in by_key[row["review_key"]]
                if json.loads(r["specimen_rule"])["kind"] == "same_specimen_value"
            ]
            condition = "NOT (" + " OR ".join(f"({s})" for s, _ in other) + ")"
            args = [v for _, values in other for v in values]
        else:
            condition, args = selector_sql(rule)
        duplicate = json.loads(row["duplicate_rule"])
        numeric = json.loads(row["numeric_result_rule"])
        selected = (
            "SELECT DISTINCT a.sample_id,a.lab_record_key,a.labevent_id,a.specimen_id,a.value,a.numeric_value,a.text_numeric "
            f"FROM associations a WHERE a.itemid=? AND a.valueuom=? AND ({condition}) "
            "AND a.availability_reason='available' AND a.ownership_reason='owned'"
        )
        parameters = [row["itemid"], row["source_unit"], *args]
        counts = query_rows(
            connection,
            "WITH selected AS ("
            + selected
            + ") SELECT count(DISTINCT lab_record_key) FILTER(WHERE NOT regexp_full_match(specimen_id,'[1-9][0-9]*')) AS missing_specimen,"
            "count(DISTINCT lab_record_key) FILTER(WHERE text_numeric IS NOT NULL AND numeric_value IS NOT NULL AND text_numeric<>numeric_value) AS numeric_disagreement,"
            "count(DISTINCT lab_record_key) FILTER(WHERE NOT regexp_full_match(labevent_id,'[1-9][0-9]*') OR EXISTS(SELECT 1 FROM associations e WHERE e.sample_id=selected.sample_id AND e.labevent_id=selected.labevent_id AND e.lab_record_key<>selected.lab_record_key AND e.availability_reason='available' AND e.ownership_reason='owned')) AS conflicting_id FROM selected",
            parameters,
        )[0]
        if (
            (duplicate["missing_specimen"] == "error" and counts["missing_specimen"])
            or (duplicate["conflicting_id"] == "error" and counts["conflicting_id"])
            or (
                numeric["text_disagreement"] == "error"
                and counts["numeric_disagreement"]
            )
        ):
            raise ValueError(
                f"Retained rule {row['review_key']}/{row['branch_id']} leaves observed conflicts unresolved: {counts}"
            )
        conversion = query_rows(
            connection,
            "WITH selected AS ("
            + selected
            + ") SELECT count(DISTINCT lab_record_key) FILTER(WHERE numeric_value IS NOT NULL) AS numeric_records,"
            "count(DISTINCT lab_record_key) FILTER(WHERE numeric_value IS NOT NULL AND NOT isfinite(numeric_value*?+?)) AS nonfinite_conversions,"
            "min(numeric_value*?+?) AS converted_min,max(numeric_value*?+?) AS converted_max FROM selected",
            [
                *parameters,
                row["conversion_multiplier"],
                row["conversion_offset"],
                row["conversion_multiplier"],
                row["conversion_offset"],
                row["conversion_multiplier"],
                row["conversion_offset"],
            ],
        )[0]
        if conversion["nonfinite_conversions"]:
            raise ValueError("Reviewed conversion produces non-finite values")
        reports.append(
            {
                "review_key": row["review_key"],
                "branch_id": row["branch_id"],
                "feature_name": row["feature_name"],
                **counts,
                **conversion,
            }
        )
    return reports


def dictionary_table(decisions: list[dict], version: str) -> pa.Table:
    records = []
    for row in decisions:
        item = {k: row[k] for k in DICTIONARY_SCHEMA.names}
        item["itemid"] = int(item["itemid"])
        item["dictionary_version"] = version
        records.append(item)
    return pa.Table.from_pylist(records, schema=DICTIONARY_SCHEMA)


def decisions_hash(decisions: list[dict], acknowledgment: dict | None) -> str:
    return json_hash({"decisions": decisions, "scope_acknowledgment": acknowledgment})
