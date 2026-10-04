"""Construct the clinical decisions explicitly approved on 2026-10-05."""

from __future__ import annotations

from datetime import date

from src.ehr.lab_review import REVIEW_COLUMNS

# These identities and source units were reviewed against the bound audit.
APPROVED_FEATURES = {
    "50861": ("alt", "IU/L"),
    "50868": ("anion_gap", "mEq/L"),
    "50878": ("ast", "IU/L"),
    "50882": ("bicarbonate", "mEq/L"),
    "50885": ("total_bilirubin", "mg/dL"),
    "50893": ("total_calcium", "mg/dL"),
    "50902": ("chloride", "mEq/L"),
    "50912": ("creatinine", "mg/dL"),
    "50931": ("glucose", "mg/dL"),
    "50960": ("magnesium", "mg/dL"),
    "50970": ("phosphate", "mg/dL"),
    "50971": ("potassium", "mEq/L"),
    "50983": ("sodium", "mEq/L"),
    "51006": ("bun", "mg/dL"),
    "51221": ("hematocrit", "%"),
    "51222": ("hemoglobin", "g/dL"),
    "51237": ("inr", "1"),
    "51248": ("mch", "pg"),
    "51249": ("mchc", "g/dL"),
    "51250": ("mcv", "fL"),
    "51265": ("platelet_count", "K/uL"),
    "51275": ("ptt", "sec"),
    "51277": ("rdw", "%"),
    "51279": ("rbc", "m/uL"),
    "51301": ("wbc", "K/uL"),
}
DUPLICATE_RULE = (
    '{"kind":"latest_available","conflicting_id":"error","missing_specimen":"error"}'
)
NUMERIC_RULE = (
    '{"kind":"valuenum_only","comparator":"exclude","text_disagreement":"error"}'
)
RANGE_BASIS = (
    "No reliable hard validity bounds approved for this version. Reference ranges "
    "are descriptive evidence only; preserve finite extreme values."
)


def build_approved_reviews(record: dict, selection: list[dict]) -> tuple[list, list]:
    """Return complete branch decisions and the separately approved scope record."""
    selected = {r["itemid"] for r in selection if r["selection_status"] == "selected"}
    if selected != APPROVED_FEATURES.keys():
        raise ValueError(
            "Candidate scope differs from the explicitly approved 25 items"
        )
    inventory = record["inventory"]
    by_id = {r["itemid"]: r for r in selection}
    if len(by_id) != len(selection) or set(by_id) != {r["itemid"] for r in inventory}:
        raise ValueError("Selection does not cover the complete unique item inventory")
    for row in selection:
        if row["audit_fingerprint"] != record["audit_fingerprint"]:
            raise ValueError("Selection audit fingerprint changed")
        if row["selection_status"] not in ("selected", "deferred"):
            raise ValueError("Unknown selection status")
    keys = [r["review_key"] for r in inventory]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicated audit branch key")
    rows = []
    provenance = {
        "audit_fingerprint": record["audit_fingerprint"],
        "reviewer": "ZE Xiaokang",
        "reviewed_on": date(2026, 10, 5).isoformat(),
        "review_version": "lab_dictionary_reviews_2026-10-05_v1",
    }
    for entry in sorted(inventory, key=lambda r: (int(r["itemid"]), r["source_unit"])):
        itemid, unit = entry["itemid"], entry["source_unit"]
        row = dict.fromkeys(REVIEW_COLUMNS, "")
        row.update(provenance)
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
        if any(entry[k] != by_id[itemid][k] for k in ("label", "fluid", "category")):
            raise ValueError("Selection and audit item identities disagree")
        row.update(
            record_type="decision",
            branch_id="default",
            specimen_rule='{"kind":"item_identity"}',
        )
        if itemid in selected:
            feature, canonical = APPROVED_FEATURES[itemid]
            allowed = (
                {"%", "g/dL"}
                if itemid == "51249"
                else {""}
                if itemid == "51237"
                else {canonical}
            )
            if (
                unit not in allowed
                or entry["fluid"] != "Blood"
                or not entry["dictionary_item_present"]
            ):
                raise ValueError(
                    "Retained source unit or specimen identity was not approved"
                )
            row.update(
                keep="1",
                feature_name=feature,
                canonical_unit=canonical,
                conversion_multiplier="1",
                conversion_offset="0",
                valid_range_basis=RANGE_BASIS,
                duplicate_rule=DUPLICATE_RULE,
                numeric_result_rule=NUMERIC_RULE,
                missing_unit_rule="explicit_item_unit" if itemid == "51237" else "",
                rationale="User approved candidate scope, item identity, current units, latest available revision and valuenum-only rules on 2026-10-05.",
            )
            if itemid == "51249":
                row["rationale"] += (
                    " Both MCHC source branches use the explicitly reviewed local reporting scale; preserve raw units and pairing discrepancy evidence."
                )
            elif itemid == "51237":
                row["rationale"] += (
                    " INR is the explicitly reviewed dimensionless ratio; missing source unit is approved for this item only."
                )
        else:
            row.update(
                keep="0",
                missing_unit_rule="exclude" if not unit.strip() else "",
                exclusion_reason="Deferred from this dictionary version under approved candidate selection; eligible for future reassessment.",
                rationale=by_id[itemid]["rationale"],
            )
        rows.append(row)
    ack = dict.fromkeys(REVIEW_COLUMNS, "")
    ack.update(provenance)
    ack.update(
        record_type="scope_ack",
        review_key="scope_" + record["scope_inventory_sha256"],
        scope_inventory_sha256=record["scope_inventory_sha256"],
        rationale="User confirmed the complete bound inventory on 2026-10-05: retain the 25 selected itemids and exclude all remaining 343 itemids from this dictionary version, with future reassessment permitted. Explicit per-branch decisions preserve the full inventory.",
    )
    return rows, [ack]
