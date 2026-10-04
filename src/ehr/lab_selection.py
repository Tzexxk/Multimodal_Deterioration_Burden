"""Training-only candidate selection, separate from clinical dictionary approval."""

from __future__ import annotations

import json
from pathlib import Path

from src.ehr.lab_audit import query_rows


def missingness_band(covered: int, total: int) -> str:
    """Compare exact integer fractions; rounding never changes a decision."""
    if type(covered) is not int or type(total) is not int:
        raise ValueError("Coverage counts must be integers")
    if total <= 0 or not 0 <= covered <= total:
        raise ValueError("Invalid study coverage denominator or numerator")
    missing = 100 * (total - covered)
    if missing < 30 * total:
        return "below_30"
    if missing <= 50 * total:
        return "30_to_50"
    if missing <= 80 * total:
        return "above_50_to_80"
    return "above_80"


def select_candidates(connection, inventory: list[dict], decisions: dict) -> list[dict]:
    """Read audited associations and frozen patient_splits; emit aggregate evidence.

    Numeric coverage precedes clinical cleaning and unit/specimen decisions.
    An item with several units is counted once per study without converting values.
    Manual decisions are explicit item-level choices in the 30% through 80% bands.
    """
    definitions = {}
    for row in inventory:
        itemid = row["itemid"]
        if itemid in definitions and any(
            row[k] != definitions[itemid][k] for k in ("label", "fluid", "category")
        ):
            raise ValueError("Conflicting inventory item identity")
        definitions[itemid] = row
    if not definitions or set(decisions) - definitions.keys():
        raise ValueError("Empty inventory or unknown manual decision itemid")
    for choice in decisions.values():
        if (
            type(choice.get("include")) is not bool
            or not isinstance(choice.get("rationale"), str)
            or not choice["rationale"].strip()
        ):
            raise ValueError("Manual decisions require a boolean and rationale")
    invalid = connection.execute(
        "SELECT count(*) FROM (SELECT subject_id FROM patient_splits "
        "GROUP BY subject_id HAVING count(*)<>1 OR "
        "count(*) FILTER(WHERE split IN ('train','validation','test'))<>1)"
    ).fetchone()[0]
    if invalid:
        raise ValueError("Invalid or duplicated patient split assignment")
    if connection.execute(
        "SELECT count(*) FROM study_keys s LEFT JOIN patient_splits p "
        "USING(subject_id) WHERE p.subject_id IS NULL"
    ).fetchone()[0]:
        raise ValueError("Study patient has no split assignment")
    denominator = connection.execute(
        "SELECT count(*),count(DISTINCT sample_id),count(DISTINCT s.subject_id) "
        "FROM study_keys s JOIN patient_splits p USING(subject_id) WHERE p.split='train'"
    ).fetchone()
    total, unique, patients = denominator
    if not total or total != unique:
        raise ValueError("Empty training cohort or duplicate training sample key")
    coverage = {
        row["itemid"]: row
        for row in query_rows(
            connection,
            "SELECT a.itemid,count(DISTINCT a.sample_id) AS covered_studies,"
            "count(DISTINCT s.subject_id) AS covered_patients "
            "FROM associations a JOIN study_keys s USING(sample_id) "
            "JOIN patient_splits p ON p.subject_id=s.subject_id "
            "WHERE p.split='train' AND a.availability_reason='available' "
            "AND a.ownership_reason='owned' AND a.numeric_value IS NOT NULL "
            "AND isfinite(a.numeric_value) GROUP BY a.itemid",
        )
    }
    if set(coverage) - definitions.keys():
        raise ValueError("Coverage contains an item absent from the audit inventory")
    rows = []
    for itemid in sorted(definitions, key=int):
        definition = definitions[itemid]
        observed = coverage.get(itemid, {})
        covered = observed.get("covered_studies", 0)
        band = missingness_band(covered, total)
        choice = decisions.get(itemid)
        if choice and band not in ("30_to_50", "above_50_to_80"):
            raise ValueError("Manual decision changed missingness band; reassess")
        if band == "below_30":
            selected, basis = True, "automatic_threshold"
            rationale = "Accepted by the reviewed training missingness <30% policy."
        elif band == "above_80":
            selected, basis = False, "automatic_defer"
            rationale = "Deferred by the reviewed training missingness >80% policy."
        elif choice:
            selected, basis = choice["include"], "explicit_user_choice"
            rationale = choice["rationale"]
        else:
            selected, basis = False, "pending_individual_choice"
            rationale = (
                "Individual selection remains pending; omission is not approval."
            )
        units = sorted({r["source_unit"] for r in inventory if r["itemid"] == itemid})
        rows.append(
            {
                "itemid": itemid,
                **{k: definition[k] for k in ("label", "fluid", "category")},
                "source_units": json.dumps(units),
                "training_studies": total,
                "training_patients": patients,
                "covered_training_studies": covered,
                "covered_training_patients": observed.get("covered_patients", 0),
                "missing_training_studies": total - covered,
                "missing_fraction": (total - covered) / total,
                "missingness_band": band,
                "selection_status": "selected" if selected else "deferred",
                "decision_basis": basis,
                "rationale": rationale,
                "clinical_rules_approved": False,
                "pending_unit_review": selected and (len(units) > 1 or "" in units),
            }
        )
    return rows


def publish_selection(output: Path, csv_content: bytes, metadata: dict) -> None:
    """Reuse identical inputs and recover a missing member of a versioned pair."""
    artifacts = {
        output: csv_content,
        output.with_suffix(".json"): (
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8"),
    }
    for path, content in artifacts.items():
        if path.exists() and path.read_bytes() != content:
            raise ValueError(
                f"Changed selection cannot overwrite versioned input: {path}"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, content in artifacts.items():
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(content)


def main() -> int:
    """Publish independent selection inputs using verified caches only."""
    import argparse
    import csv
    import io
    from datetime import date

    import duckdb
    import pyarrow.parquet as pq

    from src.cohort.common import _file_hash, _verify_run
    from src.ehr import build_ehr_features as step
    from src.ehr import lab_audit as audit

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Versioned selection CSV"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix != ".csv" or not output.is_relative_to(root / "configs/reviews"):
        raise ValueError("Selection CSV must be under configs/reviews")
    policy_path = args.policy.resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_hash = _file_hash(policy_path)
    settings = step.load_settings(root / "configs/lab_processing_v1.json")
    if policy["reviewer"] != settings["reviewer"] or not policy["selection_version"]:
        raise ValueError("Missing version or unauthorized selection reviewer")
    if date.fromisoformat(policy["reviewed_on"]).isoformat() != policy["reviewed_on"]:
        raise ValueError("Selection review date must be an ISO date")
    configs = [
        root / f"configs/{n}.json" for n in ("paths.local", "cohort", "ehr", "imaging")
    ]
    context = step.verify_frozen_inputs(
        root,
        configs,
        root / "configs/image_processing_v1.json",
        root / "configs/patient_split_v1.json",
        settings,
    )
    processed, outputs = context["processed"], context["outputs"]
    _verify_run(outputs, 10)
    directory = outputs / "audit/burden_v2/labs" / settings["audit_version"]
    manifest = directory / "lab_audit_manifest.json"
    record = json.loads(manifest.read_text())
    sources = {k: Path(v) for k, v in record["source_binding"]["source_paths"].items()}
    binding = step.source_binding(
        context, sources, root / "configs/lab_processing_v1.json", settings
    )
    cache = (
        processed / "../cache/burden_v2/labs" / settings["audit_version"]
    ).resolve()
    step.load_audit(manifest, binding, cache, outputs)
    split_path = processed / "splits/burden_v2/patient_split.parquet"
    patients = pq.read_table(split_path)
    with duckdb.connect(":memory:") as connection:
        connection.execute(f"SET memory_limit='{settings['memory_limit']}'")
        audit.load_cache(
            connection, {k: Path(v) for k, v in record["cache_paths"].items()}
        )
        audit.build_views(connection, context["cohort"])
        connection.register("patient_splits", patients)
        rows = select_candidates(
            connection, record["inventory"], policy["individual_decisions"]
        )
        train_keys = connection.execute(
            "SELECT sample_id FROM study_keys JOIN patient_splits USING(subject_id) "
            "WHERE split='train' ORDER BY sample_id"
        ).fetchall()
    if (
        _file_hash(policy_path) != policy_hash
        or step.source_binding(
            context, sources, root / "configs/lab_processing_v1.json", settings
        )
        != binding
    ):
        raise ValueError("Selection policy or audit sources changed during calculation")
    step.load_audit(manifest, binding, cache, outputs)
    _verify_run(outputs, 8)
    _verify_run(outputs, 9)
    _verify_run(outputs, 10)
    for row in rows:
        row.update(
            selection_version=policy["selection_version"],
            reviewer=policy["reviewer"],
            reviewed_on=policy["reviewed_on"],
            audit_fingerprint=record["audit_fingerprint"],
        )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    csv_content = stream.getvalue().encode("utf-8")
    import hashlib

    metadata = {
        "selection_version": policy["selection_version"],
        "policy": policy,
        "policy_sha256": policy_hash,
        "audit_fingerprint": record["audit_fingerprint"],
        "audit_manifest_sha256": _file_hash(manifest),
        "source_binding": binding,
        "selection_code_sha256": _file_hash(Path(__file__)),
        "patient_split_sha256": _file_hash(split_path),
        "training_sample_keys_sha256": audit.json_hash(train_keys),
        "training_studies": rows[0]["training_studies"],
        "training_patients": rows[0]["training_patients"],
        "inventory_itemids": len(rows),
        "selected_itemids": sum(r["selection_status"] == "selected" for r in rows),
        "thresholds_percent": [30, 50, 80],
        "coverage_stage": "eligible_finite_valuenum_before_clinical_cleaning",
        "selection_csv_sha256": hashlib.sha256(csv_content).hexdigest(),
        "clinical_rules_approved": False,
        "dictionary_frozen": False,
        "ehr_features_ready": False,
    }
    publish_selection(output, csv_content, metadata)
    print(
        f"Selection recorded: {metadata['selected_itemids']} of {len(rows)} itemids; clinical rules pending"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
