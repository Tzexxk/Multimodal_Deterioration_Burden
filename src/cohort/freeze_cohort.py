"""Step 08: validate the common cohort and publish immutable release snapshots."""

from __future__ import annotations

import copy
import json
import re
import shutil
import tempfile
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort import audit_burden_labels as audit
from src.cohort import build_labels
from src.cohort.burden_io import COMPONENTS, open_store
from src.cohort.common import (
    _config_hash,
    _file_hash,
    _file_version,
    _git_commit,
    _json_file,
    _OutputRun,
    _progress,
    _under_root,
    _verify_run,
    _write_csv,
)
from src.imaging import prepare_images as images
from src.imaging.preprocessing import validate_pixels, validate_settings

PUBLICATION_FIELDS = (
    "decision_version rules_version dictionary_version source_audit_run_id label_schema_version"
).split()
QC_FIELDS = {
    "npy_relpath": "npy_relpath",
    "npy_sha256": "npy_sha256",
    "source_dicom_sha256": "source_dicom_sha256",
    "processing_config_sha256": "processing_config_sha256",
    "processing_version": "processing_version",
    "processing_status": "processing_status",
    "qc_status": "qc_status",
    "qc_reason": "qc_reason",
    "qc_reviewer": "reviewer",
    "qc_reviewed_on": "reviewed_on",
    "qc_review_version": "review_version",
    "qc_review_rationale": "review_rationale",
    "qc_review_source_run_id": "review_source_run_id",
}
TABLE_NAMES = ("cohort.parquet", "cohort_audit.parquet")
REPORT_NAMES = (
    "cohort_flow.csv",
    "cohort_qc_exclusions.csv",
    "training_feasibility.csv",
    "source_overlap.csv",
    "step_08_report.md",
)


def _code_hash() -> str:
    return _config_hash(
        [
            Path(__file__),
            Path(__file__).parents[2] / "scripts/08_freeze_common_cohort.py",
        ]
    )


def _counts(rows: list[dict]) -> dict:
    return {
        "studies": len(rows),
        "admissions": len({r["hadm_id"] for r in rows}),
        "patients": len({r["subject_id"] for r in rows}),
    }


def verify_inputs(root: Path, configs: list[Path], processing: Path) -> dict:
    """Verify upstream content; hash review CSVs without parsing them or raw data."""
    labels, indexed, context = images.load_inputs(root, configs)
    outputs, processed = context["outputs"], context["processed"]
    data = processed / "cohort/burden_v2"
    formal_manifest = _json_file(outputs / "audit/step_06_manifest.json")
    if str(data / "event_evidence.parquet") not in formal_manifest["files"]:
        raise ValueError("Formal evidence is not covered by the Step 06 manifest")
    image_dir = processed / "imaging/burden_v2"
    qc_dir = outputs / "qc/burden_v2"
    metadata_path = qc_dir / "run_metadata.json"
    required = [
        metadata_path,
        image_dir / "image_processing.parquet",
        image_dir / "image_qc_status.parquet",
        outputs / "audit/step_07_csv_tail_checks.csv",
    ]
    _verify_run(outputs, 7, metadata_path)
    image_manifest = _json_file(outputs / "audit/step_07_manifest.json")
    if any(str(path) not in image_manifest["files"] for path in required):
        raise ValueError("Step 07 required artifacts are not covered by its manifest")
    metadata = _json_file(metadata_path)
    settings = _json_file(processing)
    validate_settings(settings)
    if (
        metadata["run_id"]
        != _json_file(outputs / "audit/step_07_manifest.json")["run_id"]
        or metadata["source_hashes"] != context["source_hashes"]
        or metadata["source_label_run_id"] != context["source_label_run_id"]
        or metadata["processing_config_sha256"] != _file_hash(processing)
        or Path(metadata["processing_config_path"]).resolve() != processing.resolve()
        or metadata["processing_settings"] != settings
        or metadata["processing_version"] != settings["processing_version"]
        or metadata["code_sha256"]
        != _config_hash(
            [
                Path(images.__file__),
                Path(images.__file__).parent / "preprocessing.py",
                Path(images.__file__).parents[2]
                / "scripts/07_qc_and_prepare_images.py",
            ]
        )
    ):
        raise ValueError("Step 07 configuration, code or upstream provenance changed")
    if not all(
        metadata.get(k) is True
        for k in (
            "image_checks_passed",
            "qc_review_complete",
            "ready_for_cohort_freeze",
        )
    ):
        raise ValueError("Image QC is not ready for cohort freeze")
    review_path = (
        Path(metadata["reviews_path"]).resolve() if metadata["reviews_path"] else None
    )
    if (review_path and _file_hash(review_path)) != metadata["reviews_sha256"]:
        raise ValueError("Step 07 manual review input changed")
    tables = [pq.read_table(path) for path in required[1:3]]
    if any(table.schema != images.SCHEMA for table in tables) or not tables[0].equals(
        tables[1]
    ):
        raise ValueError(
            "Image processing and QC tables disagree or have invalid schemas"
        )
    records = tables[1].to_pylist()
    by_sample = {r["sample_id"]: r for r in records}
    if len(by_sample) != len(records) or set(by_sample) != {
        r["sample_id"] for r in labels
    }:
        raise ValueError("QC keys are missing, duplicated or extra")
    reviews = {}
    for label in labels:
        row = by_sample[label["sample_id"]]
        if any(row[k] != label[k] for k in images.KEY_COLUMNS):
            raise ValueError("QC image or sample identity disagrees with formal labels")
        if (
            row["qc_status"] not in ("pass", "fail")
            or row["review_complete"] is not True
        ):
            raise ValueError("Unresolved image QC")
        if (
            row["processing_config_sha256"] != metadata["processing_config_sha256"]
            or row["processing_version"] != metadata["processing_version"]
            or json.loads(row["source_version"])
            != _file_version(indexed[row["sample_id"]]["path"])
        ):
            raise ValueError("Image source version or processing binding changed")
        if row["reviewer"]:
            if not review_path or not all(
                row[k] for k in ("reviewed_on", "review_version", "review_rationale")
            ):
                raise ValueError("Incomplete manual QC provenance")
            date.fromisoformat(row["reviewed_on"])
            if row["review_source_run_id"] and not re.fullmatch(
                r"[0-9a-f]{32}", row["review_source_run_id"]
            ):
                raise ValueError("Invalid manual review source run")
            reviews[row["sample_id"]] = {
                **{k: row[k] for k in images.REVIEW_COLUMNS[:4]},
                "decision": "pass" if row["qc_status"] == "pass" else "exclude",
                "rationale": row["review_rationale"],
                "reviewer": row["reviewer"],
                "reviewed_on": row["reviewed_on"],
                "version": row["review_version"],
                "source_run_id": row["review_source_run_id"],
            }
    replay = copy.deepcopy(records)
    for row in replay:
        row["issues"] = json.loads(row["issues"])
    images.assign_qc(replay, settings, reviews)
    for original, repeated in zip(records, replay):
        if any(
            original[k] != repeated[k]
            for k in (
                "qc_status",
                "auto_qc_status",
                "qc_reason",
                "review_required",
                "review_complete",
                "sampled_for_review",
            )
        ):
            raise ValueError(
                "QC decisions do not reproduce reviewed processing results"
            )
    counts = {
        status: _counts([r for r in records if r["qc_status"] == status])
        for status in ("pass", "fail", "pending_review")
    }
    if (
        counts != metadata["qc_counts"]
        or _counts(records)
        != {k: metadata[f"total_{k}"] for k in ("studies", "admissions", "patients")}
        or metadata["unreviewed_studies"] != 0
        or metadata["review_required_studies"]
        != sum(r["review_required"] for r in records)
        or not counts["pass"]["studies"]
    ):
        raise ValueError("QC counts do not reconcile")
    manifest_files = image_manifest["files"]
    with _progress(
        "Verify included NPY", unit="image", total=counts["pass"]["studies"]
    ) as progress:
        for row in records:
            if row["qc_status"] != "pass":
                continue
            expected = f"images/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}.npy"
            if row["processing_status"] != "complete" or row["npy_relpath"] != expected:
                raise ValueError(
                    "Included image is incomplete or has an invalid NPY layout"
                )
            path = _under_root(image_dir, row["npy_relpath"])
            if (
                not row["npy_sha256"]
                or manifest_files.get(str(path)) != row["npy_sha256"]
                or _file_hash(path) != row["npy_sha256"]
            ):
                raise ValueError("Included NPY hash changed or is not published")
            validate_pixels(np.load(path, allow_pickle=False), settings["size"])
            progress.update(1)
    decision = _json_file(outputs / "audit/burden_v2/endpoint_decision.json")
    source05 = _json_file(outputs / "audit/burden_v2/run_metadata.json")
    source06 = _json_file(outputs / "audit/burden_v2/step_06_run_metadata.json")
    binding = {
        "source_hashes": context["source_hashes"],
        "step_07_manifest_sha256": _file_hash(outputs / "audit/step_07_manifest.json"),
        "step_07_metadata_sha256": _file_hash(metadata_path),
        "source_image_run_id": metadata["run_id"],
        "config_sha256": _config_hash(configs),
        "config_files": {str(path): _file_hash(path) for path in configs},
        "processing_config_path": str(processing),
        "processing_config_sha256": _file_hash(processing),
        "processing_version": settings["processing_version"],
        "reviews_path": str(review_path) if review_path else None,
        "reviews_sha256": metadata["reviews_sha256"],
        "code_sha256": _code_hash(),
        "horizon_hours": decision["horizon_hours"],
    }
    context.update(
        decision=decision,
        labels_path=data / "study_burden_labels.parquet",
        evidence_path=data / "event_evidence.parquet",
        qc_records=records,
        source05=source05,
        source06=source06,
        source07=metadata,
        binding=binding,
        fingerprint=build_labels._json_hash(binding),
        image_dir=image_dir,
    )
    return context


def validate_cohort(connection, context: dict) -> tuple[pa.Table, pa.Table]:
    """Reconstruct final evidence, then retain original label types in the intersection."""
    decision = context["decision"]
    artifacts = {**context["artifacts"], "evidence": context["evidence_path"]}
    counts = build_labels.validate_labels(
        connection, artifacts, decision["horizon_hours"]
    )
    build_labels._reject(
        connection,
        "SELECT count(*) FROM evidence WHERE review_status IS NULL OR review_status NOT IN ('not_required','reviewed') OR event_status IS NULL OR event_status NOT IN ('positive','preexisting_only','no_qualifying_record')",
        "Unresolved final evidence status",
    )
    if counts != decision["window_comparison"][str(decision["horizon_hours"])]:
        raise ValueError("Formal label counts disagree with frozen decision")
    labels = pq.read_table(context["labels_path"])
    wide_schema = pq.read_schema(context["artifacts"]["wide"])
    if (
        len(set(labels.column_names)) != len(labels.column_names)
        or set(labels.column_names) != set(wide_schema.names) | set(PUBLICATION_FIELDS)
        or any(
            labels.schema.field(k).type != field.type
            for k, field in zip(wide_schema.names, wide_schema)
        )
    ):
        raise ValueError("Formal label schema changed")
    connection.register("formal", labels)
    mismatch = " OR ".join(f"l.{c} IS DISTINCT FROM w.{c}" for c in wide_schema.names)
    build_labels._reject(
        connection,
        f"SELECT count(*) FROM formal l FULL JOIN scored w USING(sample_id) WHERE l.sample_id IS NULL OR w.sample_id IS NULL OR {mismatch}",
        "Formal labels disagree with reconstructed final evidence",
    )
    build_labels._reject(
        connection,
        "SELECT count(*) FROM (SELECT sample_id FROM formal GROUP BY sample_id HAVING count(*)<>1)",
        "Duplicate formal sample key",
    )
    build_labels._reject(
        connection,
        "SELECT count(*) FROM evidence e LEFT JOIN formal l USING(sample_id) WHERE l.sample_id IS NULL",
        "Formal evidence contains excluded candidates",
    )
    versions = dict(
        decision_version=decision["decision_version"],
        rules_version=decision["rules_version"],
        dictionary_version=decision["dictionary_version"],
        source_audit_run_id=context["source05"]["run_id"],
        label_schema_version=build_labels.LABEL_SCHEMA_VERSION,
    )
    label_rows = sorted(labels.to_pylist(), key=lambda r: r["sample_id"])
    if any(any(row[k] != v for k, v in versions.items()) for row in label_rows):
        raise ValueError("Formal label version provenance mismatch")
    qc = {r["sample_id"]: r for r in context["qc_records"]}
    formal = {r["sample_id"]: r for r in label_rows}
    extra_fields = [(k, pa.string()) for k in QC_FIELDS] + [
        ("cohort_version", pa.string())
    ]
    included = []
    for row in label_rows:
        if qc[row["sample_id"]]["qc_status"] == "pass":
            included.append(
                {
                    **row,
                    **{k: qc[row["sample_id"]][v] for k, v in QC_FIELDS.items()},
                    "cohort_version": context["release_version"],
                }
            )
    cohort = pa.Table.from_pylist(
        included, schema=pa.schema(list(labels.schema) + extra_fields)
    )
    audit_fields = (
        [(k, labels.schema.field(k).type) for k in PUBLICATION_FIELDS]
        + extra_fields
        + [
            ("cohort_included", pa.uint8()),
            ("qc_performed", pa.uint8()),
            ("cohort_exclusion_stage", pa.string()),
            ("cohort_exclusion_reason", pa.string()),
        ]
    )
    rows = []
    for row in pq.read_table(context["artifacts"]["wide"]).to_pylist():
        key = row["sample_id"]
        image = qc.get(key)
        selected = image is not None and image["qc_status"] == "pass"
        rows.append(
            {
                **row,
                **{
                    k: formal[key][k] if key in formal else None
                    for k in PUBLICATION_FIELDS
                },
                **{k: image[v] if image else None for k, v in QC_FIELDS.items()},
                "cohort_version": context["release_version"],
                "cohort_included": int(selected),
                "qc_performed": int(image is not None),
                "cohort_exclusion_stage": ""
                if selected
                else "image_qc"
                if image
                else "upstream_candidate",
                "cohort_exclusion_reason": ""
                if selected
                else (image["review_rationale"] or image["qc_reason"])
                if image
                else row["exclusion_reason"],
            }
        )
    cohort_audit = pa.Table.from_pylist(
        sorted(rows, key=lambda r: r["sample_id"]),
        schema=pa.schema(list(wide_schema) + audit_fields),
    )
    connection.register("common", cohort)
    connection.register("cohort_audit", cohort_audit)
    return cohort, cohort_audit


def write_reports(
    connection,
    run: _OutputRun,
    directory: Path,
    context: dict,
    cohort: pa.Table,
    candidate_audit: pa.Table,
) -> dict:
    rows, candidates = cohort.to_pylist(), candidate_audit.to_pylist()
    excluded = [r for r in candidates if r["cohort_exclusion_stage"] == "image_qc"]
    flow = [
        {"stage": stage, **_counts(group)}
        for stage, group in (
            ("all_candidates", candidates),
            ("upstream_excluded", [r for r in candidates if r["qc_performed"] == 0]),
            ("formal_labels", [r for r in candidates if r["qc_performed"] == 1]),
            ("qc_excluded", excluded),
            ("common_cohort", rows),
        )
    ]
    removed = {
        "wholly_removed_admissions": len(
            {r["hadm_id"] for r in excluded} - {r["hadm_id"] for r in rows}
        ),
        "wholly_removed_patients": len(
            {r["subject_id"] for r in excluded} - {r["subject_id"] for r in rows}
        ),
    }
    _write_csv(
        run.path(directory / "cohort_flow.csv"),
        ["stage", "studies", "admissions", "patients"],
        flow,
    )
    exclusion_columns = [
        *images.KEY_COLUMNS,
        "burden_score",
        "phys_death",
        "high_burden",
        "qc_reason",
        "qc_review_rationale",
        "qc_reviewer",
        "qc_reviewed_on",
        "qc_review_version",
        "qc_review_source_run_id",
    ]
    _write_csv(
        run.path(directory / "cohort_qc_exclusions.csv"),
        exclusion_columns,
        [{k: r[k] for k in exclusion_columns} for r in excluded],
    )
    feasibility = []
    for kind, values in (
        (
            "cohort",
            [("total", rows), ("nonzero", [r for r in rows if r["burden_score"] > 0])],
        ),
        (
            "score",
            [
                (str(score), [r for r in rows if r["burden_score"] == score])
                for score in range(9)
            ],
        ),
        (
            "endpoint",
            [
                (name, [r for r in rows if r[name] == 1])
                for name in (
                    *COMPONENTS,
                    "phys_death",
                    "high_burden",
                    "early_discharge",
                )
            ],
        ),
        (
            "coverage",
            [
                (name, [r for r in rows if r[name] == 1])
                for name in (
                    "spo2_source_available",
                    "emar_source_available",
                    "procedure_source_available",
                )
            ],
        ),
        ("qc", [("excluded", excluded)]),
        (
            "early_3h",
            [
                (
                    "any",
                    [
                        r
                        for r in rows
                        if any(
                            r[f"{c}_first_event_time"] is not None
                            and r[f"{c}_first_event_time"]
                            <= min(r["t0"] + timedelta(hours=3), r["observation_end"])
                            for c in COMPONENTS
                        )
                    ],
                )
            ],
        ),
    ):
        feasibility.extend(
            {"category": kind, "value": value, **_counts(group)}
            for value, group in values
        )
    per_patient = Counter(r["subject_id"] for r in rows)
    for count in sorted(set(per_patient.values())):
        feasibility.append(
            {
                "category": "studies_per_patient",
                "value": str(count),
                **_counts([r for r in rows if per_patient[r["subject_id"]] == count]),
            }
        )
    _write_csv(
        run.path(directory / "training_feasibility.csv"),
        [
            "category",
            "value",
            "studies",
            "admissions",
            "patients",
            "denominator_studies",
            "study_proportion",
        ],
        [
            {
                **r,
                "denominator_studies": len(context["qc_records"])
                if r["category"] == "qc"
                else len(rows),
                "study_proportion": r["studies"]
                / (len(context["qc_records"]) if r["category"] == "qc" else len(rows)),
            }
            for r in feasibility
        ],
    )
    overlap = audit.query_rows(
        connection,
        "SELECT e.component,e.source_file,e.source_record_key,count(DISTINCT e.sample_id) linked_studies,count(DISTINCT e.hadm_id) linked_admissions,count(DISTINCT e.subject_id) linked_patients,count(DISTINCT e.sample_id) FILTER(WHERE e.qualifies=1) qualifying_linked_studies,string_agg(DISTINCT e.sample_id, ';' ORDER BY e.sample_id) sample_ids FROM evidence e JOIN common c USING(sample_id) GROUP BY e.component,e.source_file,e.source_record_key ORDER BY e.component,e.source_record_key",
    )
    # One stable source key identifies one file within its component.
    build_labels._reject(
        connection,
        "SELECT count(*) FROM (SELECT e.component,e.source_record_key FROM evidence e JOIN common c USING(sample_id) GROUP BY ALL HAVING count(DISTINCT source_file)<>1 OR count(DISTINCT e.subject_id)<>1 OR count(DISTINCT e.hadm_id)<>1)",
        "Conflicting stable source ownership",
    )
    _write_csv(
        run.path(directory / "source_overlap.csv"),
        [
            "component",
            "source_file",
            "source_record_key",
            "linked_studies",
            "linked_admissions",
            "linked_patients",
            "qualifying_linked_studies",
            "sample_ids",
        ],
        overlap,
    )
    summary = {
        "counts": _counts(rows),
        "candidate_counts": _counts(candidates),
        "flow": flow,
        "qc_loss": {**_counts(excluded), **removed},
        "nonzero_studies": sum(r["burden_score"] > 0 for r in rows),
        "phys_death_studies": sum(r["phys_death"] for r in rows),
        "high_burden_studies": sum(r["high_burden"] for r in rows),
        "feasibility": feasibility,
        "unique_component_source_records": len(overlap),
        "shared_component_source_records": sum(
            r["linked_studies"] > 1 for r in overlap
        ),
        "qualifying_component_source_records": sum(
            r["qualifying_linked_studies"] > 0 for r in overlap
        ),
        "shared_qualifying_component_source_records": sum(
            r["qualifying_linked_studies"] > 1 for r in overlap
        ),
        "source_counts": dict(Counter(r["component"] for r in overlap)),
        "early_discharge_proportion": sum(r["early_discharge"] for r in rows)
        / len(rows),
    }
    warnings = context["source05"]["csv_tail_checks"]
    report = (
        f"# Step 08 common cohort freeze\n\nRelease: `{context['release_version']}`. Frozen H: {context['decision']['horizon_hours']} hours.\n\n"
        + "| Stage | Studies | Admissions | Patients |\n| --- | ---: | ---: | ---: |\n"
        + "\n".join(
            f"| {r['stage']} | {r['studies']} | {r['admissions']} | {r['patients']} |"
            for r in flow
        )
        + f"\n\nQC exclusion affects {len(excluded)} studies, {summary['qc_loss']['admissions']} admissions and {summary['qc_loss']['patients']} patients; it wholly removes {removed['wholly_removed_admissions']} admissions and {removed['wholly_removed_patients']} patients. Stage-specific patient/admission counts overlap and must not be added. See cohort_qc_exclusions.csv for sample-specific human reasons.\n\n"
        + f"Nonzero scores: {summary['nonzero_studies']}; phys_death: {summary['phys_death_studies']}; high_burden: {summary['high_burden_studies']}. training_feasibility.csv includes all scores 0-8, components, source coverage, alive early discharge, early events and study counts per patient. Events in the first three hours and before alive early discharge remain in the original labels.\n\n"
        + f"source_overlap.csv lists {len(overlap)} distinct (component, stable source key) records, including baseline and excluded evidence, with {summary['shared_component_source_records']} linked to multiple studies. Of these, {summary['qualifying_component_source_records']} qualify as events and {summary['shared_qualifying_component_source_records']} qualify for multiple studies. Source records and study associations are not independent patients. Alive early discharge proportion: {summary['early_discharge_proportion']:.6f}.\n\n"
        + "All formal fields and types are preserved. Upstream exclusions retain null labels and null QC fields (qc_performed=0). Image exclusions retain their formal labels. No additional exclusions or operational zeros are introduced.\n\n"
        + f"Step 08 parses no CSV: ../step_08_csv_tail_checks.csv records no_csv_inputs. Step 05 accepted {warnings['incomplete_file_count']} incomplete source files and skipped {warnings['skipped_record_count']} confirmed EOF records; see ../step_05_csv_tail_checks.csv. Step 07 review CSV checks: {context['source07']['csv_tail_checks']['status']}; see ../step_07_csv_tail_checks.csv. Missing records do not prove clinical absence; review source provenance manually.\n\n"
        + "Snapshots bind sources, configurations, code, human review and output hashes. The unchanged version can only be verified/reused; changes require an explicit new release. NPYs remain in the image cache and are not copied into snapshots. Snapshots are not portable training packages. All subsequent model comparisons must use this same common cohort. Patient splits, EHR features and final image manifests are deferred.\n"
    )
    run.path(directory / "step_08_report.md").write_text(report, encoding="utf-8")
    return summary


def freeze_common_cohort(
    root: Path,
    paths_file: Path,
    cohort_file: Path,
    ehr_file: Path,
    imaging_file: Path,
    processing_file: Path,
    release_version: str | None = None,
) -> int:
    started = datetime.now(timezone.utc)
    configs = [paths_file, cohort_file, ehr_file, imaging_file]
    context = verify_inputs(root, configs, processing_file)
    horizon = context["decision"]["horizon_hours"]
    version = release_version or f"burden_v2_h{horizon}_r01"
    if not re.fullmatch(rf"burden_v2_h{horizon}_r(?:0[1-9]|[1-9][0-9]+)", version):
        raise ValueError(
            "Release version must match the frozen horizon and a positive revision"
        )
    context["release_version"] = version
    data = context["processed"] / "cohort/burden_v2"
    directory = context["outputs"] / "audit/burden_v2"
    snapshot_data = _under_root(data, f"releases/{version}")
    snapshot_audit = _under_root(directory, f"releases/{version}")
    freeze_path = snapshot_audit / "cohort_freeze.json"
    print(f"Verify common cohort for {version}; CSV tail checks: no_csv_inputs")
    with _OutputRun(context["outputs"], 8, context["processed"]) as run:
        if run.manifest.exists():
            _verify_run(context["outputs"], 8)
        with tempfile.TemporaryDirectory(
            prefix=".step_08_", dir=run.manifest.parent
        ) as temporary:
            with open_store(Path(temporary), context["memory_limit"]) as connection:
                cohort, candidate_audit = validate_cohort(connection, context)
                locations = {name: snapshot_data / name for name in TABLE_NAMES} | {
                    name: snapshot_audit / name for name in REPORT_NAMES
                }
                reused = freeze_path.exists()
                if reused:
                    frozen = _json_file(freeze_path)
                    if (
                        frozen["release_version"] != version
                        or frozen["horizon_hours"] != horizon
                        or frozen["source_fingerprint"] != context["fingerprint"]
                        or frozen["source_binding"] != context["binding"]
                        or frozen.get("common_cohort_frozen") is not True
                        or set(frozen["artifact_sha256"]) != set(locations)
                    ):
                        raise ValueError(
                            "Frozen release sources changed; specify a new release version"
                        )
                    for name, path in locations.items():
                        if _file_hash(path) != frozen["artifact_sha256"][name]:
                            raise ValueError("Frozen snapshot hash changed")
                    if not pq.read_table(locations["cohort.parquet"]).equals(
                        cohort
                    ) or not pq.read_table(locations["cohort_audit.parquet"]).equals(
                        candidate_audit
                    ):
                        raise ValueError("Frozen snapshot samples or labels changed")
                else:
                    if any(p.exists() for p in locations.values()):
                        raise ValueError(
                            "Incomplete existing release snapshot; preserve it for investigation"
                        )
                    pq.write_table(
                        cohort,
                        run.path(locations["cohort.parquet"]),
                        compression="zstd",
                    )
                    pq.write_table(
                        candidate_audit,
                        run.path(locations["cohort_audit.parquet"]),
                        compression="zstd",
                    )
                    summary = write_reports(
                        connection,
                        run,
                        snapshot_audit,
                        context,
                        cohort,
                        candidate_audit,
                    )
                    frozen = {
                        "release_version": version,
                        "horizon_hours": horizon,
                        "common_cohort_frozen": True,
                        "original_freeze_run_id": run.run_id,
                        "frozen_utc": started.isoformat(),
                        "git_commit": _git_commit(root),
                        "source_fingerprint": context["fingerprint"],
                        "source_binding": context["binding"],
                        "summary": summary,
                        "image_cache_root": str(context["image_dir"]),
                        "source_csv_tail_checks": context["source05"][
                            "csv_tail_checks"
                        ],
                        "review_csv_tail_checks": context["source07"][
                            "csv_tail_checks"
                        ],
                        "artifact_sha256": {
                            name: _file_hash(run.path(path))
                            for name, path in locations.items()
                        },
                    }
                    run.path(freeze_path).write_text(
                        json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
                    )
                for name, path in locations.items():
                    source = path if reused else run.path(path)
                    current = data / name if name in TABLE_NAMES else directory / name
                    shutil.copyfile(source, run.path(current))
                shutil.copyfile(
                    freeze_path if reused else run.path(freeze_path),
                    run.path(directory / "cohort_freeze.json"),
                )
                # The current freeze record binds verified snapshot hashes; reused snapshots stay untouched.
                latest = verify_inputs(root, configs, processing_file)
                if latest["binding"] != context["binding"]:
                    raise ValueError("Frozen inputs changed during Step 08")
                metadata = {
                    "run_id": run.run_id,
                    "original_freeze_run_id": frozen["original_freeze_run_id"],
                    "verification_run_id": run.run_id,
                    "snapshot_reused": reused,
                    "common_cohort_frozen": True,
                    "cohort_checks_passed": True,
                    "release_version": version,
                    "horizon_hours": horizon,
                    "source_binding": context["binding"],
                    "source_fingerprint": context["fingerprint"],
                    "summary": frozen["summary"],
                    "artifact_sha256": frozen["artifact_sha256"],
                    "freeze_sha256": _file_hash(
                        freeze_path if reused else run.path(freeze_path)
                    ),
                    "snapshot_data_path": str(snapshot_data),
                    "snapshot_audit_path": str(snapshot_audit),
                    "csv_tail_checks": run.csv_tail_summary(),
                    "source_csv_tail_checks": context["source05"]["csv_tail_checks"],
                    "review_csv_tail_checks": context["source07"]["csv_tail_checks"],
                    "started_utc": started.isoformat(),
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "code_sha256": _code_hash(),
                    "git_commit": _git_commit(root),
                }
                run.path(directory / "step_08_run_metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )
    print(
        f"Step 08 complete: {version}; {cohort.num_rows} common studies; snapshot {'reused' if reused else 'frozen'}"
    )
    return 0
