"""Step 07 publication, deterministic sampling and hash-bound image review."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import pyarrow as pa
import pyarrow.parquet as pq
import pydicom
from PIL import Image
from pydicom import config as dicom_config

from src.cohort import build_labels
from src.cohort.burden_io import clinical_time
from src.cohort.common import (
    _acquisition_time,
    _config_hash,
    _csv_reader,
    _file_hash,
    _file_version,
    _git_commit,
    _json_file,
    _OutputRun,
    _progress,
    _resolve_config_path,
    _under_root,
    _verify_run,
    _write_csv,
)
from src.imaging.preprocessing import (
    process_dataset,
    validate_pixels,
    validate_settings,
)

REVIEW_COLUMNS = (
    "sample_id source_dicom_sha256 processing_config_sha256 npy_sha256 decision "
    "rationale reviewer reviewed_on version"
).split()
KEY_COLUMNS = (
    "sample_id subject_id hadm_id study_id dicom_id dicom_path view_position t0".split()
)
STRING_COLUMNS = (
    "sample_id dicom_id dicom_path view_position source_dicom_sha256 source_version "
    "decoded_pixel_sha256 transfer_syntax_uid photometric_interpretation "
    "modality_path voi_path presentation_path display_metadata processing_status error_code "
    "issues processing_config_sha256 processing_version npy_relpath npy_sha256 before_relpath "
    "after_relpath before_method duplicate_groups auto_qc_status qc_status qc_reason "
    "reviewer reviewed_on review_version review_rationale review_source_run_id"
).split()
INT_COLUMNS = (
    "original_height original_width resized_height resized_width pad_top pad_bottom pad_left pad_right"
).split()
FLOAT_COLUMNS = (
    "p1 p99 valid_pixel_fraction clipped_low_fraction clipped_high_fraction".split()
)
SCHEMA = pa.schema(
    [(name, pa.string()) for name in STRING_COLUMNS]
    + [(name, pa.int64()) for name in ("subject_id", "hadm_id", "study_id")]
    + [("t0", pa.timestamp("us"))]
    + [(name, pa.int32()) for name in INT_COLUMNS]
    + [(name, pa.float64()) for name in FLOAT_COLUMNS]
    + [
        (name, pa.bool_())
        for name in (
            "inversion_applied",
            "review_required",
            "review_complete",
            "sampled_for_review",
        )
    ]
)


def load_inputs(root: Path, configs: list[Path]) -> tuple[list[dict], dict, dict]:
    decision, _, context = build_labels.verify_frozen_inputs(root, configs)
    outputs, processed = context["outputs"], context["processed"]
    label_path = processed / "cohort/burden_v2/study_burden_labels.parquet"
    index_path = processed / "index/cxr_index.parquet"
    selection_path = processed / "index/study_image_selection.parquet"
    for step, path in ((2, index_path), (3, selection_path), (6, label_path)):
        _verify_run(outputs, step, path)
    selection_summary = outputs / "audit/study_image_selection_summary.json"
    _verify_run(outputs, 3, selection_summary)
    if _json_file(selection_summary)["input_parquet_sha256"] != _file_hash(index_path):
        raise ValueError("Selected images refer to a different Step 02 index")
    metadata_path = outputs / "audit/burden_v2/step_06_run_metadata.json"
    metadata = _json_file(metadata_path)
    if (
        metadata["run_id"]
        != _json_file(outputs / "audit/step_06_manifest.json")["run_id"]
        or metadata["decision_sha256"] != context["decision_sha256"]
        or metadata["source_audit"] != decision["source_audit"]
        or metadata["selected_window_artifact_sha256"]
        != decision["selected_window_artifact_sha256"]
    ):
        raise ValueError("Formal label provenance does not match the frozen decision")
    if (
        not all(
            metadata.get(k) is True
            for k in (
                "formal_labels_published",
                "label_checks_passed",
                "horizon_frozen",
                "label_rules_frozen",
            )
        )
        or metadata["horizon_hours"] != decision["horizon_hours"]
        or metadata["counts"]
        != decision["window_comparison"][str(decision["horizon_hours"])]
    ):
        raise ValueError("Formal labels are not ready for image QC")
    directory = Path(build_labels.__file__).parents[2]
    if metadata["code_sha256"] != _config_hash(
        [Path(build_labels.__file__), directory / "scripts/06_build_burden_labels.py"]
    ):
        raise ValueError("Step 06 code changed since label publication")
    labels = pq.read_table(
        label_path, columns=KEY_COLUMNS + ["label_status", "candidate_status"]
    ).to_pylist()
    if (
        len(labels) != metadata["counts"]["eligible_studies"]
        or len({r["sample_id"] for r in labels}) != len(labels)
        or len({(r["subject_id"], r["hadm_id"], r["study_id"]) for r in labels})
        != len(labels)
    ):
        raise ValueError("Formal label keys or counts are invalid")
    selected = {}
    for row in pq.read_table(selection_path).to_pylist():
        key = (row["subject_id"], row["study_id"])
        if key in selected:
            raise ValueError("Duplicate selected study")
        selected[key] = row
    index = {}
    columns = "subject_id study_id dicom_id dicom_path view_position acquisition_datetime rows columns".split()
    for row in pq.read_table(index_path, columns=columns).to_pylist():
        key = (row["subject_id"], row["study_id"], row["dicom_id"])
        if key in index:
            raise ValueError("Duplicate indexed image")
        index[key] = row
    paths = _json_file(configs[0])
    imaging = _json_file(configs[3])
    cxr = _resolve_config_path(paths["mimic_cxr"], root, raw=True)
    dicom_dir = _under_root(cxr, imaging["raw_audit"]["dicom_directory"])
    matched = {}
    for row in labels:
        if (
            row["label_status"] != "complete"
            or row["candidate_status"] != "eligible"
            or any(
                type(row[k]) is not int or row[k] <= 0
                for k in ("subject_id", "hadm_id", "study_id")
            )
        ):
            raise ValueError("Invalid formal label eligibility")
        sample = f"p{row['subject_id']}_h{row['hadm_id']}_s{row['study_id']}"
        chosen = selected.get((row["subject_id"], row["study_id"]))
        indexed = index.get((row["subject_id"], row["study_id"], row["dicom_id"]))
        if (
            row["sample_id"] != sample
            or not chosen
            or not indexed
            or row["dicom_id"] != chosen["selected_dicom_id"]
            or row["view_position"] not in ("AP", "PA")
        ):
            raise ValueError("Formal labels do not match selected/indexed images")
        for other, time_field in (
            (chosen, "acquisition_datetime"),
            (indexed, "acquisition_datetime"),
        ):
            if any(row[k] != other[k] for k in ("dicom_path", "view_position")) or row[
                "t0"
            ] != clinical_time(other[time_field]):
                raise ValueError("Selected image path, view or t0 changed")
        path = _under_root(cxr, row["dicom_path"])
        tail = (
            f"p{row['subject_id']}",
            f"s{row['study_id']}",
            row["dicom_id"] + ".dcm",
        )
        standard = ("files", "p" + str(row["subject_id"])[:2], *tail)
        if (
            not re.fullmatch(r"[0-9a-fA-F-]+", row["dicom_id"])
            or not path.is_relative_to(dicom_dir)
            or path.relative_to(dicom_dir).parts not in (tail, standard)
        ):
            raise ValueError("Selected DICOM path/key mismatch")
        matched[sample] = {**indexed, "path": path}
    context["source_hashes"] = {
        "decision_sha256": context["decision_sha256"],
        "label_sha256": _file_hash(label_path),
        "metadata_sha256": _file_hash(metadata_path),
        "index_sha256": _file_hash(index_path),
        "selection_sha256": _file_hash(selection_path),
        "selection_summary_sha256": _file_hash(selection_summary),
        "upstream_manifests": {
            f"step_{step:02d}": {
                "sha256": _file_hash(outputs / f"audit/step_{step:02d}_manifest.json"),
                "run_id": _json_file(outputs / f"audit/step_{step:02d}_manifest.json")[
                    "run_id"
                ],
            }
            for step in range(2, 7)
        },
    }
    context["source_label_run_id"] = metadata["run_id"]
    return sorted(labels, key=lambda r: r["sample_id"]), matched, context


def load_reviews(path: Path | None, run: _OutputRun) -> dict:
    if path is None:
        return {}
    if any(path.is_relative_to(p) for p in run.roots):
        raise ValueError("Keep versioned review inputs outside generated output roots")
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = _csv_reader(stream, run.check_csv(path))
        if (
            not reader.fieldnames
            or len(reader.fieldnames) != len(set(reader.fieldnames))
            or set(REVIEW_COLUMNS) - set(reader.fieldnames)
        ):
            raise ValueError("Missing or duplicate image review columns")
        for row in reader:
            if None in row:
                raise ValueError("Unexpected extra fields in image review CSV")
            source_run_id = (row.get("source_run_id") or "").strip()
            row = {k: row[k].strip() for k in REVIEW_COLUMNS}
            if source_run_id and not re.fullmatch(r"[0-9a-f]{32}", source_run_id):
                raise ValueError("Invalid image review source run ID")
            row["source_run_id"] = source_run_id
            if (
                row["sample_id"] in result
                or row["decision"] not in {"pass", "exclude"}
                or not all(
                    row[k]
                    for k in (
                        "sample_id",
                        "rationale",
                        "reviewer",
                        "reviewed_on",
                        "version",
                        "processing_config_sha256",
                    )
                )
            ):
                raise ValueError("Invalid or duplicate image review decision")
            date.fromisoformat(row["reviewed_on"])
            for key in (
                "source_dicom_sha256",
                "npy_sha256",
                "processing_config_sha256",
            ):
                if row[key] and not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                    raise ValueError("Invalid image review hash")
            result[row["sample_id"]] = row
    if len({r["version"] for r in result.values()}) > 1:
        raise ValueError("Use one review version per CSV")
    return result


def assign_qc(records: list[dict], settings: dict, reviews: dict) -> list[dict]:
    """Sampling and duplicate review use imaging metadata only, never endpoint labels."""
    for row in records:
        row["_duplicate_groups"] = []
        row["sampled_for_review"] = False
        for field in (
            "reviewer",
            "reviewed_on",
            "review_version",
            "review_rationale",
            "review_source_run_id",
        ):
            row[field] = ""
    duplicates = []
    for field in ("source_dicom_sha256", "decoded_pixel_sha256", "npy_sha256"):
        groups = defaultdict(list)
        for row in records:
            if row.get(field):
                groups[row[field]].append(row)
        for digest, group in sorted(groups.items()):
            if len(group) > 1:
                group_id = field + ":" + digest
                for row in group:
                    row["_duplicate_groups"].append(group_id)
                    duplicates.append(
                        dict(
                            group_id=group_id,
                            hash_type=field,
                            sample_id=row["sample_id"],
                        )
                    )
    strata = defaultdict(list)
    for row in records:
        if (
            row["processing_status"] == "complete"
            and not row["issues"]
            and not row["_duplicate_groups"]
        ):
            strata[
                tuple(
                    row[k]
                    for k in (
                        "view_position",
                        "photometric_interpretation",
                        "modality_path",
                        "voi_path",
                        "presentation_path",
                    )
                )
            ].append(row)
    for group in strata.values():
        ranked = sorted(
            group,
            key=lambda r: hashlib.sha256(
                f"{settings['review_seed']}:{r['sample_id']}".encode()
            ).hexdigest(),
        )
        for row in ranked[: settings["review_per_group"]]:
            row["sampled_for_review"] = True
    ids = {r["sample_id"] for r in records}
    if set(reviews) - ids:
        raise ValueError("Image review references an unpublished study")
    for row in records:
        reasons = list(row["issues"])
        if row["error_code"]:
            reasons.append(row["error_code"])
        if row["_duplicate_groups"]:
            reasons.append("duplicate_image")
        if row["sampled_for_review"]:
            reasons.append("stratified_sample")
        row["review_required"] = bool(reasons) or row["processing_status"] != "complete"
        row["auto_qc_status"] = (
            "fail"
            if row["processing_status"] == "fail"
            else "pending_review"
            if row["review_required"]
            else "pass"
        )
        row["qc_status"] = row["auto_qc_status"]
        row["qc_reason"] = ";".join(reasons) or "automatic_checks_passed"
        row["review_complete"] = not row["review_required"]
        decision = reviews.get(row["sample_id"])
        if decision:
            for field in (
                "source_dicom_sha256",
                "processing_config_sha256",
                "npy_sha256",
            ):
                if decision[field] != (row.get(field) or ""):
                    raise ValueError(f"Stale image review: {row['sample_id']} {field}")
            if decision["decision"] == "pass" and (
                row["processing_status"] != "complete" or not row.get("npy_sha256")
            ):
                raise ValueError(
                    "Manual pass cannot bypass a hard failure or incomplete transform"
                )
            row.update(
                qc_status="pass" if decision["decision"] == "pass" else "fail",
                review_complete=True,
                reviewer=decision["reviewer"],
                reviewed_on=decision["reviewed_on"],
                review_version=decision["version"],
                review_rationale=decision["rationale"],
                review_source_run_id=decision.get("source_run_id", ""),
            )
    return duplicates


def _preview(run: _OutputRun, path: Path, pixels: np.ndarray, maximum: int) -> None:
    image = Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8))
    image.thumbnail((maximum, maximum), Image.Resampling.BILINEAR)
    image.save(run.path(path), format="PNG")


def _gallery(records: list[dict]) -> str:
    cards = []
    for row in sorted(
        records, key=lambda r: (not r["review_required"], r["sample_id"])
    ):
        pictures = []
        for field, caption in (
            ("before_relpath", row["before_method"]),
            ("after_relpath", "Prepared float32 NPY, displayed as PNG"),
        ):
            if row.get(field):
                pictures.append(
                    f'<figure><img loading="lazy" src="{html.escape(row[field], quote=True)}" alt="{html.escape(caption, quote=True)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
                )
        cards.append(
            f'<article data-required="{str(row["review_required"]).lower()}"><h2>{html.escape(row["sample_id"])}</h2><p>{html.escape(row["view_position"])} | {html.escape(row["qc_status"])} | {html.escape(row["qc_reason"])}</p><p>Polarity: {html.escape(row.get("photometric_interpretation", ""))}; modality: {html.escape(row.get("modality_path", ""))}; VOI: {html.escape(row.get("voi_path", ""))}; presentation: {html.escape(row.get("presentation_path", ""))}.</p><div class="pair">{"".join(pictures) or "No reliable preview available; inspect the failure reason."}</div></article>'
        )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Image QC review</title><style>body{font:16px system-ui;margin:24px;background:#f5f5f5;color:#17202a}header{position:sticky;top:0;background:#fff;padding:12px;border-bottom:1px solid #aaa}article{background:white;padding:16px;margin:16px 0;border:1px solid #ccc}.pair{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0;max-width:48%}img{max-width:100%;max-height:640px;background:#000}figcaption{max-width:640px}h2{font-size:18px}button{padding:8px;cursor:pointer}</style><header><h1>Image QC review</h1><p>Compare display direction, lung coverage, proportions, padding, clipping, artifacts, text and AP/PA. Outcome labels are hidden.</p><p>Record decisions in an independent versioned CSV using the generated template. A pass requires a validated NPY; failures need acknowledgment or correction.</p><button id="toggle">Show all images</button></header>"""
        + "".join(cards)
        + """<script>let requiredOnly=true;function update(){document.querySelectorAll('article').forEach(e=>e.hidden=requiredOnly&&e.dataset.required!=='true');document.querySelector('#toggle').textContent=requiredOnly?'Show all images':'Show required reviews only'}document.querySelector('#toggle').onclick=()=>{requiredOnly=!requiredOnly;update()};update()</script></html>\n"""
    )


def prepare_images(
    root: Path,
    paths_file: Path,
    cohort_file: Path,
    ehr_file: Path,
    imaging_file: Path,
    processing_file: Path,
    reviews_file: Path | None = None,
) -> int:
    started = datetime.now(timezone.utc)
    configs = [paths_file, cohort_file, ehr_file, imaging_file]
    settings = _json_file(processing_file)
    validate_settings(settings)
    processing_hash = _file_hash(processing_file)
    review_hash = _file_hash(reviews_file) if reviews_file else None
    code_files = [
        Path(__file__),
        Path(__file__).with_name("preprocessing.py"),
        Path(__file__).parents[2] / "scripts/07_qc_and_prepare_images.py",
    ]
    code_hash = _config_hash(code_files)
    labels, index, context = load_inputs(root, configs)
    outputs, processed = context["outputs"], context["processed"]
    image_dir = processed / "imaging/burden_v2"
    qc_dir = outputs / "qc/burden_v2"
    with _OutputRun(outputs, 7, processed) as run:
        reviews = load_reviews(reviews_file, run)
        records = []
        progress = _progress(
            "Prepare selected label images", unit="image", total=len(labels)
        )
        try:
            for label in labels:
                row = {k: label[k] for k in KEY_COLUMNS}
                row.update(
                    processing_status="fail",
                    error_code="",
                    issues=[],
                    _duplicate_groups=[],
                    sampled_for_review=False,
                    processing_version=settings["processing_version"],
                    processing_config_sha256=processing_hash,
                    before_method="unavailable",
                    modality_path="not_applied",
                    voi_path="not_applied",
                    presentation_path="not_applied",
                    source_dicom_sha256="",
                    npy_sha256="",
                    review_source_run_id=run.run_id,
                )
                info = index[row["sample_id"]]
                path = info["path"]
                version = _file_version(path)
                row["_source_version"] = version
                if not version["exists"]:
                    row["error_code"] = "missing_source_dicom"
                else:
                    try:
                        row["source_dicom_sha256"] = _file_hash(path)
                        dataset = pydicom.dcmread(path, force=False)
                    except Exception as error:
                        row["error_code"] = (
                            f"metadata_read_error:{type(error).__name__}"
                        )
                    else:
                        time, _, _ = _acquisition_time(dataset)
                        if (
                            clinical_time(time) != row["t0"]
                            or str(dataset.get("ViewPosition", "")).strip().upper()
                            != row["view_position"]
                            or dataset.get("Rows") != info["rows"]
                            or dataset.get("Columns") != info["columns"]
                        ):
                            raise ValueError(
                                f"DICOM source metadata changed: {row['sample_id']}"
                            )
                        row["transfer_syntax_uid"] = (
                            str(dataset.file_meta.get("TransferSyntaxUID", ""))
                            if hasattr(dataset, "file_meta")
                            else ""
                        )
                        result = process_dataset(dataset, settings)
                        pixels, before = result.pop("pixels"), result.pop("before")
                        row.update(result)
                        if pixels is not None:
                            relative = f"images/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}.npy"
                            target = _under_root(image_dir, relative)
                            staged = run.path(target)
                            with staged.open("wb") as stream:
                                np.save(stream, pixels, allow_pickle=False)
                            validate_pixels(np.load(staged, allow_pickle=False))
                            row.update(
                                npy_relpath=relative, npy_sha256=_file_hash(staged)
                            )
                            relative = f"images/after/{row['sample_id']}.png"
                            _preview(
                                run,
                                _under_root(qc_dir, relative),
                                pixels,
                                settings["preview_size"],
                            )
                            row["after_relpath"] = relative
                        if before is not None:
                            relative = f"images/before/{row['sample_id']}.png"
                            _preview(
                                run,
                                _under_root(qc_dir, relative),
                                before,
                                settings["preview_size"],
                            )
                            row["before_relpath"] = relative
                if _file_version(path) != version:
                    raise ValueError("A source DICOM changed during processing")
                records.append(row)
                progress.update(1)
        finally:
            progress.close()
        duplicates = assign_qc(records, settings, reviews)
        required = [r for r in records if r["review_required"]]
        template = []
        for row in required:
            decision = reviews.get(row["sample_id"], {})
            template.append(
                {
                    **{k: row.get(k, "") or "" for k in REVIEW_COLUMNS[:4]},
                    **{k: decision.get(k, "") for k in REVIEW_COLUMNS[4:]},
                    "dicom_id": row["dicom_id"],
                    "qc_reason": row["qc_reason"],
                    "auto_qc_status": row["auto_qc_status"],
                    "source_run_id": run.run_id,
                }
            )
        _write_csv(
            run.path(qc_dir / "image_qc_review.csv"),
            REVIEW_COLUMNS
            + ["dicom_id", "qc_reason", "auto_qc_status", "source_run_id"],
            template,
        )
        _write_csv(
            run.path(qc_dir / "duplicate_images.csv"),
            ["group_id", "hash_type", "sample_id"],
            duplicates,
        )
        review_fields = [
            "sample_id",
            "view_position",
            "photometric_interpretation",
            "modality_path",
            "voi_path",
            "presentation_path",
            "qc_reason",
        ]
        _write_csv(
            run.path(qc_dir / "review_sample.csv"),
            review_fields,
            [
                {key: r.get(key) for key in review_fields}
                for r in records
                if r["sampled_for_review"]
            ],
        )
        failure_fields = [
            "sample_id",
            "dicom_id",
            "processing_status",
            "error_code",
            "qc_status",
            "review_complete",
        ]
        _write_csv(
            run.path(qc_dir / "dicom_failures.csv"),
            failure_fields,
            [
                {key: r.get(key) for key in failure_fields}
                for r in records
                if r["processing_status"] != "complete"
            ],
        )
        run.path(qc_dir / "image_qc_review.html").write_text(
            _gallery(records), encoding="utf-8"
        )
        serialized = []
        for row in records:
            serialized.append(
                {
                    **{k: row.get(k) for k in SCHEMA.names},
                    "source_version": json.dumps(
                        row["_source_version"], sort_keys=True
                    ),
                    "display_metadata": json.dumps(
                        row.get("display_metadata", {}), sort_keys=True
                    ),
                    "issues": json.dumps(row["issues"]),
                    "duplicate_groups": json.dumps(row["_duplicate_groups"]),
                }
            )
        table = pa.Table.from_pylist(serialized, schema=SCHEMA)
        for name in ("image_processing.parquet", "image_qc_status.parquet"):
            pq.write_table(table, run.path(image_dir / name), compression="zstd")
        _, _, latest = load_inputs(root, configs)
        if (
            latest != context
            or _file_hash(processing_file) != processing_hash
            or _config_hash(code_files) != code_hash
            or (_file_hash(reviews_file) if reviews_file else None) != review_hash
            or any(
                _file_version(index[r["sample_id"]]["path"]) != r["_source_version"]
                for r in records
            )
        ):
            raise ValueError(
                "An upstream, configuration, review or image source changed during Step 07"
            )
        counts = {
            status: dict(
                studies=sum(r["qc_status"] == status for r in records),
                admissions=len(
                    {r["hadm_id"] for r in records if r["qc_status"] == status}
                ),
                patients=len(
                    {r["subject_id"] for r in records if r["qc_status"] == status}
                ),
            )
            for status in ("pass", "fail", "pending_review")
        }
        metadata = {
            "run_id": run.run_id,
            "processing_version": settings["processing_version"],
            "processing_settings": settings,
            "processing_config_path": str(processing_file),
            "processing_config_sha256": processing_hash,
            "reviews_path": str(reviews_file) if reviews_file else None,
            "reviews_sha256": review_hash,
            "source_label_run_id": context["source_label_run_id"],
            "source_hashes": context["source_hashes"],
            "csv_tail_checks": run.csv_tail_summary(),
            "image_checks_passed": True,
            "qc_review_complete": all(r["review_complete"] for r in records),
            "common_cohort_frozen": False,
            "ready_for_cohort_freeze": all(r["review_complete"] for r in records)
            and counts["pass"]["studies"] > 0
            and counts["pending_review"]["studies"] == 0,
            "total_studies": len(records),
            "total_admissions": len({r["hadm_id"] for r in records}),
            "total_patients": len({r["subject_id"] for r in records}),
            "qc_counts": counts,
            "processing_counts": dict(Counter(r["processing_status"] for r in records)),
            "view_counts": dict(Counter(r["view_position"] for r in records)),
            "review_required_studies": len(required),
            "unreviewed_studies": sum(not r["review_complete"] for r in records),
            "sampled_studies": sum(r["sampled_for_review"] for r in records),
            "duplicate_studies": len({r["sample_id"] for r in duplicates}),
            "npy_count": sum(bool(r.get("npy_relpath")) for r in records),
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(root),
            "code_sha256": code_hash,
            "runtime_versions": {
                "numpy": np.__version__,
                "pillow": PIL.__version__,
                "pydicom": pydicom.__version__,
            },
            "decoder_handlers": {
                m.HANDLER_NAME: m.is_available()
                for m in dicom_config.pixel_data_handlers
            },
        }
        run.path(qc_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        report = (
            "# Step 07 image preparation and QC\n\nImage QC is independent of outcome scores. "
            "All selected formal-label studies are retained in the QC tables, including failures. "
            "A complete run does not freeze the common training cohort.\n\n"
            + f"- Studies: {len(records)}\n- Admissions: {metadata['total_admissions']}\n- Patients: {metadata['total_patients']}\n"
            + "\n".join(
                f"- {status}: {value['studies']} studies, {value['admissions']} admissions, {value['patients']} patients"
                for status, value in counts.items()
            )
            + f"\n- Required reviews: {len(required)}\n- Unreviewed studies: {metadata['unreviewed_studies']}\n- QC review complete: {metadata['qc_review_complete']}\n\n"
            + f"Processing results: {json.dumps(metadata['processing_counts'], sort_keys=True)}. "
            + f"Validated NPY files: {metadata['npy_count']}.\n\n"
            + "Open image_qc_review.html for image pairs and reasons. Normal-image sampling uses a fixed seed and "
            "view/polarity/modality/VOI/presentation groups. All failures, metadata issues and exact duplicates require review. "
            "Missing or unprocessable pixels have no NPY; manual pass cannot bypass incomplete processing.\n\n"
            "Copy image_qc_review.csv to an independent versioned review input, fill decisions, reviewer, date, rationale "
            "and version, then rerun Step 07 with --reviews. Decisions are bound to source/configuration/NPY hashes. "
            "Do not edit the Parquet files or overwrite the input CSV with a generated report.\n\n"
            + f"CSV checks: {run.csv_tail_summary()['status']}; see ../../audit/step_07_csv_tail_checks.csv. "
            "The source course CSV warnings remain in the Step 05 audit; this step does not rescan them.\n\n"
            "Use only artifacts referenced by the current manifest and tables; old unreferenced NPY/previews "
            "are not current outputs. Steps 01-06, formal labels and the endpoint decision are unchanged.\n"
        )
        run.path(qc_dir / "qc_report.md").write_text(report, encoding="utf-8")
    print(
        f"Step 07 complete: {len(records)} studies; {metadata['unreviewed_studies']} required reviews remain. Open {qc_dir / 'image_qc_review.html'}"
    )
    return 0
