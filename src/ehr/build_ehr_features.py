"""Step 10A orchestration and immutable laboratory dictionary publication."""

from __future__ import annotations

import json
import platform
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort import build_labels, freeze_cohort
from src.cohort import create_patient_split as split
from src.cohort.burden_io import open_store
from src.cohort.common import (
    _config_hash,
    _file_hash,
    _file_version,
    _git_commit,
    _json_file,
    _OutputRun,
    _protect_outputs,
    _resolve_config_path,
    _under_root,
    _verify_run,
    _write_csv,
)
from src.ehr import lab_audit as audit
from src.ehr import lab_review as review


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def code_hash() -> str:
    base = Path(__file__).parents[2]
    return _config_hash(
        [
            Path(__file__),
            Path(audit.__file__),
            Path(review.__file__),
            base / "scripts/10_build_ehr_features.py",
            base / "src/cohort/common.py",
            base / "src/cohort/burden_io.py",
        ]
    )


def load_settings(path: Path) -> dict:
    settings = _json_file(path)
    required = {
        "algorithm_version": "lab_audit_review_v1",
        "lookback_hours": 24,
        "reviewer": "ZE Xiaokang",
    }
    if any(
        settings.get(k) != v or type(settings.get(k)) is not type(v)
        for k, v in required.items()
    ):
        raise ValueError(
            "Laboratory settings do not match the frozen research decisions"
        )
    cohort_version = settings.get("cohort_version", "")
    if (
        not isinstance(cohort_version, str)
        or not re.fullmatch(r"burden_v2_h72_r(?:0[1-9]|[1-9][0-9]+)", cohort_version)
        or not isinstance(settings.get("split_version"), str)
        or not re.fullmatch(
            re.escape(cohort_version) + r"_split_v[1-9][0-9]*",
            settings["split_version"],
        )
    ):
        raise ValueError("Laboratory cohort and split versions must agree")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", settings.get("audit_version", "")):
        raise ValueError("Invalid laboratory audit version")
    for name, limit in (
        ("batch_item_limit", 10),
        ("example_limit", 100),
        ("comment_max_chars", 4096),
    ):
        if type(settings.get(name)) is not int or not 1 <= settings[name] <= limit:
            raise ValueError(f"Invalid {name}")
    if not re.fullmatch(r"[1-9][0-9]*(MB|GB)", settings.get("memory_limit", "")):
        raise ValueError("Invalid laboratory memory limit")
    if set(settings.get("tables", {})) != {"d_labitems", "labevents", "admissions"}:
        raise ValueError("Laboratory source table configuration is incomplete")
    if "vital_audit" in settings:
        vital = settings["vital_audit"]
        if (set(vital) != {"tables", "items"}
                or set(vital["tables"]) != {"chartevents", "d_items", "icustays"}
                or not vital["items"]
                or any(not re.fullmatch(r"[1-9][0-9]*", k)
                       or not isinstance(v, str) or not v.strip()
                       for k, v in vital["items"].items())):
            raise ValueError("Invalid vital-sign audit scope")
    return settings


def verify_frozen_inputs(
    root: Path,
    configs: list[Path],
    processing: Path,
    split_config: Path,
    settings: dict,
) -> dict:
    """Read-only upstream replay; no raw clinical CSV scanning or publication."""
    context = split.load_frozen_cohort(root, configs, processing)
    outputs, processed = context["outputs"], context["processed"]
    split_settings = split.load_settings(split_config)
    primary_table = split.split_table_names(split_settings)[0]
    patient_path = processed / "splits/burden_v2" / primary_table
    _verify_run(outputs, 9, patient_path)
    directory = outputs / "audit/burden_v2"
    frozen = _json_file(directory / "split_freeze.json")
    metadata = _json_file(directory / "step_09_run_metadata.json")
    binding = {
        "cohort": context["cohort_identity"],
        "split_config_sha256": _file_hash(split_config),
        "settings": split_settings,
        "code_sha256": split._code_hash(),
    }
    if (
        frozen.get("patient_split_frozen") is not True
        or metadata.get("patient_split_frozen") is not True
        or frozen["source_binding"] != binding
        or metadata["source_binding"] != binding
        or frozen["source_fingerprint"] != build_labels._json_hash(binding)
        or metadata["source_fingerprint"] != frozen["source_fingerprint"]
        or metadata["original_split_run_id"] != frozen["original_split_run_id"]
        or settings["split_version"] != frozen["split_version"]
        or settings["cohort_version"] != context["release_version"]
        or context["decision"]["horizon_hours"] != 72
    ):
        raise ValueError("Frozen patient split identity or sources changed")
    manifest = _json_file(outputs / "audit/step_09_manifest.json")
    if (
        metadata["run_id"] != manifest["run_id"]
        or metadata["split_checks_passed"] is not True
    ):
        raise ValueError("Step 09 publication metadata disagrees")
    patients = pq.read_table(patient_path)
    with duckdb.connect(":memory:") as connection:
        cohort, cohort_audit = freeze_cohort.validate_cohort(connection, context)
        if not cohort.equals(context["cohort"]) or not cohort_audit.equals(
            context["cohort_audit"]
        ):
            raise ValueError("Frozen common cohort reconstruction failed")
        expected, quotas = split.allocate_patients(cohort, split_settings)
        if not expected.equals(patients):
            raise ValueError("Frozen patient assignments changed")
        checks = split.check_assignments(cohort, patients, split_settings, quotas)
        split.check_source_leakage(connection, patients)
        if "cross_validation" in quotas:
            for fold in range(1, 6):
                split.check_source_leakage(connection, split.select_fold(
                    patients, quotas["cross_validation"], fold))
        checks["source_records_do_not_cross_splits"] = True
        _, checks = split.split_reports(cohort, patients, checks)
    if (
        checks != frozen["checks"]
        or checks != metadata["checks"]
        or checks != _json_file(directory / "split_checks.json")
    ):
        raise ValueError("Frozen split checks changed")
    table_names = split.split_table_names(split_settings)
    if set(frozen["artifact_sha256"]) != {*table_names, *split.split_report_names(split_settings)}:
        raise ValueError("Incomplete frozen split artifact set")
    for name, digest in frozen["artifact_sha256"].items():
        current = (
            processed / "splits/burden_v2" / name
            if name in table_names
            else directory / name
        )
        base = processed / "splits/burden_v2" if name in table_names else directory
        snapshot = base / "releases" / frozen["split_version"] / name
        if _file_hash(current) != digest or _file_hash(snapshot) != digest:
            raise ValueError("Frozen split snapshot hash changed")
        if name in table_names and name != split.TABLE_NAME:
            fold = int(Path(name).parts[0].removeprefix("fold_"))
            expected_fold = split.select_fold(patients, quotas["cross_validation"], fold)
            if not pq.read_table(current).equals(expected_fold):
                raise ValueError("Frozen fold patient assignments changed")
        if name == split.CV_REPORT:
            bundle = _json_file(current)
            split.validate_cv_indices(cohort, bundle, patients)
            if bundle != quotas.get("cross_validation"):
                raise ValueError("Frozen CV indices disagree with reproducible allocation")
    freeze_hash = _file_hash(directory / "split_freeze.json")
    if freeze_hash != metadata["freeze_sha256"] or freeze_hash != _file_hash(
        directory / "releases" / frozen["split_version"] / "split_freeze.json"
    ):
        raise ValueError("Frozen split record hash changed")
    context["split_identity"] = {
        "split_version": frozen["split_version"],
        "original_split_run_id": frozen["original_split_run_id"],
        "split_freeze_sha256": freeze_hash,
        "patient_split_sha256": frozen["artifact_sha256"][primary_table],
        "source_fingerprint": frozen["source_fingerprint"],
    }
    return context


def source_binding(
    context: dict, sources: dict[str, Path], config: Path, settings: dict
) -> dict:
    versions = {name: _file_version(path) for name, path in sources.items()}
    if any(v.get("exists") is not True for v in versions.values()):
        raise FileNotFoundError("Missing laboratory source table")
    return {
        "cohort": context["cohort_identity"],
        "split": context["split_identity"],
        "source_versions": versions,
        "source_paths": {k: str(v) for k, v in sources.items()},
        "dictionary_source_sha256": _file_hash(sources["d_labitems"]),
        "configuration_sha256": _file_hash(config),
        "settings": settings,
        "code_sha256": code_hash(),
        "runtime_versions": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "pyarrow": pa.__version__,
        },
        "cache_filter": "all laboratory records and all admissions of frozen-cohort patients; inventory by inclusive charttime study windows",
    }


def load_audit(path: Path, binding: dict, processed: Path, outputs: Path) -> dict:
    data = _json_file(path)
    if data["source_binding"] != binding or data[
        "audit_fingerprint"
    ] != audit.json_hash(binding):
        raise ValueError(
            "Stale laboratory cache or changed sources/rules; use a new audit_version"
        )
    for name, digest in data["artifact_sha256"].items():
        target = Path(name).resolve()
        if not any(target.is_relative_to(p.resolve()) for p in (processed, outputs)):
            raise ValueError("Unsafe audit artifact path")
        if _file_hash(target) != digest:
            raise ValueError(f"Laboratory audit artifact hash changed: {target.name}")
    return data


def perform_audit(
    connection,
    run,
    cohort: pa.Table,
    sources: dict,
    cache: Path,
    directory: Path,
    settings: dict,
    binding: dict,
) -> dict:
    dictionary = audit.read_small_csv(
        sources["d_labitems"], ["itemid", "label", "fluid", "category"], run
    )
    dictionary = [
        {k: r[k] for k in ("itemid", "label", "fluid", "category")} for r in dictionary
    ]
    ids = [r["itemid"] for r in dictionary]
    if len(set(ids)) != len(ids) or any(
        not re.fullmatch(r"[1-9][0-9]*", value) for value in ids
    ):
        raise ValueError("Duplicate or invalid laboratory dictionary itemid")
    patients = pa.table({"subject_id": sorted(set(cohort["subject_id"].to_pylist()))})
    connection.register("audit_patients", patients)
    condition = "EXISTS(SELECT 1 FROM audit_patients p WHERE p.subject_id=try_cast(r.subject_id AS BIGINT))"
    audit.scan_csv(
        connection,
        sources["admissions"],
        "admissions",
        ["subject_id", "hadm_id", "admittime", "dischtime"],
        [],
        condition,
        run,
    )
    audit.scan_csv(
        connection,
        sources["labevents"],
        "labs_raw",
        audit.LAB_REQUIRED,
        audit.LAB_OPTIONAL,
        condition,
        run,
    )
    audit.prepare_labs(connection, settings)
    audit.build_views(connection, cohort)
    inventory, batches, outside = audit.inventory(connection, dictionary, settings)
    if any(not r["itemid"].isdigit() for r in inventory):
        raise ValueError("Invalid itemid in laboratory source records")
    fingerprint = audit.json_hash(binding)
    cache_paths = {
        "labs": cache / "labs_candidate_subset.parquet",
        "admissions": cache / "admissions.parquet",
        "dictionary": cache / "d_labitems.parquet",
    }
    for name in ("labs", "admissions"):
        audit.copy_parquet(connection, name, run.path(cache_paths[name]))
    dictionary_schema = pa.schema(
        [(k, pa.string()) for k in ("itemid", "label", "fluid", "category")]
    )
    pq.write_table(
        pa.Table.from_pylist(dictionary, schema=dictionary_schema),
        run.path(cache_paths["dictionary"]),
    )
    counts, examples, specimens = audit.write_reports(
        connection, run, directory, inventory, batches, outside, settings
    )
    vital_record = None
    if settings.get("vital_audit"):
        vital_record = audit.audit_vitals(
            connection, run, cohort, sources, cache, directory, settings, fingerprint)
    scope_hash = _file_hash(run.path(directory / "out_of_scope_inventory.csv"))
    review.write_templates(run, directory, inventory, fingerprint, scope_hash, batches)
    audit.write_html(
        run.path(directory / "lab_review.html"),
        inventory,
        batches,
        examples,
        specimens,
        fingerprint,
    )
    if vital_record is not None:
        run.path(directory / "step10a_review.html").write_text(
            '<!doctype html><meta charset="utf-8"><h1>Step 10A joint evidence review</h1>'
            '<p><a href="lab_review.html">Laboratory evidence and review templates</a></p>'
            '<p><a href="vital_review.html">Vital-sign evidence and pending review template</a></p>'
            '<p>Audit only. Laboratory approval does not approve vital-sign features. '
            'Both reviews must be discussed before defining the expanded EHR feature set.</p>',
            encoding="utf-8")
    text = (
        f"# Step 10A laboratory audit\n\nAudit: `{settings['audit_version']}`. Fingerprint: `{fingerprint}`.\n\n"
        "All observed window itemids are inventoried before clinical mapping. Navigation groups use names only. "
        "No numeric value, unit, specimen identity, conversion, valid range, or feature whitelist is approved by this audit.\n\n"
        "| Count | Value |\n| --- | ---: |\n"
        + "\n".join(f"| {k} | {v} |" for k, v in counts.items())
        + "\n\nPatient-scan counts describe the cached patient history. Window source counts deduplicate exact logical source records using every CSV field; raw-row counts retain their multiplicity. Study associations may overlap and are not independent measurements.\n\n"
        "Missing/invalid charttime records cannot be mapped to a study and appear in lab_time_failures.csv. "
        "lab_availability_summary.csv counts missing/inconsistent/future storetime and ownership failures independently. "
        "Specimen evidence lists raw same-specimen observations; its presence alone does not establish availability at t0. "
        "Reference ranges are descriptive evidence. Severe numeric extremes are preserved.\n\n"
        "Open lab_review.html. Review at most ten itemids per batch, retaining all unit branches. "
        "Copy completed decisions and the signed scope acknowledgment to independent versioned CSV inputs. "
        "Review and freeze commands do not rescan labevents.\n\n"
        f"CSV tail checks: `{json.dumps(run.csv_tail_summary(), sort_keys=True)}`. "
        "See ../../../../step_10_csv_tail_checks.csv and the audit manifest's source_csv_checks for record-level diagnostics. "
        "A passing final-record check establishes structural validity, not dataset completeness. "
        "Review source files manually when incomplete-record warnings are reported.\n"
    )
    run.path(directory / "lab_audit_report.md").write_text(
        text.replace("\n", "\r\n"), encoding="utf-8", newline=""
    )
    return {
        "audit_version": settings["audit_version"],
        "audit_fingerprint": fingerprint,
        "source_binding": binding,
        "original_audit_run_id": run.run_id,
        "cache_paths": {k: str(v) for k, v in cache_paths.items()},
        "inventory": inventory,
        "batches": batches,
        "counts": counts,
        "vital_audit": vital_record,
        "scope_inventory_sha256": scope_hash,
        "source_csv_tail_checks": run.csv_tail_summary(),
        "source_csv_checks": list(run.csv_checks.values()),
        "artifact_sha256": {
            str(path): _file_hash(staged) for path, staged in run.files.items()
        },
    }


def dictionary_releases(context: dict) -> tuple[Path, dict]:
    """Keep historical snapshot checksums in the current transactional manifest."""
    index_path = context["outputs"] / "audit/burden_v2/lab_dictionary_releases.json"
    index = _json_file(index_path) if index_path.exists() else {}
    for version, entry in index.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", version):
            raise ValueError("Invalid dictionary release index")
        for kind in ("dictionary", "freeze"):
            path = Path(entry[kind + "_path"]).resolve()
            root = context["processed"] if kind == "dictionary" else context["outputs"]
            if (
                not path.is_relative_to(root.resolve())
                or _file_hash(path) != entry[kind + "_sha256"]
            ):
                raise ValueError("Historical dictionary snapshot checksum changed")
    current = context["outputs"] / "audit/burden_v2/lab_dictionary_freeze.json"
    if current.exists():
        frozen = _json_file(current)
        entry = index.get(frozen["dictionary_version"])
        table = context["processed"] / "ehr/burden_v2/lab_dictionary.parquet"
        if (
            not entry
            or _file_hash(current) != entry["freeze_sha256"]
            or _file_hash(table) != entry["dictionary_sha256"]
        ):
            raise ValueError("Current dictionary and immutable snapshots disagree")
    return index_path, index


def validate_feature_scope(path, record, decisions, settings, audit_path):
    """Accept a separate, explicitly reviewed exclusion scope, not an event decision."""
    scope = _json_file(path)
    retained = {str(r["itemid"]) for r in decisions if r["keep"]}
    if (scope.get("vital_feature_scope") != "exclude_all"
            or scope.get("audit_fingerprint") != record["audit_fingerprint"]
            or scope.get("audit_manifest_sha256") != _file_hash(audit_path)
            or scope.get("reviewer") != settings["reviewer"]
            or not scope.get("rationale", "").strip()
            or not scope.get("review_version", "").strip()
            or scope.get("cohort_version") != settings["cohort_version"]
            or scope.get("split_version") != settings["split_version"]
            or sorted(scope.get("retained_lab_itemids", [])) != sorted(retained)
            or sorted(scope.get("excluded_vital_itemids", [])) != sorted(
                settings.get("vital_audit", {}).get("items", {}))
            or scope.get("static_features_unchanged") is not True
            or scope.get("no_study_excluded_for_missing_features") is not True
            or scope.get("pending_raw_vital_issues_resolved") is not False):
        raise ValueError("Feature scope is stale, incomplete, or differs from reviewed exclusions")
    from datetime import date
    date.fromisoformat(scope["reviewed_on"])
    return {"path": str(path), "sha256": _file_hash(path), "vital_feature_scope": "exclude_all"}


def freeze_dictionary(
    run,
    context: dict,
    directory: Path,
    audit_record: dict,
    decisions: list[dict],
    acknowledgment: dict | None,
    review_hashes: dict,
    version: str,
    feature_scope: dict | None = None,
) -> dict:
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", version):
        raise ValueError("Invalid dictionary version")
    expanded = review.expanded_decisions(
        audit_record["inventory"], decisions, acknowledgment
    )
    retained = [r for r in expanded if r["keep"]]
    feature_units = {}
    for row in retained:
        unit = feature_units.setdefault(row["feature_name"], row["canonical_unit"])
        if unit != row["canonical_unit"]:
            raise ValueError("One feature has inconsistent canonical units")
    table = review.dictionary_table(expanded, version)
    data = context["processed"] / "ehr/burden_v2"
    table_path = data / "releases" / version / "lab_dictionary.parquet"
    freeze_path = (
        directory / "dictionary_releases" / version / "lab_dictionary_freeze.json"
    )
    binding = {
        "audit_fingerprint": audit_record["audit_fingerprint"],
        "audit_manifest_sha256": _file_hash(directory / "lab_audit_manifest.json"),
        "review_files_sha256": review_hashes,
        "decisions_sha256": review.decisions_hash(decisions, acknowledgment),
        "dictionary_version": version,
        "code_sha256": code_hash(),
    }
    if feature_scope is not None:
        binding["feature_scope_review"] = feature_scope
    index_path, index = dictionary_releases(context)
    if freeze_path.exists():
        frozen = _json_file(freeze_path)
        if (
            frozen["source_binding"] != binding
            or frozen["source_fingerprint"] != audit.json_hash(binding)
            or frozen.get("dictionary_frozen") is not True
        ):
            raise ValueError("Frozen dictionary identity cannot accept changed content")
        if _file_hash(table_path) != frozen["dictionary_sha256"] or not pq.read_table(
            table_path
        ).equals(table):
            raise ValueError("Frozen dictionary snapshot changed")
        source_table, source_freeze = table_path, freeze_path
    else:
        if (
            table_path.exists()
            or freeze_path.parent.exists()
            and any(freeze_path.parent.iterdir())
        ):
            raise ValueError(
                "Incomplete dictionary release; preserve it for investigation"
            )
        source_table = run.path(table_path)
        pq.write_table(table, source_table)
        frozen = {
            "dictionary_frozen": True,
            "dictionary_version": version,
            "original_freeze_run_id": run.run_id,
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "source_binding": binding,
            "source_fingerprint": audit.json_hash(binding),
            "dictionary_sha256": _file_hash(source_table),
            "retained_rule_rows": len(retained),
            "excluded_rule_rows": len(expanded) - len(retained),
            "ehr_features_ready": False,
        }
        source_freeze = run.path(freeze_path)
        write_json(source_freeze, frozen)
    shutil.copyfile(source_table, run.path(data / "lab_dictionary.parquet"))
    shutil.copyfile(
        source_freeze,
        run.path(context["outputs"] / "audit/burden_v2/lab_dictionary_freeze.json"),
    )
    entry = {
        "dictionary_path": str(table_path),
        "dictionary_sha256": _file_hash(source_table),
        "freeze_path": str(freeze_path),
        "freeze_sha256": _file_hash(source_freeze),
    }
    if version in index and index[version] != entry:
        raise ValueError("Frozen dictionary release index changed")
    index[version] = entry
    write_json(run.path(index_path), index)
    return frozen


def run_step10(
    root: Path,
    mode: str,
    configs: list[Path],
    processing: Path,
    split_config: Path,
    lab_config: Path,
    reviews: list[Path],
    dictionary_version: str | None = None,
    feature_scope_review: Path | None = None,
) -> int:
    if (
        mode not in ("audit", "review", "freeze")
        or (mode != "audit" and not reviews)
        or (mode == "audit" and reviews)
        or (mode == "freeze" and not dictionary_version)
        or (mode != "freeze" and dictionary_version)
        or (mode == "audit" and feature_scope_review is not None)
    ):
        raise ValueError("Invalid Step 10 mode arguments")
    settings = load_settings(lab_config)
    context = verify_frozen_inputs(root, configs, processing, split_config, settings)
    processed, outputs = context["processed"], context["outputs"]
    paths = _json_file(configs[0])
    raw = _resolve_config_path(paths["mimic_iv"], root, raw=True)
    sources = {k: _under_root(raw, v) for k, v in settings["tables"].items()}
    if settings.get("vital_audit"):
        sources.update({k: _under_root(raw, v)
                        for k, v in settings["vital_audit"]["tables"].items()})
    cache_root = (
        _resolve_config_path(paths["cache"], root, raw=False)
        if paths.get("cache")
        else processed.parent / "cache"
    )
    cache = cache_root / "burden_v2/labs" / settings["audit_version"]
    cache = cache.resolve()
    directory = outputs / "audit/burden_v2/labs" / settings["audit_version"]
    _protect_outputs(paths, root, processed, outputs, cache)
    for path in reviews + [lab_config] + ([feature_scope_review] if feature_scope_review else []):
        if any(
            path.resolve().is_relative_to(p.resolve())
            for p in (processed, outputs, cache)
        ):
            raise ValueError(
                "Configuration and review inputs must be outside generated directories"
            )
    binding = source_binding(context, sources, lab_config, settings)
    audit_path = directory / "lab_audit_manifest.json"
    print(f"Step 10A {mode}: {settings['audit_version']}; no EHR aggregation")
    started = datetime.now(timezone.utc)
    with _OutputRun(outputs, 10, processed, cache) as run:
        if run.manifest.exists():
            _verify_run(outputs, 10)
        index_path, index = dictionary_releases(context)
        if index:
            shutil.copyfile(index_path, run.path(index_path))
        if audit_path.exists():
            record = load_audit(audit_path, binding, cache, outputs)
        else:
            if mode != "audit":
                raise ValueError("Run audit mode before review or freeze")
            if any(p.exists() and any(p.iterdir()) for p in (cache, directory)):
                raise ValueError(
                    "Incomplete laboratory audit; preserve it for investigation"
                )
            record = None
        state = {"review_complete": False, "scope_acknowledgment_complete": False}
        decisions, acknowledgment = [], None
        frozen = None
        feature_scope = None
        run.manifest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".step_10_", dir=run.manifest.parent
        ) as temporary:
            with open_store(Path(temporary), settings["memory_limit"]) as connection:
                if record is None:
                    record = perform_audit(
                        connection,
                        run,
                        context["cohort"],
                        sources,
                        cache,
                        directory,
                        settings,
                        binding,
                    )
                    write_json(run.path(audit_path), record)
                else:
                    shutil.copyfile(audit_path, run.path(audit_path))
                if mode == "freeze" and record.get("vital_audit") and feature_scope_review is None:
                    raise ValueError("Joint vital-sign review is pending; supply the explicitly reviewed feature scope")
                if mode != "audit":
                    dictionary = audit.load_cache(
                        connection,
                        {k: Path(v) for k, v in record["cache_paths"].items()},
                    )
                    audit.build_views(connection, context["cohort"])
                    review_hashes = {str(path): _file_hash(path) for path in reviews}
                    decisions, acknowledgment = review.parse_reviews(
                        reviews, record["inventory"], dictionary, record, settings, run
                    )
                    if feature_scope_review:
                        review_hashes[str(feature_scope_review)] = _file_hash(feature_scope_review)
                        feature_scope = validate_feature_scope(
                            feature_scope_review, record, decisions, settings, audit_path)
                    if mode == "freeze" and record.get("vital_audit") and feature_scope is None:
                        raise ValueError("Joint vital-sign review is pending; supply the explicitly reviewed feature scope")
                    state, progress = review.review_progress(
                        connection, record["inventory"], decisions, acknowledgment
                    )
                    conversions = review.validate_retained_conflicts(
                        connection, decisions
                    )
                    _write_csv(
                        run.path(directory / "review_progress.csv"),
                        ["review_key", "itemid", "source_unit", "status"],
                        progress,
                    )
                    _write_csv(
                        run.path(directory / "lab_conversion_checks.csv"),
                        [
                            "review_key",
                            "branch_id",
                            "feature_name",
                            "missing_specimen",
                            "numeric_disagreement",
                            "conflicting_id",
                            "numeric_records",
                            "nonfinite_conversions",
                            "converted_min",
                            "converted_max",
                        ],
                        conversions,
                    )
                    write_json(
                        run.path(directory / "review_state.json"),
                        {
                            **state,
                            "decisions_sha256": review.decisions_hash(
                                decisions, acknowledgment
                            ),
                            "review_files_sha256": review_hashes,
                        },
                    )
                    if any(
                        _file_hash(Path(path)) != digest
                        for path, digest in review_hashes.items()
                    ):
                        raise ValueError("Review input changed during execution")
                    if mode == "freeze":
                        if dictionary_version is None:
                            raise ValueError(
                                "Dictionary version is required for freezing"
                            )
                        if not state["review_complete"]:
                            raise ValueError(
                                "Dictionary freeze blocked by incomplete clinical review or scope acknowledgment"
                            )
                        frozen = freeze_dictionary(
                            run,
                            context,
                            directory,
                            record,
                            decisions,
                            acknowledgment,
                            review_hashes,
                            dictionary_version,
                            feature_scope,
                        )
                if frozen is None:
                    current = outputs / "audit/burden_v2/lab_dictionary_freeze.json"
                    existing = _json_file(current) if current.exists() else None
                    if (
                        existing
                        and existing["source_binding"]["audit_fingerprint"]
                        == record["audit_fingerprint"]
                    ):
                        matches = mode == "audit" or existing["source_binding"][
                            "decisions_sha256"
                        ] == review.decisions_hash(decisions, acknowledgment)
                        if matches:
                            frozen = existing
                            dictionary_version = existing["dictionary_version"]
                            if mode == "audit":
                                state.update(
                                    review_complete=True,
                                    scope_acknowledgment_complete=True,
                                )
                            for path in (
                                outputs / "audit/burden_v2/lab_dictionary_freeze.json",
                                processed / "ehr/burden_v2/lab_dictionary.parquet",
                            ):
                                shutil.copyfile(path, run.path(path))
        # Bind publication to a second read-only upstream/source check.
        latest = verify_frozen_inputs(root, configs, processing, split_config, settings)
        if (
            source_binding(latest, sources, lab_config, load_settings(lab_config))
            != binding
        ):
            raise ValueError(
                "Frozen inputs, laboratory sources, or rules changed during Step 10"
            )
        metadata = {
            "run_id": run.run_id,
            "mode": mode,
            "audit_version": settings["audit_version"],
            "original_audit_run_id": record["original_audit_run_id"],
            "audit_fingerprint": record["audit_fingerprint"],
            "audit_manifest_path": str(audit_path),
            "audit_checks_passed": True,
            **state,
            "dictionary_frozen": frozen is not None,
            "dictionary_version": dictionary_version,
            "ehr_features_ready": False,
            "vital_review_complete": (
                record["vital_audit"]["review_complete"] if record.get("vital_audit") else None),
            "vital_feature_scope": "exclude_all" if feature_scope else "pending" if record.get("vital_audit") else None,
            "joint_review_complete": state["review_complete"] and (
                not record.get("vital_audit") or feature_scope is not None),
            "cohort_version": settings["cohort_version"],
            "split_version": settings["split_version"],
            "horizon_hours": 72,
            "lookback_hours": 24,
            "source_binding": binding,
            "csv_tail_checks": run.csv_tail_summary(),
            "source_csv_tail_checks": record["source_csv_tail_checks"],
            "source_csv_checks": record["source_csv_checks"],
            "upstream_source_csv_tail_checks": context.get("source05", {}).get(
                "csv_tail_checks"
            ),
            "upstream_review_csv_tail_checks": context.get("source07", {}).get(
                "csv_tail_checks"
            ),
            "code_sha256": code_hash(),
            "git_commit": _git_commit(root),
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        if frozen:
            metadata["original_dictionary_freeze_run_id"] = frozen[
                "original_freeze_run_id"
            ]
            metadata["dictionary_sha256"] = frozen["dictionary_sha256"]
        write_json(
            run.path(outputs / "audit/burden_v2/step_10_run_metadata.json"), metadata
        )
    print(
        f"Step 10A complete: {mode}; review_complete={state['review_complete']}; dictionary_frozen={frozen is not None}; ehr_features_ready=false"
    )
    return 0
