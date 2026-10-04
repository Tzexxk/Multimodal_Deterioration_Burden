"""Step 09: deterministic patient allocation and recoverable split freezing."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort import build_labels, freeze_cohort
from src.cohort.burden_io import COMPONENTS, open_store
from src.cohort.common import (
    _config_hash,
    _file_hash,
    _git_commit,
    _json_file,
    _OutputRun,
    _under_root,
    _verify_run,
    _write_csv,
)

SPLITS = ("train", "validation", "test")
ALGORITHM = "patient_stratified_pcg64_v1"
SCHEMA = pa.schema(
    [
        ("subject_id", pa.int64()),
        ("split", pa.string()),
        ("seed", pa.int64()),
        ("split_version", pa.string()),
        ("cohort_version", pa.string()),
        ("study_count", pa.int64()),
        ("max_burden_score", pa.float32()),
        ("ever_phys_death", pa.uint8()),
        ("ever_high_burden", pa.uint8()),
    ]
)
TABLE_NAME = "patient_split.parquet"
REPORT_NAMES = ("split_summary.csv", "split_checks.json", "step_09_report.md")


def _code_hash() -> str:
    return _config_hash(
        [
            Path(__file__),
            Path(__file__).parents[2] / "scripts/09_create_patient_split.py",
        ]
    )


def load_settings(path: Path) -> dict:
    settings = _json_file(path)
    expected = {
        "algorithm_version": ALGORITHM,
        "weights": [70, 15, 15],
        "seed": 42,
        "primary_stratum": "ever_phys_death",
        "require_high_burden_in_each_split": True,
        "confirmed": True,
    }
    if any(
        settings.get(k) != v or type(settings.get(k)) is not type(v)
        for k, v in expected.items()
    ):
        raise ValueError(
            "Patient split configuration does not match the approved rules"
        )
    if any(type(v) is not int for v in settings["weights"]):
        raise ValueError("Split weights must be integers")
    cohort_version = settings.get("cohort_version", "")
    if not isinstance(cohort_version, str) or not re.fullmatch(
        r"burden_v2_h(?:48|72)_r(?:0[1-9]|[1-9][0-9]+)", cohort_version
    ):
        raise ValueError("Invalid cohort version in split configuration")
    if not isinstance(settings.get("split_version"), str) or not re.fullmatch(
        re.escape(cohort_version) + r"_split_v[1-9][0-9]*", settings["split_version"]
    ):
        raise ValueError("Split version must belong to the configured cohort version")
    if not all(
        isinstance(settings.get(k), str) and settings[k].strip()
        for k in ("reviewer", "reviewed_on", "rationale")
    ):
        raise ValueError("Missing split confirmation provenance")
    date.fromisoformat(settings["reviewed_on"])
    return settings


def apportion(
    n: int, weights: list[int], minimum: list[int], capacity: list[int]
) -> list[int]:
    """Bounded integer largest-deficit rounding; ties follow SPLITS order."""
    if (
        type(n) is not int
        or n < 0
        or any(len(v) != 3 for v in (weights, minimum, capacity))
        or any(type(x) is not int for v in (weights, minimum, capacity) for x in v)
        or any(w <= 0 for w in weights)
        or sum(weights) != 100
        or any(lo < 0 or cap < lo for lo, cap in zip(minimum, capacity))
        or sum(minimum) > n
        or sum(capacity) < n
    ):
        raise ValueError("Patient allocation constraints cannot be satisfied")
    allocation = [
        min(cap, max(lo, n * w // 100))
        for w, lo, cap in zip(weights, minimum, capacity)
    ]
    while sum(allocation) < n:
        group = min(
            (i for i in range(3) if allocation[i] < capacity[i]),
            key=lambda i: (100 * allocation[i] - n * weights[i], i),
        )
        allocation[group] += 1
    while sum(allocation) > n:
        group = min(
            (i for i in range(3) if allocation[i] > minimum[i]),
            key=lambda i: (n * weights[i] - 100 * allocation[i], i),
        )
        allocation[group] -= 1
    return allocation


def aggregate_patients(cohort: pa.Table, version: str) -> list[dict]:
    required = {
        "sample_id": pa.string(),
        "subject_id": pa.int64(),
        "hadm_id": pa.int64(),
        "study_id": pa.int64(),
        "burden_score": pa.float32(),
        "phys_death": pa.uint8(),
        "high_burden": pa.uint8(),
        "cohort_version": pa.string(),
        **dict.fromkeys(COMPONENTS, pa.uint8()),
    }
    if len(set(cohort.column_names)) != len(cohort.column_names) or any(
        k not in cohort.column_names
        or cohort.schema.field(k).type != v
        or cohort[k].null_count
        for k, v in required.items()
    ):
        raise ValueError("Missing, null or invalid common-cohort fields")
    patients, samples, studies = {}, set(), set()
    for row in cohort.to_pylist():
        key = (row["subject_id"], row["hadm_id"], row["study_id"])
        if (
            any(v <= 0 for v in key)
            or row["sample_id"] != f"p{key[0]}_h{key[1]}_s{key[2]}"
            or row["sample_id"] in samples
            or key in studies
            or row["cohort_version"] != version
        ):
            raise ValueError("Invalid, duplicated or mixed-version study identity")
        flags = [row[c] for c in COMPONENTS]
        if any(f not in (0, 1) for f in flags):
            raise ValueError("Invalid event flag in common cohort")
        t, r, c, d = max(flags[:2]), max(flags[2:4]), flags[4], flags[5]
        score = t + 2 * r + 2 * c + 3 * d
        if (
            row["burden_score"] != score
            or row["phys_death"] != int(r == 1 or c == 1 or d == 1)
            or row["high_burden"] != int(score >= 3)
        ):
            raise ValueError("Invalid common-cohort score or derived endpoint")
        samples.add(row["sample_id"])
        studies.add(key)
        patient = patients.setdefault(
            key[0],
            {
                "subject_id": key[0],
                "study_count": 0,
                "max_burden_score": 0.0,
                "ever_phys_death": 0,
                "ever_high_burden": 0,
            },
        )
        patient["study_count"] += 1
        patient["max_burden_score"] = max(patient["max_burden_score"], score)
        for flag in ("phys_death", "high_burden"):
            patient["ever_" + flag] = max(patient["ever_" + flag], row[flag])
    if not patients:
        raise ValueError("The common cohort is empty")
    return [patients[key] for key in sorted(patients)]


def allocate_patients(cohort: pa.Table, settings: dict) -> tuple[pa.Table, dict]:
    patients = aggregate_patients(cohort, settings["cohort_version"])
    n = len(patients)
    positives = sum(p["ever_phys_death"] for p in patients)
    high = sum(p["ever_high_burden"] for p in patients)
    if high < 3:
        raise ValueError(
            "At least three high-burden patients are required; do not change the seed or split patients"
        )
    total = apportion(n, settings["weights"], [1] * 3, [n] * 3)
    phys = apportion(positives, settings["weights"], [1] * 3, total)
    high_counts = apportion(high, settings["weights"], [1] * 3, phys)
    strata = (
        ([p for p in patients if p["ever_high_burden"]], high_counts),
        (
            [p for p in patients if p["ever_phys_death"] and not p["ever_high_burden"]],
            [a - b for a, b in zip(phys, high_counts)],
        ),
        (
            [p for p in patients if not p["ever_phys_death"]],
            [a - b for a, b in zip(total, phys)],
        ),
    )
    rng = np.random.Generator(np.random.PCG64(settings["seed"]))
    assignments = []
    for layer, quotas in strata:
        order = rng.permutation(len(layer))
        start = 0
        for split, count in zip(SPLITS, quotas):
            for position in order[start : start + count]:
                assignments.append(
                    {
                        **layer[int(position)],
                        "split": split,
                        "seed": settings["seed"],
                        "split_version": settings["split_version"],
                        "cohort_version": settings["cohort_version"],
                    }
                )
            start += count
        if start != len(layer):
            raise ValueError("Stratum allocation did not cover all patients")
    table = pa.Table.from_pylist(
        sorted(assignments, key=lambda p: p["subject_id"]), schema=SCHEMA
    )
    return table, {
        "total": total,
        "ever_phys_death": phys,
        "ever_high_burden": high_counts,
    }


def check_assignments(
    cohort: pa.Table, patients: pa.Table, settings: dict, quotas: dict
) -> dict:
    if patients.schema != SCHEMA or any(patients[k].null_count for k in SCHEMA.names):
        raise ValueError("Invalid patient split schema or null fields")
    summaries = {
        r["subject_id"]: r
        for r in aggregate_patients(cohort, settings["cohort_version"])
    }
    records = patients.to_pylist()
    by_id = {r["subject_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != set(summaries):
        raise ValueError("Patient split coverage is missing, duplicated or extra")
    for row in records:
        if (
            row["split"] not in SPLITS
            or row["seed"] != settings["seed"]
            or row["split_version"] != settings["split_version"]
            or row["cohort_version"] != settings["cohort_version"]
            or any(row[k] != value for k, value in summaries[row["subject_id"]].items())
        ):
            raise ValueError("Invalid patient assignment or aggregation provenance")
    actual = {
        name: [
            sum(
                1 if name == "total" else r[name]
                for r in records
                if r["split"] == split
            )
            for split in SPLITS
        ]
        for name in quotas
    }
    if (
        actual != quotas
        or any(v <= 0 for v in actual["total"])
        or any(v <= 0 for v in actual["ever_high_burden"])
    ):
        raise ValueError("Patient quotas or high-burden coverage failed")
    return {
        "patient_coverage_complete": True,
        "patient_groups_disjoint": True,
        "each_study_has_one_split": True,
        "groups_nonempty": True,
        "quotas_match": True,
        "high_burden_in_each_split": True,
        "quotas": quotas,
    }


def load_frozen_cohort(root: Path, configs: list[Path], processing: Path) -> dict:
    context = freeze_cohort.verify_inputs(root, configs, processing)
    outputs, processed = context["outputs"], context["processed"]
    data = processed / "cohort/burden_v2"
    directory = outputs / "audit/burden_v2"
    _verify_run(outputs, 8, data / "cohort.parquet")
    manifest = _json_file(outputs / "audit/step_08_manifest.json")
    metadata = _json_file(directory / "step_08_run_metadata.json")
    frozen = _json_file(directory / "cohort_freeze.json")
    if any(
        str(p) not in manifest["files"]
        for p in (
            directory / "step_08_run_metadata.json",
            directory / "cohort_freeze.json",
            data / "cohort_audit.parquet",
        )
    ):
        raise ValueError("Step 08 manifest omits required cohort artifacts")
    if (
        metadata["run_id"] != manifest["run_id"]
        or not all(
            metadata.get(k) is True
            for k in ("common_cohort_frozen", "cohort_checks_passed")
        )
        or frozen.get("common_cohort_frozen") is not True
    ):
        raise ValueError("Common cohort is not frozen and verified")
    horizon = context["decision"]["horizon_hours"]
    version = frozen["release_version"]
    if (
        not re.fullmatch(rf"burden_v2_h{horizon}_r(?:0[1-9]|[1-9][0-9]+)", version)
        or metadata["release_version"] != version
        or metadata["horizon_hours"] != horizon
        or frozen["horizon_hours"] != horizon
    ):
        raise ValueError("Common-cohort release or horizon mismatch")
    if (
        metadata["source_binding"] != frozen["source_binding"]
        or frozen["source_binding"] != context["binding"]
        or metadata["source_fingerprint"] != frozen["source_fingerprint"]
        or frozen["source_fingerprint"] != context["fingerprint"]
        or metadata["original_freeze_run_id"] != frozen["original_freeze_run_id"]
        or metadata["summary"] != frozen["summary"]
        or metadata["artifact_sha256"] != frozen["artifact_sha256"]
    ):
        raise ValueError("Common-cohort frozen provenance changed")
    snapshot_data = _under_root(data, f"releases/{version}")
    snapshot_audit = _under_root(directory, f"releases/{version}")
    freeze_hash = _file_hash(directory / "cohort_freeze.json")
    if freeze_hash != metadata["freeze_sha256"] or freeze_hash != _file_hash(
        snapshot_audit / "cohort_freeze.json"
    ):
        raise ValueError("Common-cohort freeze record hash changed")
    names = (*freeze_cohort.TABLE_NAMES, *freeze_cohort.REPORT_NAMES)
    if set(frozen["artifact_sha256"]) != set(names):
        raise ValueError("Incomplete common-cohort snapshot")
    for name in names:
        base, snapshot = (
            (data, snapshot_data)
            if name in freeze_cohort.TABLE_NAMES
            else (directory, snapshot_audit)
        )
        if (
            _file_hash(base / name) != frozen["artifact_sha256"][name]
            or _file_hash(snapshot / name) != frozen["artifact_sha256"][name]
        ):
            raise ValueError("Current cohort and frozen snapshot hashes disagree")
    context.update(
        release_version=version,
        cohort=pq.read_table(data / "cohort.parquet"),
        cohort_audit=pq.read_table(data / "cohort_audit.parquet"),
        step08=metadata,
        cohort_identity={
            "cohort_version": version,
            "horizon_hours": horizon,
            "cohort_freeze_sha256": freeze_hash,
            "cohort_sha256": frozen["artifact_sha256"]["cohort.parquet"],
            "cohort_audit_sha256": frozen["artifact_sha256"]["cohort_audit.parquet"],
            "original_cohort_freeze_run_id": frozen["original_freeze_run_id"],
            "cohort_source_fingerprint": frozen["source_fingerprint"],
        },
    )
    return context


def split_reports(
    cohort: pa.Table, patients: pa.Table, checks: dict
) -> tuple[list[dict], dict]:
    by_id = {p["subject_id"]: p for p in patients.to_pylist()}
    rows = cohort.to_pylist()
    summary, evaluability = [], {}
    for split in SPLITS:
        group = [r for r in rows if by_id[r["subject_id"]]["split"] == split]
        people = [p for p in by_id.values() if p["split"] == split]
        categories = [("total", "all", group)]
        categories += [
            (
                "study_score",
                str(score),
                [r for r in group if r["burden_score"] == score],
            )
            for score in range(9)
        ]
        categories += [
            (
                "patient_max_score",
                str(score),
                [
                    r
                    for r in group
                    if by_id[r["subject_id"]]["max_burden_score"] == score
                ],
            )
            for score in range(9)
        ]
        categories += [
            ("endpoint", flag, [r for r in group if r[flag] == 1])
            for flag in (*COMPONENTS, "phys_death", "high_burden", "early_discharge")
        ]
        summary.extend(
            {
                "split": split,
                "category": cat,
                "value": value,
                **freeze_cohort._counts(selected),
            }
            for cat, value, selected in categories
        )
        evaluability[split] = {}
        for flag in (*COMPONENTS, "phys_death", "high_burden"):
            positive = [r for r in group if r[flag] == 1]
            positive_patients = len({r["subject_id"] for r in positive})
            evaluability[split][flag] = {
                "positive_studies": len(positive),
                "negative_studies": len(group) - len(positive),
                "positive_patients": positive_patients,
                "negative_patients": len(people) - positive_patients,
                "discrimination_evaluable": 0 < len(positive) < len(group),
                "patient_classes_available": 0 < positive_patients < len(people),
                "reason": "both_classes_present"
                if 0 < len(positive) < len(group)
                else "no_positive_studies"
                if not positive
                else "no_negative_studies",
            }
    return summary, {
        **checks,
        "cohort_counts": freeze_cohort._counts(rows),
        "endpoint_evaluability": evaluability,
    }


def check_source_leakage(connection, patients: pa.Table) -> None:
    connection.register("patient_split", patients)
    build_labels._reject(
        connection,
        "SELECT count(*) FROM (SELECT e.component,e.source_record_key FROM evidence e JOIN common c USING(sample_id) JOIN patient_split p ON p.subject_id=e.subject_id GROUP BY e.component,e.source_record_key HAVING count(DISTINCT p.split)>1)",
        "A stable component/source record crosses patient splits",
    )


def create_patient_split(
    root: Path,
    paths_file: Path,
    cohort_file: Path,
    ehr_file: Path,
    imaging_file: Path,
    processing_file: Path,
    split_file: Path,
) -> int:
    started = datetime.now(timezone.utc)
    configs = [paths_file, cohort_file, ehr_file, imaging_file]
    settings = load_settings(split_file)
    context = load_frozen_cohort(root, configs, processing_file)
    if any(
        split_file.resolve().is_relative_to(context[key])
        for key in ("outputs", "processed")
    ):
        raise ValueError("Keep the split configuration outside generated output roots")
    if settings["cohort_version"] != context["release_version"]:
        raise ValueError("Split configuration targets a different frozen cohort")
    binding = {
        "cohort": context["cohort_identity"],
        "split_config_sha256": _file_hash(split_file),
        "settings": settings,
        "code_sha256": _code_hash(),
    }
    fingerprint = build_labels._json_hash(binding)
    version = settings["split_version"]
    data = context["processed"] / "splits/burden_v2"
    directory = context["outputs"] / "audit/burden_v2"
    snapshot_data = _under_root(data, f"releases/{version}")
    snapshot_audit = _under_root(directory, f"releases/{version}")
    freeze_path = snapshot_audit / "split_freeze.json"
    print(f"Verify patient split for {version}; CSV tail checks: no_csv_inputs")
    with _OutputRun(context["outputs"], 9, context["processed"]) as run:
        if run.manifest.exists():
            _verify_run(context["outputs"], 9)
        with tempfile.TemporaryDirectory(
            prefix=".step_09_", dir=run.manifest.parent
        ) as tmp:
            with open_store(Path(tmp), context["memory_limit"]) as connection:
                expected, expected_audit = freeze_cohort.validate_cohort(
                    connection, context
                )
                if not context["cohort"].equals(expected) or not context[
                    "cohort_audit"
                ].equals(expected_audit):
                    raise ValueError(
                        "Frozen cohort does not reproduce final labels and QC"
                    )
                patients, quotas = allocate_patients(context["cohort"], settings)
                checks = check_assignments(
                    context["cohort"], patients, settings, quotas
                )
                check_source_leakage(connection, patients)
                checks["source_records_do_not_cross_splits"] = True
                summary, checks = split_reports(context["cohort"], patients, checks)
                locations = {TABLE_NAME: snapshot_data / TABLE_NAME} | {
                    name: snapshot_audit / name for name in REPORT_NAMES
                }
                reused = freeze_path.exists()
                if reused:
                    frozen = _json_file(freeze_path)
                    if (
                        frozen.get("patient_split_frozen") is not True
                        or frozen["split_version"] != version
                        or frozen["source_binding"] != binding
                        or frozen["source_fingerprint"] != fingerprint
                        or set(frozen["artifact_sha256"]) != set(locations)
                    ):
                        raise ValueError(
                            "Frozen split sources or rules changed; an approved new split version is required"
                        )
                    for name, path in locations.items():
                        if _file_hash(path) != frozen["artifact_sha256"][name]:
                            raise ValueError(
                                "Frozen patient split snapshot hash changed"
                            )
                    if (
                        not pq.read_table(locations[TABLE_NAME]).equals(patients)
                        or _json_file(locations["split_checks.json"]) != checks
                    ):
                        raise ValueError("Frozen patient assignments or checks changed")
                else:
                    if any(path.exists() for path in locations.values()):
                        raise ValueError(
                            "Incomplete existing split snapshot; preserve it for investigation"
                        )
                    pq.write_table(
                        patients, run.path(locations[TABLE_NAME]), compression="zstd"
                    )
                    _write_csv(
                        run.path(locations["split_summary.csv"]),
                        [
                            "split",
                            "category",
                            "value",
                            "studies",
                            "admissions",
                            "patients",
                        ],
                        summary,
                    )
                    run.path(locations["split_checks.json"]).write_text(
                        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
                    )
                    report = (
                        f"# Step 09 frozen patient split\n\nCohort: `{settings['cohort_version']}`. Split: `{version}`. Seed: 42. Algorithm: `{ALGORITHM}`.\n\n"
                        "| Group | Patients | Ever phys_death | Ever high_burden |\n| --- | ---: | ---: | ---: |\n"
                        + "\n".join(
                            f"| {split} | {quotas['total'][i]} | {quotas['ever_phys_death'][i]} | {quotas['ever_high_burden'][i]} |"
                            for i, split in enumerate(SPLITS)
                        )
                        + "\n\nAllocation uses integer 70/15/15 weights and bounded largest-deficit rounding, with train/validation/test tie priority. One PCG64 generator processes sorted high-burden, other phys_death-positive and phys_death-negative patients in that order. No study-count optimization or seed search is performed.\n\n"
                        + "All studies and admissions of a patient remain in one group. Coverage, disjointness, quotas, high-burden coverage and stable source leakage checks passed. split_summary.csv distinguishes study scores from patient maximum scores; patients/admissions in study-score bins may overlap and must not be added.\n\n"
                        + "High-burden validation/test patient counts remain small; one positive patient permits class coverage but does not establish stable evaluation. split_checks.json records missing classes for each binary component/endpoint without resampling. Patient counts, rather than correlated studies, describe the independent positive subjects.\n\n"
                        + "Steps 01-08 and the common cohort are unchanged. All three models must join these same assignments by subject_id. Patient IDs, labels and future observation data are not EHR features. No feature extraction or training package is produced.\n\n"
                        + f"CSV checks: no_csv_inputs; see ../step_09_csv_tail_checks.csv. Step 05 accepted {context['source05']['csv_tail_checks']['incomplete_file_count']} incomplete source files; see ../step_05_csv_tail_checks.csv. Step 07 review checks remain in ../step_07_csv_tail_checks.csv. No raw CSV records are parsed.\n\n"
                        + "Identical split versions reuse immutable snapshots. Changed cohort content, parameters or code cannot overwrite a frozen split. A Step 08 verification-only rerun or a different Git HEAD does not change cohort identity when bound contents are unchanged.\n"
                    )
                    run.path(locations["step_09_report.md"]).write_text(
                        report, encoding="utf-8"
                    )
                    frozen = {
                        "patient_split_frozen": True,
                        "split_version": version,
                        "cohort_version": settings["cohort_version"],
                        "original_split_run_id": run.run_id,
                        "frozen_utc": started.isoformat(),
                        "source_binding": binding,
                        "source_fingerprint": fingerprint,
                        "checks": checks,
                        "runtime_versions": {
                            "numpy": np.__version__,
                            "pyarrow": pa.__version__,
                        },
                        "artifact_sha256": {
                            name: _file_hash(run.path(path))
                            for name, path in locations.items()
                        },
                        "git_commit": _git_commit(root),
                    }
                    run.path(freeze_path).write_text(
                        json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
                    )
                for name, path in locations.items():
                    current = data / name if name == TABLE_NAME else directory / name
                    shutil.copyfile(
                        path if reused else run.path(path), run.path(current)
                    )
                shutil.copyfile(
                    freeze_path if reused else run.path(freeze_path),
                    run.path(directory / "split_freeze.json"),
                )
                latest = load_frozen_cohort(root, configs, processing_file)
                latest_binding = {
                    "cohort": latest["cohort_identity"],
                    "split_config_sha256": _file_hash(split_file),
                    "settings": load_settings(split_file),
                    "code_sha256": _code_hash(),
                }
                if latest_binding != binding:
                    raise ValueError("Frozen inputs changed during Step 09")
                metadata = {
                    "run_id": run.run_id,
                    "original_split_run_id": frozen["original_split_run_id"],
                    "verification_run_id": run.run_id,
                    "snapshot_reused": reused,
                    "patient_split_frozen": True,
                    "split_checks_passed": True,
                    "split_version": version,
                    "cohort_version": settings["cohort_version"],
                    "source_binding": binding,
                    "split_config_path": str(split_file),
                    "source_fingerprint": fingerprint,
                    "checks": checks,
                    "artifact_sha256": frozen["artifact_sha256"],
                    "freeze_sha256": _file_hash(
                        freeze_path if reused else run.path(freeze_path)
                    ),
                    "latest_cohort_run_id": latest["step08"]["run_id"],
                    "latest_cohort_manifest_sha256": _file_hash(
                        context["outputs"] / "audit/step_08_manifest.json"
                    ),
                    "snapshot_data_path": str(snapshot_data),
                    "snapshot_audit_path": str(snapshot_audit),
                    "csv_tail_checks": run.csv_tail_summary(),
                    "source_csv_tail_checks": context["source05"]["csv_tail_checks"],
                    "review_csv_tail_checks": context["source07"]["csv_tail_checks"],
                    "started_utc": started.isoformat(),
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "runtime_versions": {
                        "numpy": np.__version__,
                        "pyarrow": pa.__version__,
                    },
                    "code_sha256": _code_hash(),
                    "git_commit": _git_commit(root),
                }
                run.path(directory / "step_09_run_metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )
    print(
        f"Step 09 complete: {version}; patient counts {dict(zip(SPLITS, quotas['total']))}; snapshot {'reused' if reused else 'frozen'}"
    )
    return 0
