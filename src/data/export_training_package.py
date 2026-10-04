"""Step 11: frozen-source export, immutable publication and portable ZIP."""

from __future__ import annotations

import hashlib
import math
import platform
import re
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.burden_io import COMPONENTS
from src.cohort.common import (
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
from src.ehr import build_ehr_features as upstream
from src.ehr import step10b as ehr
from src.ehr.lab_audit import json_hash
from src.imaging.preprocessing import validate_pixels

SPLITS = ("train", "validation", "test")
SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("subject_id", pa.int64()),
        ("hadm_id", pa.int64()),
        ("study_id", pa.int64()),
        ("dicom_id", pa.string()),
        ("split", pa.string()),
        ("image_relpath", pa.string()),
        ("burden_score", pa.float32()),
        *[(k, pa.uint8()) for k in COMPONENTS],
        ("phys_death", pa.uint8()),
        ("high_burden", pa.uint8()),
        ("t0", pa.timestamp("us")),
        ("observation_end", pa.timestamp("us")),
        ("observed_hours", pa.float32()),
        ("early_discharge", pa.uint8()),
    ]
)
IMAGE_SCHEMA = pa.schema(
    [
        *[
            (k, pa.string())
            for k in [
                "sample_id",
                "dicom_id",
                "dicom_path",
                "source_dicom_sha256",
                "npy_path",
                "npy_relpath",
                "npy_sha256",
                "view_position",
                "qc_status",
                "processing_config_sha256",
                "processing_version",
                "split",
            ]
        ],
        *[(k, pa.int64()) for k in ["subject_id", "hadm_id", "study_id"]],
        ("t0", pa.timestamp("us")),
    ]
)
SUMMARY_COLUMNS = ["split", "metric", "value", "unit", "denominator"]
PACKAGE_TABLES = [
    "samples.parquet",
    "ehr_features.parquet",
    "feature_schema.json",
    "summary.csv",
    "README.md",
    "dataset_metadata.json",
]


def load_settings(path: Path) -> dict:
    settings = _json_file(path)
    expected = {
        "interface_version",
        "release_id",
        "cohort_version",
        "split_version",
        "dictionary_version",
        "feature_set_version",
    }
    if (
        set(settings) != expected
        or settings["interface_version"] != "1.0"
        or not re.fullmatch(
            r"burden_v2_h72_r(?:0[1-9]|[1-9][0-9]+)", settings["release_id"]
        )
    ):
        raise ValueError("Invalid training export configuration")
    for name in expected - {"interface_version"}:
        ehr.identity(settings[name])
    return settings


def code_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    return json_hash(
        {
            str(p.relative_to(root)): _file_hash(p)
            for p in [
                Path(__file__),
                root / "scripts/11_export_training_package.py",
                root / "src/data/__init__.py",
            ]
        }
    )


def expected_image_path(row: dict) -> str:
    return f"images/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}.npy"


def build_samples(cohort: pa.Table, patients: pa.Table) -> pa.Table:
    records = patients.to_pylist()
    assignments = {r["subject_id"]: r["split"] for r in records}
    if (
        len(assignments) != len(records)
        or any(v not in SPLITS for v in assignments.values())
        or set(assignments) != set(cohort["subject_id"].to_pylist())
    ):
        raise ValueError("Patient split keys or assignments disagree with cohort")
    rows = []
    for original in sorted(cohort.to_pylist(), key=lambda r: r["sample_id"]):
        row = {
            k: original[k]
            for k in SAMPLE_SCHEMA.names
            if k not in {"split", "image_relpath"}
        }
        row.update(
            split=assignments[row["subject_id"]], image_relpath=original["npy_relpath"]
        )
        if row["image_relpath"] != expected_image_path(row):
            raise ValueError("Frozen image path does not follow the package layout")
        rows.append(row)
    table = pa.Table.from_pylist(rows, schema=SAMPLE_SCHEMA)
    validate_samples(table)
    return table


def validate_samples(samples: pa.Table) -> None:
    if (
        samples.schema != SAMPLE_SCHEMA
        or not samples.num_rows
        or any(samples[k].null_count for k in samples.column_names)
    ):
        raise ValueError("Samples schema or required fields are invalid")
    seen, triples, patients, paths = set(), set(), {}, set()
    for row in samples.to_pylist():
        key = row["sample_id"]
        triple = tuple(row[k] for k in ["subject_id", "hadm_id", "study_id"])
        expected_key = f"p{triple[0]}_h{triple[1]}_s{triple[2]}"
        if (
            key != expected_key
            or key in seen
            or triple in triples
            or any(v <= 0 for v in triple)
            or not row["dicom_id"]
        ):
            raise ValueError("Invalid or duplicated sample identity")
        seen.add(key)
        triples.add(triple)
        patient, split = row["subject_id"], row["split"]
        if split not in SPLITS or patient in patients and patients[patient] != split:
            raise ValueError("A patient crosses splits or has an invalid split")
        patients[patient] = split
        path = row["image_relpath"]
        if path != expected_image_path(row) or path in paths:
            raise ValueError("Invalid or duplicated package image path")
        paths.add(path)
        flags = [
            row[k]
            for k in [*COMPONENTS, "phys_death", "high_burden", "early_discharge"]
        ]
        if any(v not in (0, 1) for v in flags):
            raise ValueError("Invalid binary label")
        t = max(row[COMPONENTS[0]], row[COMPONENTS[1]])
        resp = max(row[COMPONENTS[2]], row[COMPONENTS[3]])
        circ, death = row[COMPONENTS[4]], row[COMPONENTS[5]]
        score = t + 2 * resp + 2 * circ + 3 * death
        if (
            row["burden_score"] != score
            or row["phys_death"] != int(bool(resp or circ or death))
            or row["high_burden"] != int(score >= 3)
        ):
            raise ValueError("Exported target formula is inconsistent")
        hours = (row["observation_end"] - row["t0"]).total_seconds() / 3600
        if (
            not 0 <= hours <= 72
            or not math.isfinite(row["observed_hours"])
            or not math.isclose(
                row["observed_hours"], hours, rel_tol=1e-6, abs_tol=1e-4
            )
            or row["observation_end"] > row["t0"] + timedelta(hours=72)
        ):
            raise ValueError("Exported observation times are inconsistent")


def summary_rows(samples: pa.Table) -> list[dict]:
    result = []
    for split in ["all", *SPLITS]:
        rows = [r for r in samples.to_pylist() if split == "all" or r["split"] == split]
        patients = {r["subject_id"] for r in rows}

        def add(metric, value, unit, denominator):
            result.append(
                dict(
                    split=split,
                    metric=metric,
                    value=value,
                    unit=unit,
                    denominator=denominator,
                )
            )

        add("studies", len(rows), "study", len(rows))
        add(
            "admissions",
            len({r["hadm_id"] for r in rows}),
            "admission",
            len({r["hadm_id"] for r in rows}),
        )
        add("patients", len(patients), "patient", len(patients))
        for score in range(9):
            add(
                f"score_{score}_studies",
                sum(r["burden_score"] == score for r in rows),
                "study",
                len(rows),
            )
            add(
                f"maximum_score_{score}_patients",
                sum(
                    max(r["burden_score"] for r in rows if r["subject_id"] == p)
                    == score
                    for p in patients
                ),
                "patient",
                len(patients),
            )
        for field in [*COMPONENTS, "phys_death", "high_burden", "early_discharge"]:
            add(field + "_studies", sum(r[field] for r in rows), "study", len(rows))
            add(
                field + "_patients",
                len({r["subject_id"] for r in rows if r[field]}),
                "patient",
                len(patients),
            )
    return result


def image_manifest(context: dict, samples: pa.Table, cxr: Path) -> pa.Table:
    cohort = {r["sample_id"]: r for r in context["cohort"].to_pylist()}
    qc = {r["sample_id"]: r for r in context["qc_records"]}
    if len(qc) != len(context["qc_records"]):
        raise ValueError("Duplicate QC keys")
    rows = []
    for sample in samples.to_pylist():
        row, checked = cohort[sample["sample_id"]], qc.get(sample["sample_id"])
        if (
            checked is None
            or checked["qc_status"] != "pass"
            or checked["processing_status"] != "complete"
        ):
            raise ValueError("Included study lacks accepted image QC")
        fields = [
            "subject_id",
            "hadm_id",
            "study_id",
            "dicom_id",
            "dicom_path",
            "t0",
            "npy_relpath",
            "npy_sha256",
            "source_dicom_sha256",
            "processing_config_sha256",
            "processing_version",
            "qc_status",
            "view_position",
        ]
        if any(checked[k] != row[k] for k in fields):
            raise ValueError("Cohort image identity disagrees with QC")
        record = {
            k: row[k]
            for k in IMAGE_SCHEMA.names
            if k not in {"dicom_path", "npy_path", "split"}
        }
        record.update(
            dicom_path=str(_under_root(cxr, row["dicom_path"])),
            npy_path=str(_under_root(context["image_dir"], row["npy_relpath"])),
            split=sample["split"],
        )
        rows.append(record)
    return pa.Table.from_pylist(rows, schema=IMAGE_SCHEMA)


def copy_image(source: Path, target: Path, digest: str) -> None:
    if _file_hash(source) != digest:
        raise ValueError("Source NPY checksum changed")
    validate_pixels(np.load(source, allow_pickle=False))
    shutil.copyfile(source, target)
    if _file_hash(source) != digest or _file_hash(target) != digest:
        raise ValueError("Source or copied NPY changed during export")
    validate_pixels(np.load(target, allow_pickle=False))


def archive_files(target: Path, files: dict[str, Path], release: str) -> None:
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        for name in sorted(files):
            if (
                name.replace("\\", "/") != name
                or name.startswith("/")
                or any(p in {"", ".", ".."} for p in name.split("/"))
                or ":" in name
            ):
                raise ValueError("Unsafe ZIP entry")
            info = zipfile.ZipInfo(f"{release}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with (
                archive.open(info, "w", force_zip64=True) as destination,
                files[name].open("rb") as source,
            ):
                shutil.copyfileobj(source, destination)


def verify_archive(path: Path, hashes: dict[str, str], release: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        expected = {f"{release}/{k}": v for k, v in hashes.items()}
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("ZIP contents differ from the frozen package")
        for name, digest in expected.items():
            hasher = hashlib.sha256()
            with archive.open(name) as source:
                while block := source.read(1024 * 1024):
                    hasher.update(block)
            if hasher.hexdigest() != digest:
                raise ValueError("ZIP member checksum differs")


def assert_portable(value) -> None:
    """Public JSON must not embed machine-specific absolute paths."""
    if isinstance(value, dict):
        for k, v in value.items():
            assert_portable(k)
            assert_portable(v)
    elif isinstance(value, list):
        for v in value:
            assert_portable(v)
    elif isinstance(value, str) and (
        re.search(r"[A-Za-z]:[\\/]", value) or value.startswith(("\\\\", "/"))
    ):
        raise ValueError("Public metadata contains an absolute machine path")


def verify_inputs(root: Path, args, settings: dict) -> dict:
    configs = [
        getattr(args, k).resolve() for k in ["paths", "cohort", "ehr", "imaging"]
    ]
    context = upstream.verify_frozen_inputs(
        root,
        configs,
        args.processing,
        args.split_config,
        upstream.load_settings(args.lab_config),
    )
    outputs, processed = context["outputs"], context["processed"]
    for step in range(1, 11):
        _verify_run(outputs, step)
    for stage in ["audit", "build"]:
        _verify_run(outputs / f"step_10b/{stage}", 10)
    feature_settings = ehr.load_settings(args.feature_config)
    directory = outputs / "audit/burden_v2/ehr" / feature_settings["audit_version"]
    data = processed / "ehr/burden_v2"
    upstream.dictionary_releases(context)
    dictionary = _json_file(outputs / "audit/burden_v2/lab_dictionary_freeze.json")
    paths = _json_file(args.paths)
    raw = _resolve_config_path(paths["mimic_iv"], root, raw=True)
    lab_settings = upstream.load_settings(args.lab_config)
    lab_sources = {k: _under_root(raw, v) for k, v in lab_settings["tables"].items()}
    lab_directory = outputs / "audit/burden_v2/labs" / lab_settings["audit_version"]
    lab_binding = upstream.source_binding(
        context, lab_sources, args.lab_config, lab_settings
    )
    lab_record = upstream.load_audit(
        lab_directory / "lab_audit_manifest.json",
        lab_binding,
        processed.parent,
        outputs,
    )
    if (
        dictionary["dictionary_version"] != settings["dictionary_version"]
        or dictionary["source_binding"]["code_sha256"] != upstream.code_hash()
    ):
        raise ValueError("Frozen laboratory dictionary identity changed")
    if dictionary["source_binding"]["audit_fingerprint"] != lab_record[
        "audit_fingerprint"
    ] or dictionary["source_binding"]["audit_manifest_sha256"] != _file_hash(
        lab_directory / "lab_audit_manifest.json"
    ):
        raise ValueError("Frozen dictionary laboratory evidence changed")
    ehr.verify_releases(data, directory.parent / "ehr_features_releases.json")
    frozen = _json_file(directory / "ehr_features_freeze.json")
    if (
        not frozen["ehr_features_ready"]
        or frozen["feature_set_version"] != settings["feature_set_version"]
        or settings["cohort_version"] != context["release_version"]
        or settings["split_version"] != context["split_identity"]["split_version"]
    ):
        raise ValueError(
            "Training sources are not frozen under the configured identities"
        )
    audit = _json_file(directory / "ehr_audit_manifest.json")
    binding = audit["binding"]
    ehr.verify_audit(audit, binding, (processed.parent, outputs))
    if (
        not audit["audit_checks_passed"]
        or audit["blocking_conflicts"]
        or frozen["binding"]
        != {
            **binding,
            "audit_manifest_sha256": _file_hash(directory / "ehr_audit_manifest.json"),
        }
    ):
        raise ValueError("EHR freeze is not bound to a passed audit")
    if (
        binding["code_sha256"] != ehr.code_hash()
        or binding["configuration_sha256"] != _file_hash(args.feature_config)
        or binding["cohort"] != context["cohort_identity"]
        or binding["split"] != context["split_identity"]
    ):
        raise ValueError("EHR source configuration or code changed")
    ehr.verify_sources(
        {k: Path(v) for k, v in binding["source_paths"].items()},
        binding["source_versions"],
    )
    freeze_path = (
        data / "releases" / settings["feature_set_version"] / "ehr_features_freeze.json"
    )
    if _file_hash(freeze_path) != _file_hash(directory / "ehr_features_freeze.json"):
        raise ValueError("Current EHR freeze record differs from its snapshot")
    # Manifests establish completed publication; immutable records bind identity.
    # A verification-only run ID change must not create a different package.
    files = {}
    files.update(
        cohort=processed / "cohort/burden_v2/cohort.parquet",
        cohort_freeze=outputs / "audit/burden_v2/cohort_freeze.json",
        split_freeze=outputs / "audit/burden_v2/split_freeze.json",
        endpoint_decision=outputs / "audit/burden_v2/endpoint_decision.json",
        split=processed / "splits/burden_v2/patient_split.parquet",
        ehr=data / "ehr_features_raw.parquet",
        schema=data / "feature_schema.json",
        feature_freeze=directory / "ehr_features_freeze.json",
        dictionary=data / "lab_dictionary.parquet",
        dictionary_freeze=outputs / "audit/burden_v2/lab_dictionary_freeze.json",
        lab_audit=outputs
        / "audit/burden_v2/labs"
        / upstream.load_settings(args.lab_config)["audit_version"]
        / "lab_audit_manifest.json",
    )
    for name, digest in frozen["artifact_sha256"].items():
        if _file_hash(data / name) != digest:
            raise ValueError("Current EHR file differs from its freeze")
    if (
        binding["dictionary_sha256"] != _file_hash(files["dictionary"])
        or binding["dictionary_freeze_sha256"] != _file_hash(files["dictionary_freeze"])
        or binding["lab_audit_sha256"] != _file_hash(files["lab_audit"])
    ):
        raise ValueError("EHR dictionary or audit evidence changed")
    schema = _json_file(files["schema"])
    assert_portable(schema)
    ehr.validate_wide(pq.read_table(files["ehr"]), context["cohort"], schema)
    context.update(
        schema=schema,
        feature_freeze=frozen,
        source_files=files,
        source_hashes_export={k: _file_hash(v) for k, v in files.items()},
        lab_source_paths=lab_sources,
        lab_source_versions=lab_binding["source_versions"],
        export_config_hashes={
            p.name: _file_hash(p)
            for p in [
                *configs,
                args.processing,
                args.split_config,
                args.lab_config,
                args.feature_config,
                args.export_config,
            ]
        },
        split_settings=_json_file(args.split_config),
        image_settings=_json_file(args.processing),
    )
    return context


def check_sources(context: dict) -> None:
    if {k: _file_hash(p) for k, p in context["source_files"].items()} != context[
        "source_hashes_export"
    ]:
        raise ValueError("Frozen source files changed during export")
    for row in context["cohort"].to_pylist():
        if (
            _file_hash(_under_root(context["image_dir"], row["npy_relpath"]))
            != row["npy_sha256"]
        ):
            raise ValueError("Source NPY changed during publication")
    binding = context["feature_freeze"]["binding"]
    ehr.verify_sources(
        {k: Path(v) for k, v in binding["source_paths"].items()},
        binding["source_versions"],
    )
    if {k: _file_version(p) for k, p in context["lab_source_paths"].items()} != context[
        "lab_source_versions"
    ]:
        raise ValueError("Laboratory sources changed during export")


def package_readme(settings: dict, generated_utc: str) -> str:
    return f"""# Training package {settings["release_id"]}

Interface 1.0. Outcome window: 72 hours. EHR lookback: 24 hours.
Generated UTC: {generated_utc}.
Feature set: {settings["feature_set_version"]}. The three models use identical samples,
labels and patient assignments. Read actual counts from summary.csv and metadata.

```python
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq

DATA_ROOT = Path("path/to/{settings["release_id"]}")
samples = pq.read_table(DATA_ROOT / "samples.parquet").to_pandas()
ehr = pq.read_table(DATA_ROOT / "ehr_features.parquet").to_pandas()
schema = json.loads((DATA_ROOT / "feature_schema.json").read_text())
features = schema["numeric_features"] + schema["categorical_features"]
aligned = samples[["sample_id"]].merge(ehr, on="sample_id", validate="one_to_one")
x = aligned[features]
gray = np.load(DATA_ROOT / samples.iloc[0]["image_relpath"], allow_pickle=False)
image = np.repeat(gray[None, :, :], 3, axis=0)
```

NPY files are float32, 224x224, single-channel, finite and in [0,1]. Do not
divide by 255 again. Model normalization and training augmentation belong to
the model loader. No random augmentation is stored in these files.

EHR numeric missing values are NaN; categorical missing values are __MISSING__.
Fit imputation, scaling and category vocabularies on train only. CatBoost uses
the raw mixed-type whitelist; the fusion MLP saves its train-fitted preprocessor.
Never use IDs, labels, split, observation duration or post-imaging fields as EHR inputs.

The target is operational deterioration burden, not a validated severity scale
or an event probability. Missing records do not establish clinical absence.
Hypoxemia uses supplied ICU chartevents only. Live early discharge truncates
follow-up; absent recorded events before the cutoff are operational zeros.
Deidentified anchor ages cannot recover true ages above 89. Services uses
transfertime as its availability proxy because no storetime is supplied.

ZIP contains this entire directory. After moving or extracting it, change only
DATA_ROOT. Export completed Step 11; independent Step 12 acceptance is pending.
"""


def metadata(
    context: dict,
    settings: dict,
    hashes: dict,
    summaries: list,
    root: Path,
    generated_utc: str,
) -> dict:
    decision = context["decision"]
    split = context["split_settings"]
    result = {
        "interface_version": "1.0",
        "release_id": settings["release_id"],
        "outcome_version": decision["outcome_version"],
        "cohort_version": settings["cohort_version"],
        "feature_set_version": settings["feature_set_version"],
        "dictionary_version": settings["dictionary_version"],
        "horizon_hours": 72,
        "ehr_lookback_hours": 24,
        "target_column": "burden_score",
        "target_range": [0, 8],
        "target_formula": decision["target_formula"],
        "split_column": "split",
        "split_values": list(SPLITS),
        "split_unit": "subject_id",
        "split_version": settings["split_version"],
        "split_rules": {
            k: split[k]
            for k in [
                "algorithm_version",
                "seed",
                "weights",
                "primary_stratum",
                "require_high_burden_in_each_split",
            ]
        },
        "image": {
            "format": "npy",
            "dtype": "float32",
            "shape": [224, 224],
            "value_range": [0.0, 1.0],
            "model_normalization_applied": False,
        },
        "image_config": {
            **context["image_settings"],
            "intensity_normalization": "After frozen DICOM display transforms, valid pixels are mapped by clip((value-p1)/(p99-p1),0,1); excluded pixels are zero.",
            "geometry": "Aspect-preserving bilinear resize, nearest-neighbor valid-pixel mask and centered zero padding.",
        },
        "hypoxemia_source": decision["hypoxemia_source"],
        "live_discharge_rule": decision["live_discharge_rule"],
        "feature_counts": {
            "numeric": len(context["schema"]["numeric_features"]),
            "categorical": len(context["schema"]["categorical_features"]),
        },
        "split_counts": {
            group: {
                x["metric"]: x["value"]
                for x in summaries
                if x["split"] == group
                and x["metric"] in ["studies", "admissions", "patients"]
            }
            for group in ["all", *SPLITS]
        },
        "generated_utc": generated_utc,
        "git_commit": _git_commit(root),
        "export_code_sha256": code_hash(),
        "configuration_sha256": context["export_config_hashes"],
        "upstream_file_sha256": context["source_hashes_export"],
        "file_sha256": hashes,
        "ehr_preprocessing_applied": False,
        "independent_validation_completed": False,
        "limitations": context["schema"].get("limitations", []),
    }
    assert_portable(result)
    return result


def verify_release(training: Path, audit: Path, release: str, record: dict) -> None:
    package = training / release
    expected = record["package_file_sha256"]
    actual = {
        p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()
    }
    if actual != set(expected):
        raise ValueError("Frozen package file inventory changed")
    for name, digest in expected.items():
        if _file_hash(_under_root(package, name)) != digest:
            raise ValueError("Frozen package checksum changed")
    archive = training / f"{release}.zip"
    if _file_hash(archive) != record["zip_sha256"]:
        raise ValueError("Frozen ZIP checksum changed")
    verify_archive(archive, expected, release)
    if (
        _file_hash(audit / "releases" / release / "image_manifest.parquet")
        != record["image_manifest_sha256"]
    ):
        raise ValueError("Frozen local image manifest changed")


def run_export(root: Path, args) -> int:
    settings = load_settings(args.export_config)
    context = verify_inputs(root, args, settings)
    processed, outputs = context["processed"], context["outputs"]
    training = processed / "training"
    release = settings["release_id"]
    package = training / release
    audit = outputs / "audit/burden_v2/training"
    snapshot = audit / "releases" / release
    paths = _json_file(args.paths)
    _protect_outputs(paths, root, processed, outputs)
    configs = [
        getattr(args, k).resolve()
        for k in [
            "paths",
            "cohort",
            "ehr",
            "imaging",
            "processing",
            "split_config",
            "lab_config",
            "feature_config",
            "export_config",
        ]
    ]
    if any(
        p.is_relative_to(processed.resolve()) or p.is_relative_to(outputs.resolve())
        for p in configs
    ):
        raise ValueError("Export configuration must be outside generated directories")
    samples = build_samples(
        context["cohort"], pq.read_table(context["source_files"]["split"])
    )
    ehr.validate_wide(
        pq.read_table(context["source_files"]["ehr"]), samples, context["schema"]
    )
    cxr = _resolve_config_path(paths["mimic_cxr"], root, raw=True)
    images = image_manifest(context, samples, cxr)
    binding = {
        "settings": settings,
        "configuration_sha256": context["export_config_hashes"],
        "source_sha256": context["source_hashes_export"],
        "cohort": context["cohort_identity"],
        "split": context["split_identity"],
        "feature_freeze_sha256": _file_hash(context["source_files"]["feature_freeze"]),
        "code_sha256": code_hash(),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
    }
    registry_path = audit / "training_package_releases.json"
    print(
        f"Step 11 export: {release}; directory and ZIP; no pixel processing or model fitting"
    )
    with _OutputRun(outputs, 11, processed) as run:
        if run.manifest.exists():
            _verify_run(outputs, 11)
        registry = _json_file(registry_path) if registry_path.exists() else {}
        for version, entry in registry.items():
            if not re.fullmatch(r"burden_v2_h72_r(?:0[1-9]|[1-9][0-9]+)", version):
                raise ValueError("Unsafe package registry version")
            record_path = audit / "releases" / version / "training_package_freeze.json"
            if _file_hash(record_path) != entry["freeze_sha256"]:
                raise ValueError("Historical package freeze record changed")
            verify_release(training, audit, version, _json_file(record_path))
        frozen_path = snapshot / "training_package_freeze.json"
        if release in registry:
            record = _json_file(frozen_path)
            if record["binding"] != binding:
                raise ValueError(
                    "Changed sources or code cannot overwrite this package identity"
                )
            if (
                _file_hash(processed / "imaging/burden_v2/image_manifest.parquet")
                != record["image_manifest_sha256"]
            ):
                raise ValueError("Current image manifest differs from frozen package")
            print("Identical frozen package and ZIP verified and reused.")
        else:
            if (
                package.exists()
                and any(p.is_file() for p in package.rglob("*"))
                or (training / f"{release}.zip").exists()
                or frozen_path.exists()
            ):
                raise ValueError("Unregistered package content cannot be overwritten")
            files = {}
            generated_utc = datetime.now(timezone.utc).isoformat()

            def target(name):
                path = run.path(_under_root(package, name))
                files[name] = path
                return path

            pq.write_table(samples, target("samples.parquet"), compression="zstd")
            shutil.copyfile(
                context["source_files"]["ehr"], target("ehr_features.parquet")
            )
            shutil.copyfile(
                context["source_files"]["schema"], target("feature_schema.json")
            )
            summaries = summary_rows(pq.read_table(files["samples.parquet"]))
            _write_csv(target("summary.csv"), SUMMARY_COLUMNS, summaries)
            target("README.md").write_text(
                package_readme(settings, generated_utc), encoding="utf-8"
            )
            for row in images.to_pylist():
                copy_image(
                    Path(row["npy_path"]), target(row["npy_relpath"]), row["npy_sha256"]
                )
            hashes = {name: _file_hash(path) for name, path in files.items()}
            upstream.write_json(
                target("dataset_metadata.json"),
                metadata(context, settings, hashes, summaries, root, generated_utc),
            )
            hashes = {name: _file_hash(path) for name, path in files.items()}
            archive = run.path(training / f"{release}.zip")
            archive_files(archive, files, release)
            verify_archive(archive, hashes, release)
            pq.write_table(
                images,
                run.path(snapshot / "image_manifest.parquet"),
                compression="zstd",
            )
            shutil.copyfile(
                run.path(snapshot / "image_manifest.parquet"),
                run.path(processed / "imaging/burden_v2/image_manifest.parquet"),
            )
            record = {
                "release_id": release,
                "training_package_exported": True,
                "independent_validation_completed": False,
                "binding": binding,
                "original_export_run_id": run.run_id,
                "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "package_file_sha256": hashes,
                "zip_sha256": _file_hash(archive),
                "image_manifest_sha256": _file_hash(
                    run.path(snapshot / "image_manifest.parquet")
                ),
            }
            upstream.write_json(run.path(frozen_path), record)
            registry[release] = {"freeze_sha256": _file_hash(run.path(frozen_path))}
        check_sources(context)
        if (
            context["export_config_hashes"] != {p.name: _file_hash(p) for p in configs}
            or binding["code_sha256"] != code_hash()
        ):
            raise ValueError("Export configuration or code changed during publication")
        # Keep the historical registry covered by every verification-run manifest.
        upstream.write_json(run.path(registry_path), registry)
        upstream.write_json(run.path(audit / "training_package_freeze.json"), record)
        upstream.write_json(
            run.path(audit / "step_11_run_metadata.json"),
            {
                "run_id": run.run_id,
                "release_id": release,
                "training_package_exported": True,
                "independent_validation_completed": False,
                "original_export_run_id": record["original_export_run_id"],
                "freeze_sha256": registry[release]["freeze_sha256"],
                "csv_tail_checks": run.csv_tail_summary(),
            },
        )
    print(
        f"Export complete: {package}; ZIP: {training / (release + '.zip')}. Step 12 remains pending."
    )
    return 0
