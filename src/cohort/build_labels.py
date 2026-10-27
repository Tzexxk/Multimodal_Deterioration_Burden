"""Step 06: verify a frozen audit and transactionally publish formal labels."""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort import audit_burden_labels as audit
from src.cohort.build_candidates import CANDIDATE_COLUMNS, OBSERVATION_COLUMNS
from src.cohort.burden_events import EVIDENCE_COLUMNS
from src.cohort.burden_io import COMPONENTS, open_store
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
)

LABEL_SCHEMA_VERSION = "burden_v2_labels_v1"
SCORES = "T R C D burden_score phys_death high_burden".split()
READY_FLAGS = (
    "audit_checks_passed",
    "dictionary_handling_review_complete",
    "audit_ready_for_window_decision",
)


def _json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _audit_code_hash() -> str:
    directory = Path(audit.__file__).parent
    return _config_hash(
        [
            directory / "audit_burden_labels.py",
            *(
                directory / name
                for name in (
                    "burden_io.py",
                    "build_candidates.py",
                    "burden_events.py",
                    "common.py",
                )
            ),
            directory.parent.parent / "scripts/05_audit_burden_labels.py",
        ]
    )


def verify_frozen_inputs(
    root: Path,
    config_files: list[Path],
    *,
    decision_file: Path | None = None,
    window_file: Path | None = None,
) -> tuple[dict, dict, dict]:
    """Hash review inputs and stat raw sources; never parse or scan raw CSVs."""
    paths, cohort, ehr, imaging = map(_json_file, config_files)
    rules = audit.validate_rules(cohort, ehr, imaging)
    processed = _resolve_config_path(paths["processed"], root, raw=False)
    outputs = _resolve_config_path(paths["outputs"], root, raw=False)
    _protect_outputs(paths, root, processed, outputs)
    decision_path = outputs / "audit/burden_v2/endpoint_decision.json"
    decision = _json_file(decision_file or decision_path)
    expected = {
        "decision_status": "frozen",
        "horizon_frozen": True,
        "label_rules_frozen": True,
        "outcome_version": "burden_v2",
        "sample_unit": "study",
        "ehr_lookback_hours": 24,
        "event_window": "(t0, min(t0+H, dischtime)]",
        "transfer_weight": 1,
        "respiratory_weight": 2,
        "circulatory_weight": 2,
        "death_weight": 3,
        "domain_capping": True,
        "target_column": "burden_score",
        "target_range": [0, 8],
        "high_burden_threshold": 3,
        "early_3h_events_included": True,
        "hypoxemia_source": "MIMIC-IV ICU chartevents only",
        "hypoxemia_itemid": 220277,
        "hypoxemia_threshold_percent": 85,
        "target_formula": "T + 2*R + 2*C + 3*D; T=max(icu,stepdown), R=max(imv,hypoxemia), C=infusion, D=death",
        "phys_death_rule": "int(R == 1 or C == 1 or D == 1)",
        "live_discharge_rule": "Retain events in (t0, dischtime] when alive discharge precedes t0+H; absent recorded events are operational zeros.",
    }
    if any(
        decision.get(k) != v or type(decision.get(k)) is not type(v)
        for k, v in expected.items()
    ):
        raise ValueError("Endpoint decision is not frozen under the supported V2 rules")
    horizon = decision.get("horizon_hours")
    if type(horizon) is not int or horizon not in (48, 72):
        raise ValueError("Unsupported frozen horizon")
    if not all(
        decision.get(k)
        for k in ("decision_version", "reviewer", "reviewed_on", "rationale")
    ):
        raise ValueError("Missing endpoint review provenance")
    source = decision["source_audit"]
    meta_path = outputs / "audit/burden_v2/run_metadata.json"
    manifest_path = outputs / "audit/step_05_manifest.json"
    for key, path in (("manifest", manifest_path), ("metadata", meta_path)):
        if (
            _under_root(root, source[f"{key}_path"]) != path
            or _file_hash(path) != source[f"{key}_sha256"]
        ):
            raise ValueError(f"Frozen audit {key} changed")
    metadata = _json_file(meta_path)
    for step in (3, 4, 5):
        _verify_run(outputs, step)
    if (
        metadata["run_id"] != source["run_id"]
        or _json_file(manifest_path)["run_id"] != source["run_id"]
    ):
        raise ValueError("Frozen audit run mismatch")
    if not all(metadata.get(k) is True and source.get(k) is True for k in READY_FLAGS):
        raise ValueError("Frozen audit is not ready for formal labels")
    if metadata["csv_tail_checks"] != source["csv_tail_checks"]:
        raise ValueError("Frozen source CSV warnings mismatch")
    for key, path in (
        ("window_decision", outputs / "audit/burden_v2/window_decision.md"),
    ):
        if (
            _under_root(root, decision[f"{key}_path"]) != path
            or _file_hash(window_file or path) != decision[f"{key}_sha256"]
        ):
            raise ValueError("Window decision changed")
    selection = decision["selection_record"]
    if _file_hash(_under_root(root, selection["path"])) != selection["sha256"]:
        raise ValueError("Window selection record changed")
    extra_files = []
    for field, hash_field in (
        ("overrides_path", "override_sha256"),
        ("candidate_exclusions_path", "candidate_exclusions_sha256"),
        *audit.REVIEW_PATHS.items(),
    ):
        path = (
            _resolve_config_path(rules[field], root, raw=False)
            if rules.get(field)
            else None
        )
        digest = _file_hash(path) if path else None
        if digest != metadata.get(hash_field) or digest != decision.get(hash_field):
            raise ValueError(f"Frozen {field} changed")
        if path:
            extra_files.append(path)
    for name, digest in metadata.get("review_snapshot_bindings", {}).items():
        path = _under_root(root, name)
        if _file_hash(path) != digest:
            raise ValueError("Frozen review snapshot binding changed")
        audit.verify_snapshot(path)
    if decision.get("baseline_reference_bindings", {}) != metadata.get(
        "baseline_reference_bindings", {}
    ):
        raise ValueError("Baseline reference set changed")
    for name, digest in metadata.get("baseline_reference_bindings", {}).items():
        if _file_hash(_under_root(root, name)) != digest:
            raise ValueError("Baseline reference changed")
    if decision.get("review_snapshot_bindings", {}) != metadata.get(
        "review_snapshot_bindings", {}
    ):
        raise ValueError("Frozen decision review snapshot set changed")
    for name, digest in decision.get("review_records_sha256", {}).items():
        if _file_hash(_under_root(root, name)) != digest:
            raise ValueError("Endpoint review record changed")
    dictionary = {
        name: rules[name]
        for name in (*audit.MAPPING_ACTIONS, "medications", "spo2_units", "items")
    }
    hashes = {
        "config_sha256": _config_hash(config_files + extra_files),
        "rules_config_sha256": _file_hash(config_files[1]),
        "dictionary_sha256": _json_hash(dictionary),
        "dictionary_handling_reviews_sha256": _json_hash(
            rules.get("dictionary_handling_reviews", {})
        ),
        "code_sha256": _audit_code_hash(),
    }
    for key, digest in hashes.items():
        frozen = source if key in ("config_sha256", "code_sha256") else decision
        if digest != metadata[key] or digest != frozen[key]:
            raise ValueError(f"Frozen {key} changed")
    for key in ("rules_version", "dictionary_version"):
        if rules[key] != metadata[key] or rules[key] != decision[key]:
            raise ValueError(f"Frozen {key} mismatch")
    iv = _resolve_config_path(paths["mimic_iv"], root, raw=True)
    tables = ehr["burden_v2"]["tables"]
    versions = {name: _file_version(_under_root(iv, p)) for name, p in tables.items()}
    if (
        not all(v["exists"] for v in versions.values())
        or tables != metadata["source_tables"]
        or versions != metadata["input_versions"]
        or versions != source["input_versions"]
    ):
        raise ValueError("Raw source versions changed; repeat the audit and decision")
    if metadata["upstream"] != source["upstream"]:
        raise ValueError("Frozen upstream provenance mismatch")
    for key, path, step in (
        ("selection_sha256", processed / "index/study_image_selection.parquet", 3),
        ("matches_sha256", processed / "index/cxr_admission_matches.parquet", 4),
    ):
        _verify_run(outputs, step, path)
        if _file_hash(path) != source["upstream"][key]:
            raise ValueError("Frozen selection or matching changed")
    data = processed / "cohort/burden_v2"
    artifacts = {
        "candidates": data / "candidate_studies.parquet",
        "candidate_evidence": data / "candidate_evidence.parquet",
        "observations": data / f"window_{horizon}h/candidate_observation.parquet",
        "wide": data / f"window_{horizon}h/study_event_audit.parquet",
        "evidence": data / f"window_{horizon}h/event_evidence.parquet",
    }
    frozen_artifacts = decision["selected_window_artifact_sha256"]
    if {_under_root(root, p) for p in frozen_artifacts} != set(artifacts.values()):
        raise ValueError("Frozen selected artifact set mismatch")
    for p, digest in frozen_artifacts.items():
        path = _under_root(root, p)
        if (
            str(path) not in _json_file(manifest_path)["files"]
            or _file_hash(path) != digest
        ):
            raise ValueError("Frozen selected artifact changed")
    return (
        decision,
        metadata,
        {
            "processed": processed,
            "outputs": outputs,
            "artifacts": artifacts,
            "memory_limit": ehr["burden_v2"]["memory_limit"],
            "decision_sha256": _file_hash(decision_file or decision_path),
        },
    )


def _reject(connection, condition: str, message: str) -> None:
    if connection.execute(condition).fetchone()[0]:
        raise ValueError(message)


def validate_labels(connection, artifacts: dict, horizon: int) -> dict:
    """Reconstruct binary flags and scores from final evidence, independently of raw ingestion."""
    for name, required in (
        ("candidates", CANDIDATE_COLUMNS),
        ("observations", OBSERVATION_COLUMNS),
        ("wide", audit.WIDE_COLUMNS),
        ("evidence", EVIDENCE_COLUMNS),
    ):
        schema = pq.read_schema(artifacts[name])
        if len(schema.names) != len(set(schema.names)) or set(required) - set(
            schema.names
        ):
            raise ValueError(f"Missing {name} fields")
        if schema.field("sample_id").type != pa.string():
            raise ValueError("Sample keys must be strings")
        for field in schema:
            if pa.types.is_timestamp(field.type) and field.type != pa.timestamp("us"):
                raise ValueError(
                    "Clinical timestamps must be timezone-free microseconds"
                )
        for field in ("subject_id", "hadm_id", "study_id"):
            if schema.field(field).type != pa.int64():
                raise ValueError("Identifiers must be int64")
        connection.read_parquet(str(artifacts[name])).create_view(name)
        if name != "evidence":
            _reject(
                connection,
                f"SELECT count(*) FROM (SELECT sample_id FROM {name} GROUP BY sample_id HAVING count(*)<>1 OR sample_id IS NULL)",
                "Duplicate or null sample key",
            )
    for name, columns in (
        ("observations", CANDIDATE_COLUMNS),
        ("wide", OBSERVATION_COLUMNS),
    ):
        reference = "candidates" if name == "observations" else "observations"
        mismatch = " OR ".join(f"a.{c} IS DISTINCT FROM b.{c}" for c in columns)
        _reject(
            connection,
            f"SELECT count(*) FROM {reference} a FULL JOIN {name} b USING(sample_id) WHERE a.sample_id IS NULL OR b.sample_id IS NULL OR {mismatch}",
            "Candidate, observation and audit rows disagree",
        )
    _reject(
        connection,
        "SELECT count(*) FROM wide WHERE subject_id IS NULL OR hadm_id IS NULL OR study_id IS NULL OR sample_id<>concat('p',subject_id,'_h',hadm_id,'_s',study_id) OR candidate_status IS NULL OR candidate_status NOT IN ('eligible','excluded')",
        "Invalid sample identity or unresolved candidate",
    )
    _reject(
        connection,
        "SELECT count(*) FROM (SELECT subject_id,hadm_id,study_id FROM wide GROUP BY ALL HAVING count(*)<>1)",
        "Duplicate study identifiers",
    )
    _reject(
        connection,
        f"SELECT count(*) FROM wide WHERE horizon_hours IS DISTINCT FROM {horizon}",
        "Mixed observation horizons",
    )
    null_labels = " OR ".join(
        f"{c} IS NOT NULL"
        for c in (*COMPONENTS, *(f"{c}_first_event_time" for c in COMPONENTS), *SCORES)
    )
    _reject(
        connection,
        f"SELECT count(*) FROM wide WHERE candidate_status='excluded' AND (label_status IS DISTINCT FROM 'excluded' OR {null_labels})",
        "Excluded studies must retain null labels",
    )
    _reject(
        connection,
        "SELECT count(*) FROM wide WHERE candidate_status='eligible' AND label_status IS DISTINCT FROM 'complete'",
        "Eligible studies require completed labels",
    )
    connection.execute(
        "CREATE VIEW eligible AS SELECT * FROM wide WHERE candidate_status='eligible'"
    )
    _reject(
        connection,
        f"SELECT count(*) FROM eligible WHERE dicom_id IS NULL OR dicom_id='' OR dicom_path IS NULL OR dicom_path='' OR t0 IS NULL OR admittime IS NULL OR dischtime IS NULL OR t0<admittime OR t0>=dischtime OR horizon_end IS DISTINCT FROM t0+INTERVAL '{horizon} hours' OR observation_end IS DISTINCT FROM least(horizon_end,dischtime) OR observed_hours IS NULL OR NOT isfinite(observed_hours) OR abs(observed_hours-epoch(observation_end-t0)/3600)>0.0001 OR early_discharge IS DISTINCT FROM cast(dischtime<horizon_end AND hospital_expire_flag='0' AND NOT contains(exclusion_reason,'death_time_conflict') AS UTINYINT)",
        "Invalid observation boundaries or discharge status",
    )
    _reject(
        connection,
        "SELECT count(*) FROM evidence e LEFT JOIN wide w USING(sample_id) WHERE w.sample_id IS NULL OR e.subject_id IS DISTINCT FROM w.subject_id OR e.hadm_id IS DISTINCT FROM w.hadm_id OR e.study_id IS DISTINCT FROM w.study_id OR e.horizon_hours IS DISTINCT FROM w.horizon_hours OR e.source_record_key IS NULL OR e.source_record_key=''",
        "Invalid evidence ownership or source key",
    )
    allowed = ",".join(f"'{c}'" for c in COMPONENTS)
    _reject(
        connection,
        f"SELECT count(*) FROM evidence WHERE component IS NULL OR component NOT IN ({allowed}) OR (qualifies IS NOT NULL AND qualifies NOT IN (0,1))",
        "Invalid evidence component or flag",
    )
    _reject(
        connection,
        "SELECT count(*) FROM evidence e JOIN eligible w USING(sample_id) WHERE (qualifies IS NULL AND coalesce(blocks_event,1)<>0) OR (qualifies=1 AND (event_time IS NULL OR first_event_time IS NULL OR event_time<=t0 OR event_time>observation_end OR first_event_time<=t0 OR first_event_time>observation_end))",
        "Unresolved or out-of-window evidence",
    )
    verify_nonblocking_evidence(
        pq.read_table(artifacts["wide"]).to_pylist(),
        pq.read_table(artifacts["evidence"]).to_pylist(),
    )
    for c in COMPONENTS:
        _reject(
            connection,
            f"WITH d AS (SELECT sample_id, coalesce(max(qualifies),0) flag, min(first_event_time) FILTER(WHERE qualifies=1) first_time FROM evidence WHERE component='{c}' GROUP BY sample_id) SELECT count(*) FROM eligible w LEFT JOIN d USING(sample_id) WHERE w.{c} IS DISTINCT FROM coalesce(d.flag,0) OR w.{c}_first_event_time IS DISTINCT FROM d.first_time OR w.{c}_event_status IS NULL OR w.{c}_event_status NOT IN ('positive','preexisting_only','no_qualifying_record') OR (w.{c}=1 AND w.{c}_event_status<>'positive') OR (w.{c}=0 AND w.{c}_event_status='positive')",
            f"Evidence does not reproduce {c}",
        )
    a, b, c, d, e, f = COMPONENTS
    connection.execute(
        f"CREATE VIEW recalculated AS SELECT * EXCLUDE(T,R,C,D,burden_score,phys_death,high_burden), cast(greatest({a},{b}) AS UTINYINT) T, cast(greatest({c},{d}) AS UTINYINT) R, cast({e} AS UTINYINT) C, cast({f} AS UTINYINT) D FROM eligible"
    )
    connection.execute(
        "CREATE VIEW scored AS SELECT *, cast(T+2*R+2*C+3*D AS FLOAT) burden_score, cast(R=1 OR C=1 OR D=1 AS UTINYINT) phys_death, cast(T+2*R+2*C+3*D>=3 AS UTINYINT) high_burden FROM recalculated"
    )
    mismatch = " OR ".join(f"w.{c} IS DISTINCT FROM s.{c}" for c in SCORES)
    _reject(
        connection,
        f"SELECT count(*) FROM eligible w JOIN scored s USING(sample_id) WHERE {mismatch}",
        "Audit scores disagree with recalculated labels",
    )
    for field in (
        *COMPONENTS,
        "T",
        "R",
        "C",
        "D",
        "phys_death",
        "high_burden",
        "early_discharge",
    ):
        if pq.read_schema(artifacts["wide"]).field(field).type != pa.uint8():
            raise ValueError("Binary flags must be uint8")
    for field in ("burden_score", "observed_hours"):
        if pq.read_schema(artifacts["wide"]).field(field).type != pa.float32():
            raise ValueError("Scores and observation hours must be float32")
    counts = audit.query_rows(
        connection,
        "SELECT count(*) eligible_studies, count(DISTINCT hadm_id) eligible_admissions, count(DISTINCT subject_id) eligible_patients, count(*) FILTER(WHERE burden_score>0) nonzero_studies, count(*) FILTER(WHERE phys_death=1) phys_death_studies, count(*) FILTER(WHERE high_burden=1) high_burden_studies, 0 unresolved_studies FROM scored",
    )[0]
    if not counts["eligible_studies"]:
        raise ValueError("No formal labels to publish")
    return counts


def verify_nonblocking_evidence(studies: list[dict], evidence_rows: list[dict]) -> None:
    """Recheck retained uncertainty from its raw ownership, time and positive evidence."""
    indexed = {r["sample_id"]: r for r in studies}
    groups = {}
    for row in evidence_rows:
        groups.setdefault((row["sample_id"], row["component"]), []).append(row)
    reasons = {
        "indirect_ventilation_only",
        "spo2_artifact_review",
        "conflicting_spo2_at_same_time",
        "unconfirmed_transfer_mapping_or_origin",
    }
    for (sample, component), records in groups.items():
        study = indexed[sample]
        if study["candidate_status"] != "eligible":
            continue
        positive = [r["event_time"] for r in records if r["qualifies"] == 1]
        for row in records:
            if row["qualifies"] is not None:
                continue
            raw = json.loads(row["raw_evidence"])
            if row["blocks_event"] != 0 or (
                str(raw.get("subject_id")) != str(study["subject_id"])
                or str(raw.get("hadm_id")) != str(study["hadm_id"])
            ):
                raise ValueError(
                    "Unresolved evidence cannot be independently justified"
                )
            if row["uncertainty_relevance"] == "independent_after_confirmed_positive":
                valid = (
                    positive
                    and row["event_time"] is not None
                    and row["event_time"] > min(positive)
                    and row["exclusion_reason"] in reasons
                )
            elif row["uncertainty_relevance"] == "history_without_window_infusion":
                valid = (
                    component == COMPONENTS[4]
                    and row["event_time"] is not None
                    and row["event_time"] <= study["t0"]
                    and not any(
                        r["event_time"] is not None
                        and study["t0"] < r["event_time"] <= study["observation_end"]
                        and r["exclusion_reason"]
                        != "excluded_medication_or_administration"
                        for r in records
                    )
                )
            else:
                valid = False
            if not valid:
                raise ValueError(
                    "Unresolved evidence has an invalid nonblocking classification"
                )


def _write_query(connection, run: _OutputRun, path: Path, query: str) -> None:
    if path.suffix == ".parquet":
        connection.execute(
            f"COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(run.path(path))],
        )
    else:
        audit.write_query(connection, run, path, query)


def build_burden_labels(
    root: Path, paths_file: Path, cohort_file: Path, ehr_file: Path, imaging_file: Path
) -> int:
    started = datetime.now(timezone.utc)
    configs = [paths_file, cohort_file, ehr_file, imaging_file]
    decision, source_metadata, context = verify_frozen_inputs(root, configs)
    horizon = decision["horizon_hours"]
    data = context["processed"] / "cohort/burden_v2"
    audit_dir = context["outputs"] / "audit/burden_v2"
    print(
        f"Verified frozen audit {source_metadata['run_id']}; building H={horizon} labels"
    )
    print("CSV tail checks: no_csv_inputs; see the retained Step 05 source checks")
    with (
        _OutputRun(context["outputs"], 6, context["processed"]) as run,
        tempfile.TemporaryDirectory(
            prefix=".step_06_", dir=run.manifest.parent
        ) as temporary,
    ):
        with open_store(Path(temporary), context["memory_limit"]) as connection:
            counts = validate_labels(connection, context["artifacts"], horizon)
            if counts != decision["window_comparison"][str(horizon)]:
                raise ValueError("Label counts differ from the frozen decision")
            connection.execute(
                "CREATE TABLE publication_version(decision_version VARCHAR, rules_version VARCHAR, dictionary_version VARCHAR, source_audit_run_id VARCHAR, label_schema_version VARCHAR)"
            )
            connection.execute(
                "INSERT INTO publication_version VALUES (?,?,?,?,?)",
                [
                    decision["decision_version"],
                    decision["rules_version"],
                    decision["dictionary_version"],
                    source_metadata["run_id"],
                    LABEL_SCHEMA_VERSION,
                ],
            )
            _write_query(
                connection,
                run,
                data / "study_burden_labels.parquet",
                "SELECT * FROM scored CROSS JOIN publication_version ORDER BY sample_id",
            )
            _write_query(
                connection,
                run,
                data / "event_evidence.parquet",
                "SELECT e.* FROM evidence e JOIN eligible USING(sample_id) ORDER BY sample_id,component,source_record_key,event_time",
            )
            flags = [f"early_3h_{c}" for c in COMPONENTS]
            early = ",".join(
                f"cast(coalesce({c}_first_event_time<=least(t0+INTERVAL '3 hours',observation_end),false) AS UTINYINT) {name}"
                for c, name in zip(COMPONENTS, flags)
            )
            times = ",".join(f"{c}_first_event_time" for c in COMPONENTS)
            connection.execute(
                f"CREATE VIEW early AS SELECT *, {early}, least({times}) first_early_event_time FROM scored"
            )
            formula = f"greatest({flags[0]},{flags[1]})+2*greatest({flags[2]},{flags[3]})+2*{flags[4]}+3*{flags[5]}"
            audit.write_query(
                connection,
                run,
                audit_dir / "early_3h_events.csv",
                f"SELECT sample_id,subject_id,hadm_id,study_id,t0,horizon_hours,observation_end,{','.join(flags)},first_early_event_time,{formula} early_3h_score,burden_score FROM early WHERE first_early_event_time<=least(t0+INTERVAL '3 hours',observation_end) ORDER BY sample_id",
            )
            audit.write_query(
                connection,
                run,
                audit_dir / "early_discharge.csv",
                "SELECT sample_id,subject_id,hadm_id,study_id,t0,dischtime,horizon_hours,observation_end,observed_hours,"
                + ",".join(COMPONENTS)
                + ",burden_score,phys_death,high_burden FROM scored WHERE early_discharge=1 ORDER BY sample_id",
            )
            component_counts = audit.query_rows(
                connection,
                "SELECT "
                + ",".join(
                    f"count(*) FILTER(WHERE {c}=1) {c}_studies,count(DISTINCT subject_id) FILTER(WHERE {c}=1) {c}_patients"
                    for c in COMPONENTS
                )
                + " FROM scored",
            )[0]
            source_event_counts = audit.query_rows(
                connection,
                "SELECT component,count(DISTINCT source_record_key) source_record_count "
                "FROM evidence JOIN eligible USING(sample_id) WHERE qualifies=1 "
                "GROUP BY component ORDER BY component",
            )
            distribution = audit.query_rows(
                connection,
                "SELECT r.range score,count(s.sample_id) study_count FROM range(9) r LEFT JOIN scored s ON s.burden_score=r.range GROUP BY r.range ORDER BY r.range",
            )
            early_summary = audit.query_rows(
                connection,
                "SELECT count(*) study_count,count(DISTINCT subject_id) patient_count FROM early WHERE first_early_event_time<=least(t0+INTERVAL '3 hours',observation_end)",
            )[0]
            early_scores = audit.query_rows(
                connection,
                f"SELECT {formula} early_3h_score,count(*) study_count FROM early WHERE first_early_event_time<=least(t0+INTERVAL '3 hours',observation_end) GROUP BY ALL ORDER BY 1",
            )
            discharge_summary = audit.query_rows(
                connection,
                "SELECT count(*) study_count,count(DISTINCT subject_id) patient_count,avg(cast(burden_score=0 AS FLOAT)) zero_score_proportion,min(observed_hours) min_hours,median(observed_hours) median_hours,max(observed_hours) max_hours FROM scored WHERE early_discharge=1",
            )[0]
            coverage = audit.query_rows(
                connection,
                "SELECT "
                + ",".join(
                    f"count(*) FILTER(WHERE {c}=0) {c}_missing_studies"
                    for c in (
                        "spo2_source_available",
                        "emar_source_available",
                        "procedure_source_available",
                    )
                )
                + " FROM scored",
            )[0]
            # Recheck before the transaction commits, including sources only statted above.
            latest_decision, latest_source, latest_context = verify_frozen_inputs(
                root, configs
            )
            if (latest_decision, latest_source, latest_context) != (
                decision,
                source_metadata,
                context,
            ):
                raise ValueError("Frozen inputs changed during Step 06")
            metadata = {
                "run_id": run.run_id,
                "outcome_version": "burden_v2",
                "label_schema_version": LABEL_SCHEMA_VERSION,
                "horizon_hours": horizon,
                "horizon_frozen": True,
                "label_rules_frozen": True,
                "formal_labels_published": True,
                "label_checks_passed": True,
                "common_cohort_frozen": False,
                "image_qc_performed": False,
                "decision_sha256": context["decision_sha256"],
                "decision_version": decision["decision_version"],
                "source_audit": decision["source_audit"],
                "selected_window_artifact_sha256": decision[
                    "selected_window_artifact_sha256"
                ],
                "rules_version": decision["rules_version"],
                "dictionary_version": decision["dictionary_version"],
                "csv_tail_checks": run.csv_tail_summary(),
                "source_csv_tail_checks": source_metadata["csv_tail_checks"],
                "counts": counts,
                "component_counts": component_counts,
                "qualifying_source_counts": source_event_counts,
                "score_distribution": distribution,
                "early_3h": early_summary,
                "early_3h_score_distribution": early_scores,
                "early_discharge": discharge_summary,
                "source_coverage": coverage,
                "started_utc": started.isoformat(),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": _git_commit(root),
                "code_sha256": _config_hash(
                    [
                        Path(__file__),
                        Path(__file__).parents[2] / "scripts/06_build_burden_labels.py",
                    ]
                ),
            }
            run.path(audit_dir / "step_06_run_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            component_table = "\n".join(
                f"| {c} | {component_counts[c + '_studies']} | {component_counts[c + '_patients']} |"
                for c in COMPONENTS
            )
            score_table = "\n".join(
                f"| {r['score']} | {r['study_count']} |" for r in distribution
            )
            report = (
                f"# Step 06 formal burden labels\n\nFrozen horizon: {horizon} hours. "
                f"Source audit: `{source_metadata['run_id']}`. Decision: `{decision['decision_version']}`.\n\n"
                "Evidence, first times, observation boundaries, scores, keys and frozen hashes passed validation. "
                "These formal labels precede image QC and the common training cohort.\n\n"
                + "\n".join(f"- {k}: {v}" for k, v in counts.items())
                + "\n\n## Components and scores\n\n| Component | Studies | Patients |\n| --- | ---: | ---: |\n"
                + component_table
                + "\n\n| Score | Studies |\n| ---: | ---: |\n"
                + score_table
                + "\n\n## Observation summaries\n\nSee step_06_run_metadata.json for deduplicated "
                "qualifying source counts and early-event score distributions.\n\n"
                + f"- Early events: {early_summary['study_count']} studies, {early_summary['patient_count']} patients.\n"
                + f"- Alive early discharge: {discharge_summary['study_count']} studies; zero-score proportion: {discharge_summary['zero_score_proportion']}.\n"
                + f"- Early discharge observed hours (min/median/max): {discharge_summary['min_hours']}/{discharge_summary['median_hours']}/{discharge_summary['max_hours']}.\n"
                + "\n".join(f"- {k}: {v}" for k, v in coverage.items())
                + "\n\n"
                "## CSV integrity and provenance\n\nStep 06 parses no CSV inputs: "
                "../step_06_csv_tail_checks.csv records no_csv_inputs. Review CSVs are hashed only. "
                "The source audit's ../step_05_csv_tail_checks.csv remains authoritative for source warnings: "
                + f"{source_metadata['csv_tail_checks']['incomplete_file_count']} incomplete files; "
                + f"{source_metadata['csv_tail_checks']['skipped_record_count']} confirmed EOF records skipped. "
                "Review the file source manually. Tail checks do not prove dataset completeness. "
                "Missing recorded events are operational zeros, not proof of clinical absence.\n\n"
                "Unknown identities remain unknown under reviewed handling policies. "
                "The frozen decision and Step 05 metadata are historical records and remain unchanged. "
                "Publication status is recorded by Step 06 metadata and its manifest.\n"
            )
            run.path(audit_dir / "step_06_report.md").write_text(
                report, encoding="utf-8"
            )
    print(
        f"Step 06 complete: {counts['eligible_studies']} formal studies; H={horizon}. Image QC is next."
    )
    return 0
