"""Disk-backed laboratory inventory; no clinical mapping or value cleaning."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.common import _csv_reader, _file_hash, _progress, _validate_csv_rejects, _write_csv

LAB_REQUIRED: Sequence[str] = (
    "labevent_id subject_id hadm_id specimen_id itemid charttime storetime value valuenum valueuom".split()
)
LAB_OPTIONAL: Sequence[str] = (
    "ref_range_lower ref_range_upper flag priority comments".split()
)
GROUPS = (
    ("blood_count", r"white blood|\bwbc\b|hemoglobin|haemoglobin|platelet"),
    ("electrolytes_metabolism", r"sodium|potassium|chloride|bicarbonate|glucose"),
    ("renal", r"creatinine|urea nitrogen|\bbun\b"),
    ("liver_nutrition", r"bilirubin.*total|total.*bilirubin|albumin"),
    ("inr", r"\binr\b|international normalized ratio"),
    (
        "blood_gas_lactate",
        r"^ph$|\bpo2\b|\bpco2\b|oxygen.*pressure|carbon dioxide.*pressure|lactate|lactic acid",
    ),
)


def json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def review_key(itemid: str, source_unit: str) -> str:
    return "lab_" + json_hash([itemid, source_unit])


def suggested_group(label: str) -> str:
    matches = [name for name, pattern in GROUPS if re.search(pattern, label, re.I)]
    return (
        matches[0]
        if len(matches) == 1
        else "grouping_review"
        if matches
        else "out_of_scope"
    )


def query_rows(connection, sql: str, parameters=None) -> list[dict]:
    result = connection.execute(sql, parameters or [])
    names = [c[0] for c in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def read_small_csv(
    path: Path, required: Sequence[str], run, *, exact: bool = False
) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = _csv_reader(stream, run.check_csv(path))
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or set(required) - set(header):
            raise ValueError(f"Missing or duplicate required columns: {path}")
        if exact and set(header) != set(required):
            raise ValueError(f"Unexpected CSV columns: {path}")
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError(f"Extra CSV fields: {path}")
    return rows


def scan_csv(
    connection,
    path: Path,
    name: str,
    required: Sequence[str],
    optional: Sequence[str],
    condition: str,
    run,
) -> None:
    """Parse explicit string columns and verify every parser reject before use."""
    check = run.check_csv(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)) or set(required) - set(header):
        raise ValueError(f"Missing or duplicate required columns: {path}")
    projection = ", ".join(
        f'r."{k}"' if k in header else f"NULL::VARCHAR AS {k}"
        for k in [*required, *optional]
    )
    if name in ("labs_raw", "vitals_raw"):
        projection += ", sha256(to_json(r)) AS raw_record_sha256"
    with _progress(f"Read Step 10 {name}", unit="table", total=1) as progress:
        connection.execute(
            f"CREATE TABLE {name} AS SELECT {projection} FROM read_csv(?, columns=?, "
            "auto_detect=false, header=true, delim=',', quote='\"', escape='\"', "
            "strict_mode=true, null_padding=false, ignore_errors=true, store_rejects=true, "
            f"rejects_limit=0, rejects_table='rejects_{name}', rejects_scan='scans_{name}') r WHERE {condition}",
            [str(path), dict.fromkeys(header, "VARCHAR")],
        )
        rejects = connection.execute(
            f"SELECT line_byte_position,csv_line FROM rejects_{name}"
        ).fetchall()
        _validate_csv_rejects(path, check, rejects)
        connection.execute(f"DROP TABLE rejects_{name}")
        connection.execute(f"DROP TABLE scans_{name}")
        progress.update(1)


def timestamp_sql(field: str) -> str:
    # DuckDB's general timestamp cast also accepts dates/offsets; forbid those here.
    return (
        f"CASE WHEN regexp_full_match({field}, "
        "'\\d{4}-\\d{2}-\\d{2}[ T]\\d{2}:\\d{2}:\\d{2}(\\.\\d{1,6})?') "
        f"THEN try_cast({field} AS TIMESTAMP) END"
    )


def prepare_labs(connection, settings: dict) -> None:
    fields = [*LAB_REQUIRED, *LAB_OPTIONAL]
    projection = ",".join(
        f"substring(coalesce(comments,''),1,{settings['comment_max_chars']}) AS comments"
        if k == "comments"
        else f"coalesce({k},'') AS {k}"
        for k in fields
    )
    connection.execute(
        "CREATE TABLE labs AS SELECT *, count(*)::BIGINT AS raw_row_count FROM (SELECT "
        f"raw_record_sha256 AS lab_record_key,{projection},"
        f"length(coalesce(comments,''))>{settings['comment_max_chars']} AS comments_truncated "
        "FROM labs_raw) GROUP BY ALL"
    )
    connection.execute("DROP TABLE labs_raw")


def build_views(connection, cohort: pa.Table) -> None:
    connection.register(
        "study_keys", cohort.select(["sample_id", "subject_id", "hadm_id", "t0"])
    )
    connection.execute(
        "CREATE VIEW admission_times AS SELECT try_cast(subject_id AS BIGINT) AS patient_id,"
        "try_cast(hadm_id AS BIGINT) AS admission_id,"
        f"{timestamp_sql('admittime')} AS admitted,{timestamp_sql('dischtime')} AS discharged FROM admissions"
    )
    invalid = connection.execute(
        "SELECT count(*) FROM admission_times WHERE patient_id IS NULL OR admission_id IS NULL "
        "OR admitted IS NULL OR discharged IS NULL OR discharged<admitted"
    ).fetchone()[0]
    conflicts = connection.execute(
        "SELECT count(*) FROM (SELECT admission_id FROM admission_times GROUP BY admission_id "
        "HAVING count(DISTINCT struct_pack(p:=patient_id,a:=admitted,d:=discharged))<>1)"
    ).fetchone()[0]
    if invalid or conflicts:
        raise ValueError("Invalid or conflicting admission identity/time intervals")
    connection.execute(
        "CREATE VIEW parsed_labs AS SELECT *, try_cast(subject_id AS BIGINT) AS patient_id,"
        "try_cast(hadm_id AS BIGINT) AS provided_hadm_id,"
        f"{timestamp_sql('charttime')} AS chart_at,{timestamp_sql('storetime')} AS stored_at,"
        "CASE WHEN isfinite(try_cast(valuenum AS DOUBLE)) THEN try_cast(valuenum AS DOUBLE) END AS numeric_value,"
        "CASE WHEN isfinite(try_cast(value AS DOUBLE)) THEN try_cast(value AS DOUBLE) END AS text_numeric "
        "FROM labs"
    )
    connection.execute(
        "CREATE TABLE assigned_labs AS WITH inferred AS (SELECT l.lab_record_key,"
        "count(DISTINCT a.admission_id) AS admission_matches,max(a.admission_id) AS inferred_hadm "
        "FROM parsed_labs l LEFT JOIN admission_times a ON a.patient_id=l.patient_id "
        "AND l.chart_at BETWEEN a.admitted AND a.discharged AND trim(l.hadm_id)='' GROUP BY l.lab_record_key) "
        "SELECT l.*,i.admission_matches,CASE WHEN trim(l.hadm_id)<>'' THEN l.provided_hadm_id "
        "WHEN i.admission_matches=1 THEN i.inferred_hadm END AS assigned_hadm_id,"
        "CASE WHEN trim(l.hadm_id)<>'' THEN 'provided' WHEN i.admission_matches=1 THEN 'unique_patient_charttime' "
        "ELSE 'unresolved' END AS hadm_assignment_source,"
        "EXISTS(SELECT 1 FROM admission_times a WHERE a.patient_id=l.patient_id "
        "AND a.admission_id=l.provided_hadm_id) AS provided_owner_valid "
        "FROM parsed_labs l JOIN inferred i USING(lab_record_key)"
    )
    connection.execute(
        "CREATE TABLE associations AS SELECT l.*,s.sample_id,s.hadm_id AS study_hadm_id,s.t0,"
        "CASE WHEN trim(l.storetime)='' THEN 'missing_storetime' WHEN l.stored_at IS NULL THEN 'invalid_storetime' "
        "WHEN l.stored_at<l.chart_at THEN 'storetime_before_charttime' WHEN l.stored_at>s.t0 THEN 'future_storetime' "
        "ELSE 'available' END AS availability_reason,"
        "CASE WHEN trim(l.hadm_id)<>'' AND l.provided_hadm_id IS NULL THEN 'invalid_hadm_id' "
        "WHEN trim(l.hadm_id)<>'' AND NOT l.provided_owner_valid THEN 'provided_owner_conflict' "
        "WHEN trim(l.hadm_id)='' AND l.admission_matches=0 THEN 'no_admission_match' "
        "WHEN trim(l.hadm_id)='' AND l.admission_matches>1 THEN 'ambiguous_admission' "
        "WHEN l.assigned_hadm_id<>s.hadm_id THEN 'different_admission' ELSE 'owned' END AS ownership_reason "
        "FROM assigned_labs l JOIN study_keys s ON l.patient_id=s.subject_id "
        "AND l.chart_at BETWEEN s.t0-INTERVAL '24 hours' AND s.t0"
    )
    connection.execute(
        "CREATE VIEW window_labs AS SELECT l.* FROM assigned_labs l "
        "WHERE EXISTS(SELECT 1 FROM associations a WHERE a.lab_record_key=l.lab_record_key)"
    )


def inventory(
    connection, dictionary: list[dict], settings: dict
) -> tuple[list[dict], list[dict], list[dict]]:
    by_id = {r["itemid"]: r for r in dictionary}
    if len(by_id) != len(dictionary) or any(
        not re.fullmatch(r"[0-9]+", k) for k in by_id
    ):
        raise ValueError("Duplicate or invalid laboratory dictionary itemid")
    rows = query_rows(
        connection,
        "SELECT itemid,valueuom AS source_unit,count(*) AS source_records,sum(raw_row_count) AS raw_rows,"
        "count(DISTINCT NULLIF(trim(specimen_id),'')) AS specimens,"
        "count(*) FILTER(WHERE NOT regexp_full_match(specimen_id,'[1-9][0-9]*')) AS missing_specimen_records,"
        "count(*) FILTER(WHERE numeric_value IS NULL) AS numeric_failures,"
        "count(*) FILTER(WHERE regexp_matches(value,'[<>≤≥]')) AS comparator_records,"
        "count(*) FILTER(WHERE text_numeric IS NOT NULL AND numeric_value IS NOT NULL AND text_numeric<>numeric_value) AS numeric_text_disagreements,"
        "min(numeric_value) AS raw_min,max(numeric_value) AS raw_max,avg(numeric_value) AS raw_mean,"
        "quantile_cont(numeric_value,0.5) AS raw_median FROM window_labs GROUP BY itemid,valueuom ORDER BY try_cast(itemid AS BIGINT),itemid,valueuom",
    )
    associations = {
        (r["itemid"], r["source_unit"]): r
        for r in query_rows(
            connection,
            "SELECT itemid,valueuom AS source_unit,count(*) AS study_window_associations,"
            "count(*) FILTER(WHERE availability_reason='available' AND ownership_reason='owned') AS eligible_associations "
            "FROM associations GROUP BY itemid,valueuom",
        )
    }
    batches, outside = [], []
    item_groups = {}
    for row in rows:
        definition = by_id.get(row["itemid"], {})
        row.update({k: definition.get(k, "") for k in ("label", "fluid", "category")})
        row["dictionary_item_present"] = bool(definition)
        row["suggested_group"] = (
            suggested_group(row["label"]) if definition else "grouping_review"
        )
        row["review_key"] = review_key(row["itemid"], row["source_unit"])
        row.update(
            {
                k: v
                for k, v in associations[(row["itemid"], row["source_unit"])].items()
                if k not in ("itemid", "source_unit")
            }
        )
        item_groups[row["itemid"]] = row["suggested_group"]
        if row["suggested_group"] == "out_of_scope":
            outside.append(row.copy())
    connection.register(
        "item_definitions",
        pa.Table.from_pylist(
            dictionary,
            schema=pa.schema(
                [(k, pa.string()) for k in ("itemid", "label", "fluid", "category")]
            ),
        ),
    )
    for group in [g[0] for g in GROUPS] + ["grouping_review", "out_of_scope"]:
        ids = sorted(
            (k for k, v in item_groups.items() if v == group),
            key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else k),
        )
        for start in range(0, len(ids), settings["batch_item_limit"]):
            batch_id = f"{group}_{start // settings['batch_item_limit'] + 1:03d}"
            for itemid in ids[start : start + settings["batch_item_limit"]]:
                batches.append(
                    {"batch_id": batch_id, "suggested_group": group, "itemid": itemid}
                )
    return rows, batches, outside


def write_query_csv(connection, run, target: Path, sql: str, parameters=None) -> None:
    result = connection.execute(sql, parameters or [])
    columns = [c[0] for c in result.description]
    with run.path(target).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        while rows := result.fetchmany(4096):
            writer.writerows(rows)


def copy_parquet(connection, table: str, target: Path) -> None:
    connection.execute(
        f"COPY {table} TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(target)]
    )


def load_cache(connection, paths: dict) -> list[dict]:
    for name in ("labs", "admissions"):
        connection.execute(
            f"CREATE TABLE {name} AS SELECT * FROM read_parquet(?)", [str(paths[name])]
        )
    return pq.read_table(paths["dictionary"]).to_pylist()


def write_reports(
    connection,
    run,
    directory: Path,
    rows: list[dict],
    batches: list[dict],
    outside: list[dict],
    settings: dict,
) -> tuple[dict, list[dict], list[dict]]:
    columns = (
        list(rows[0])
        if rows
        else (
            "itemid source_unit source_records raw_rows specimens missing_specimen_records numeric_failures comparator_records numeric_text_disagreements raw_min raw_max raw_mean raw_median label fluid category dictionary_item_present suggested_group review_key study_window_associations eligible_associations".split()
        )
    )
    _write_csv(run.path(directory / "lab_dictionary_audit.csv"), columns, rows)
    _write_csv(run.path(directory / "out_of_scope_inventory.csv"), columns, outside)
    _write_csv(
        run.path(directory / "review_batches.csv"),
        ["batch_id", "suggested_group", "itemid"],
        batches,
    )
    queries = {
        "lab_availability_summary.csv": "SELECT availability_reason,ownership_reason,hadm_assignment_source,count(*) AS study_window_associations,count(DISTINCT lab_record_key) AS distinct_source_records FROM associations GROUP BY ALL ORDER BY ALL",
        "lab_time_failures.csv": "SELECT lab_record_key,labevent_id,itemid,subject_id,charttime,storetime,CASE WHEN trim(charttime)='' THEN 'missing_charttime' ELSE 'invalid_charttime' END AS reason FROM assigned_labs WHERE chart_at IS NULL ORDER BY lab_record_key",
        "lab_unit_conflicts.csv": "SELECT itemid,count(DISTINCT valueuom) AS unit_variants,string_agg(DISTINCT valueuom,' | ') AS observed_units FROM window_labs GROUP BY itemid HAVING count(DISTINCT valueuom)>1 OR bool_or(trim(valueuom)='') ORDER BY itemid",
        "lab_duplicate_reports.csv": "SELECT 'exact_duplicate' AS issue,lab_record_key AS group_key,itemid,raw_row_count AS raw_rows,1::BIGINT AS source_records FROM window_labs WHERE raw_row_count>1 UNION ALL SELECT 'conflicting_labevent_id',labevent_id,itemid,sum(raw_row_count),count(*) FROM window_labs WHERE EXISTS(SELECT 1 FROM assigned_labs e WHERE e.labevent_id=window_labs.labevent_id AND e.lab_record_key<>window_labs.lab_record_key) GROUP BY labevent_id,itemid UNION ALL SELECT 'specimen_time_revision',sha256(to_json(struct_pack(p:=patient_id,i:=itemid,s:=specimen_id,c:=chart_at))),itemid,sum(raw_row_count),count(*) FROM window_labs WHERE regexp_full_match(specimen_id,'[1-9][0-9]*') GROUP BY patient_id,itemid,specimen_id,chart_at HAVING count(*)>1 ORDER BY issue,group_key,itemid",
        "lab_specimen_evidence.csv": "SELECT w.itemid,w.valueuom AS source_unit,e.itemid AS evidence_itemid,d.label AS evidence_label,e.value AS evidence_value,count(DISTINCT w.lab_record_key) AS target_records,count(DISTINCT e.lab_record_key) AS evidence_records FROM window_labs w JOIN assigned_labs e ON e.patient_id=w.patient_id AND regexp_full_match(w.specimen_id,'[1-9][0-9]*') AND e.specimen_id=w.specimen_id AND e.lab_record_key<>w.lab_record_key LEFT JOIN item_definitions d ON d.itemid=e.itemid GROUP BY ALL ORDER BY ALL",
        "lab_examples.csv": f"SELECT * EXCLUDE(example_rank) FROM (SELECT lab_record_key,labevent_id,itemid,valueuom AS source_unit,specimen_id,charttime,storetime,value,valuenum,ref_range_lower,ref_range_upper,flag,priority,comments,comments_truncated,row_number() OVER(PARTITION BY itemid,valueuom ORDER BY CASE WHEN numeric_value IS NULL OR regexp_matches(value,'[<>≤≥]') OR text_numeric<>numeric_value THEN 0 ELSE 1 END,numeric_value NULLS FIRST,lab_record_key) AS example_rank FROM window_labs) WHERE example_rank<={settings['example_limit']} ORDER BY itemid,source_unit,example_rank",
        "lab_window_failures.csv": "SELECT sample_id,t0,study_hadm_id,lab_record_key,labevent_id,itemid,charttime,storetime,hadm_id,hadm_assignment_source,availability_reason,ownership_reason FROM associations WHERE availability_reason<>'available' OR ownership_reason<>'owned' ORDER BY sample_id,lab_record_key",
        "lab_specimen_failures.csv": "SELECT lab_record_key,labevent_id,itemid,specimen_id,CASE WHEN trim(specimen_id)='' THEN 'missing_specimen_id' ELSE 'invalid_specimen_id' END AS reason FROM window_labs WHERE NOT regexp_full_match(specimen_id,'[1-9][0-9]*') ORDER BY itemid,lab_record_key",
    }
    for name, sql in queries.items():
        write_query_csv(connection, run, directory / name, sql)
    stats = query_rows(
        connection,
        "SELECT sum(raw_row_count) AS patient_scan_raw_rows,count(*) AS patient_scan_distinct_records,"
        "count(DISTINCT NULLIF(labevent_id,'')) AS patient_scan_labevent_ids,"
        "count(*) FILTER(WHERE chart_at IS NULL) AS patient_scan_invalid_charttime_records FROM assigned_labs",
    )[0]
    stats.update(
        query_rows(
            connection,
            "SELECT count(*) AS window_distinct_records,coalesce(sum(raw_row_count),0) AS window_raw_rows FROM window_labs",
        )[0]
    )
    stats.update(
        query_rows(
            connection,
            "SELECT count(*) AS study_window_associations,count(*) FILTER(WHERE availability_reason='available' AND ownership_reason='owned') AS eligible_associations FROM associations",
        )[0]
    )
    examples = query_rows(connection, queries["lab_examples.csv"])
    specimens = query_rows(
        connection,
        "SELECT * EXCLUDE(example_rank) FROM (SELECT *,row_number() OVER(PARTITION BY itemid,source_unit ORDER BY target_records DESC,evidence_itemid,evidence_value) AS example_rank FROM ("
        + queries["lab_specimen_evidence.csv"]
        + f")) WHERE example_rank<={settings['example_limit']}",
    )
    return stats, examples, specimens


def audit_vitals(connection, run, cohort, sources, cache, directory, settings, fingerprint):
    """Inventory pre-t0 ICU observations; report evidence, never approve features."""
    items = settings["vital_audit"]["items"]
    definitions = read_small_csv(sources["d_items"], ["itemid", "label", "linksto", "unitname"], run)
    for itemid, label in items.items():
        matches = [r for r in definitions if r["itemid"] == itemid]
        if len(matches) != 1 or matches[0]["label"] != label or matches[0]["linksto"] != "chartevents":
            raise ValueError(f"Vital item dictionary identity changed: {itemid}")
    connection.register("vital_items", pa.table({"itemid": list(items)}))
    connection.register("vital_patients", cohort.select(["subject_id"]))
    patient_filter = "EXISTS(SELECT 1 FROM vital_patients p WHERE p.subject_id=try_cast(r.subject_id AS BIGINT))"
    fields = "subject_id hadm_id stay_id charttime storetime itemid value valuenum valueuom warning".split()
    scan_csv(connection, sources["chartevents"], "vitals_raw", fields, [],
             patient_filter + " AND r.itemid IN (SELECT itemid FROM vital_items)", run)
    scan_csv(connection, sources["icustays"], "vital_stays",
             "subject_id hadm_id stay_id intime outtime".split(), [], patient_filter, run)
    connection.execute(
        "CREATE TABLE vital_records AS SELECT raw_record_sha256 AS source_record_key,"
        + ",".join(f"coalesce({f},'') AS {f}" for f in fields)
        + ",count(*) AS raw_row_count FROM vitals_raw GROUP BY ALL")
    connection.execute("DROP TABLE vitals_raw")
    connection.register("vital_studies", cohort.select(
        [k for k in ("sample_id", "subject_id", "hadm_id", "t0", "careunit_at_t0")
         if k in cohort.column_names]))
    connection.execute(
        "CREATE VIEW parsed_vitals AS SELECT *,"
        + timestamp_sql("charttime") + " AS chart_at,"
        + timestamp_sql("storetime") + " AS stored_at,"
        "try_cast(valuenum AS DOUBLE) AS numeric_value FROM vital_records")
    # Keep invalid/unplaced history separately; it is not attached to an invented window.
    write_query_csv(connection, run, directory / "vital_unplaced_records.csv",
                    "SELECT * FROM parsed_vitals WHERE chart_at IS NULL ORDER BY source_record_key")
    connection.execute(
        "CREATE TABLE vital_associations AS SELECT v.*,s.sample_id,s.t0,s.hadm_id AS study_hadm_id,"
        "CASE WHEN v.hadm_id<>cast(s.hadm_id AS VARCHAR) THEN 'different_admission' "
        "WHEN (SELECT count(DISTINCT struct_pack(p:=i.subject_id,h:=i.hadm_id,a:=i.intime,b:=i.outtime)) "
        "FROM vital_stays i WHERE i.stay_id=v.stay_id)<>1 THEN 'conflicting_or_missing_stay' "
        "WHEN NOT EXISTS(SELECT 1 FROM vital_stays i WHERE i.subject_id=v.subject_id "
        "AND i.hadm_id=v.hadm_id AND i.stay_id=v.stay_id "
        "AND v.chart_at BETWEEN " + timestamp_sql("i.intime") + " AND " + timestamp_sql("i.outtime")
        + ") THEN 'unresolved_stay_or_interval' ELSE 'owned' END AS ownership_status,"
        "CASE WHEN trim(v.storetime)='' THEN 'missing_storetime' WHEN v.stored_at IS NULL "
        "THEN 'invalid_storetime' WHEN v.stored_at<v.chart_at THEN 'storetime_before_charttime' "
        "WHEN v.stored_at>s.t0 THEN 'future_storetime' ELSE 'available' END AS availability_status,"
        "CASE WHEN NOT isfinite(v.numeric_value) OR v.numeric_value IS NULL THEN 'no_finite_valuenum' "
        "WHEN regexp_matches(v.value,'[<>≤≥]') THEN 'comparator_value' "
        "WHEN try_cast(v.value AS DOUBLE) IS NOT NULL AND try_cast(v.value AS DOUBLE)<>v.numeric_value "
        "THEN 'text_numeric_disagreement' ELSE 'numeric' END AS numeric_status,"
        "CASE WHEN trim(v.warning)='0' THEN 'unflagged' WHEN trim(v.warning)='' THEN 'unknown' "
        "ELSE 'flagged_or_invalid' END AS warning_status "
        "FROM parsed_vitals v JOIN vital_studies s ON try_cast(v.subject_id AS BIGINT)=s.subject_id "
        "AND v.chart_at BETWEEN s.t0-INTERVAL '24 hours' AND s.t0")
    connection.execute(
        "CREATE VIEW vital_conflict_flags AS SELECT a.*,CASE WHEN EXISTS(SELECT 1 FROM vital_associations b "
        "WHERE b.sample_id=a.sample_id AND b.itemid=a.itemid AND b.chart_at=a.chart_at "
        "AND b.valueuom=a.valueuom AND isfinite(a.numeric_value) AND isfinite(b.numeric_value) "
        "AND b.numeric_value<>a.numeric_value) THEN 'conflicting_values' ELSE 'no_value_conflict' END "
        "AS duplicate_status FROM vital_associations a")
    connection.execute(
        "CREATE VIEW vital_candidates AS SELECT *,ownership_status='owned' AND availability_status='available' "
        "AND numeric_status='numeric' AND warning_status='unflagged' "
        "AND duplicate_status='no_value_conflict' AS screening_candidate "
        "FROM vital_conflict_flags")
    evidence = cache / "vital_evidence.parquet"
    copy_parquet(connection, "vital_candidates", run.path(evidence))
    denominator = cohort.num_rows
    patients = len(set(cohort["subject_id"].to_pylist()))
    observed = query_rows(connection,
        "SELECT itemid,valueuom AS source_unit,count(*) AS study_associations,"
        "count(DISTINCT source_record_key) AS distinct_records,"
        "count(DISTINCT sample_id) AS observed_studies,"
        "count(DISTINCT sample_id) FILTER(WHERE screening_candidate) AS candidate_studies,"
        "count(DISTINCT subject_id) FILTER(WHERE screening_candidate) AS candidate_patients,"
        "min(numeric_value) FILTER(WHERE isfinite(numeric_value)) AS raw_min,"
        "max(numeric_value) FILTER(WHERE isfinite(numeric_value)) AS raw_max "
        "FROM vital_candidates GROUP BY itemid,valueuom ORDER BY itemid,valueuom")
    inventory = []
    for itemid, label in items.items():
        branches = [r for r in observed if r["itemid"] == itemid] or [dict(
            itemid=itemid, source_unit="", study_associations=0, distinct_records=0,
            observed_studies=0, candidate_studies=0, candidate_patients=0, raw_min=None, raw_max=None)]
        inventory.extend({**r, "label": label, "total_studies": denominator,
                          "total_patients": patients, "review_status": "pending"} for r in branches)
    _write_csv(run.path(directory / "vital_inventory.csv"), list(inventory[0]), inventory)
    write_query_csv(connection, run, directory / "vital_status_counts.csv",
        "SELECT itemid,valueuom AS source_unit,ownership_status,availability_status,numeric_status,warning_status,duplicate_status,"
        "count(*) AS study_associations,count(DISTINCT source_record_key) AS distinct_records "
        "FROM vital_candidates GROUP BY ALL ORDER BY ALL")
    write_query_csv(connection, run, directory / "vital_duplicates_conflicts.csv",
        "SELECT sample_id,itemid,chart_at,valueuom AS source_unit,count(*) AS distinct_records,"
        "sum(raw_row_count) AS raw_rows,count(DISTINCT numeric_value) AS distinct_numeric_values "
        "FROM vital_candidates GROUP BY ALL HAVING count(*)>1 OR sum(raw_row_count)>1 ORDER BY ALL")
    write_query_csv(connection, run, directory / "vital_examples.csv",
        "SELECT * EXCLUDE(example_rank) FROM (SELECT *,row_number() OVER(PARTITION BY itemid,valueuom,"
        "ownership_status,availability_status,numeric_status,warning_status ORDER BY sample_id,chart_at,source_record_key) "
        f"AS example_rank FROM vital_candidates) WHERE example_rank<={settings['example_limit']}")
    # Denominators include studies without any measurements; no outcome labels are used.
    from src.cohort import create_patient_split as split
    processed = next(p for p in run.roots if (p / "splits/burden_v2").exists())
    split_dir = processed / "splits/burden_v2"
    table_paths = sorted(split_dir.glob("fold_*/patient_split.parquet"))
    if not table_paths:
        table_paths = [split_dir / split.TABLE_NAME]
    coverage = []
    for path in table_paths:
        table = pq.read_table(path).select(["subject_id", "split"])
        connection.register("vital_split", table)
        coverage.extend(query_rows(connection,
            "SELECT ? AS fold,p.split,i.itemid,count(DISTINCT s.sample_id) AS total_studies,"
            "count(DISTINCT s.subject_id) AS total_patients,"
            "count(DISTINCT v.sample_id) FILTER(WHERE v.screening_candidate) AS candidate_studies,"
            "count(DISTINCT s.subject_id) FILTER(WHERE v.screening_candidate) AS candidate_patients "
            "FROM vital_studies s JOIN vital_split p USING(subject_id) CROSS JOIN vital_items i "
            "LEFT JOIN vital_candidates v ON v.sample_id=s.sample_id AND v.itemid=i.itemid "
            "GROUP BY p.split,i.itemid ORDER BY p.split,i.itemid", [path.parent.name]))
    _write_csv(run.path(directory / "vital_fold_coverage.csv"), list(coverage[0]), coverage)
    if "careunit_at_t0" in cohort.column_names:
        write_query_csv(connection, run, directory / "vital_careunit_coverage.csv",
            "SELECT s.careunit_at_t0,i.itemid,count(DISTINCT s.sample_id) AS total_studies,"
            "count(DISTINCT s.subject_id) AS total_patients,"
            "count(DISTINCT v.sample_id) FILTER(WHERE v.screening_candidate) AS candidate_studies "
            "FROM vital_studies s CROSS JOIN vital_items i LEFT JOIN vital_candidates v "
            "ON v.sample_id=s.sample_id AND v.itemid=i.itemid GROUP BY s.careunit_at_t0,i.itemid ORDER BY ALL")
    review_rows = [{"audit_fingerprint": fingerprint,
                   "review_key": "vital_" + json_hash([r["itemid"], r["source_unit"]]),
                   "itemid": r["itemid"], "label": r["label"], "source_unit": r["source_unit"],
                   "decision": "pending", "canonical_unit": "", "conversion_rule": "",
                   "warning_rule": "", "duplicate_rule": "", "range_rule": "",
                   "rationale": "", "reviewer": "", "reviewed_on": "", "review_version": "",
                   "evidence_path": str(evidence)} for r in inventory]
    _write_csv(run.path(directory / "vital_review_template.csv"), list(review_rows[0]), review_rows)
    links = ["vital_inventory.csv", "vital_fold_coverage.csv", "vital_status_counts.csv",
             "vital_duplicates_conflicts.csv", "vital_examples.csv", "vital_unplaced_records.csv",
             "vital_review_template.csv"]
    run.path(directory / "vital_review.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Vital-sign audit: pending review</h1>'
        '<p>Screening counts are not approved features or clinical validity judgments. '
        'Units and devices remain separate. Missing warning is unknown. No values are imputed, '
        'converted or automatically excluded by range. Missing observations are not normal values.</p>'
        + ''.join(f'<p><a href="{name}">{name}</a></p>' for name in links)
        + '<pre>' + html.escape(json.dumps(inventory, indent=2, default=str)) + '</pre>', encoding="utf-8")
    return {"review_complete": False, "features_approved": False, "lookback_hours": 24,
            "inventory": inventory, "evidence_path": str(evidence),
            "evidence_sha256": _file_hash(run.path(evidence)),
            "interpretation": "screening only; no clinical rules or features approved"}


def write_html(
    path: Path,
    rows: list[dict],
    batches: list[dict],
    evidence: list[dict],
    specimens: list[dict],
    fingerprint: str,
) -> None:
    """Self-contained local evidence page, with escaped source text and no outcomes."""
    body = [
        "<!doctype html><meta charset='utf-8'><title>Laboratory review</title>",
        "<style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:12px}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #aaa;padding:6px;vertical-align:top}pre{white-space:pre-wrap}summary{cursor:pointer}input{padding:8px;width:60%}</style>",
        "<h1>Step 10A laboratory review</h1><p>Navigation suggestions require explicit review. Raw ranges are descriptive; no clinical mapping, conversion, or exclusion is approved.</p>",
        f"<p>Audit fingerprint: <code>{fingerprint}</code></p>",
        "<p>Review decisions are saved in versioned CSV inputs. All units of an item remain in the same batch. Use lab_specimen_evidence.csv, lab_examples.csv, and conflict reports for complete local evidence. Specimen evidence is descriptive; final rules must check availability at each study's t0.</p>",
        "<input id='search' placeholder='Filter by itemid, name, unit, or batch' oninput=\"document.querySelectorAll('details').forEach(e=>e.hidden=!e.textContent.toLowerCase().includes(this.value.toLowerCase()))\">",
        "<p><a href='review_template.csv'>Review template</a> · <a href='scope_ack_template.csv'>Scope acknowledgment</a> · <a href='out_of_scope_inventory.csv'>Scope inventory</a> · <a href='lab_duplicate_reports.csv'>Duplicate and revision groups</a> · <a href='lab_unit_conflicts.csv'>Unit conflicts</a> · <a href='lab_specimen_evidence.csv'>All specimen evidence</a> · <a href='lab_specimen_failures.csv'>Missing/invalid specimens</a> · <a href='lab_window_failures.csv'>Time and ownership failures</a></p>",
    ]
    for batch_id in dict.fromkeys(b["batch_id"] for b in batches):
        ids = {b["itemid"] for b in batches if b["batch_id"] == batch_id}
        selected = [r for r in rows if r["itemid"] in ids]
        body.append(
            f"<details><summary>{html.escape(batch_id)} ({len(ids)} itemids)</summary>"
        )
        body.append(
            f"<p><a href='batch_{batch_id}_review.csv'>Download this batch's review template</a></p>"
        )
        for itemid in sorted(
            ids, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s)
        ):
            item = [r for r in selected if r["itemid"] == itemid]
            body.append(
                f"<h2>{html.escape(itemid)} - {html.escape(item[0]['label'])}</h2><pre>{html.escape(json.dumps(item, indent=2, default=str))}</pre>"
            )
            examples = [r for r in evidence if r["itemid"] == itemid]
            body.append(
                f"<h3>Bounded source examples</h3><pre>{html.escape(json.dumps(examples, indent=2, default=str))}</pre>"
            )
            related = [r for r in specimens if r["itemid"] == itemid]
            body.append(
                f"<h3>Bounded same-specimen evidence (availability not yet approved)</h3><pre>{html.escape(json.dumps(related, indent=2, default=str))}</pre>"
            )
        body.append("</details>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
