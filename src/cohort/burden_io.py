"""Bounded raw CSV ingestion and typed audit output, without DICOM decoding."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.common import (
    _check_csv_tail,
    _OutputRun,
    _progress,
    _under_root,
    _validate_csv_rejects,
)

COMPONENTS = (
    "new_icu_ccu_transfer",
    "new_stepdown_transfer",
    "new_invasive_ventilation",
    "new_severe_hypoxemia",
    "new_iv_vaso_inotrope_infusion",
    "in_hospital_death",
)
TABLE_COLUMNS = {
    "patients": "subject_id gender anchor_age anchor_year anchor_year_group dod",
    "admissions": "subject_id hadm_id admittime dischtime deathtime "
    "hospital_expire_flag discharge_location",
    "transfers": "subject_id hadm_id transfer_id eventtype careunit intime outtime",
    "icustays": "subject_id hadm_id stay_id intime outtime",
    "d_items": "itemid label linksto",
    "procedureevents": "subject_id hadm_id stay_id itemid starttime endtime storetime "
    "orderid linkorderid statusdescription continueinnextdept",
    "chartevents": "subject_id hadm_id stay_id itemid charttime storetime value "
    "valuenum valueuom warning",
    "emar": "subject_id hadm_id emar_id emar_seq pharmacy_id charttime medication "
    "event_txt scheduletime storetime",
    "emar_detail": "subject_id emar_id emar_seq parent_field_ordinal "
    "administration_type complete_dose_not_given dose_given route "
    "prior_infusion_rate infusion_rate infusion_complete new_iv_bag_hung "
    "continued_infusion_in_other_location",
    "pharmacy": "subject_id hadm_id pharmacy_id poe_id medication starttime stoptime "
    "route proc_type status",
    "prescriptions": "subject_id hadm_id pharmacy_id poe_id poe_seq starttime stoptime "
    "drug_type drug route",
}


def clinical_time(value: Any) -> datetime | None:
    """Reject date-only and timezone-aware values; never silently strip offsets."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and ("T" in value or " " in value):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed.tzinfo is None else None


def record_key(table: str, row: dict) -> str:
    body = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
    return f"{table}:{hashlib.sha256(body.encode()).hexdigest()}"


def confirmed(mapping: dict, value: str) -> dict | None:
    entry = mapping.get(value)
    if entry and entry.get("confirmed") is True and entry.get("reviewer"):
        return entry
    return None


def score(flags: list[int | None]) -> dict:
    if len(flags) != 6:
        raise ValueError("Expected six component flags")
    if any(v is not None and (type(v) is not int or v not in (0, 1)) for v in flags):
        raise ValueError("Expected six binary component flags")
    if any(flag is None for flag in flags):
        return dict(
            T=None,
            R=None,
            C=None,
            D=None,
            burden_score=None,
            phys_death=None,
            high_burden=None,
        )
    transfer, respiratory = max(flags[:2]), max(flags[2:4])
    burden = transfer + 2 * respiratory + 2 * flags[4] + 3 * flags[5]
    return dict(
        T=transfer,
        R=respiratory,
        C=flags[4],
        D=flags[5],
        burden_score=float(burden),
        phys_death=int(bool(respiratory or flags[4] or flags[5])),
        high_burden=int(burden >= 3),
    )


def identifier(value: Any) -> int | None:
    return int(value) if str(value).isdigit() else None


def open_store(directory: Path, memory_limit: str) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(str(directory / "events.duckdb"))
    connection.execute("SET memory_limit = ?", [memory_limit])
    connection.execute("SET temp_directory = ?", [str(directory / "spill")])
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET threads = 2")
    return connection


def ingest(
    connection: Any,
    iv_root: Path,
    settings: dict,
    matches: list[dict],
    run: _OutputRun | None = None,
) -> dict:
    """Project CSVs into disk tables; keep complete candidate-admission history."""
    connection.register(
        "candidate_keys_arrow",
        pa.Table.from_pylist(
            [
                {"subject_id": row["subject_id"], "hadm_id": row["hadm_id"]}
                for row in matches
            ],
            schema=pa.schema([("subject_id", pa.int64()), ("hadm_id", pa.int64())]),
        ),
    )
    connection.execute(
        "CREATE TABLE candidate_keys AS SELECT DISTINCT * FROM candidate_keys_arrow"
    )
    paths = {}
    for name, columns in TABLE_COLUMNS.items():
        path = _under_root(iv_root, settings["tables"][name])
        if not path.is_file():
            raise FileNotFoundError(f"Missing V2 source: {path}")
        check = run.check_csv(path) if run else _check_csv_tail(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        required = columns.split()
        if len(header) != len(set(header)) or set(required) - set(header):
            raise ValueError(f"Missing or duplicate V2 columns in {name}")
        paths[name] = path
        if name == "d_items":
            condition = "true"
        elif name == "emar_detail":
            condition = "EXISTS (SELECT 1 FROM emar e WHERE e.emar_id = r.emar_id)"
        elif name in {"pharmacy", "prescriptions"}:
            condition = (
                "EXISTS (SELECT 1 FROM emar e WHERE coalesce(trim(e.medication),'')='' "
            )
            condition += "AND trim(e.subject_id)=trim(r.subject_id) "
            condition += "AND trim(e.hadm_id)=trim(r.hadm_id) "
            condition += "AND trim(e.pharmacy_id)=trim(r.pharmacy_id) "
            condition += "AND coalesce(trim(e.subject_id),'')<>'' "
            condition += "AND coalesce(trim(e.hadm_id),'')<>'' "
            condition += "AND coalesce(trim(e.pharmacy_id),'')<>'')"
            if name == "prescriptions":
                condition += (
                    " AND NOT EXISTS (SELECT 1 FROM pharmacy p WHERE "
                    "trim(p.subject_id)=trim(r.subject_id) AND "
                    "trim(p.hadm_id)=trim(r.hadm_id) AND "
                    "trim(p.pharmacy_id)=trim(r.pharmacy_id) "
                    "HAVING count(*)=1 AND count(nullif(trim(p.medication),''))=1)"
                )
        elif name in ("patients", "admissions"):
            condition = "EXISTS (SELECT 1 FROM candidate_keys k WHERE "
            condition += "k.subject_id = try_cast(r.subject_id AS BIGINT))"
        else:
            condition = "EXISTS (SELECT 1 FROM candidate_keys k WHERE "
            condition += "k.subject_id = try_cast(r.subject_id AS BIGINT) AND "
            condition += "(k.hadm_id = try_cast(r.hadm_id AS BIGINT) OR "
            condition += "coalesce(r.hadm_id, '') = ''))"
            if name == "chartevents":
                condition += " AND (r.itemid = '220277' OR r.itemid IN "
                condition += "(SELECT itemid FROM d_items WHERE linksto='chartevents' "
                condition += "AND regexp_matches(lower(label), "
                condition += "'airway|intubation|ventilat')) )"
        projection = ", ".join(f'r."{col}"' for col in required)
        with _progress(f"Read V2 {name}", unit="table", total=1) as progress:
            try:
                connection.execute(
                    f"CREATE TABLE {name} AS SELECT {projection} "
                    f"FROM read_csv(?, columns=?, auto_detect=false, delim=',', "
                    f"header=true, quote='\"', escape='\"', "
                    f"null_padding=false, strict_mode=true, ignore_errors=true, "
                    f"store_rejects=true, rejects_limit=0, "
                    f"rejects_table='rejects_{name}', rejects_scan='scans_{name}') "
                    f"r WHERE {condition}",
                    [str(path), dict.fromkeys(header, "VARCHAR")],
                )
            except duckdb.InvalidInputException as exc:
                raise ValueError(
                    f"Cannot read V2 source CSV: {path}. Only an incomplete final "
                    f"record may be skipped. DuckDB diagnostic: {exc}"
                ) from exc
            rejects = connection.execute(
                f"SELECT line_byte_position, csv_line FROM rejects_{name}"
            ).fetchall()
            _validate_csv_rejects(path, check, rejects)
            connection.execute(f"DROP TABLE rejects_{name}")
            connection.execute(f"DROP TABLE scans_{name}")
            progress.update(1)
    return paths


def rows(
    connection: Any, name: str, subject: int | None = None, admission: int | None = None
) -> list[dict]:
    query, params = f"SELECT * FROM {name}", []
    if subject is not None:
        query += " WHERE try_cast(subject_id AS BIGINT)=?"
        params.append(subject)
        if admission is not None:
            query += " AND (try_cast(hadm_id AS BIGINT)=? OR coalesce(hadm_id,'')='')"
            params.append(admission)
    result = connection.execute(query, params)
    columns = [col[0] for col in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def frequencies(connection: Any, table: str, fields: list[str]) -> list[dict]:
    results = []
    for field in fields:
        query = f'SELECT coalesce("{field}", \'\') AS "value", count(*) n FROM {table}'
        query += f' GROUP BY "{field}" ORDER BY n DESC, "value"'
        for value, count in connection.execute(query).fetchall():
            results.append(
                dict(source_table=table, field=field, value=value, record_count=count)
            )
    return results


ID_FIELDS = {"subject_id", "hadm_id", "study_id", "horizon_hours", "age_at_t0"}
TIME_FIELDS = {
    "t0",
    "admittime",
    "dischtime",
    "deathtime",
    "observation_end",
    "horizon_end",
    "first_event_time",
    "event_time",
    "episode_start",
    "episode_end",
    "first_early_event_time",
    "first_hypoxemia_time",
}


def schema_for(columns: list[str]) -> pa.Schema:
    fields = []
    for column in columns:
        if column in TIME_FIELDS or column.endswith("_first_event_time"):
            dtype = pa.timestamp("us")
        elif (
            column in COMPONENTS
            or column.startswith("early_3h_new_")
            or column
            in {
                "early_3h_in_hospital_death",
                "early_discharge",
                "prior_icu_before_t0",
                "T",
                "R",
                "C",
                "D",
                "phys_death",
                "high_burden",
                "outtime_imputed",
                "spo2_source_available",
                "emar_source_available",
                "procedure_source_available",
                "qualifies",
                "automated_qualifies",
                "event_flag",
            }
        ):
            dtype = pa.uint8()
        elif (
            column in ID_FIELDS
            or column.endswith("_count")
            or column == ("spo2_measurements_in_window")
        ):
            dtype = pa.int64()
        elif column in {
            "burden_score",
            "early_3h_score",
            "observed_hours",
            "minutes_from_first_icu_transfer_to_hypoxemia",
        }:
            dtype = pa.float32()
        else:
            dtype = pa.string()
        fields.append(pa.field(column, dtype))
    return pa.schema(fields)


class ParquetSink:
    """Append admission-sized batches, including an explicitly typed empty table."""

    def __init__(self, path: Path, columns: list[str]) -> None:
        self.schema = schema_for(columns)
        self.writer = pq.ParquetWriter(path, self.schema, compression="zstd")
        self.count = 0

    def write(self, values: list[dict]) -> None:
        if values:
            self.writer.write_table(pa.Table.from_pylist(values, schema=self.schema))
            self.count += len(values)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
