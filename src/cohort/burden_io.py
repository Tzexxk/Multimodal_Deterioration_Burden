"""Bounded raw CSV ingestion and typed audit output, without DICOM decoding."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.common import (
    _check_csv_tail,
    _file_version,
    _OutputRun,
    _progress,
    _resolve_config_path,
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
SUPPLEMENTARY_COLUMNS = {
    "icustays": "first_careunit last_careunit",
    "emar_detail": "pharmacy_id product_code product_description "
    "product_description_other infusion_rate_unit dose_given_unit",
    "prescriptions": "formulary_drug_cd gsn ndc prod_strength form_rx "
    "dose_val_rx dose_unit_rx form_val_disp form_unit_disp",
}

# These sources supply review evidence only, never automatic positive labels.
REVIEW_SOURCES = {
    "inputevents_full": (
        "mimic_iv",
        "subject_id hadm_id itemid starttime endtime amount rate orderid linkorderid statusdescription",
    ),
    "ingredientevents_full": (
        "mimic_iv",
        "subject_id hadm_id itemid starttime endtime orderid linkorderid",
    ),
    "poe_full": ("mimic_iv", "subject_id hadm_id poe_id ordertime"),
    "poe_detail_full": ("mimic_iv", "subject_id poe_id field_name field_value"),
    "pharmacy_full": ("mimic_iv", "subject_id hadm_id pharmacy_id poe_id medication"),
    "prescriptions_full": (
        "mimic_iv",
        "subject_id hadm_id pharmacy_id poe_id drug drug_type route",
    ),
    "emar_full": (
        "mimic_iv",
        "subject_id hadm_id emar_id poe_id pharmacy_id charttime",
    ),
    "emar_detail_full": (
        "mimic_iv",
        "subject_id emar_id administration_type infusion_rate_adjustment",
    ),
    "discharge_notes_full": ("mimic_note", "subject_id hadm_id note_id charttime text"),
}
REVIEW_TARGET_ITEMS = {
    "221289": "Epinephrine",
    "221653": "Dobutamine",
    "221662": "Dopamine",
    "221749": "Phenylephrine",
    "221906": "Norepinephrine",
    "222315": "Vasopressin",
    "229617": "Epinephrine.",
    "229630": "Phenylephrine (50/250)",
    "229631": "Phenylephrine (200/250)_OLD_1",
    "229632": "Phenylephrine (200/250)",
}


def review_source_paths(root: Path, paths: dict, settings: dict) -> dict:
    sources = settings.get("review_evidence_sources", {})
    if not isinstance(sources, dict) or set(sources) - set(REVIEW_SOURCES):
        raise ValueError("Unsupported review evidence source")
    for child, parent in (
        ("poe_detail_full", "poe_full"),
        ("emar_detail_full", "emar_full"),
    ):
        if child in sources and parent not in sources:
            raise ValueError("Review detail requires its ownership parent")
    return {
        name: _under_root(
            _resolve_config_path(paths[REVIEW_SOURCES[name][0]], root, raw=True), value
        )
        for name, value in sources.items()
    }


def ingest_review_sources(connection: Any, sources: dict, run: _OutputRun) -> dict:
    """Read full source rows with strict CSV validation and unique parent ownership."""
    bindings = {}
    # Parents precede detail tables even when config ordering differs.
    order = [n for n in sources if n not in {"poe_detail_full", "emar_detail_full"}]
    order += [n for n in sources if n not in order]
    for name in order:
        path = sources[name]
        print(f"Read review source {name}: {path}", flush=True)
        before = _file_version(path)
        if not before["exists"]:
            raise FileNotFoundError(path)
        check = run.check_csv(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if (
            len(header) != len(set(header))
            or "review_owner_hadm_id" in header
            or set(REVIEW_SOURCES[name][1].split()) - set(header)
        ):
            raise ValueError(f"Missing or duplicate review columns: {name}")
        join = ""
        if name in {"poe_detail_full", "emar_detail_full"}:
            parent, key = (
                ("poe_full", "poe_id")
                if name == "poe_detail_full"
                else ("emar_full", "emar_id")
            )
            # Aggregate the retained parent once, before scanning the full detail CSV.
            # A correlated aggregate here can materialize keys from the entire CSV.
            owners = f"review_owners_{name}"
            connection.execute(
                f"CREATE OR REPLACE TEMP TABLE {owners} AS SELECT subject_id,{key}, "
                "CASE WHEN count(DISTINCT hadm_id)=1 AND "
                "count(*)=count(nullif(hadm_id,'')) THEN min(hadm_id) "
                f"END AS review_owner_hadm_id FROM {parent} GROUP BY subject_id,{key}"
            )
            owner = "o.review_owner_hadm_id"
            join = (
                f" INNER JOIN {owners} o ON o.subject_id=r.subject_id "
                f"AND o.{key}=r.{key}"
            )
            condition = "true"
        elif name in {"poe_full", "emar_full"}:
            # Conflicting parents outside a selected admission still affect ownership.
            owner = "r.hadm_id"
            condition = "EXISTS(SELECT 1 FROM candidate_keys k WHERE k.subject_id=try_cast(r.subject_id AS BIGINT))"
        else:
            owner = "r.hadm_id"
            condition = "EXISTS(SELECT 1 FROM candidate_keys k WHERE k.subject_id=try_cast(r.subject_id AS BIGINT) AND k.hadm_id=try_cast(r.hadm_id AS BIGINT))"
        if name == "inputevents_full":
            condition += (
                " AND r.itemid IN ("
                + ",".join("'" + k + "'" for k in REVIEW_TARGET_ITEMS)
                + ")"
            )
        disabled = (
            connection.execute(
                "SELECT current_setting('disabled_optimizers')"
            ).fetchone()[0]
            if join
            else None
        )
        try:
            if disabled is not None:
                # CSV cardinality estimates can make DuckDB build on the full file.
                # Keep the retained ownership map on the build side for this query only.
                guarded = set(disabled.split(",")) | {
                    "join_order",
                    "build_side_probe_side",
                }
                connection.execute(
                    "SET disabled_optimizers = ?", [",".join(sorted(guarded - {""}))]
                )
            connection.execute(
                f"CREATE TABLE {name} AS SELECT r.*, {owner} AS review_owner_hadm_id "
                "FROM read_csv(?,columns=?,auto_detect=false,delim=',',header=true,"
                "quote='\"',escape='\"',null_padding=false,strict_mode=true,ignore_errors=true,"
                f"store_rejects=true,rejects_limit=0,rejects_table='reject_{name}',rejects_scan='scan_{name}') r{join} WHERE {condition}",
                [str(path), dict.fromkeys(header, "VARCHAR")],
            )
        except duckdb.OutOfMemoryException as exc:
            raise duckdb.OutOfMemoryException(
                f"Review source {name} ({path}): {exc}"
            ) from exc
        finally:
            if disabled is not None:
                connection.execute("SET disabled_optimizers = ?", [disabled])
        rejects = connection.execute(
            f"SELECT DISTINCT line_byte_position,csv_line FROM reject_{name}"
        ).fetchall()
        _validate_csv_rejects(path, check, rejects)
        if name == "inputevents_full":
            for item, label, target in connection.execute(
                "SELECT DISTINCT r.itemid,d.label,d.linksto FROM inputevents_full r "
                "LEFT JOIN d_items d ON r.itemid=d.itemid"
            ).fetchall():
                if label != REVIEW_TARGET_ITEMS[item] or target != "inputevents":
                    raise ValueError("Review target item identity changed")
        if before != _file_version(path):
            raise ValueError("Review source changed during ingestion")
        bindings[name] = {"path": str(path), "version": before, "columns": header}
        print(f"Review source {name}: complete", flush=True)
    return bindings


def verify_review_sources(
    root: Path, paths: dict, settings: dict, bindings: dict
) -> None:
    expected = review_source_paths(root, paths, settings)
    if set(expected) != set(bindings) or any(
        Path(bindings[n]["path"]) != p or _file_version(p) != bindings[n]["version"]
        for n, p in expected.items()
    ):
        raise ValueError("Frozen review evidence sources changed")


def review_records(
    connection: Any, sources: dict, subject: int, admission: int
) -> list[dict]:
    result = []
    for name in sources:
        query = connection.execute(
            f"SELECT * FROM {name} WHERE try_cast(subject_id AS BIGINT)=? AND "
            "(try_cast(review_owner_hadm_id AS BIGINT)=? OR review_owner_hadm_id IS NULL)",
            [subject, admission],
        )
        for values in query.fetchall():
            record = dict(zip([c[0] for c in query.description], values, strict=True))
            owner = record.pop("review_owner_hadm_id")
            payload = {
                "subject_id": record["subject_id"],
                "hadm_id": owner,
                "record": record,
            }
            result.append({**payload, "source_record_key": record_key(name, payload)})
    return result


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
    known = [flag for flag in flags if flag is not None]
    transfer, respiratory = max(known[:2]), max(known[2:4])
    burden = transfer + 2 * respiratory + 2 * known[4] + 3 * known[5]
    return dict(
        T=transfer,
        R=respiratory,
        C=known[4],
        D=known[5],
        burden_score=float(burden),
        phys_death=int(bool(respiratory or known[4] or known[5])),
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
    supplementary: bool = False,
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
        if supplementary:
            required += SUPPLEMENTARY_COLUMNS.get(name, "").split()
        if len(header) != len(set(header)) or set(required) - set(header):
            raise ValueError(f"Missing or duplicate V2 columns in {name}")
        paths[name] = path
        if name == "d_items":
            condition = "true"
        elif name == "emar_detail":
            condition = "EXISTS (SELECT 1 FROM emar e WHERE e.emar_id = r.emar_id)"
        elif name in {"pharmacy", "prescriptions"} and supplementary:
            condition = (
                "EXISTS (SELECT 1 FROM emar e WHERE "
                "trim(e.subject_id)=trim(r.subject_id) AND "
                "trim(e.hadm_id)=trim(r.hadm_id) AND "
                "trim(e.pharmacy_id)=trim(r.pharmacy_id) AND "
                "coalesce(trim(e.pharmacy_id),'')<>'')"
            )
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
                condition += (
                    "'airway|intubation|ventilat|(^|[^a-z])ett([^a-z]|$)|"
                    "o2 delivery device|extubat')) )"
                )
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


def schema_for(columns: Sequence[str]) -> pa.Schema:
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
                "blocks_event",
                "possible_earlier_event",
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

    def __init__(self, path: Path, columns: Sequence[str]) -> None:
        self.schema = schema_for(columns)
        self.writer = pq.ParquetWriter(path, self.schema, compression="zstd")
        self.count = 0

    def write(self, values: list[dict]) -> None:
        if values:
            if self.writer is None:
                raise ValueError("Cannot write to a closed Parquet sink")
            self.writer.write_table(pa.Table.from_pylist(values, schema=self.schema))
            self.count += len(values)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
