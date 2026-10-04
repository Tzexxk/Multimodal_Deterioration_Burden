"""V2 eligibility and clinical intervals, independent of future labels."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from src.cohort.burden_io import clinical_time, confirmed, identifier, record_key

CANDIDATE_COLUMNS = (
    "sample_id subject_id hadm_id study_id dicom_id dicom_path view_position "
    "selection_reason t0 t0_raw admittime dischtime deathtime hospital_expire_flag "
    "discharge_location age_at_t0 anchor_year_group careunit_at_t0 care_level_at_t0 "
    "prior_icu_before_t0 outtime_imputed candidate_status exclusion_reason "
    "review_status candidate_evidence_keys outcome_version "
    "candidate_exclusion_reviewer candidate_exclusion_rationale candidate_exclusion_version"
).split()
OBSERVATION_COLUMNS = (
    CANDIDATE_COLUMNS
    + (
        "horizon_hours horizon_end observation_end observed_hours early_discharge"
    ).split()
)


def deduplicate(table: str, values: list[dict]) -> list[dict]:
    grouped = {}
    for value in values:
        key = record_key(table, value)
        if key not in grouped:
            grouped[key] = {**value, "source_record_key": key, "duplicate_count": 0}
        grouped[key]["duplicate_count"] += 1
    return sorted(grouped.values(), key=lambda row: row["source_record_key"])


def ownership(
    row: dict, admission: dict, admissions: list[dict], event_time: datetime | None
) -> str:
    """Assign missing hadm_id only with one containing admission, never by patient alone."""
    if identifier(row.get("subject_id")) != identifier(admission["subject_id"]):
        return "subject_mismatch"
    if row.get("hadm_id") not in (None, ""):
        if identifier(row["hadm_id"]) != identifier(admission["hadm_id"]):
            return "other_admission"
        start, end = (
            clinical_time(admission["admittime"]),
            clinical_time(admission["dischtime"]),
        )
        if event_time is not None and (
            start is None or end is None or not start <= event_time <= end
        ):
            return "event_outside_admission"
        return ""
    if event_time is None:
        return "missing_hadm_unresolved_time"
    containing = []
    for other in admissions:
        if identifier(other["subject_id"]) != identifier(row.get("subject_id")):
            continue
        start, end = (
            clinical_time(other["admittime"]),
            clinical_time(other["dischtime"]),
        )
        if start is not None and end is not None and start <= event_time <= end:
            containing.append(identifier(other["hadm_id"]))
    if containing == [identifier(admission["hadm_id"])]:
        return ""
    return "other_admission" if not containing else "ambiguous_missing_hadm"


def prepare_intervals(
    table: str, values: list[dict], admission: dict, admissions: list[dict], rules: dict
) -> list[dict]:
    result = deduplicate(table, values)
    result.sort(
        key=lambda row: (
            clinical_time(row.get("intime")) or datetime.min,
            clinical_time(row.get("outtime")) or datetime.max,
            row.get("transfer_id") or row.get("stay_id") or "",
            row["source_record_key"],
        )
    )
    discharged = clinical_time(admission["dischtime"])
    admitted = clinical_time(admission["admittime"])
    for index, row in enumerate(result):
        row["start"], row["end"] = (
            clinical_time(row.get("intime")),
            clinical_time(row.get("outtime")),
        )
        # Explicit admission IDs identify intervals even when a boundary crosses ADT.
        row["reason"] = ownership(
            row,
            admission,
            admissions,
            None if row.get("hadm_id") not in (None, "") else row["start"],
        )
        if table == "transfers" and not row["reason"]:
            action = confirmed(
                rules["transfer_event_actions"], row.get("eventtype") or ""
            )
            if action is None:
                row["reason"] = "unconfirmed_transfer_event_type"
            elif action["action"] == "exclude":
                row["reason"] = "excluded_nonlocation_record"
        row["outtime_imputed"] = 0
        row["imputation_basis"] = ""
        if (
            not row["reason"]
            and row["end"] is None
            and not row.get("outtime")
            and table == "transfers"
        ):
            later = [
                other_start
                for other in result[index + 1 :]
                if (other_start := clinical_time(other.get("intime"))) is not None
                and row["start"] is not None
                and other_start > row["start"]
                and not ownership(
                    other,
                    admission,
                    admissions,
                    None
                    if other.get("hadm_id") not in (None, "")
                    else clinical_time(other["intime"]),
                )
                and (
                    action := confirmed(
                        rules["transfer_event_actions"], other.get("eventtype") or ""
                    )
                )
                and action["action"] == "location"
            ]
            row["end"] = min([discharged, *later]) if discharged else None
            row["outtime_imputed"] = 1
            row["imputation_basis"] = "next_intime_or_dischtime"
        if not row["reason"] and (
            row["start"] is None
            or row["end"] is None
            or admitted is None
            or discharged is None
            or discharged <= admitted
            or row["end"] <= row["start"]
        ):
            row["reason"] = "invalid_care_interval"
        row["admission_overlap_start"] = None
        row["admission_overlap_end"] = None
        if not row["reason"]:
            start, end = row["start"], row["end"]
            assert start is not None and end is not None
            assert admitted is not None and discharged is not None
            if end <= admitted or start > discharged:
                row["reason"] = "interval_outside_admission"
            else:
                row["admission_overlap_start"] = max(start, admitted)
                row["admission_overlap_end"] = min(end, discharged)
        row["starts_before_admission"] = int(
            row["start"] is not None
            and admitted is not None
            and row["start"] < admitted
        )
        row["ends_after_discharge"] = int(
            row["end"] is not None
            and discharged is not None
            and row["end"] > discharged
        )
        mapping = confirmed(rules["careunits"], row.get("careunit") or "")
        row["level"] = (
            (mapping["care_level"] if mapping else "unknown")
            if (table == "transfers")
            else "icu_ccu"
        )
    return result


def active_intervals(values: list[dict], time: datetime) -> list[dict]:
    return [
        row for row in values if not row["reason"] and row["start"] <= time < row["end"]
    ]


def interval_may_cover(row: dict, time: datetime | None) -> bool:
    """Scope uncertain intervals to current location, leaving future events to labels."""
    start, end = row["start"], row["end"]
    if time is None:
        return True
    if start is not None and end is not None and end < start:
        return end <= time <= start
    return (start is None or start <= time) and (end is None or time < end)


def death_conflict(admission: dict, patient: dict) -> str:
    start, end, death = (
        clinical_time(admission.get(field))
        for field in ("admittime", "dischtime", "deathtime")
    )
    expire = str(admission.get("hospital_expire_flag") or "")
    if expire not in {"0", "1"} or start is None or end is None:
        return "death_time_conflict"
    if expire == "1" and (death is None or not start <= death <= end):
        return "death_time_conflict"
    if expire == "0" and admission.get("deathtime"):
        return "death_time_conflict"
    if (
        expire == "0"
        and (admission.get("discharge_location") or "").strip().casefold() == "died"
    ):
        return "death_time_conflict"
    if death and patient.get("dod"):
        try:
            if datetime.fromisoformat(patient["dod"]).date() != death.date():
                return "death_time_conflict"
        except ValueError:
            return "death_date_conflict"
    return ""


def build_candidate(
    match: dict,
    selection: dict,
    admission: dict,
    patient: dict,
    transfers: list[dict],
    stays: list[dict],
    rules: dict,
    image_exists: bool,
) -> dict:
    t0 = clinical_time(match["t0"])
    start, end, death = (
        clinical_time(admission.get(field))
        for field in ("admittime", "dischtime", "deathtime")
    )
    rejected, review = [], []
    if t0 is None or start is None or end is None or not start <= t0 < end:
        review.append("admission_time_conflict")
    age = None
    if (
        t0
        and identifier(patient.get("anchor_age")) is not None
        and (identifier(patient.get("anchor_year")) is not None)
    ):
        age = int(patient["anchor_age"]) + t0.year - int(patient["anchor_year"])
    if age is None or age < 0:
        review.append("unresolved_age")
    elif age < rules["minimum_age"]:
        rejected.append("underage")
    if not image_exists:
        rejected.append("selected_image_unusable")
    if death and t0 and death <= t0:
        rejected.append("death_before_or_at_t0")
    conflict = death_conflict(admission, patient)
    if t0 and patient.get("dod"):
        try:
            if datetime.fromisoformat(patient["dod"]).date() < t0.date():
                conflict = "death_date_conflict"
        except ValueError:
            conflict = "death_date_conflict"
    if conflict:
        review.append(conflict)
    current = active_intervals(transfers, t0) if t0 else []
    icu = active_intervals(stays, t0) if t0 else []
    careunits = {row.get("careunit") or "" for row in current}
    levels = {row["level"] for row in current}
    current_mapping = (
        confirmed(rules["careunits"], next(iter(careunits)))
        if len(careunits) == 1
        else None
    )
    excluded_careunit = bool(
        current_mapping and current_mapping.get("exclude_at_t0", False)
    )
    if excluded_careunit:
        rejected.append("excluded_careunit_at_t0")
    if len(careunits) != 1 or ("unknown" in levels and not excluded_careunit):
        review.append("unresolved_careunit_at_t0")
    level = next(iter(levels)) if len(levels) == 1 else "unknown"
    if level == "icu_ccu":
        rejected.append("icu_at_t0")
    if bool(icu) != (level == "icu_ccu"):
        review.append("transfers_icustays_conflict")
    if any(
        row["reason"]
        not in {
            "",
            "other_admission",
            "excluded_nonlocation_record",
            "interval_outside_admission",
        }
        and interval_may_cover(row, t0)
        for row in transfers + stays
    ):
        review.append("care_interval_conflict")
    if current and level in {"other", "unknown"} and not excluded_careunit:
        review.append("unresolved_careunit_at_t0")
    prior = int(
        bool(
            t0
            and any(
                not row["reason"] and row["level"] == "icu_ccu" and row["start"] < t0
                for row in transfers + stays
            )
        )
    )
    evidence_keys = [row["source_record_key"] for row in current + icu]
    evidence_keys += [
        row["source_record_key"]
        for row in transfers + stays
        if t0
        and (
            row["start"] is not None and row["start"] < t0 and row["level"] == "icu_ccu"
        )
    ]
    evidence_keys += [
        record_key("admissions", admission),
        record_key("patients", patient),
    ]
    reasons = sorted(set(rejected + review))
    return {
        **{
            field: match[field]
            for field in ("subject_id", "hadm_id", "study_id", "dicom_id")
        },
        "sample_id": f"p{match['subject_id']}_h{match['hadm_id']}_s{match['study_id']}",
        **{
            field: selection[field]
            for field in ("dicom_path", "view_position", "selection_reason")
        },
        "t0": t0,
        "t0_raw": match["t0"],
        "admittime": start,
        "dischtime": end,
        "deathtime": death,
        "hospital_expire_flag": str(admission.get("hospital_expire_flag") or ""),
        "discharge_location": admission.get("discharge_location") or "",
        "age_at_t0": age,
        "anchor_year_group": patient.get("anchor_year_group") or "",
        "careunit_at_t0": next(iter(careunits)) if len(careunits) == 1 else "",
        "care_level_at_t0": level,
        "prior_icu_before_t0": prior,
        "outtime_imputed": int(any(row["outtime_imputed"] for row in current)),
        "candidate_status": "excluded"
        if rejected
        else "needs_review"
        if review
        else "eligible",
        "exclusion_reason": ";".join(reasons),
        "review_status": "pending" if review else "not_required",
        "candidate_evidence_keys": json.dumps(sorted(set(evidence_keys))),
        "outcome_version": "burden_v2",
        "candidate_exclusion_reviewer": "",
        "candidate_exclusion_rationale": "",
        "candidate_exclusion_version": "",
    }


def observation(candidate: dict, horizon: int) -> dict:
    result = {**candidate, "horizon_hours": horizon}
    t0, discharged = candidate["t0"], candidate["dischtime"]
    cutoff = t0 + timedelta(hours=horizon) if t0 else None
    end = min(cutoff, discharged) if cutoff and discharged and discharged > t0 else None
    result.update(
        horizon_end=cutoff,
        observation_end=end,
        observed_hours=(end - t0).total_seconds() / 3600 if end else None,
        early_discharge=int(discharged < cutoff)
        if end
        and (
            candidate["hospital_expire_flag"] == "0"
            and "death_time_conflict" not in candidate["exclusion_reason"]
        )
        else 0,
    )
    return result
