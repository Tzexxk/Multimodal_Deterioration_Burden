"""Evidence-first V2 event rules; ambiguous onsets remain unresolved."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime

from src.cohort.build_candidates import (
    active_intervals,
    deduplicate,
    ownership,
)
from src.cohort.burden_io import (
    COMPONENTS,
    clinical_time,
    confirmed,
    identifier,
    record_key,
    score,
)

EVIDENCE_COLUMNS = (
    "sample_id subject_id hadm_id study_id horizon_hours component event_time "
    "first_event_time baseline_status event_status source_file source_record_key "
    "source_record_keys source_itemid source_value qualifies exclusion_reason "
    "review_status raw_evidence override_version automated_qualifies "
    "automated_event_status automated_exclusion_reason"
).split()
EVENT_COLUMNS = ["sample_id", "subject_id", "hadm_id", "study_id", "horizon_hours"]
for _component in COMPONENTS:
    EVENT_COLUMNS += [
        _component,
        f"{_component}_first_event_time",
        f"{_component}_event_status",
        f"{_component}_baseline_status",
    ]
EVENT_COLUMNS += (
    "T R C D burden_score phys_death high_burden label_status "
    "spo2_measurements_in_window spo2_source_available hypoxemia_evidence_status "
    "first_hypoxemia_time minutes_from_first_icu_transfer_to_hypoxemia "
    "emar_source_available procedure_source_available"
).split()


def number(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def boolean(value: object) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"1", "y", "yes", "true"}:
        return True
    if text in {"", "0", "n", "no", "false"}:
        return False
    return None


def evidence(
    study: dict,
    component: str,
    table: str,
    row: dict,
    time: datetime | None,
    qualifies: int | None,
    reason: str = "",
    baseline: str = "not_observed",
    source_file: str | None = None,
) -> dict:
    key = row.get("source_record_key") or record_key(table, row)
    status = (
        "unresolved"
        if qualifies is None
        else "positive"
        if qualifies
        else (
            "preexisting_only"
            if reason == "preexisting_support"
            else "no_qualifying_record"
        )
    )
    return {
        **{name: study[name] for name in EVENT_COLUMNS[:5]},
        "component": component,
        "event_time": time,
        "first_event_time": time if qualifies == 1 else None,
        "baseline_status": baseline,
        "event_status": status,
        "automated_qualifies": qualifies,
        "automated_event_status": status,
        "automated_exclusion_reason": reason,
        "source_file": source_file or table,
        "source_record_key": key,
        "source_record_keys": json.dumps(row.get("source_record_keys", [key])),
        "source_itemid": str(row.get("itemid") or ""),
        "source_value": str(
            row.get("value") or row.get("medication") or row.get("careunit") or ""
        ),
        "qualifies": qualifies,
        "exclusion_reason": reason,
        "review_status": "pending" if qualifies is None else "not_required",
        "raw_evidence": json.dumps(row, sort_keys=True, default=str),
        "override_version": "",
    }


def in_window(study: dict, time: datetime | None) -> bool:
    return bool(time and study["t0"] < time <= study["observation_end"])


def relevant(study: dict, start: datetime | None, end: datetime | None = None) -> bool:
    return (
        start is None
        or start <= study["observation_end"]
        and (start >= study["t0"] or end is None or end > study["t0"])
    )


def stay_reason(row: dict, time: datetime | None, context: dict) -> str:
    reason = ownership(row, context["admission"], context["admissions"], time)
    if reason:
        return reason
    matches = [
        stay
        for stay in context["stays"]
        if identifier(stay.get("stay_id")) == identifier(row.get("stay_id"))
    ]
    if len(matches) != 1 or matches[0]["reason"]:
        return "unresolved_stay_ownership"
    if time is None or not matches[0]["start"] <= time <= matches[0]["end"]:
        return "event_outside_icu_stay"
    return ""


def transfer_evidence(study: dict, context: dict) -> list[dict]:
    result = []
    transfers = context["transfers"]
    for target in transfers:
        start = target["start"]
        if target["reason"] in {
            "other_admission",
            "excluded_nonlocation_record",
            "interval_outside_admission",
        } or not relevant(study, start, target["end"]):
            continue
        if start is not None and not in_window(study, start):
            continue
        previous = (
            [
                row
                for row in transfers
                if not row["reason"] and row["start"] < start and row["end"] == start
            ]
            if start
            else []
        )
        sources = {row.get("careunit") for row in previous}
        from_level = previous[0]["level"] if len(sources) == 1 else "unknown"
        enriched = {
            **target,
            "from_transfer_ids": [row["transfer_id"] for row in previous],
            "from_careunits": sorted(str(value or "") for value in sources),
            "from_level": from_level,
            "to_level": target["level"],
        }
        reason = target["reason"]
        if (
            start
            and len({row.get("careunit") for row in active_intervals(transfers, start)})
            > 1
        ):
            reason = reason or "conflicting_transfer_destination"
        if target["level"] == "unknown" or from_level == "unknown":
            reason = reason or "unconfirmed_transfer_mapping_or_origin"
        if (
            target["level"] == "icu_ccu"
            and start
            and not active_intervals(context["stays"], start)
        ):
            reason = reason or "transfers_icustays_conflict"
        for component, destination, origins in (
            (COMPONENTS[0], "icu_ccu", {"ward", "stepdown_intermediate"}),
            (COMPONENTS[1], "stepdown_intermediate", {"ward"}),
        ):
            qualifies = (
                None
                if reason
                else int(target["level"] == destination and from_level in origins)
            )
            care_level = study.get("care_level_at_t0", "unknown")
            baseline = (
                "active_documented"
                if care_level == destination
                else (
                    "inactive_documented"
                    if care_level in {"ward", "stepdown_intermediate", "icu_ccu"}
                    else "not_observed"
                )
            )
            result.append(
                evidence(
                    study,
                    component,
                    "transfers",
                    enriched,
                    start,
                    qualifies,
                    reason or ("" if qualifies else "not_an_upgrade"),
                    baseline,
                )
            )
    return result


def ventilation_episodes(context: dict, rules: dict) -> tuple[list[dict], list[dict]]:
    episodes, rejected = [], []
    for row in deduplicate("procedureevents", context["procedureevents"]):
        if str(row["itemid"]) not in {"225792", "225794"}:
            continue
        start, end = clinical_time(row["starttime"]), clinical_time(row["endtime"])
        reason = stay_reason(row, start, context)
        if reason == "other_admission":
            continue
        item = confirmed(rules["items"], str(row["itemid"]))
        action = confirmed(
            rules["procedure_status_actions"], row["statusdescription"] or ""
        )
        row.update(start=start, end=end, reason=reason)
        if not item or not action:
            row["reason"] = reason or "unconfirmed_ventilation_dictionary"
        elif row["itemid"] == "225794" or action["action"] == "exclude":
            row["reason"] = "excluded_noninvasive_or_cancelled"
        elif start is None or end is None or end <= start:
            row["reason"] = reason or "invalid_ventilation_interval"
        elif boolean(row["continueinnextdept"]) is None:
            row["reason"] = reason or "unresolved_department_continuation"
        row["continuation"] = boolean(row["continueinnextdept"]) is True
        (rejected if row["reason"] else episodes).append(row)
    # Conflicting order IDs cannot become independent starts.
    orders = defaultdict(list)
    for row in episodes:
        orders[row.get("orderid") or row["source_record_key"]].append(row)
    for group in orders.values():
        if len(group) > 1:
            for row in group:
                row["reason"] = "conflicting_ventilation_order"
    rejected += [row for row in episodes if row["reason"]]
    valid = sorted(
        [row for row in episodes if not row["reason"]],
        key=lambda row: (row["start"], row["end"], row["source_record_key"]),
    )
    merged = []
    for row in valid:
        if merged and row["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
            merged[-1]["source_record_keys"].append(row["source_record_key"])
            merged[-1]["raw_records"].append(row)
        else:
            merged.append(
                {
                    **row,
                    "source_record_keys": [row["source_record_key"]],
                    "raw_records": [row],
                    "reason": (
                        "unresolved_department_continuation"
                        if row["continuation"]
                        else ""
                    ),
                }
            )
    for row in merged:
        row["source_record_key"] = record_key(
            "ventilation_episode", {"keys": sorted(row["source_record_keys"])}
        )
        row.update(episode_start=row["start"], episode_end=row["end"])
    return merged, rejected


def ventilation_evidence(study: dict, context: dict) -> list[dict]:
    result = []
    for row in context["ventilation_episodes"] + context["ventilation_rejected"]:
        start, end = row["start"], row["end"]
        if not relevant(study, start, end):
            continue
        baseline = (
            "active_documented"
            if start and end and start <= study["t0"] < end
            else "not_observed"
        )
        reason = row["reason"]
        qualifies = None if reason else int(in_window(study, start))
        if reason == "excluded_noninvasive_or_cancelled":
            qualifies = 0
        if not reason and baseline == "active_documented":
            reason = "preexisting_support"
        result.append(
            evidence(
                study,
                COMPONENTS[2],
                "procedureevents",
                row,
                start,
                qualifies,
                reason,
                baseline,
            )
        )
    for row in context["chartevents"]:
        if str(row["itemid"]) == "220277":
            continue
        time = clinical_time(row["charttime"])
        if time is not None and not in_window(study, time):
            continue
        covered = time is not None and any(
            not episode["reason"] and episode["start"] <= time <= episode["end"]
            for episode in context["ventilation_episodes"]
        )
        result.append(
            evidence(
                study,
                COMPONENTS[2],
                "chartevents",
                row,
                time,
                0 if covered else None,
                "direct_episode_crosscheck" if covered else "indirect_ventilation_only",
            )
        )
    return result


def hypoxemia_evidence(study: dict, context: dict, rules: dict) -> list[dict]:
    records = deduplicate(
        "chartevents",
        [row for row in context["chartevents"] if str(row["itemid"]) == "220277"],
    )
    records.sort(
        key=lambda row: (
            clinical_time(row["charttime"]) or datetime.min,
            row["source_record_key"],
        )
    )
    groups = defaultdict(list)
    for row in records:
        groups[clinical_time(row["charttime"])].append(row)
    states, baseline_states, result = {}, {}, []
    item = confirmed(rules["items"], "220277")
    threshold = rules["spo2_threshold"]
    for time, group in groups.items():
        if time and time > study["observation_end"]:
            continue
        valid_values = [number(row["valuenum"]) for row in group]
        valid_values = [
            value for value in valid_values if value is not None and 0 < value <= 100
        ]
        conflict = bool(
            valid_values and min(valid_values) <= threshold < max(valid_values)
        )
        updates = {}
        for row in group:
            stay = identifier(row["stay_id"])
            active_at_t0 = any(
                identifier(value["stay_id"]) == stay
                for value in active_intervals(context["stays"], study["t0"])
            )
            if time and time <= study["t0"] and not active_at_t0:
                continue
            previous = states.get(stay)
            baseline_value = baseline_states.get(stay)
            baseline = (
                "not_observed"
                if baseline_value is None
                else ("active_documented" if baseline_value else "inactive_documented")
            )
            reason = stay_reason(row, time, context)
            if reason == "other_admission":
                continue
            value = number(row["valuenum"])
            if not reason and (
                not item or not confirmed(rules["spo2_units"], row["valueuom"] or "")
            ):
                reason = "unconfirmed_spo2_dictionary_or_unit"
            if not reason and (value is None or not 0 <= value <= 100):
                reason = "invalid_spo2_value"
            if not reason and (value == 0 or boolean(row["warning"]) is not False):
                reason = "spo2_artifact_review"
            if not reason and conflict:
                reason = "conflicting_spo2_at_same_time"
            qualifies = (
                None
                if reason
                else int(
                    in_window(study, time)
                    and value <= threshold
                    and previous is not True
                )
            )
            if reason == "invalid_spo2_value":
                qualifies = 0
            if not reason:
                updates[stay] = value <= threshold
                if time and time <= study["t0"]:
                    baseline_states[stay] = value <= threshold
                    baseline = (
                        "active_documented"
                        if value <= threshold
                        else "inactive_documented"
                    )
                if value <= threshold and previous is True:
                    reason = "preexisting_support"
            result.append(
                evidence(
                    study,
                    COMPONENTS[3],
                    "chartevents",
                    row,
                    time,
                    qualifies,
                    reason,
                    baseline,
                )
            )
        states.update(updates)
    return result


def medication_identity(row: dict, sources: dict, rules: dict) -> None:
    """Supplement blank names with one exact source row, pharmacy before prescriptions."""
    name = (row.get("medication") or "").strip()
    row.update(
        resolved_medication=name,
        medication_name_source="emar" if name else "unresolved",
    )
    key = tuple(
        (row.get(field) or "").strip()
        for field in ("subject_id", "hadm_id", "pharmacy_id")
    )
    for source, field in (("pharmacy", "medication"), ("prescriptions", "drug")):
        status = f"{source}_lookup_status"
        count = f"{source}_match_count"
        row[status] = "not_needed" if row["resolved_medication"] else "disabled"
        row[count], row[f"{source}_records"] = 0, []
        policy = rules.get(f"{source}_identity_lookup", {})
        if (
            row["resolved_medication"]
            or not policy.get("confirmed")
            or not policy.get("reviewer")
        ):
            continue
        row[f"{source}_lookup_version"] = policy["version"]
        if not all(key):
            row[status] = "missing_lookup_key"
            continue
        matches = sources[source].get(key, [])
        row[f"{source}_records"] = matches
        row[count] = sum(value["duplicate_count"] for value in matches)
        row["source_record_keys"] += [value["source_record_key"] for value in matches]
        if (
            source == "prescriptions"
            and row["pharmacy_lookup_status"] == "ambiguous_match"
        ):
            row[status] = "blocked_pharmacy_conflict"
        elif row[count] != 1:
            row[status] = "ambiguous_match" if matches else "no_match"
        elif not (matches[0].get(field) or "").strip():
            row[status] = f"blank_{source}_{field}"
        else:
            row["resolved_medication"] = matches[0][field].strip()
            row["medication_name_source"] = source
            row[status] = "resolved"


def aggregate_medications(
    context: dict, detail_rows: list[dict], rules: dict
) -> list[dict]:
    details = defaultdict(list)
    for row in detail_rows:
        details[row["emar_id"]].append(row)
    sources = {source: defaultdict(list) for source in ("pharmacy", "prescriptions")}
    for source, index in sources.items():
        for row in deduplicate(source, context.get(source, [])):
            key = tuple(
                (row.get(field) or "").strip()
                for field in ("subject_id", "hadm_id", "pharmacy_id")
            )
            if all(key):
                index[key].append(row)
    events = deduplicate("emar", context["emar"])
    id_counts = defaultdict(int)
    for row in events:
        id_counts[row["emar_id"]] += 1
    for row in events:
        attached = deduplicate("emar_detail", details[row["emar_id"]])
        row["details"] = attached
        row["source_record_keys"] = [row["source_record_key"]] + [
            value["source_record_key"] for value in attached
        ]
        row["time"] = clinical_time(row["charttime"])
        row["reason"] = ownership(
            row, context["admission"], context["admissions"], row["time"]
        )
        if id_counts[row["emar_id"]] > 1:
            row["reason"] = "conflicting_emar_id"
        if any(
            value["subject_id"] != row["subject_id"]
            or value["emar_seq"] != row["emar_seq"]
            for value in attached
        ):
            row["reason"] = "emar_detail_key_mismatch"
        medication_identity(row, sources, rules)
        drug = confirmed(rules["medications"], row["resolved_medication"].casefold())
        row["medication_mapping_confirmed"] = bool(drug)
        row["canonical_drug"] = drug.get("canonical_drug") if drug else None
        row["action"] = "review"
        if drug and drug.get("canonical_drug") is None:
            row["action"] = "exclude"
            continue
        types = {
            value["administration_type"]
            for value in attached
            if value["administration_type"]
        }
        mapped = [confirmed(rules["administration_types"], value) for value in types]
        actions = {value["action"] for value in mapped if value}
        type_resolved = bool(types) and all(mapped) and len(actions) == 1
        if type_resolved and actions == {"exclude"}:
            row["action"] = "exclude"
            continue
        event = confirmed(rules["medication_event_actions"], row["event_txt"] or "")
        given = {boolean(value["complete_dose_not_given"]) for value in attached}
        no_administration = (event and event["action"] == "exclude") or (
            given == {True} and row["event_txt"] != "Partial Administered"
        )
        # Identity cannot qualify a record already confirmed as not administered.
        # Ownership, administration conflicts and temporal checks still apply.
        if not drug and not no_administration:
            row["reason"] = row["reason"] or "unconfirmed_medication_name"
        review_type_nonadministration = (
            event
            and event["action"] == "exclude"
            and row["event_txt"] in {"Not Given", "Not Started"}
            and type_resolved
            and len(types) == 1
            and actions == {"review"}
        )
        if review_type_nonadministration:
            doses = [
                number(value["dose_given"])
                for value in attached
                if value["dose_given"] not in (None, "")
            ]
            if any(value is None or value < 0 for value in doses):
                row["reason"] = row["reason"] or "invalid_not_given_dose"
            elif any(value > 0 for value in doses):
                row["reason"] = row["reason"] or "conflicting_not_given_dose"
        if not type_resolved or (
            actions == {"review"} and not review_type_nonadministration
        ):
            row["reason"] = row["reason"] or "unresolved_administration_type"
        if None in given or len(given) > 1:
            row["reason"] = row["reason"] or "conflicting_given_status"
        elif given == {True}:
            if (
                event
                and event["action"] == "given"
                and row["event_txt"] == "Partial Administered"
            ):
                row["reason"] = row["reason"] or "unresolved_partial_administration"
            else:
                row["action"] = "exclude"
                continue
        if not event:
            row["reason"] = row["reason"] or "unconfirmed_medication_event"
        else:
            row["action"] = event["action"]
            if row["action"] == "review":
                row["reason"] = row["reason"] or "unresolved_medication_event"
        routes = {value["route"] for value in attached if value["route"]}
        if any(
            not confirmed(rules["iv_routes"], value)
            or confirmed(rules["iv_routes"], value)["action"] != "iv"
            for value in routes
        ):
            row["reason"] = row["reason"] or "route_conflict_or_unconfirmed"
        prior_rates = [
            number(value["prior_infusion_rate"])
            for value in attached
            if value["prior_infusion_rate"] not in (None, "")
        ]
        row["prior_active"] = any(
            value is not None and value > 0 for value in prior_rates
        )
        if any(value is None or value < 0 for value in prior_rates):
            row["reason"] = row["reason"] or "invalid_prior_infusion_rate"
        continuing = [
            boolean(value["continued_infusion_in_other_location"]) for value in attached
        ]
        complete = [boolean(value["infusion_complete"]) for value in attached]
        if None in continuing + complete:
            row["reason"] = row["reason"] or "unresolved_infusion_status"
        row["continuing"] = any(continuing)
        row["completed"] = any(complete)
    events.sort(key=lambda row: (row["time"] or datetime.min, row["source_record_key"]))
    simultaneous = defaultdict(list)
    for row in events:
        if row["time"] and row["canonical_drug"] and row["action"] != "exclude":
            simultaneous[row["time"], row["canonical_drug"]].append(row)
    for group in simultaneous.values():
        if len(group) > 1:
            for row in group:
                row["reason"] = row["reason"] or "simultaneous_infusion_records"
    active, episode_starts = {}, {}
    for row in events:
        drug = row["canonical_drug"]
        row["onset"] = None
        row["active_before"] = active.get(drug)
        row["episode_start_before"] = episode_starts.get(drug)
        if row["reason"] or row["time"] is None:
            row["reason"] = row["reason"] or "invalid_medication_time"
            continue
        action = row["action"]
        if action == "exclude":
            row["onset"] = False
            continue
        if action in {"start", "restart"}:
            if (
                row.get("prior_active")
                or row.get("continuing")
                or active.get(drug) is True
            ):
                row["reason"] = "conflicting_infusion_start"
            else:
                row["onset"] = True
                active[drug] = True
                episode_starts[drug] = row["time"]
        elif action == "stop":
            row["onset"] = False
            active[drug] = False
            episode_starts.pop(drug, None)
        elif action in {"continue", "given"}:
            if (
                active.get(drug) is True
                or row.get("prior_active")
                or row.get("continuing")
            ):
                row["onset"] = False
                active[drug] = True
            else:
                row["reason"] = "unresolved_infusion_onset"
        else:
            row["reason"] = "unresolved_infusion_action"
        if row.get("completed"):
            active[drug] = False
            episode_starts.pop(drug, None)
    return events


def infusion_evidence(study: dict, context: dict) -> list[dict]:
    result, baseline = [], {}
    for row in context["medication_events"]:
        time, drug = row["time"], row["canonical_drug"]
        if (
            row["reason"] == "other_admission"
            or time
            and time > study["observation_end"]
        ):
            continue
        if time and time <= study["t0"]:
            if not row["reason"]:
                if row["action"] in {"start", "restart", "continue", "given"}:
                    baseline[drug] = (
                        "active_documented"
                        if row["onset"] is not None
                        else "not_observed"
                    )
                if row["action"] == "stop" or row.get("completed"):
                    baseline[drug] = "inactive_documented"
            later_stop = any(
                other["canonical_drug"] == drug
                and not other["reason"]
                and other["time"]
                and time < other["time"] <= study["t0"]
                and (other["action"] == "stop" or other.get("completed"))
                for other in context["medication_events"]
            )
            if later_stop:
                continue
            if not row["reason"] and baseline.get(drug) != "active_documented":
                continue
        state = baseline.get(drug, "not_observed")
        reason = row["reason"]
        qualifies = (
            None if reason else int(in_window(study, time) and row["onset"] is True)
        )
        # A confirmed exclusion cannot establish target support or change its history.
        if row["action"] == "exclude" and reason == "event_outside_admission":
            qualifies = 0
        if not reason and time and time <= study["t0"]:
            reason = "preexisting_support"
        elif not reason and row["action"] == "exclude":
            reason = "excluded_medication_or_administration"
        elif not reason and row["onset"] is False and row["action"] != "stop":
            started = row.get("episode_start_before")
            if started is not None and started > study["t0"]:
                reason = "continuation_of_window_start"
            elif state == "not_observed" and time and in_window(study, time):
                qualifies, reason = None, "unresolved_infusion_onset"
            else:
                reason = "preexisting_support"
        result.append(
            evidence(study, COMPONENTS[4], "emar", row, time, qualifies, reason, state)
        )
    return result


def death_evidence(study: dict, context: dict) -> list[dict]:
    row = context["admission"]
    time = clinical_time(row["deathtime"])
    conflict = "death_time_conflict" in study["exclusion_reason"] or (
        "death_date_conflict" in study["exclusion_reason"]
    )
    return [
        evidence(
            study,
            COMPONENTS[5],
            "admissions",
            row,
            time,
            None if conflict else int(in_window(study, time)),
            "death_time_conflict" if conflict else "",
        )
    ]


def apply_override(
    study: dict, component: str, values: list[dict], override: dict
) -> None:
    if (
        not override.get("reviewer")
        or not override.get("rationale")
        or not override.get("version")
    ):
        raise ValueError("Overrides require reviewer, rationale and version")
    flag = identifier(override.get("event_flag"))
    time = clinical_time(override.get("first_event_time"))
    scope = override.get("override_scope") or "component"
    if (
        scope not in {"component", "source_record"}
        or scope == "source_record"
        and (
            flag != 0
            or override.get("first_event_time")
            or override.get("baseline_status")
        )
    ):
        raise ValueError("Invalid override scope or source-record exclusion")
    if (
        flag not in (0, 1)
        or flag == 1
        and not in_window(study, time)
        or flag == 0
        and (override.get("first_event_time"))
    ):
        raise ValueError("Invalid override flag or time")
    keys = json.loads(override.get("source_record_keys") or "[]")
    available = {key for row in values for key in json.loads(row["source_record_keys"])}
    available |= {row["source_record_key"] for row in values}
    if (
        not isinstance(keys, list)
        or not keys
        or any(not isinstance(key, str) or not key for key in keys)
        or not set(keys) <= available
    ):
        raise ValueError("Override evidence keys do not identify component evidence")
    if scope == "source_record":
        # Exact evidence rows only: a constituent key cannot exclude a merged episode.
        if not set(keys) <= {row["source_record_key"] for row in values}:
            raise ValueError("Source-record exclusion requires exact evidence row keys")
        for row in values:
            if row["source_record_key"] in keys:
                raw = json.loads(row["raw_evidence"])
                if (
                    identifier(raw.get("subject_id")) != study["subject_id"]
                    or identifier(raw.get("hadm_id")) != study["hadm_id"]
                ):
                    raise ValueError(
                        "Source-record exclusion cites another patient or admission"
                    )
                row.update(
                    qualifies=0,
                    first_event_time=None,
                    event_status="no_qualifying_record",
                    review_status="reviewed",
                    exclusion_reason="manual_source_exclusion",
                    override_version=override["version"],
                )
        manual = evidence(
            study, component, "overrides", override, None, 0, "manual_source_exclusion"
        )
        manual.update(
            review_status="reviewed",
            override_version=override["version"],
            source_record_keys=json.dumps(keys),
            source_record_key=record_key("reviewed_source_records", override),
            automated_qualifies=None,
            automated_event_status="manual_source_exclusion",
            automated_exclusion_reason="",
        )
        values.append(manual)
        return
    if override.get("baseline_status") and override["baseline_status"] not in {
        "not_observed",
        "active_documented",
        "inactive_documented",
    }:
        raise ValueError("Invalid override baseline_status")
    for row in values:
        row.update(
            qualifies=0,
            first_event_time=None,
            event_status="no_qualifying_record",
            review_status="reviewed",
            exclusion_reason="manual_override",
            override_version=override["version"],
        )
    manual = evidence(
        study,
        component,
        "overrides",
        override,
        time,
        flag,
        "manual_override",
        override.get("baseline_status") or "not_observed",
    )
    manual.update(
        review_status="reviewed",
        override_version=override["version"],
        source_record_keys=json.dumps(keys),
        source_record_key=record_key(
            "reviewed_event",
            {"component": component, "keys": sorted(keys), "event_time": time},
        ),
        automated_qualifies=None,
        automated_event_status="manual",
        automated_exclusion_reason="",
    )
    values.append(manual)


def evaluate(
    study: dict, context: dict, rules: dict, overrides: dict
) -> tuple[dict, list[dict]]:
    if study["observation_end"] is None or study["candidate_status"] == "excluded":
        values = []
    else:
        values = transfer_evidence(study, context) + ventilation_evidence(
            study, context
        )
        values += hypoxemia_evidence(study, context, rules) + infusion_evidence(
            study, context
        )
        values += death_evidence(study, context)
    result = {name: study[name] for name in EVENT_COLUMNS[:5]}
    output = []
    for component in COMPONENTS:
        records = [row for row in values if row["component"] == component]
        override = overrides.get(
            (study["sample_id"], study["horizon_hours"], component)
        )
        if override:
            if study["candidate_status"] != "eligible":
                raise ValueError("Event overrides cannot bypass candidate eligibility")
            apply_override(study, component, records, override)
        pending = any(row["qualifies"] is None for row in records)
        positive_times = [
            row["first_event_time"] for row in records if row["qualifies"] == 1
        ]
        if study["observation_end"] is None or study["candidate_status"] == "excluded":
            pending = True
        flag = None if pending else int(bool(positive_times))
        baseline = {row["baseline_status"] for row in records}
        state = (
            "active_documented"
            if "active_documented" in baseline
            else (
                "inactive_documented"
                if "inactive_documented" in baseline
                else "not_observed"
            )
        )
        manual = [row for row in records if row["automated_event_status"] == "manual"]
        if manual:
            state = manual[-1]["baseline_status"]
        elif component in COMPONENTS[:2]:
            care_level = study.get("care_level_at_t0", "unknown")
            target = (
                "icu_ccu" if component == COMPONENTS[0] else "stepdown_intermediate"
            )
            state = (
                "active_documented"
                if care_level == target
                else (
                    "inactive_documented"
                    if care_level in {"ward", "stepdown_intermediate", "icu_ccu"}
                    else "not_observed"
                )
            )
        status = (
            "unresolved"
            if pending
            else "positive"
            if flag
            else (
                "preexisting_only"
                if any(row["event_status"] == "preexisting_only" for row in records)
                else "no_qualifying_record"
            )
        )
        result.update(
            {
                component: flag,
                f"{component}_first_event_time": (
                    min(positive_times) if positive_times and not pending else None
                ),
                f"{component}_event_status": status,
                f"{component}_baseline_status": state,
            }
        )
        output += records
    flags = [result[component] for component in COMPONENTS]
    result.update(
        score(flags if study["candidate_status"] == "eligible" else [None] * 6)
    )
    result["label_status"] = (
        "complete"
        if result["burden_score"] is not None
        else ("excluded" if study["candidate_status"] == "excluded" else "unresolved")
    )
    spo2 = [
        row
        for row in output
        if row["component"] == COMPONENTS[3] and in_window(study, row["event_time"])
    ]
    result.update(
        spo2_measurements_in_window=len(spo2),
        spo2_source_available=int(
            bool(spo2)
            or any(
                row["component"] == COMPONENTS[3] and row["event_time"] is None
                for row in output
            )
        ),
        hypoxemia_evidence_status=result[f"{COMPONENTS[3]}_event_status"],
        first_hypoxemia_time=result[f"{COMPONENTS[3]}_first_event_time"],
        emar_source_available=int(
            any(
                row["reason"] != "other_admission"
                for row in context["medication_events"]
            )
        ),
        procedure_source_available=int(bool(context["procedureevents"])),
        minutes_from_first_icu_transfer_to_hypoxemia=None,
    )
    icu_time, hypox_time = (
        result[f"{component}_first_event_time"]
        for component in (COMPONENTS[0], COMPONENTS[3])
    )
    if icu_time and hypox_time:
        result["minutes_from_first_icu_transfer_to_hypoxemia"] = (
            hypox_time - icu_time
        ).total_seconds() / 60
    for row in output:
        row["source_file"] = context["source_names"].get(
            row["source_file"], row["source_file"]
        )
    return result, output
