"""ICU timeline assessment and deterministic cohort exclusion reasons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from src.cohort.build_labels import comparable, evaluate_label, parse_time


@dataclass(frozen=True)
class IcuInterval:
    start: datetime
    end: datetime | None
    event_id: int | None


@dataclass(frozen=True)
class IcuTimeline:
    intervals: tuple[IcuInterval, ...]
    invalid_reason: str | None = None
    mixed_intervals: tuple[IcuInterval, ...] = ()

    @property
    def first_time(self) -> datetime | None:
        return min((item.start for item in self.intervals), default=None)

    @property
    def first_event_id(self) -> int | None:
        first = min(self.intervals, key=lambda item: item.start, default=None)
        return first.event_id if first is not None else None


def build_icu_timeline(
    rows: list[dict[str, Any]], careunits: frozenset[str], source: str
) -> IcuTimeline:
    """Read listed ICU units; icustays with a mixed first/last unit need review."""
    intervals: list[IcuInterval] = []
    mixed: list[IcuInterval] = []
    invalid: str | None = None
    for row in rows:
        if source == "transfers":
            if row["careunit"] not in careunits:
                continue
            is_mixed = False
            id_field = "transfer_id"
        elif source == "icustays":
            first_in = row["first_careunit"] in careunits
            last_in = row["last_careunit"] in careunits
            if not first_in and not last_in:
                continue
            is_mixed = not first_in and last_in
            id_field = "stay_id"
        else:
            raise ValueError(f"Unknown ICU source: {source}")
        start = parse_time(row.get("intime"))
        end = parse_time(row.get("outtime"))
        if start is None:
            invalid = "invalid_icu_intime"
            continue
        if row.get("outtime") and end is None:
            invalid = "invalid_icu_outtime"
            continue
        if end is not None and (not comparable(start, end) or end <= start):
            invalid = "invalid_icu_interval"
            continue
        try:
            event_id = int(row[id_field]) if row.get(id_field) else None
        except (TypeError, ValueError):
            invalid = "invalid_icu_event_id"
            continue
        interval = IcuInterval(start, end, event_id)
        (mixed if is_mixed else intervals).append(interval)
    return IcuTimeline(tuple(intervals), invalid, tuple(mixed))


def icu_signature(
    timeline: IcuTimeline, t0: datetime, min_hours: int, max_hours: int
) -> str:
    """Map a source's first ICU event to the cohort decision at one t0."""
    if timeline.invalid_reason:
        return "unknown"
    if any(not comparable(t0, item.start, item.end) for item in timeline.intervals):
        return "unknown"
    if any(
        item.end is None and item.start <= t0 for item in timeline.intervals
    ):
        return "unknown"
    for item in timeline.intervals:
        if item.end is not None and item.start <= t0 < item.end:
            return "at_t0"
    first = timeline.first_time
    if first is None:
        return "negative"
    if first <= t0:
        return "prior"
    if first <= t0 + timedelta(hours=min_hours):
        return "early"
    if first <= t0 + timedelta(hours=max_hours):
        return "positive"
    return "negative"


def cross_audit(
    transfer_timeline: IcuTimeline,
    stay_timeline: IcuTimeline,
    t0: datetime,
    min_hours: int,
    max_hours: int,
    dischtime: datetime | None = None,
    deathtime: datetime | None = None,
    hospital_expire_flag: int | None = None,
) -> tuple[str, bool]:
    """Return the discrepancy type and whether it changes this study's decision."""
    if stay_timeline.invalid_reason:
        return stay_timeline.invalid_reason, True
    mixed_relevant = any(
        not comparable(t0, item.start, item.end)
        or item.start <= t0 + timedelta(hours=max_hours)
        for item in stay_timeline.mixed_intervals
    )
    if mixed_relevant:
        return "mixed_icu_stay", True
    transfer_first = transfer_timeline.first_time
    stay_first = stay_timeline.first_time
    transfer_class = icu_signature(transfer_timeline, t0, min_hours, max_hours)
    stay_class = icu_signature(stay_timeline, t0, min_hours, max_hours)
    if transfer_first == stay_first and transfer_class == stay_class:
        return "agree", False
    if (transfer_first is None) != (stay_first is None):
        reason = "presence_disagreement"
    elif transfer_first != stay_first:
        reason = "time_disagreement"
    else:
        reason = "interval_disagreement"
    changes_decision = transfer_class != stay_class or stay_class == "unknown"
    if dischtime is not None and not changes_decision:
        transfer_outcome = evaluate_label(
            t0, dischtime, transfer_first, deathtime, hospital_expire_flag,
            min_hours, max_hours,
        )
        stay_outcome = evaluate_label(
            t0, dischtime, stay_first, deathtime, hospital_expire_flag,
            min_hours, max_hours,
        )
        changes_decision = (
            transfer_outcome.label != stay_outcome.label
            or transfer_outcome.status != stay_outcome.status
        )
    return reason, changes_decision


EXCLUSION_PRIORITY = (
    ("eligible_admission_match", "admission_match_failed", False),
    ("eligible_frontal", "ineligible_frontal_image", False),
    ("dicom_valid", "invalid_dicom", False),
    ("eligible_age", "underage_or_unknown_age", False),
    ("icu_timeline_valid", "invalid_icu_timeline", False),
    ("icu_at_t0", "icu_at_t0", True),
    ("prior_icu", "prior_icu", True),
    ("icu_within_6h", "icu_within_6h", True),
    ("icu_source_changes_decision", "icu_source_conflict", True),
    ("death_within_H_without_icu", "death_within_H_without_icu", True),
)


def exclusion_reason(flags: dict[str, bool], label_status: str) -> str | None:
    for field, reason, exclude_when in EXCLUSION_PRIORITY:
        if flags[field] == exclude_when:
            return reason
    if label_status not in ("positive", "negative"):
        return label_status
    return None
