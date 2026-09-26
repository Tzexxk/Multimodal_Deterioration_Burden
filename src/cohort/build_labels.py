"""Pure, study-level ICU outcome rules for cohort audits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class LabelResult:
    label: int | None
    status: str
    hours_to_icu: float | None
    icu_within_6h: bool
    death_within_H_without_icu: bool


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def comparable(*values: datetime | None) -> bool:
    present = [value for value in values if value is not None]
    return not present or all(
        (value.tzinfo is None) == (present[0].tzinfo is None)
        for value in present
    )


def evaluate_label(
    t0: datetime,
    dischtime: datetime,
    first_icu_time: datetime | None,
    deathtime: datetime | None,
    hospital_expire_flag: int | None,
    min_hours: int,
    max_hours: int,
) -> LabelResult:
    """Classify one study without choosing or discarding any other study."""
    if not comparable(t0, dischtime, first_icu_time, deathtime):
        return LabelResult(None, "incomparable_timezones", None, False, False)
    if dischtime <= t0 or min_hours <= 0 or max_hours <= min_hours:
        return LabelResult(None, "invalid_time_window", None, False, False)
    if hospital_expire_flag not in (0, 1):
        return LabelResult(None, "unknown_death_status", None, False, False)
    if (hospital_expire_flag == 1) != (deathtime is not None):
        return LabelResult(None, "conflicting_death_status", None, False, False)
    if deathtime is not None and not t0 < deathtime <= dischtime:
        return LabelResult(None, "invalid_death_time", None, False, False)
    if first_icu_time is not None and first_icu_time >= dischtime:
        return LabelResult(None, "icu_outside_admission", None, False, False)

    hours_to_icu = (
        (first_icu_time - t0).total_seconds() / 3600
        if first_icu_time is not None
        else None
    )
    if hours_to_icu is not None and hours_to_icu <= 0:
        return LabelResult(None, "icu_at_or_before_t0", hours_to_icu, False, False)
    if hours_to_icu is not None and hours_to_icu <= min_hours:
        return LabelResult(None, "icu_within_6h", hours_to_icu, True, False)

    horizon = t0 + timedelta(hours=max_hours)
    if deathtime is not None and deathtime <= horizon:
        if first_icu_time is None or deathtime < first_icu_time:
            return LabelResult(
                None, "death_within_H_without_icu", hours_to_icu, False, True
            )
    if first_icu_time is not None and first_icu_time <= horizon:
        return LabelResult(1, "positive", hours_to_icu, False, False)
    # An alive discharge before the horizon or observation through the horizon
    # is sufficient to establish the absence of an ICU event in that window.
    return LabelResult(0, "negative", hours_to_icu, False, False)
