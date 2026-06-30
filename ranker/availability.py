"""
Availability multiplier from redrob_signals.

This is deliberately a *multiplier*, not another additive scoring
component: the JD frames availability as a gate on usefulness ("for
hiring purposes, not actually available"), not as another axis of merit.
A candidate who is maximally responsive and active but a weak skills/title
fit should not be able to multiply their way into the top ranks; the
ceiling (config.AVAILABILITY_MULTIPLIER_CEILING) keeps that contained,
while the floor keeps one bad signal from being treated as disqualifying
on its own.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Tuple

from . import config as cfg
from .features import _parse_date


def _bucket(value: float, buckets) -> float:
    for limit, mult in buckets:
        if value <= limit:
            return mult
    return buckets[-1][1]


def compute_availability_multiplier(signals: Dict[str, Any], today: date) -> Tuple[float, Dict[str, Any]]:
    last_active = _parse_date(signals.get("last_active_date"))
    days_inactive = (today - last_active).days if last_active else 9999

    recency_mult = _bucket(days_inactive, cfg.RECENCY_BUCKETS_DAYS)
    open_to_work_mult = cfg.OPEN_TO_WORK_TRUE_MULT if signals.get("open_to_work_flag") else cfg.OPEN_TO_WORK_FALSE_MULT
    response_rate = signals.get("recruiter_response_rate", 0.0) or 0.0
    response_mult = cfg.RESPONSE_RATE_BASE_MULT + cfg.RESPONSE_RATE_SCALE * response_rate
    response_time_hours = signals.get("avg_response_time_hours", 72) or 72
    response_time_mult = _bucket(response_time_hours, cfg.RESPONSE_TIME_BUCKETS_HOURS)
    notice_days = signals.get("notice_period_days", 60) or 60
    notice_mult = _bucket(notice_days, cfg.NOTICE_PERIOD_BUCKETS_DAYS)
    interview_rate = signals.get("interview_completion_rate", 0.5) or 0.0
    interview_mult = cfg.INTERVIEW_COMPLETION_BASE_MULT + cfg.INTERVIEW_COMPLETION_SCALE * interview_rate

    verification_bonus = sum(
        cfg.VERIFICATION_BONUS_EACH
        for flag in ("verified_email", "verified_phone", "linkedin_connected")
        if signals.get(flag)
    )

    raw_mult = recency_mult * open_to_work_mult * response_mult * response_time_mult * notice_mult * interview_mult
    raw_mult += verification_bonus
    final_mult = max(cfg.AVAILABILITY_MULTIPLIER_FLOOR, min(cfg.AVAILABILITY_MULTIPLIER_CEILING, raw_mult))

    evidence = {
        "days_since_active": days_inactive,
        "open_to_work_flag": signals.get("open_to_work_flag"),
        "recruiter_response_rate": response_rate,
        "avg_response_time_hours": response_time_hours,
        "notice_period_days": notice_days,
    }
    return final_mult, evidence
