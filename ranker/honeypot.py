"""
Honeypot detection.

submission_spec.md Section 7 is explicit that these are *structural
impossibilities*, not semantic judgment calls: "8 years of experience at a
company founded 3 years ago," "'expert' proficiency in 10 skills with 0
years used." Every check below is arithmetic over fields already in the
schema — no guessing, no NLP, fully deterministic and auditable.

CALIBRATION HISTORY — read this before touching any threshold below.

First pass (50-row sample): a 1.5x skill-duration-overage ratio flagged
8/50 candidates, all low-experience profiles where an ordinary skill had a
duration_months loosely exceeding total tenure. Raised to 2.5x.

Second pass (real 100K pool — this is the important one): running the
2.5x-calibrated detector against the actual pool excluded 591 candidates.
Breaking that down by which check fired, with each check's population
base rate, told a clear story:

    check                                   count    rate     real signal?
    expert_claim_near_zero_duration            84    0.08%    YES — confirmed below
    single_role_duration_exceeds_experience    21    0.02%    inconclusive, kept as weak corroboration
    career_history_sum_exceeds_experience      24    0.02%    inconclusive, kept as weak corroboration
    skill_duration_exceeds_experience        2425    2.43%    NO — see below
    signup_after_last_active                 7496    7.50%    NO — see below

`signup_after_last_active` firing on 7.5% of the ENTIRE 100,000-candidate
pool cannot possibly be tracking a "~80 designed honeypots" signal — it's
mathematically two orders of magnitude too common. `skill_duration_
exceeds_experience` at 2.43% is similarly far too common, and manually
inspecting the highest-risk candidates confirmed why: candidates with
~12 months total experience and a *random* skill among their 10-20 listed
skills happening to have a duration_months in the 30-90 range, across
totally unrelated job titles (Civil Engineer, HR Manager, Sales
Executive, QA Engineer — no AI-stuffing pattern at all). This is the
dataset generator sampling skill durations without tightly conditioning
on total years_of_experience, not a designed trap — confirmed by the fact
that even this dataset's clearest genuine strong-fit candidate in the
50-row sample (`CAND_0000031`) has the same pattern on two skills.

`expert_claim_near_zero_duration`, by contrast: manually inspecting every
one of the 21 candidates whose risk score hit the 1.0 cap *and* triggered
this specific check showed every single one had 3-5 different skills, ALL
independently rated "expert" with duration_months == 0 exactly — across
unrelated titles (Mobile Developer, HR Manager, Full Stack Developer) and
experience levels (2.0 to 13.3 years). That is a clean, mechanically
constructed signature, essentially a direct match to the spec's own
example ("'expert' proficiency in 10 skills with 0 years used"), not
something a loose random sampler would produce by chance across multiple
independent skill draws simultaneously.

Conclusion, and what changed as a result: `expert_claim_near_zero_
duration` is the only check trusted to independently cross the exclusion
threshold. Every other check's weight was reduced so that none of them —
even all three stacked together — can reach the threshold on their own;
they now only matter as corroboration once the trusted signal has already
fired. `signup_after_last_active` was removed from risk scoring entirely
(zero discriminative value demonstrated, not just suspected).

Third pass — capping repetition, not just per-instance weight: the
demoted-weight version above still excluded 33 candidates instead of the
expected ~21. 12 of those (10 at risk=0.60, 2 at risk=0.72) turned out to
be the SAME low-experience noise pattern as before, just needing more
repetitions to cross the lower-weighted threshold — a candidate with 5-6
skills each individually tripping `skill_duration_exceeds_experience`
stacks 5-6 x 0.12 back over 0.5, with zero involvement from the primary
signal. None of the 12 carried `expert_claim_near_zero_duration`. Fixed
by capping each secondary check's contribution to ONE fixed amount per
candidate regardless of how many individual skills/roles trigger it
within that check type — corroboration value is preserved (it can still
nudge risk up alongside the primary signal), but volume alone can no
longer manufacture risk. This brought the excluded count to the expected
~21.

This is a precision-over-recall choice, made deliberately: the new
detector likely under-catches relative to the true ~80 (some honeypots
may use a construction this schema genuinely can't observe, like a
company-founding-date impossibility), but it stops excluding hundreds of
ordinary low-experience candidates who happened to have one noisy field.
Re-run `dump_honeypot_flags.py` against any new candidate pool before
trusting these weights blindly — this calibration is itself a finding
from one specific 100K-row pool, not a law of nature.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

from . import config as cfg
from .features import _parse_date


def compute_honeypot_risk(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    profile = candidate.get("profile", {}) or {}
    career_history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []

    yoe = profile.get("years_of_experience", 0) or 0
    yoe_months = yoe * 12

    risk = 0.0
    reasons: List[str] = []

    # 1. PRIMARY SIGNAL — "expert" proficiency claimed with ~no time using
    #    the skill. The only check trusted to independently exclude — see
    #    module docstring for the real-data evidence behind that trust.
    for s in skills:
        if s.get("proficiency") == "expert" and (s.get("duration_months", 0) or 0) <= cfg.HONEYPOT_EXPERT_ZERO_DURATION_MAX_MONTHS:
            risk += cfg.HONEYPOT_EXPERT_ZERO_DURATION_RISK
            reasons.append(f"expert_claim_near_zero_duration:{s.get('name')}")

    # 2-4. CORROBORATION ONLY — each capped to a single fixed contribution
    #    PER CHECK TYPE, no matter how many individual skills/roles trigger
    #    it within that type. Without this cap, a candidate with enough
    #    skills (5-6) could still stack the same demoted per-instance
    #    weight back over the exclusion threshold purely through
    #    repetition — confirmed happening on the real 100K pool (10
    #    candidates at risk=0.60, 2 at risk=0.72, all ~1-year-experience
    #    profiles with 5-6 skills each individually tripping the duration
    #    ratio, none carrying the primary signal). Capping at one
    #    contribution per type closes that without losing corroboration
    #    value: it can still nudge risk up when paired with the primary
    #    signal, it just can't manufacture risk through volume alone.
    if yoe_months > 0:
        if any((s.get("duration_months", 0) or 0) > yoe_months * cfg.HONEYPOT_SKILL_DURATION_OVERAGE_RATIO
               for s in skills):
            risk += cfg.HONEYPOT_SKILL_DURATION_OVERAGE_RISK
            worst = max(skills, key=lambda s: (s.get("duration_months", 0) or 0) / max(yoe_months, 1))
            reasons.append(f"skill_duration_exceeds_experience:{worst.get('name')}")

    hist_sum = sum((h.get("duration_months", 0) or 0) for h in career_history)
    if yoe_months > 0 and hist_sum > yoe_months * cfg.HONEYPOT_CAREER_SUM_OVERAGE_RATIO:
        risk += cfg.HONEYPOT_CAREER_SUM_OVERAGE_RISK
        reasons.append("career_history_sum_exceeds_experience")

    if any((h.get("duration_months", 0) or 0) > yoe_months + cfg.HONEYPOT_SINGLE_ROLE_OVERAGE_BUFFER_MONTHS
           for h in career_history):
        risk += cfg.HONEYPOT_SINGLE_ROLE_OVERAGE_RISK
        worst_role = max(career_history, key=lambda h: h.get("duration_months", 0) or 0)
        reasons.append(f"single_role_duration_exceeds_total_experience:{worst_role.get('company')}")

    # signup_date-after-last_active_date was removed from risk scoring —
    # confirmed firing on 7.5% of the entire 100K pool, which is two
    # orders of magnitude too common to track a ~0.08% honeypot rate.
    # See module docstring. Left out of compute entirely rather than
    # zero-weighted, so it's unambiguous this isn't an oversight.

    return min(1.0, risk), reasons


def is_honeypot_suspect(risk_score: float) -> bool:
    return risk_score >= cfg.HONEYPOT_RISK_THRESHOLD
