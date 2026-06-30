"""
Per-candidate feature extraction.

Every function here takes plain dicts (the raw candidate record, or a
sub-piece of it) and returns either a float in a documented range, or a
(score, evidence) pair where `evidence` is a small dict of the literal
field values that justified the score — those evidence dicts are what
reasoning.py uses to write grounded, non-hallucinated reasoning text.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Tuple

from . import config as cfg


def _lower(s: Any) -> str:
    return str(s or "").lower()


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Title relevance
# ---------------------------------------------------------------------------

def _title_tier_weight(title: str) -> float:
    t = _lower(title)
    if any(k in t for k in cfg.TITLE_TIER_CORE):
        return 1.00
    if any(k in t for k in cfg.TITLE_TIER_ADJACENT):
        return 0.55
    if any(k in t for k in cfg.TITLE_TIER_WEAK):
        return 0.30
    return cfg.TITLE_TIER_OFF_TARGET_BASE


def title_relevance_score(profile: Dict[str, Any], career_history: List[Dict[str, Any]]
                           ) -> Tuple[float, Dict[str, Any]]:
    """Current title counts most; best historical title gives partial credit
    for a candidate who has since moved sideways but has the right lineage."""
    current_title = profile.get("current_title", "")
    current_w = _title_tier_weight(current_title)

    hist_titles = [h.get("title", "") for h in career_history]
    hist_w = max((_title_tier_weight(t) for t in hist_titles), default=0.0)

    # current title dominates; history can only lift the score, not replace it
    score = max(current_w, 0.7 * current_w + 0.3 * hist_w)
    best_hist_title = max(hist_titles, key=_title_tier_weight, default=None) if hist_titles else None
    evidence = {
        "current_title": current_title,
        "current_title_weight": round(current_w, 2),
        "best_history_title": best_hist_title,
    }
    return score, evidence


def is_architect_no_code_proxy(current_title: str) -> bool:
    t = _lower(current_title)
    return any(m in t for m in cfg.ARCHITECT_TITLE_MARKERS)


SENIORITY_MARKERS = ["senior", "staff", "principal", "lead "]


def is_title_chaser(career_history: List[Dict[str, Any]]) -> bool:
    """JD's actual complaint: 'optimizing for Senior -> Staff -> Principal
    titles by switching companies every 1.5 years.' Short average tenure
    alone is NOT this pattern — a specialist moving laterally between IC
    titles (NLP Engineer -> Search Engineer -> Recommendation Systems
    Engineer) across companies in the same domain is a normal, often
    desirable trajectory and should not be flagged. We additionally
    require at least one title in the history to carry an explicit
    seniority-ladder marker, so this only fires on the pattern the JD is
    actually describing.
    """
    if len(career_history) < cfg.TITLE_CHASER_MIN_ROLES:
        return False
    durations = [h.get("duration_months", 0) or 0 for h in career_history]
    if not durations:
        return False
    avg = sum(durations) / len(durations)
    if avg >= cfg.TITLE_CHASER_MAX_AVG_TENURE_MONTHS:
        return False
    titles = [_lower(h.get("title", "")) for h in career_history]
    return any(any(marker in t for marker in SENIORITY_MARKERS) for t in titles)


# ---------------------------------------------------------------------------
# Trust-weighted skill scoring (the anti-keyword-stuffing component)
# ---------------------------------------------------------------------------

def skill_trust_score(skills: List[Dict[str, Any]], skill_assessment_scores: Dict[str, float]
                       ) -> Tuple[float, Dict[str, Any]]:
    """Score core/desired skills, but discount a self-rated advanced/expert
    claim if the platform's own skill_assessment_scores contradicts it.

    This directly targets the dataset's keyword-stuffer trap: a skills
    array full of "advanced NLP" tags is worth little if the assessed
    score for that exact skill sits well below what "advanced" should mean.
    """
    if not skills:
        return 0.0, {"matched_core_skills": []}

    assessment = {k.lower(): v for k, v in (skill_assessment_scores or {}).items()}
    group_scores: Dict[str, float] = {}
    matched_evidence = []

    for group_name, (terms, group_weight) in cfg.CORE_SKILL_GROUPS.items():
        best_contribution = 0.0
        best_match = None
        for sk in skills:
            name = _lower(sk.get("name", ""))
            if not any(term in name for term in terms):
                continue
            prof = sk.get("proficiency", "intermediate")
            prof_w = cfg.PROFICIENCY_WEIGHT.get(prof, 0.5)
            dur = min(sk.get("duration_months", 0) or 0, cfg.SKILL_DURATION_CAP_MONTHS) \
                / cfg.SKILL_DURATION_CAP_MONTHS

            trust_mult = 1.0
            assessed = assessment.get(_lower(sk.get("name", "")))
            endorsements = sk.get("endorsements", 0) or 0
            if prof in ("advanced", "expert"):
                if assessed is not None and assessed < cfg.ASSESSMENT_TRUST_LOW_SCORE:
                    trust_mult = cfg.ASSESSMENT_DISTRUST_MULTIPLIER
                elif assessed is None and endorsements == 0:
                    trust_mult = cfg.ENDORSEMENT_ZERO_DISTRUST_MULTIPLIER

            contribution = prof_w * (0.5 + 0.5 * dur) * trust_mult
            if contribution > best_contribution:
                best_contribution = contribution
                best_match = {
                    "skill": sk.get("name"),
                    "proficiency": prof,
                    "duration_months": sk.get("duration_months", 0),
                    "assessed_score": assessed,
                    "endorsements": endorsements,
                    "trust_discounted": trust_mult < 1.0,
                }
        group_scores[group_name] = best_contribution * group_weight
        if best_match:
            matched_evidence.append(best_match)

    total_weight = sum(w for _, w in cfg.CORE_SKILL_GROUPS.values())
    raw_score = sum(group_scores.values()) / total_weight if total_weight else 0.0
    raw_score = min(1.0, raw_score)

    matched_evidence.sort(key=lambda m: -(cfg.PROFICIENCY_WEIGHT.get(m["proficiency"], 0)))
    return raw_score, {"matched_core_skills": matched_evidence[:4]}


def skill_credibility_check(skills: List[Dict[str, Any]], skill_assessment_scores: Dict[str, float]
                             ) -> Tuple[float, List[Dict[str, Any]]]:
    """Scan EVERY skill (not just the JD-core taxonomy used for scoring
    above) for a self-rated advanced/expert claim contradicted by evidence.
    Two independent corroboration sources are checked, in order of
    strength: the platform's own assessment score (strong evidence, when
    present), and — when no assessment exists at all for that skill —
    endorsement count (weaker evidence, since plenty of genuinely skilled
    people have zero endorsements simply because they never asked). Each
    produces a distinctly-typed flag so reasoning.py can phrase the softer
    one with appropriate hedging rather than stating it as fact.

    This is intentionally decoupled from skill_trust_score's per-group
    "best match wins" logic: a candidate can have one genuine core skill
    that wins its group's score *and* a separate inflated claim elsewhere
    in their skill list — the discount on the inflated claim shouldn't
    silently vanish just because a stronger skill happened to win that
    group's slot. Every flag found here is surfaced verbatim in the
    reasoning text (reasoning.py) and contributes a small, capped penalty
    here (a systematic over-claimer is itself a mild negative signal,
    independent of which specific skills are JD-relevant).
    """
    assessment = {k.lower(): v for k, v in (skill_assessment_scores or {}).items()}
    flags = []
    for sk in skills:
        prof = sk.get("proficiency")
        if prof not in ("advanced", "expert"):
            continue
        assessed = assessment.get(_lower(sk.get("name", "")))
        endorsements = sk.get("endorsements", 0) or 0
        if assessed is not None and assessed < cfg.ASSESSMENT_TRUST_LOW_SCORE:
            flags.append({
                "skill": sk.get("name"), "proficiency": prof,
                "assessed_score": assessed, "flag_type": "assessment",
            })
        elif assessed is None and endorsements == 0:
            flags.append({
                "skill": sk.get("name"), "proficiency": prof,
                "assessed_score": None, "flag_type": "endorsement",
            })
    # endorsement-based flags are weaker evidence -> smaller penalty per flag
    penalty = 0.0
    for f in flags:
        penalty -= 0.05 if f["flag_type"] == "assessment" else 0.025
    penalty = max(penalty, -0.15)
    return penalty, flags


def cv_speech_dominant_without_nlp(skills: List[Dict[str, Any]]) -> bool:
    names = [_lower(s.get("name", "")) for s in skills]
    cv_count = sum(1 for n in names if any(t in n for t in cfg.SKILLS_CV_SPEECH_ROBOTICS))
    nlp_count = sum(1 for n in names if any(t in n for t in cfg.SKILLS_NLP_IR))
    return cv_count >= cfg.CV_SPEECH_DOMINANCE_MIN_COUNT and nlp_count == 0


def recent_llm_wrapper_only(skills: List[Dict[str, Any]]) -> bool:
    has_wrapper = False
    has_substantial_pre_llm = False
    for s in skills:
        name = _lower(s.get("name", ""))
        dur = s.get("duration_months", 0) or 0
        if any(t in name for t in cfg.SKILLS_RECENT_LLM_WRAPPER) and dur <= cfg.RECENT_LLM_MAX_DURATION_MONTHS:
            has_wrapper = True
        if any(t in name for t in cfg.SKILLS_PRE_LLM_ERA_PROXY) and dur >= cfg.PRE_LLM_MIN_DURATION_MONTHS:
            has_substantial_pre_llm = True
    return has_wrapper and not has_substantial_pre_llm


# ---------------------------------------------------------------------------
# Experience-band fit (soft taper, not a hard cutoff)
# ---------------------------------------------------------------------------

def _relevant_ml_tenure_months(career_history: List[Dict[str, Any]]) -> int:
    """Months spent in a core/adjacent-relevance title (per _title_tier_weight)
    at a company that isn't a pure consulting/services firm. This is what
    the JD's "4-5 of which are in applied ML/AI roles at product companies"
    line is actually asking about — distinct from total years_of_experience,
    which says nothing about composition.
    """
    total = 0
    for h in career_history:
        title_w = _title_tier_weight(h.get("title", ""))
        if title_w < cfg.RELEVANT_TENURE_TITLE_MIN_WEIGHT:
            continue
        if is_consulting_only_career([h]):   # single-entry check = "is this one role at a consulting firm"
            continue
        total += h.get("duration_months", 0) or 0
    return total


def _relevant_tenure_band_score(months: int) -> float:
    if cfg.RELEVANT_TENURE_IDEAL_LOW_MONTHS <= months <= cfg.RELEVANT_TENURE_IDEAL_HIGH_MONTHS:
        return 1.0
    dist = (cfg.RELEVANT_TENURE_IDEAL_LOW_MONTHS - months) if months < cfg.RELEVANT_TENURE_IDEAL_LOW_MONTHS \
        else (months - cfg.RELEVANT_TENURE_IDEAL_HIGH_MONTHS)
    score = 1.0 - cfg.RELEVANT_TENURE_TAPER_PER_MONTH * dist
    return max(cfg.RELEVANT_TENURE_SCORE_FLOOR, score)


def experience_fit_score(years_of_experience: float, career_history: List[Dict[str, Any]] = None) -> float:
    y = years_of_experience or 0.0
    if cfg.EXPERIENCE_IDEAL_LOW <= y <= cfg.EXPERIENCE_IDEAL_HIGH:
        total_years_score = 1.0
    elif cfg.EXPERIENCE_BAND_LOW <= y <= cfg.EXPERIENCE_BAND_HIGH:
        total_years_score = 0.85
    else:
        dist = (cfg.EXPERIENCE_BAND_LOW - y) if y < cfg.EXPERIENCE_BAND_LOW else (y - cfg.EXPERIENCE_BAND_HIGH)
        total_years_score = max(cfg.EXPERIENCE_SCORE_FLOOR, 0.85 - cfg.EXPERIENCE_TAPER_PER_YEAR * dist)

    if not career_history:
        return total_years_score

    relevant_months = _relevant_ml_tenure_months(career_history)
    relevant_score = _relevant_tenure_band_score(relevant_months)
    return cfg.TOTAL_YEARS_BLEND_WEIGHT * total_years_score + cfg.RELEVANT_TENURE_BLEND_WEIGHT * relevant_score


# ---------------------------------------------------------------------------
# Location fit
# ---------------------------------------------------------------------------

def location_fit_score(location: str, country: str, willing_to_relocate: bool) -> float:
    loc = _lower(location)
    country_l = _lower(country)
    in_india = "india" in country_l or country_l == ""

    if any(c in loc for c in cfg.LOCATION_PREFERRED):
        return cfg.LOCATION_SCORE_PREFERRED
    if any(c in loc for c in cfg.LOCATION_WELCOME):
        return cfg.LOCATION_SCORE_WELCOME
    if in_india:
        base = cfg.LOCATION_SCORE_OTHER_INDIA
    else:
        base = cfg.LOCATION_SCORE_OUTSIDE_INDIA_BASE
    if willing_to_relocate:
        base = min(1.0, base + cfg.LOCATION_RELOCATE_BONUS)
    return base


# ---------------------------------------------------------------------------
# Career-history-level disqualifier rules
# ---------------------------------------------------------------------------

def is_consulting_only_career(career_history: List[Dict[str, Any]]) -> bool:
    """Matches if EVERY entry is either (a) a named consulting firm, or
    (b) tagged with the 'IT Services' industry. (b) was added after
    auditing real sample data: "Mindtree" appeared as a consulting-pattern
    employer but wasn't yet in the name list — any name list will always
    be incomplete against a 100K-candidate pool, whereas the schema's own
    `industry` field tagging a company "IT Services" is a direct,
    structural match for exactly the category the JD names ("TCS, Infosys,
    Wipro, Accenture, Cognizant, Capgemini, etc."). Every consulting-named
    company observed in the sample data was independently tagged this way,
    so the two signals corroborate rather than conflict.
    """
    if not career_history:
        return False
    def _is_consulting(h: Dict[str, Any]) -> bool:
        company = _lower(h.get("company", ""))
        industry = _lower(h.get("industry", ""))
        return any(firm in company for firm in cfg.CONSULTING_FIRMS) or industry == "it services"
    return all(_is_consulting(h) for h in career_history)


def is_research_only_career(career_history: List[Dict[str, Any]]) -> bool:
    if not career_history:
        return False
    def _is_research(industry: str) -> bool:
        i = _lower(industry)
        return any(tag in i for tag in cfg.RESEARCH_ONLY_INDUSTRY_TAGS)
    return all(_is_research(h.get("industry", "")) for h in career_history)


def is_closed_source_only_proxy(years_of_experience: float, github_activity_score: float) -> bool:
    return (years_of_experience or 0) >= cfg.CLOSED_SOURCE_MIN_YEARS \
        and (github_activity_score is not None) \
        and github_activity_score <= cfg.CLOSED_SOURCE_GITHUB_SCORE_THRESHOLD


def is_self_described_hobbyist(profile: Dict[str, Any], career_history: List[Dict[str, Any]]) -> bool:
    """Catches a candidate whose own summary explicitly hedges their AI/ML
    exposure as hobbyist-level (see config.py docstring for the concrete
    case this was built from), while having no career_history title that
    actually reflects ML/AI work. Both conditions are required: hedging
    language alone isn't enough (a genuine senior practitioner might
    mention a weekend project), and a generic title alone isn't enough
    either (plenty of real candidates are early in a title transition).
    It's the *combination* — hedges their own depth AND has never held a
    relevant title — that's the specific, low-false-positive signal.
    """
    text = _lower(profile.get("summary", ""))
    if not any(phrase in text for phrase in cfg.SELF_DESCRIBED_SHALLOW_PHRASES):
        return False
    titles = [h.get("title", "") for h in career_history] + [profile.get("current_title", "")]
    has_real_ml_title = any(_title_tier_weight(t) >= 0.55 for t in titles)
    return not has_real_ml_title


# ---------------------------------------------------------------------------
# Bundle: run every rule, return total bounded adjustment + reasons
# ---------------------------------------------------------------------------

def rule_adjustments(candidate: Dict[str, Any]) -> Tuple[float, List[str], List[Dict[str, Any]]]:
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    adj = 0.0
    reasons: List[str] = []

    if is_consulting_only_career(career_history):
        adj += cfg.PENALTY_CONSULTING_ONLY
        reasons.append("consulting_only_career")
    if is_research_only_career(career_history):
        adj += cfg.PENALTY_RESEARCH_ONLY
        reasons.append("research_only_no_production")
    if recent_llm_wrapper_only(skills):
        adj += cfg.PENALTY_RECENT_LLM_ONLY
        reasons.append("recent_llm_wrapper_only")
    if cv_speech_dominant_without_nlp(skills):
        adj += cfg.PENALTY_CV_SPEECH_NO_NLP
        reasons.append("cv_speech_without_nlp")
    if is_title_chaser(career_history):
        adj += cfg.PENALTY_TITLE_CHASER
        reasons.append("title_chaser_pattern")
    if is_architect_no_code_proxy(profile.get("current_title", "")):
        adj += cfg.PENALTY_ARCHITECT_NO_CODE
        reasons.append("architect_tech_lead_title")
    if is_closed_source_only_proxy(profile.get("years_of_experience", 0),
                                    signals.get("github_activity_score")):
        adj += cfg.PENALTY_CLOSED_SOURCE_ONLY
        reasons.append("closed_source_only_proxy")
    if is_self_described_hobbyist(profile, career_history):
        adj += cfg.PENALTY_SELF_DESCRIBED_HOBBYIST
        reasons.append("self_described_hobbyist")

    cred_penalty, cred_flags = skill_credibility_check(skills, signals.get("skill_assessment_scores", {}))
    if cred_flags:
        adj += cred_penalty
        for flag in cred_flags:
            reasons.append(f"skill_overclaim:{flag['skill']}")

    adj = max(cfg.RULE_ADJUSTMENT_FLOOR, adj)
    return adj, reasons, cred_flags
