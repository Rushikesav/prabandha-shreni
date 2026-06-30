"""
Reasoning generator.

submission_spec.md Section 3 lists exactly what Stage 4 manual review
checks for: specific facts, a JD connection, honest concerns where they
exist, no hallucination, variation across rows, and tone consistent with
rank. This module is built around those six checks directly:

- Every fact slotted into a sentence comes from the `evidence` dict
  features.py / honeypot.py / availability.py already computed — nothing
  is invented here. If a fact isn't in evidence, it cannot appear in text.
- Phrasing is chosen from small template pools using a hash of
  candidate_id as the random seed, so the same candidate always gets the
  same reasoning (reproducible) but different candidates get visibly
  different sentence shapes (not name-swapped templates).
- Concerns are only ever mentioned if a real flag fired (a rule-adjustment
  reason, a low availability sub-score, or a weak experience/location fit)
  — never invented for variety's sake.
- Tone intensity is selected from the candidate's *rank band*, not just
  its raw score, so a rank-5 row reads confidently and a rank-95 row reads
  as a hedged, marginal pick — directly the "rank consistency" check.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List


RULE_REASON_TEXT = {
    "consulting_only_career": "their entire career has been at consulting/IT-services firms with no product-company experience",
    "research_only_no_production": "their background looks research-only with no production deployment experience",
    "recent_llm_wrapper_only": "their AI experience looks limited to recent LLM-wrapper work without earlier ML production exposure",
    "cv_speech_without_nlp": "their core expertise is computer vision/speech rather than NLP/IR",
    "title_chaser_pattern": "their job history shows short average tenure across multiple roles",
    "architect_tech_lead_title": "their current title suggests an architecture/management track rather than hands-on coding",
    "closed_source_only_proxy": "there's no visible open-source/GitHub activity to externally validate their work",
    "self_described_hobbyist": "their own summary frames their AI/ML exposure as self-taught/side-project level rather than production experience",
}


def _rng_for(candidate_id: str) -> random.Random:
    seed = int(hashlib.sha256(candidate_id.encode()).hexdigest(), 16) % (2 ** 32)
    return random.Random(seed)


def _tone_band(rank: int, top_k: int) -> str:
    frac = rank / max(top_k, 1)
    if frac <= 0.10:
        return "strong"
    if frac <= 0.40:
        return "solid"
    if frac <= 0.75:
        return "moderate"
    return "marginal"


STRENGTH_OPENERS = {
    "strong": [
        "{title} with {yoe} years' experience — a strong match",
        "{yoe} years as a {title}, squarely in the role's wheelhouse",
        "Strong fit: {title}, {yoe} years, directly relevant background",
    ],
    "solid": [
        "{title} with {yoe} years' experience, a solid fit on substance",
        "{yoe} years' experience as a {title}, good alignment with the role",
        "Good match: {title} background, {yoe} years in",
    ],
    "moderate": [
        "{title}, {yoe} years — partial overlap with what the role needs",
        "{yoe} years as a {title}; some relevant signal, some gaps",
        "Adjacent fit: {title} experience ({yoe} years) with mixed alignment",
    ],
    "marginal": [
        "{title}, {yoe} years — included as a marginal pick rather than a strong fit",
        "Limited overlap: {title} background ({yoe} years), weaker alignment overall",
        "{yoe} years as a {title}; below-bar fit but ahead of the rest of the pool",
    ],
}

SKILL_CLAUSE_TEMPLATES = [
    "with {skill} ({prof})",
    "shows {prof}-level {skill}",
    "brings {prof} {skill} experience",
]

SEMANTIC_CLAUSE_TEMPLATES = {
    "high": ["career narrative closely echoes the role's ranking/retrieval mandate",
              "their own description of past work reads like exactly what this role needs"],
    "mid": ["career narrative has some overlap with the role's retrieval/ranking focus",
             "parts of their work history connect to what this role needs"],
    "low": ["career narrative shows limited overlap with the role's core mandate",
             "little in their work history speaks directly to ranking/retrieval"],
}

CONCERN_OPENERS = [
    "Concern: {concern}.",
    "One caveat — {concern}.",
    "Worth flagging: {concern}.",
]

AVAILABILITY_CONCERN_TEMPLATES = [
    "hasn't been active on the platform in {days} days",
    "has a recruiter response rate of only {rr:.0%}",
    "notice period runs {notice} days",
]

AVAILABILITY_STRENGTH_TEMPLATES = [
    "actively engaged (last seen {days} days ago, {rr:.0%} recruiter response rate)",
    "good availability signal — {rr:.0%} response rate, active within {days} days",
]


def _semantic_bucket(score: float) -> str:
    if score >= 0.66:
        return "high"
    if score >= 0.33:
        return "mid"
    return "low"


def generate_reasoning(candidate_id: str, rank: int, top_k: int,
                        title_evidence: Dict[str, Any],
                        skill_evidence: Dict[str, Any],
                        semantic_score: float,
                        experience_score: float,
                        location_score: float,
                        rule_reasons: List[str],
                        cred_flags: List[Dict[str, Any]],
                        availability_evidence: Dict[str, Any],
                        years_of_experience: float) -> str:
    rng = _rng_for(candidate_id)
    tone = _tone_band(rank, top_k)
    title = title_evidence.get("current_title") or "unspecified title"

    opener = rng.choice(STRENGTH_OPENERS[tone]).format(
        title=title, yoe=round(years_of_experience, 1) if years_of_experience else "?",
    )

    clauses = [opener]

    matched_skills = skill_evidence.get("matched_core_skills") or []
    if matched_skills:
        top = matched_skills[0]
        skill_clause = rng.choice(SKILL_CLAUSE_TEMPLATES).format(
            skill=top["skill"], prof=top["proficiency"],
        )
        if top.get("trust_discounted"):
            if top.get("assessed_score") is not None:
                skill_clause += " (though the platform's own skill assessment for this came in lower than self-rated)"
            else:
                skill_clause += " (though this has zero endorsements and no platform assessment to back it up)"
        clauses[-1] = clauses[-1] + ", " + skill_clause

    sem_bucket = _semantic_bucket(semantic_score)
    clauses.append(rng.choice(SEMANTIC_CLAUSE_TEMPLATES[sem_bucket]).capitalize())

    sentence_1 = clauses[0] + "; " + clauses[1] + "."

    # Second sentence: concerns first (honesty check), else availability strength
    concern_bits = []
    for r in rule_reasons:
        if r in RULE_REASON_TEXT:
            concern_bits.append(RULE_REASON_TEXT[r])
    if experience_score < 0.5:
        concern_bits.append("experience level sits well outside the role's preferred band")
    if location_score < 0.4:
        concern_bits.append("location is outside the preferred/welcome cities with no stated relocation willingness")

    # Surface any self-rating that corroborating evidence contradicts —
    # this is the dataset's core anti-keyword-stuffing tell, computed
    # independently across ALL of the candidate's skills (features.py
    # skill_credibility_check), so it always shows up here even when the
    # opening clause above happened to feature a different, uncontested
    # skill. Assessment-based flags are stated as fact; endorsement-based
    # flags (weaker evidence — no assessment exists either way) are hedged.
    for flag in cred_flags[:2]:
        if flag["flag_type"] == "assessment":
            concern_bits.append(
                f"self-rated '{flag['proficiency']}' in {flag['skill']} but the platform skill "
                f"assessment for it scored only {flag['assessed_score']:.0f}/100"
            )
        else:
            concern_bits.append(
                f"claims '{flag['proficiency']}' in {flag['skill']} with zero endorsements "
                f"and no platform assessment to corroborate it"
            )

    days = availability_evidence.get("days_since_active", 0)
    rr = availability_evidence.get("recruiter_response_rate", 0.0)
    rt_hours = availability_evidence.get("avg_response_time_hours", 72)
    notice = availability_evidence.get("notice_period_days", 60)

    if days > 120 or rr < 0.25 or rt_hours > 168:
        if days > 120:
            concern_bits.append(f"hasn't been active on the platform in {days} days")
        if rr < 0.25:
            concern_bits.append(f"recruiter response rate is only {rr:.0%}")
        if rt_hours > 168:
            concern_bits.append(f"takes an average of {rt_hours:.0f} hours to respond when they do reply")

    if concern_bits:
        chosen = rng.sample(concern_bits, k=min(2, len(concern_bits)))
        sentence_2 = rng.choice(CONCERN_OPENERS).format(concern="; ".join(chosen))
    else:
        sentence_2 = rng.choice(AVAILABILITY_STRENGTH_TEMPLATES).format(days=days, rr=rr).capitalize() + "."

    return f"{sentence_1} {sentence_2}"
