"""
Combines every per-candidate component into a final ranked list.

Score composition:

    fit_score = WEIGHT_TITLE     * title_relevance
              + WEIGHT_SEMANTIC  * semantic_fit          (TF-IDF, batch-computed)
              + WEIGHT_SKILL_TRUST * skill_trust
              + WEIGHT_EXPERIENCE * experience_fit
              + WEIGHT_LOCATION  * location_fit
              + rule_adjustments                          (bounded, can be negative)

    final_score = fit_score * availability_multiplier      (behavioral signals)

Honeypot-suspect candidates (risk_score >= HONEYPOT_RISK_THRESHOLD) are
excluded from the top-K selection entirely — not down-weighted, excluded —
since the spec scores honeypots at relevance tier 0 and the Stage 3 filter
is a hard rate cutoff, not a soft preference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from . import config as cfg
from . import features
from . import honeypot as hp
from . import availability as avail
from . import semantic as sem
from . import reasoning as reason_mod
from .io_utils import REQUIRED_HEADER


@dataclass
class Timing:
    steps: Dict[str, float] = field(default_factory=dict)

    def record(self, name: str, seconds: float) -> None:
        self.steps[name] = seconds

    def total(self) -> float:
        return sum(self.steps.values())


def score_candidates(candidates: List[Dict[str, Any]], top_k: int = 100,
                      today=None, verbose: bool = True) -> (List[Dict[str, Any]], Timing):
    """Returns (rows, timing) where `rows` is a list of dicts with exactly
    the REQUIRED_HEADER keys, already sorted/ranked and ready to write."""
    import datetime as _dt
    today = today or _dt.date.today()
    timing = Timing()

    t0 = time.perf_counter()
    rows_meta = []
    candidate_texts = []
    for c in candidates:
        profile = c.get("profile", {}) or {}
        career_history = c.get("career_history", []) or []
        skills = c.get("skills", []) or []
        signals = c.get("redrob_signals", {}) or {}

        title_score, title_ev = features.title_relevance_score(profile, career_history)
        skill_score, skill_ev = features.skill_trust_score(skills, signals.get("skill_assessment_scores", {}))
        exp_score = features.experience_fit_score(profile.get("years_of_experience", 0), career_history)
        loc_score = features.location_fit_score(
            profile.get("location", ""), profile.get("country", ""), signals.get("willing_to_relocate", False)
        )
        rule_adj, rule_reasons, cred_flags = features.rule_adjustments(c)
        risk_score, risk_reasons = hp.compute_honeypot_risk(c)
        avail_mult, avail_ev = avail.compute_availability_multiplier(signals, today)

        rows_meta.append({
            "candidate_id": c.get("candidate_id"),
            "title_score": title_score, "title_ev": title_ev,
            "skill_score": skill_score, "skill_ev": skill_ev,
            "exp_score": exp_score, "loc_score": loc_score,
            "rule_adj": rule_adj, "rule_reasons": rule_reasons, "cred_flags": cred_flags,
            "risk_score": risk_score, "risk_reasons": risk_reasons,
            "avail_mult": avail_mult, "avail_ev": avail_ev,
            "years_of_experience": profile.get("years_of_experience", 0),
        })
        candidate_texts.append(sem.build_candidate_text(c))
    timing.record("feature_extraction", time.perf_counter() - t0)

    t0 = time.perf_counter()
    semantic_scores = sem.compute_semantic_scores(candidate_texts)
    timing.record("semantic_tfidf", time.perf_counter() - t0)

    t0 = time.perf_counter()
    for meta, sem_score in zip(rows_meta, semantic_scores):
        fit = (
            cfg.WEIGHT_TITLE * meta["title_score"]
            + cfg.WEIGHT_SEMANTIC * sem_score
            + cfg.WEIGHT_SKILL_TRUST * meta["skill_score"]
            + cfg.WEIGHT_EXPERIENCE * meta["exp_score"]
            + cfg.WEIGHT_LOCATION * meta["loc_score"]
            + meta["rule_adj"]
        )
        meta["semantic_score"] = float(sem_score)
        meta["fit_score"] = fit
        meta["final_score"] = fit * meta["avail_mult"]
        meta["is_honeypot_suspect"] = hp.is_honeypot_suspect(meta["risk_score"])
    timing.record("combine_scores", time.perf_counter() - t0)

    t0 = time.perf_counter()
    eligible = [m for m in rows_meta if not m["is_honeypot_suspect"]]
    excluded_count = len(rows_meta) - len(eligible)

    eligible.sort(key=lambda m: (-m["final_score"], m["candidate_id"]))

    take_n = min(top_k, len(eligible))
    if take_n < top_k and verbose:
        print(f"[warn] only {len(eligible)} eligible candidates available "
              f"(pool size {len(candidates)}, {excluded_count} excluded as honeypot-suspect); "
              f"writing {take_n} rows instead of {top_k}. This is expected when running "
              f"against the 50-row sample file rather than the full 100K pool.")

    selected = eligible[:take_n]

    # min-max normalize final_score across the selected set into a clean
    # (0, 1] presentation range for the `score` column, preserving order
    scores = np.array([m["final_score"] for m in selected], dtype=float)
    if scores.size > 1 and (scores.max() - scores.min()) > 1e-9:
        norm = (scores - scores.min()) / (scores.max() - scores.min())
        norm = 0.05 + 0.95 * norm   # keep rank-100 above 0 rather than exactly 0
    elif scores.size > 0:
        norm = np.ones_like(scores)
    else:
        norm = np.array([])

    # IMPORTANT: tie-break on the DISPLAYED score, not the raw one.
    # Two candidates can have distinct raw final_scores that are close
    # enough to collapse to the same value once normalized and rounded to
    # 4 decimals for the CSV. The submission spec's validator checks
    # candidate_id-ascending tie-break against that *displayed* score, so
    # sorting must happen again here, after rounding — sorting once on the
    # raw score (as an earlier version of this function did) can leave two
    # rows that display an identical score in the wrong relative order.
    rounded_scores = [round(float(s), 4) for s in norm]
    order = sorted(range(len(selected)), key=lambda i: (-rounded_scores[i], selected[i]["candidate_id"]))
    selected = [selected[i] for i in order]
    rounded_scores = [rounded_scores[i] for i in order]

    rows = []
    for rank, (meta, score_val) in enumerate(zip(selected, rounded_scores), start=1):
        reasoning_text = reason_mod.generate_reasoning(
            candidate_id=meta["candidate_id"],
            rank=rank, top_k=take_n,
            title_evidence=meta["title_ev"],
            skill_evidence=meta["skill_ev"],
            semantic_score=meta["semantic_score"],
            experience_score=meta["exp_score"],
            location_score=meta["loc_score"],
            rule_reasons=meta["rule_reasons"],
            cred_flags=meta["cred_flags"],
            availability_evidence=meta["avail_ev"],
            years_of_experience=meta["years_of_experience"],
        )
        rows.append({
            "candidate_id": meta["candidate_id"],
            "rank": rank,
            "score": score_val,
            "reasoning": reasoning_text,
        })
    timing.record("rank_and_reason", time.perf_counter() - t0)

    if verbose:
        print(f"[info] pool={len(candidates)} eligible={len(eligible)} "
              f"honeypot_excluded={excluded_count} selected={len(rows)}")
        for step, secs in timing.steps.items():
            print(f"[timing] {step}: {secs:.3f}s")
        print(f"[timing] total: {timing.total():.3f}s")

    return rows, timing
