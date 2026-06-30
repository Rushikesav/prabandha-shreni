#!/usr/bin/env python3
"""
Dump every candidate the honeypot detector flagged, with the literal field
values that triggered each flag, so you can manually eyeball whether
they're real designed honeypots or false positives from the heuristics.

Usage:
    python dump_honeypot_flags.py --candidates ./candidates.jsonl.gz --out ./honeypot_review.csv

Also prints a breakdown of how many candidates each individual check fired
on — that breakdown is the most useful single piece of output here, since
it tells you WHICH check (if any) is responsible for the bulk of the 591
flagged out of 100,000, which is the thing to recalibrate if any single
check is doing most of the over-flagging.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ranker.io_utils import iter_candidates
from ranker.honeypot import compute_honeypot_risk, is_honeypot_suspect


def _evidence_for_reason(candidate, reason: str) -> str:
    """Pull out the specific field values behind a fired reason string, so
    the dump is self-contained — you shouldn't need to go re-find the raw
    candidate record to judge whether a flag looks right."""
    profile = candidate.get("profile", {}) or {}
    career_history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    yoe = profile.get("years_of_experience", 0) or 0

    if reason.startswith("expert_claim_near_zero_duration:"):
        name = reason.split(":", 1)[1]
        sk = next((s for s in skills if s.get("name") == name), {})
        return f"skill='{name}' duration_months={sk.get('duration_months')} (claimed expert)"

    if reason.startswith("skill_duration_exceeds_experience:"):
        name = reason.split(":", 1)[1]
        sk = next((s for s in skills if s.get("name") == name), {})
        return (f"skill='{name}' duration_months={sk.get('duration_months')} "
                f"vs total_experience_months={round(yoe*12,1)}")

    if reason == "career_history_sum_exceeds_experience":
        total = sum((h.get("duration_months", 0) or 0) for h in career_history)
        return f"sum(career_history durations)={total}mo vs total_experience_months={round(yoe*12,1)}"

    if reason.startswith("single_role_duration_exceeds_total_experience:"):
        company = reason.split(":", 1)[1]
        h = next((h for h in career_history if h.get("company") == company), {})
        return (f"company='{company}' role_duration_months={h.get('duration_months')} "
                f"vs total_experience_months={round(yoe*12,1)}")

    return ""


def main():
    parser = argparse.ArgumentParser(description="Dump honeypot-flagged candidates for manual review.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", default="honeypot_review.csv", help="Output CSV path")
    parser.add_argument("--include-borderline", action="store_true",
                         help="Also include candidates with risk > 0 but below the exclusion threshold")
    args = parser.parse_args()

    reason_type_counts = Counter()
    risk_histogram = Counter()
    flagged_rows = []
    total = 0

    for c in iter_candidates(args.candidates):
        total += 1
        risk, reasons = compute_honeypot_risk(c)
        if risk <= 0:
            continue

        risk_histogram[round(risk, 2)] += 1
        for r in reasons:
            reason_type = r.split(":")[0]
            reason_type_counts[reason_type] += 1

        suspect = is_honeypot_suspect(risk)
        if not suspect and not args.include_borderline:
            continue

        profile = c.get("profile", {}) or {}
        evidence_lines = [f"{r} [{_evidence_for_reason(c, r)}]" for r in reasons]

        flagged_rows.append({
            "candidate_id": c.get("candidate_id"),
            "risk_score": round(risk, 2),
            "excluded_from_topk": suspect,
            "current_title": profile.get("current_title"),
            "years_of_experience": profile.get("years_of_experience"),
            "reasons_with_evidence": " | ".join(evidence_lines),
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "candidate_id", "risk_score", "excluded_from_topk",
            "current_title", "years_of_experience", "reasons_with_evidence",
        ])
        writer.writeheader()
        writer.writerows(flagged_rows)

    excluded_count = sum(1 for r in flagged_rows if r["excluded_from_topk"])

    print(f"Scanned {total} candidates.")
    print(f"Excluded from top-K (risk >= threshold): {excluded_count}")
    print(f"Rows written to {args.out}: {len(flagged_rows)}")
    print()
    print("Breakdown by check type (a candidate can trigger more than one):")
    for reason_type, count in reason_type_counts.most_common():
        print(f"  {reason_type:50s} {count:>6}")
    print()
    print("Risk-score histogram (candidates with risk > 0):")
    for score, count in sorted(risk_histogram.items()):
        print(f"  risk={score:.2f}: {count}")
    print()
    print("Next step: open the CSV and read 15-20 rows from the highest-risk "
          "end. If most look like real fabrications (e.g. expert_claim_near_"
          "zero_duration on an otherwise strong profile), the detector is "
          "working as intended. If most look like ordinary low-experience "
          "noise (the dominant check is skill_duration_exceeds_experience "
          "on candidates with years_of_experience under ~2), that check's "
          "ratio in ranker/config.py (HONEYPOT_SKILL_DURATION_OVERAGE_RATIO) "
          "may need to go even higher than its current 2.5x.")


if __name__ == "__main__":
    main()
