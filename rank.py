#!/usr/bin/env python3
"""
Redrob Hackathon — ranking CLI.

Single reproduce command (per submission_spec.md Section 10.3):

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Also accepts the gzipped pool directly:

    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

No network access, no GPU, and no calls to any hosted model are made
anywhere in this script or in the `ranker` package it imports — the only
non-stdlib dependencies are numpy, pandas (unused directly but listed for
future extension), and scikit-learn's TfidfVectorizer, all of which run
fully offline on CPU. See README.md for the full methodology writeup and
measured runtime.
"""

from __future__ import annotations

import argparse
import sys
import time

from ranker.io_utils import load_candidates, write_submission_csv
from ranker.scoring import score_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates for the Senior AI Engineer JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Path to write the output submission CSV")
    parser.add_argument("--top_k", type=int, default=100, help="Number of ranked rows to output (default: 100)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional: only read the first N candidates from the pool (debugging/demo use only)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress/timing output")
    args = parser.parse_args()

    t_start = time.perf_counter()
    if not args.quiet:
        print(f"[info] loading candidates from {args.candidates} ...")
    candidates = load_candidates(args.candidates, limit=args.limit)
    if not args.quiet:
        print(f"[info] loaded {len(candidates)} candidates in {time.perf_counter() - t_start:.2f}s")

    rows, timing = score_candidates(candidates, top_k=args.top_k, verbose=not args.quiet)

    write_submission_csv(rows, args.out)

    total = time.perf_counter() - t_start
    if not args.quiet:
        print(f"[info] wrote {len(rows)} rows to {args.out}")
        print(f"[info] end-to-end runtime: {total:.2f}s "
              f"({'WITHIN' if total <= 300 else 'EXCEEDS'} the 5-minute ranking-step budget)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
