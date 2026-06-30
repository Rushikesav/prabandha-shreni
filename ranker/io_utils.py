"""
Loading the candidate pool and writing the submission CSV.

Kept deliberately boring: streaming JSONL parse (works for the gzipped
100K-row pool without holding the raw text twice in memory), and a CSV
writer that matches submission_spec.md Section 2 exactly (column order,
header spelling, UTF-8).
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Iterator, Dict, Any, List, Sequence

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]


def iter_candidates(path: str) -> Iterator[Dict[str, Any]]:
    """Yield candidate dicts one at a time from a .jsonl or .jsonl.gz file.

    Streaming rather than `json.load`-ing the whole file: candidates.jsonl
    is ~465MB uncompressed, and we don't need it all in memory as text —
    we turn each line into a feature row and discard the raw dict as we go.
    """
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    mode = "rt"
    with opener(p, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_candidates(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    """Materialize candidates into a list (used for the 50-candidate demo
    and for any pool small enough to fit comfortably in 16GB — the full
    100K-row pool at ~465MB uncompressed easily fits this way too)."""
    out = []
    for i, c in enumerate(iter_candidates(path)):
        if limit is not None and i >= limit:
            break
        out.append(c)
    return out


def write_submission_csv(rows: Sequence[Dict[str, Any]], out_path: str) -> None:
    """Write the final submission CSV.

    `rows` must already be in final rank order (rank 1 first) and each row
    must contain exactly the REQUIRED_HEADER keys. We don't re-sort here —
    sorting and tie-breaking is scoring.py's job, deliberately kept out of
    the IO layer so the two concerns don't get tangled.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REQUIRED_HEADER)
        for row in rows:
            writer.writerow([row[col] for col in REQUIRED_HEADER])
