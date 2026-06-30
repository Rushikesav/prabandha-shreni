"""
Smoke test: runs the full pipeline against sample_candidates.json (50
rows) and checks the structural invariants the real validate_submission.py
checks for n=100 — adapted here for n=50 since that's all the sample data
provides. Run with: python -m pytest tests/test_demo.py -v
(or just: python tests/test_demo.py)
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ranker.io_utils import load_candidates, write_submission_csv
from ranker.scoring import score_candidates


def _candidates_jsonl_path(tmp_dir: Path) -> Path:
    src = ROOT / "sample_candidates.json"
    with open(src) as f:
        data = json.load(f)
    out = tmp_dir / "sample_candidates.jsonl"
    with open(out, "w") as f:
        for c in data:
            f.write(json.dumps(c) + "\n")
    return out


def test_pipeline_runs_and_output_is_well_formed(tmp_path):
    jsonl_path = _candidates_jsonl_path(tmp_path)
    candidates = load_candidates(str(jsonl_path))
    assert len(candidates) == 50

    rows, timing = score_candidates(candidates, top_k=50, verbose=False)
    assert len(rows) == 50

    out_path = tmp_path / "demo_submission.csv"
    write_submission_csv(rows, str(out_path))

    with open(out_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["candidate_id", "rank", "score", "reasoning"]
        data_rows = list(reader)

    assert len(data_rows) == 50

    ranks = [int(r[1]) for r in data_rows]
    assert sorted(ranks) == list(range(1, 51))

    ids = [r[0] for r in data_rows]
    assert len(set(ids)) == len(ids), "duplicate candidate_id in output"
    import re
    pattern = re.compile(r"^CAND_[0-9]{7}$")
    assert all(pattern.match(i) for i in ids)

    scores = [float(r[2]) for r in data_rows]
    assert all(s1 >= s2 for s1, s2 in zip(scores, scores[1:])), "scores must be non-increasing by rank"

    # rank 1 should be the known strong match in the sample data
    assert data_rows[0][0] == "CAND_0000031", (
        "expected the sample's clearest genuine match (Recommendation Systems "
        "Engineer, Swiggy) to rank #1 — if this fails, a scoring change likely "
        "regressed something; inspect ranker/config.py weights"
    )

    reasonings = [r[3] for r in data_rows]
    assert len(set(reasonings)) > 45, "reasoning text should vary across candidates, not be templated"

    print(f"OK: {len(rows)} rows, timing={timing.steps}")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_pipeline_runs_and_output_is_well_formed(Path(d))
    print("All smoke-test assertions passed.")
