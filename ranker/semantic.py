"""
Semantic fit via TF-IDF cosine similarity.

Why TF-IDF and not a neural embedding model: see README.md "Why TF-IDF
instead of sentence-transformers." Short version — it needs zero network
access at any stage (including pre-computation), runs in milliseconds at
100K-document scale, and the dataset's actual hard cases (keyword-stuffed
skills tags, title mismatches) aren't solved by better semantics anyway —
they're solved by cross-referencing claims against corroborating evidence,
which is features.py's job. This module's only purpose is to catch the
*other* named trap: a candidate whose career narrative clearly describes
doing the work, even if their skills tags don't spell out "RAG" or
"Pinecone."

The architecture keeps this as a clean swap point: replace
`build_candidate_text` + `TfidfVectorizer` with a sentence-transformer
encoder + cosine similarity and nothing else in the pipeline needs to
change, if a real embedding model is available in your own dev environment
during the (untimed) pre-computation step.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config as cfg


def build_candidate_text(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {}) or {}
    career_history = candidate.get("career_history", []) or []
    parts = [profile.get("summary", ""), profile.get("headline", "")]
    for h in career_history:
        parts.append(h.get("description", "") or "")
        parts.append(h.get("title", "") or "")
    return " ".join(p for p in parts if p)


def compute_semantic_scores(candidate_texts: List[str], jd_text: str = cfg.JD_CORE_TEXT) -> np.ndarray:
    """Return an array of semantic-fit scores in [0, 1], one per candidate,
    aligned to the order of `candidate_texts`.

    The TF-IDF vocabulary is fit on the JD text + all candidate texts
    together, so the JD's own vocabulary is guaranteed to be represented
    even if it's rare across the candidate pool.
    """
    n = len(candidate_texts)
    if n == 0:
        return np.array([])

    corpus = [jd_text] + list(candidate_texts)
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(corpus)
    jd_vec = tfidf[0:1]
    cand_vecs = tfidf[1:]
    raw_sims = cosine_similarity(cand_vecs, jd_vec).ravel()

    return _rescale_to_unit_range(raw_sims)


def _rescale_to_unit_range(raw_sims: np.ndarray) -> np.ndarray:
    """Cosine similarities against a JD query over a large sparse TF-IDF
    space tend to cluster in a narrow low range. Rescale using robust
    percentiles so the realistic observed spread fills [0, 1] — this keeps
    the semantic component's *weight* (WEIGHT_SEMANTIC) meaningful relative
    to the other components instead of being silently crushed near 0.
    """
    if raw_sims.size == 0:
        return raw_sims
    lo = np.percentile(raw_sims, cfg.SEMANTIC_SCORE_RESCALE_PERCENTILE_LOW)
    hi = np.percentile(raw_sims, cfg.SEMANTIC_SCORE_RESCALE_PERCENTILE_HIGH)
    if hi - lo < 1e-9:
        return np.clip(raw_sims, 0, 1)
    scaled = (raw_sims - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)
