"""
All tunable constants live here, in one place, so the scoring logic in the
rest of the package stays readable and every number used in a ranking
decision can be found, justified, and defended in one sitting (Stage 5
interview readiness was a design goal, not an afterthought).

Every threshold below is commented with *why* it has the value it has,
tied back to specific lines in job_description.md / submission_spec.md /
redrob_signals_doc.md. Where a number is a genuine judgment call rather
than something derivable from the JD text, that's flagged explicitly —
those are the first things to revisit once this runs against the real
100K-candidate pool instead of the 50-candidate sample.
"""

# ---------------------------------------------------------------------------
# Top-level fit-score weights (must sum to 1.0)
#
# Rationale, in JD's own words:
#   - title is called out twice as the decisive signal ("A candidate who
#     has all the AI keywords listed as skills but whose title is
#     'Marketing Manager' is not a fit") -> highest weight.
#   - "the right answer involves reasoning about the gap between what the
#     JD says and what the JD means" -> semantic narrative fit is the
#     JD's explicit core ask, weighted second.
#   - skills are explicitly called "teachable" and de-prioritized
#     relative to title/trajectory ("Skills are teachable; the rest
#     mostly isn't") -> weighted third, and trust-corrected rather than
#     counted.
#   - experience-band is explicitly "a range, not a requirement" -> modest
#     weight, soft falloff rather than a hard cutoff.
#   - location is a real preference but JD says "flexible," "welcome to
#     apply" for a list of cities, and "case-by-case" for outside India
#     -> lowest of the five.
# ---------------------------------------------------------------------------
WEIGHT_TITLE = 0.30
WEIGHT_SEMANTIC = 0.25
WEIGHT_SKILL_TRUST = 0.20
WEIGHT_EXPERIENCE = 0.15
WEIGHT_LOCATION = 0.10

assert abs((WEIGHT_TITLE + WEIGHT_SEMANTIC + WEIGHT_SKILL_TRUST
            + WEIGHT_EXPERIENCE + WEIGHT_LOCATION) - 1.0) < 1e-9

# ---------------------------------------------------------------------------
# Rule-adjustment bounds (additive, applied after the weighted sum above)
# Penalties are negative, bonuses positive. Capped so no single rule can
# by itself flip a strong candidate to the bottom or a weak one to the top
# — these are meant to *adjust* the fit score, not override it outright.
# (Outright override is reserved for the honeypot gate, see below.)
# ---------------------------------------------------------------------------
PENALTY_CONSULTING_ONLY = -0.22          # JD: explicit disqualifier
PENALTY_RESEARCH_ONLY = -0.25            # JD: explicit, strongest-worded disqualifier
PENALTY_RECENT_LLM_ONLY = -0.18          # JD: explicit disqualifier
PENALTY_CV_SPEECH_NO_NLP = -0.15         # JD: explicit disqualifier
PENALTY_TITLE_CHASER = -0.15             # JD: explicit disqualifier
PENALTY_ARCHITECT_NO_CODE = -0.10        # JD: explicit, but we can only proxy it from title
PENALTY_CLOSED_SOURCE_ONLY = -0.08       # JD: explicit, weakest signal available in schema
PENALTY_SELF_DESCRIBED_HOBBYIST = -0.18  # see features.is_self_described_hobbyist docstring
RULE_ADJUSTMENT_FLOOR = -0.45            # multiple disqualifiers can stack, but not erase the rank entirely before sort

# ---------------------------------------------------------------------------
# Title taxonomy. Matched against current_title AND every career_history
# title (a candidate whose *current* title is generic but whose history
# shows real ranking/retrieval titles should still get credit — and vice
# versa, a candidate who is NOW in the right seat matters most).
# Matching is case-insensitive substring matching against the lowercased
# title string.
# ---------------------------------------------------------------------------
TITLE_TIER_CORE = [   # weight 1.00 — directly the role's own mandate
    "recommendation systems engineer", "recommendation engineer",
    "search engineer", "search relevance", "ranking engineer",
    "information retrieval", "retrieval engineer", "nlp engineer",
    "machine learning engineer", "ml engineer", "ai engineer",
    "applied scientist", "applied ml engineer", "applied machine learning",
    "research engineer",
]
TITLE_TIER_ADJACENT = [   # weight 0.55 — plausible feeder roles
    "data scientist", "data engineer", "backend engineer",
    "software engineer", "platform engineer", "ml platform",
    "research scientist",
]
TITLE_TIER_WEAK = [   # weight 0.30 — technical but far from the mandate
    "devops engineer", "cloud engineer", "full stack developer",
    "frontend engineer", "mobile developer", "qa engineer",
    "java developer", ".net developer", "site reliability",
]
TITLE_TIER_OFF_TARGET_BASE = 0.05   # everything else (HR Manager, Accountant, etc.)

# Architect / tech-lead proxy for the "hasn't written production code in
# 18 months" disqualifier. We cannot observe code-authorship recency from
# this schema, so title is used as an explicit, documented proxy — flagged
# as a best-effort heuristic, not a hard rule, see features.py.
ARCHITECT_TITLE_MARKERS = [
    "architect", "engineering manager", "director", "vp ", "vp engineering",
    "head of engineering", "head of ai", "head of data", "principal engineer",
]

# ---------------------------------------------------------------------------
# Skill taxonomy for trust-weighted scoring ("things you absolutely need").
# Matched case-insensitively against skill["name"].
# ---------------------------------------------------------------------------
SKILLS_EMBEDDINGS_RETRIEVAL = [
    "embeddings", "embedding", "sentence transformers", "sentence-transformers",
    "openai embeddings", "bge", "e5", "dense retrieval", "semantic search",
]
SKILLS_VECTOR_DB = [
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "hybrid search",
]
SKILLS_PYTHON = ["python"]
SKILLS_EVAL_FRAMEWORKS = [
    "ndcg", "mrr", "map", "a/b test", "ab testing", "offline-online correlation",
    "evaluation framework", "learning-to-rank", "learning to rank", "ranking evaluation",
]
# "Things we'd like but won't reject you for" — smaller weight, still tracked.
SKILLS_LLM_FINE_TUNING = ["lora", "qlora", "peft", "fine-tuning llms", "fine-tuning", "fine tuning"]
SKILLS_LEARNING_TO_RANK = ["xgboost", "lightgbm", "learning to rank", "ltr", "neural ranking"]

CORE_SKILL_GROUPS = {
    "embeddings_retrieval": (SKILLS_EMBEDDINGS_RETRIEVAL, 1.0),
    "vector_db": (SKILLS_VECTOR_DB, 1.0),
    "python": (SKILLS_PYTHON, 0.8),
    "eval_frameworks": (SKILLS_EVAL_FRAMEWORKS, 1.0),
    "llm_fine_tuning": (SKILLS_LLM_FINE_TUNING, 0.5),
    "learning_to_rank": (SKILLS_LEARNING_TO_RANK, 0.5),
}

PROFICIENCY_WEIGHT = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
SKILL_DURATION_CAP_MONTHS = 24   # duration credit saturates at 2 years of use
ASSESSMENT_TRUST_LOW_SCORE = 50.0   # below this, a self-rated advanced/expert claim is discounted
ASSESSMENT_DISTRUST_MULTIPLIER = 0.45   # multiplier applied when self-rating contradicts assessment
# Per-skill `endorsements` exists in the schema specifically as a second,
# independent corroboration signal alongside skill_assessment_scores — but
# skill_assessment_scores is sparse (most skills in most profiles have no
# entry at all), so most self-rated claims have nothing to be checked
# against. When no assessment exists, fall back to endorsements: a claimed
# advanced/expert skill with zero social corroboration AND no platform
# assessment to vouch for it either is weaker evidence of overclaiming
# than a confirmed low assessment score (some genuine experts just never
# ask colleagues for endorsements), so the discount here is intentionally
# softer than ASSESSMENT_DISTRUST_MULTIPLIER.
ENDORSEMENT_ZERO_DISTRUST_MULTIPLIER = 0.75

# CV / Speech / Robotics taxonomy, for the "no significant NLP/IR exposure" disqualifier
SKILLS_CV_SPEECH_ROBOTICS = [
    "image classification", "computer vision", "object detection", "gans",
    "speech recognition", "text-to-speech", "tts", "asr", "robotics",
    "slam", "autonomous", "ocr", "image segmentation", "pose estimation",
]
SKILLS_NLP_IR = [
    "nlp", "natural language processing", "information retrieval", "embeddings",
    "retrieval", "ranking", "search", "llm", "transformers", "bert", "gpt",
    "text classification", "named entity recognition", "ner", "topic modeling",
    "fine-tuning llms", "sentence transformers", "hugging face transformers",
]
CV_SPEECH_DOMINANCE_MIN_COUNT = 3   # need >= this many CV/speech/robotics skills to trigger

# "Recent LangChain-only" disqualifier taxonomy
SKILLS_RECENT_LLM_WRAPPER = [
    "langchain", "llamaindex", "openai api", "prompt engineering", "rag", "chatgpt api",
]
SKILLS_PRE_LLM_ERA_PROXY = [
    "machine learning", "statistical modeling", "scikit-learn", "xgboost",
    "lightgbm", "feature engineering", "deep learning", "computer vision",
    "nlp", "data science", "recommendation", "ranking", "search",
    "information retrieval",
]
PRE_LLM_MIN_DURATION_MONTHS = 30   # ~2.5 yrs of use suggests skill predates the 2023 LLM wave
RECENT_LLM_MAX_DURATION_MONTHS = 12

# ---------------------------------------------------------------------------
# Consulting-only-career disqualifier. Matched against career_history
# company names, case-insensitive, substring match. JD explicitly names a
# few and says "etc." — list extended with other well-known Indian IT
# services majors that behave the same way for this purpose.
# Rule only fires if EVERY career_history entry matches this list (JD: "if
# you're currently at one of these but have prior product-company
# experience, that's fine").
# ---------------------------------------------------------------------------
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "mindtree", "ltimindtree", "hcl", "hcltech",
    "ibm services", "genpact", "mphasis", "l&t infotech",
    "persistent systems", "zensar",
]

# Research-only disqualifier proxy (schema has no explicit "research lab"
# field; this checks career_history `industry` tags as a best-effort proxy).
RESEARCH_ONLY_INDUSTRY_TAGS = ["research", "academia", "academic", "research institute"]

# Self-described-hobbyist disqualifier. Caught a real failure mode while
# testing against the sample data: a candidate's own free-text summary can
# explicitly hedge their AI exposure ("self-learner level," "side project")
# while still claiming an "advanced" skill tag that has no corresponding
# skill_assessment_scores entry to contradict it (the credibility check in
# features.py can only catch a mismatch when an assessed score exists).
# TF-IDF similarity also can't distinguish "I built a production RAG
# system" from "I built a small RAG side project" — same vocabulary,
# opposite substance — so this phrase-based check exists specifically to
# catch the case the other two mechanisms both miss. Deliberately
# conservative: requires the hedging language in their OWN words AND the
# absence of any real ML/AI title in their history, so a genuinely senior
# practitioner who happens to mention a weekend project isn't penalized.
SELF_DESCRIBED_SHALLOW_PHRASES = [
    "self-learner", "self learner", "self-taught", "side project",
    "personal project", "online course", "hobby project", "started learning",
    "just started", "playing with", "experimenting with", "in my spare time",
    "still learning", "beginner level", "exploring ai", "exploring ml",
]

# Closed-source-only disqualifier proxy: 5+ years experience AND no GitHub
# linked (github_activity_score == -1 per schema) is the only observable
# proxy available — explicitly weak, see features.py docstring.
CLOSED_SOURCE_MIN_YEARS = 5.0
CLOSED_SOURCE_GITHUB_SCORE_THRESHOLD = 0.0   # <=0 (schema: -1 means "no GitHub linked")

# Title-chaser disqualifier: 3+ roles, average tenure under 18 months
TITLE_CHASER_MIN_ROLES = 3
TITLE_CHASER_MAX_AVG_TENURE_MONTHS = 18

# ---------------------------------------------------------------------------
# Experience-band scoring (soft, not a hard cutoff — JD: "a range, not a
# requirement... we'll seriously consider candidates outside the band if
# other signals are strong").
# ---------------------------------------------------------------------------
EXPERIENCE_IDEAL_LOW = 6.0     # JD "how to read between the lines": 6-8 ideal
EXPERIENCE_IDEAL_HIGH = 8.0
EXPERIENCE_BAND_LOW = 5.0      # JD stated band: 5-9
EXPERIENCE_BAND_HIGH = 9.0
EXPERIENCE_TAPER_PER_YEAR = 0.12   # score lost per year outside the 5-9 band
EXPERIENCE_SCORE_FLOOR = 0.15      # never fully zero out — "strong signals elsewhere" can still matter

# JD "how to read between the lines": ideal candidate has "6-8 years total
# experience, of which 4-5 are in applied ML/AI roles at product companies
# (not pure services)." Total-years-band-fit above only captures the first
# half of that sentence. This blend adds the second half: how much of the
# career was actually spent in a relevant title at a non-consulting company,
# scored against its own ideal band, same soft-taper logic.
RELEVANT_TENURE_IDEAL_LOW_MONTHS = 48    # 4 years
RELEVANT_TENURE_IDEAL_HIGH_MONTHS = 60   # 5 years
RELEVANT_TENURE_TAPER_PER_MONTH = 0.01
RELEVANT_TENURE_SCORE_FLOOR = 0.10
RELEVANT_TENURE_TITLE_MIN_WEIGHT = 0.55   # career_history entry counts as "relevant" if its title tier weight >= this (core or adjacent)
TOTAL_YEARS_BLEND_WEIGHT = 0.65
RELEVANT_TENURE_BLEND_WEIGHT = 0.35

# ---------------------------------------------------------------------------
# Location scoring. JD: Pune/Noida preferred; Hyderabad/Pune/Mumbai/Delhi
# NCR "welcome"; outside India "case-by-case," no visa sponsorship.
# ---------------------------------------------------------------------------
LOCATION_PREFERRED = ["pune", "noida"]
LOCATION_WELCOME = [
    "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
    "faridabad", "ghaziabad", "navi mumbai",
]
LOCATION_SCORE_PREFERRED = 1.0
LOCATION_SCORE_WELCOME = 0.85
LOCATION_SCORE_OTHER_INDIA = 0.55
LOCATION_SCORE_OUTSIDE_INDIA_BASE = 0.20   # no visa sponsorship is a real constraint
LOCATION_RELOCATE_BONUS = 0.15

# ---------------------------------------------------------------------------
# Honeypot detection (arithmetic / structural impossibilities only — no
# semantic guessing). See ranker/honeypot.py module docstring for the full
# real-100K-pool evidence behind this specific weight structure — short
# version: only HONEYPOT_EXPERT_ZERO_DURATION_RISK is trusted to
# independently cross HONEYPOT_RISK_THRESHOLD. The other three are capped
# low enough that all three firing simultaneously (0.12+0.12+0.12=0.36)
# still falls short of the 0.5 threshold on their own — they can only
# matter as corroboration once the primary signal has already fired.
# signup_after_last_active was removed entirely (see honeypot.py) after
# being confirmed to fire on 7.5% of the real 100K pool — two orders of
# magnitude too common to track a ~0.08% true honeypot rate.
# ---------------------------------------------------------------------------
HONEYPOT_EXPERT_ZERO_DURATION_MAX_MONTHS = 3      # "expert" claimed with <= this many months' use
HONEYPOT_EXPERT_ZERO_DURATION_RISK = 0.5          # PRIMARY — only check trusted to independently exclude
HONEYPOT_SKILL_DURATION_OVERAGE_RATIO = 2.5       # skill duration > 2.5x total experience (months)
HONEYPOT_SKILL_DURATION_OVERAGE_RISK = 0.12       # corroboration-only — see honeypot.py docstring
HONEYPOT_CAREER_SUM_OVERAGE_RATIO = 1.3           # sum(career_history durations) > 1.3x stated YOE
HONEYPOT_CAREER_SUM_OVERAGE_RISK = 0.12           # corroboration-only
HONEYPOT_SINGLE_ROLE_OVERAGE_BUFFER_MONTHS = 6    # a single role's duration vs total YOE, with slack
HONEYPOT_SINGLE_ROLE_OVERAGE_RISK = 0.12          # corroboration-only
HONEYPOT_RISK_THRESHOLD = 0.5     # >= this -> excluded from top-K entirely (see scoring.py)

# ---------------------------------------------------------------------------
# Behavioral-signal availability multiplier. JD + signals doc: "down-weight
# appropriately" a perfect-on-paper-but-unreachable candidate. This is a
# *multiplier* on the fit score (necessary-but-not-sufficient), not an
# additive term, so unavailability can meaningfully suppress an otherwise
# strong match without letting pure availability alone buy a top rank.
# ---------------------------------------------------------------------------
RECENCY_BUCKETS_DAYS = [(30, 1.00), (90, 0.85), (180, 0.65), (10**9, 0.45)]
OPEN_TO_WORK_TRUE_MULT = 1.05
OPEN_TO_WORK_FALSE_MULT = 0.90
RESPONSE_RATE_BASE_MULT = 0.85
RESPONSE_RATE_SCALE = 0.30        # final = base + scale * recruiter_response_rate
# avg_response_time_hours is listed in redrob_signals_doc.md right next to
# recruiter_response_rate as a paired signal — they measure different
# things (whether they respond at all vs. how fast when they do), and a
# candidate could score well on one while scoring poorly on the other
# (e.g. 90% response rate but a 9-day average turnaround). Both matter for
# "can we actually talk to them," so both get a sub-multiplier.
RESPONSE_TIME_BUCKETS_HOURS = [(24, 1.05), (72, 1.00), (168, 0.90), (10**9, 0.78)]
NOTICE_PERIOD_BUCKETS_DAYS = [(30, 1.05), (60, 1.00), (90, 0.92), (10**9, 0.85)]
INTERVIEW_COMPLETION_BASE_MULT = 0.90
INTERVIEW_COMPLETION_SCALE = 0.15
VERIFICATION_BONUS_EACH = 0.02    # verified_email, verified_phone, linkedin_connected
AVAILABILITY_MULTIPLIER_FLOOR = 0.45
AVAILABILITY_MULTIPLIER_CEILING = 1.18

# ---------------------------------------------------------------------------
# Semantic (TF-IDF) component. JD core text used as the "query" document.
# Pulled directly from job_description.md "What you'd actually be doing,"
# "Things you absolutely need," and "How to read between the lines"
# sections — the parts that describe substance, not culture/logistics.
# ---------------------------------------------------------------------------
JD_CORE_TEXT = """
Own the intelligence layer of the product: the ranking, retrieval, and
matching systems that decide what recruiters see when they search for
candidates and what candidates see when they search for roles. Audit an
existing BM25 plus rule-based scoring system and identify the highest
leverage fixes. Ship a v2 ranking system using embeddings, hybrid
retrieval, and LLM-based re-ranking that demonstrably improves
recruiter-engagement metrics. Build evaluation infrastructure: offline
benchmarks, online A/B testing, recruiter feedback loops. Drive long-term
architecture for candidate to job description matching at scale. Strong
production experience with embeddings-based retrieval systems such as
sentence-transformers, OpenAI embeddings, BGE, or E5, deployed to real
users, including handling embedding drift, index refresh, and retrieval
quality regression in production. Production experience with vector
databases or hybrid search infrastructure such as Pinecone, Weaviate,
Qdrant, Milvus, OpenSearch, Elasticsearch, or FAISS. Strong Python and
code quality. Hands-on experience designing evaluation frameworks for
ranking systems: NDCG, MRR, MAP, offline-to-online correlation, A/B test
interpretation. Has shipped at least one end-to-end ranking, search, or
recommendation system to real users at meaningful scale, at a product
company rather than a pure services company. Has strong opinions about
hybrid versus dense retrieval, offline versus online evaluation, and when
to fine-tune versus prompt an LLM, grounded in systems actually built and
shipped, not tutorials or demos.
""".strip()

# Cosine-similarity scores from a 100K-document TF-IDF corpus are typically
# small in absolute terms (long-tail vocabulary, sparse overlap). This
# rescaling stretches the realistic observed range into a fuller [0, 1]
# band for combination with the other components, rather than letting raw
# cosine similarities (often < 0.3) silently underweight this component.
SEMANTIC_SCORE_RESCALE_PERCENTILE_LOW = 5
SEMANTIC_SCORE_RESCALE_PERCENTILE_HIGH = 95

RANDOM_SEED = 1729   # for deterministic reasoning-template selection only; never affects ranking math
