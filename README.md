# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/YOUR_USERNAME/YOUR_REPO/main?filepath=sandbox_demo.ipynb)

A hybrid ranker for the "Senior AI Engineer — Founding Team" JD: rule-based
JD-disqualifier checks + trust-weighted skill scoring + TF-IDF semantic
similarity over free-text career narrative + a behavioral-signal
availability multiplier + an arithmetic honeypot gate.

**Two live sandboxes, for redundancy:** a Colab notebook (link in
`submission_metadata.yaml`) and the Binder badge above, which launches a
zero-login live session directly from this repo. Both run `rank.py`
against the bundled 50-candidate `sample_candidates.json` — per
`submission_spec.md` Section 10.5 ("Accept a small candidate sample
(&le;100 candidates) as input"), this is the sandbox's intended scope, not
a reduced version of the real run. The actual competition submission is
produced separately by running the identical `rank.py` command against
the real 100K-candidate pool, off-sandbox, and uploaded directly via the
portal.


## Reproduce

```
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Also accepts the gzipped pool directly (`--candidates ./candidates.jsonl.gz`).
No network access and no GPU are used anywhere in this pipeline. Runtime on
the 50-candidate demo sample is ~0.04s end-to-end; see "Performance at 100K
scale" below for how that's expected to extrapolate.

## Why this architecture

Three approaches were weighed before writing any scoring code:

**Pure rule-based scorecard.** Fast, fully explainable, trivially meets the
compute budget. Weak spot: a fixed keyword taxonomy can miss a candidate
who describes "shipped a recommendation system" without ever saying
"embeddings" or "retrieval."

**Pure neural embeddings (sentence-transformers).** Best semantic recall in
theory, but it doesn't solve this dataset's actual hard cases. The
keyword-stuffed candidate's skill tags *are* real AI words ("NLP,"
"Fine-tuning LLMs," "Speech Recognition") — an embedding model still scores
that profile as semantically AI-adjacent. The trap isn't solved by
understanding language better; it's solved by cross-checking the claim
against corroborating evidence (duration, endorsements, platform
assessment scores, actual career narrative). It also adds a model-download
dependency at exactly the point (Stage 3 Docker reproduction) where the
spec is strictest, for a benefit it doesn't clearly deliver here.

**Hybrid (chosen).** The JD frames its hardest cases as *structural*, not
*semantic*: "consulting-only career," "title says Marketing Manager,"
"expert with 0 duration," "hasn't logged in for 6 months." Those need
rules and corroboration, not vector math. TF-IDF over free-text career
descriptions still gives "beyond keywords" generalization for the
candidate-who-never-says-RAG-but-built-one case, with zero network
dependency and millisecond-scale inference at 100K-row scale.

## Score composition

```
fit_score = 0.30 * title_relevance
          + 0.25 * semantic_fit          (TF-IDF cosine similarity vs JD)
          + 0.20 * skill_trust           (trust-weighted, anti-stuffing)
          + 0.15 * experience_fit        (soft band, not a hard cutoff)
          + 0.10 * location_fit
          + rule_adjustments             (bounded, JD disqualifiers)

final_score = fit_score * availability_multiplier   (behavioral signals)
```

Honeypot-suspect candidates are excluded from the top-K selection entirely
before this combination is even sorted — not down-weighted, excluded —
since the spec scores honeypots at relevance tier 0 and the Stage 3 filter
is a hard rate cutoff (>10% of top 100), not a soft preference.

Weight rationale is documented inline in `ranker/config.py`, tied back to
specific JD language for each one (title is called out twice as the
decisive signal; skills are explicitly called "teachable"; semantic
narrative fit is the JD's stated central ask; experience-band is
explicitly "a range, not a requirement"; location is a real but flexible
preference).

### JD disqualifier rules implemented

| Rule | Detection | Source |
|---|---|---|
| Consulting-only career | every `career_history` company matches a known IT-services-firm list | explicit |
| Research-only, no production | every `career_history.industry` tag is research/academia | explicit, proxy — schema has no dedicated field |
| Recent-LLM-wrapper-only | only LangChain/RAG/prompt-engineering tags, all <12mo, no pre-LLM-era ML skill ≥30mo | explicit |
| CV/speech/robotics without NLP | ≥3 CV/speech/robotics skill tags, 0 NLP/IR tags | explicit |
| Title-chaser pattern | ≥3 roles, avg tenure <18mo, **and** at least one title carries a Senior/Staff/Principal/Lead marker | explicit (see calibration note below) |
| Architect/manager, not hands-on | current title contains Architect/Director/VP/Head of/Eng. Manager | explicit, proxy — schema can't observe code-authorship recency directly |
| Closed-source-only | ≥5 yrs experience, no GitHub linked (`github_activity_score == -1`) | explicit, weakest proxy available |
| Self-described hobbyist | summary contains hedging language ("self-learner," "side project," etc.) **and** no career title reflects real ML/AI work | found during validation, see below — not explicitly named in the JD but the same anti-stuffing spirit |
| Skill self-rating vs. platform assessment mismatch | any advanced/expert skill claim where `skill_assessment_scores` for that exact skill is <50 | explicit (the dataset's main keyword-stuffing trap) |

**Title-chaser calibration note:** an earlier version flagged *any*
candidate with 3+ roles averaging under 18 months, regardless of what the
titles actually said. That incorrectly flagged the strongest candidate in
the 50-row sample (`CAND_0000031`, a Recommendation Systems Engineer whose
four roles — Applied ML Engineer → NLP Engineer → Search Engineer →
Recommendation Systems Engineer — are lateral IC specialist moves, not
seniority-ladder climbing). The JD's actual complaint is specifically
about "optimizing for Senior → Staff → Principal titles," so the rule now
also requires an explicit seniority-ladder marker in at least one title
before firing.

## Honeypot detection

**Update after running against the real 100K pool** (this superseded the
original calibration note below — kept for the record of how the
thinking evolved): the first real-pool run excluded 591 candidates.
Breaking that down by which check fired, with population base rates,
was decisive:

| check | count | rate | verdict |
|---|---|---|---|
| `expert_claim_near_zero_duration` | 84 | 0.08% | **real** — confirmed by manual inspection |
| `single_role_duration_exceeds_experience` | 21 | 0.02% | inconclusive, demoted to corroboration-only |
| `career_history_sum_exceeds_experience` | 24 | 0.02% | inconclusive, demoted to corroboration-only |
| `skill_duration_exceeds_experience` | 2425 | 2.43% | **noise** — confirmed by manual inspection |
| `signup_after_last_active` | 7496 | 7.50% | **noise** — removed entirely |

A check firing on 7.5% of an entire 100,000-candidate pool cannot be
tracking a "~80 designed honeypots" signal — that's two orders of
magnitude too common. Manually reading the highest-risk candidates
confirmed exactly why: candidates with ~12 months total experience and
*one* skill among 10-20 listed happening to have a duration_months in the
30-90 range, across totally unrelated titles (Civil Engineer, HR
Manager, Sales Executive — no AI-stuffing pattern, no honeypot
fingerprint, just noisy data on short careers).

`expert_claim_near_zero_duration` told a completely different story:
manually reading every one of the 21 candidates whose risk hit the cap
*and* triggered this check showed every single one had 3-5 different
skills, **all** independently rated "expert" with `duration_months == 0`
exactly — across unrelated titles and experience levels. That's a clean,
mechanically constructed signature, essentially a direct match to the
spec's own example, not something noisy random sampling produces by
chance across multiple independent draws simultaneously.

**The fix:** `expert_claim_near_zero_duration` is now the only check
trusted to independently cross the exclusion threshold. The other three
structural checks had their weights cut so that even all three firing
together can't reach the threshold alone — they only matter as
corroboration once the primary signal has already fired.
`signup_after_last_active` was removed from scoring entirely. Full
before/after numbers and the exact reasoning live in
`ranker/honeypot.py`'s module docstring.

This is a precision-over-recall choice, made deliberately: the new
detector likely under-catches relative to the true ~80 (some honeypots
may use a construction this schema can't observe directly — the literal
"founded 3 years ago" example needs a company-founding-date field that
doesn't exist), but it stops excluding hundreds of ordinary candidates
over one noisy field. Re-run `dump_honeypot_flags.py` against any new
candidate pool before trusting these exact weights — this calibration is
a finding from one specific 100K-row pool, not a universal constant.

<details>
<summary>Original calibration note (50-row sample, superseded above)</summary>

The first pass used a 1.5x overage ratio for the skill-duration check and
flagged 8 of 50 sample candidates — all low-experience, off-target-title
profiles where an ordinary skill (PowerPoint, AWS, SQL) had a
`duration_months` loosely exceeding total tenure. Raised to 2.5x at the
time. This local fix turned out to be insufficient once tested against
the real 100K pool — see above.
</details>



## Reasoning generation

Every fact in a reasoning string traces back to a literal field value
computed during feature extraction — nothing is invented. Phrasing is
selected from small template pools using a hash of `candidate_id` as the
random seed, so the same candidate always gets the same reasoning text
(reproducible) but different candidates get visibly different sentence
shapes rather than name-swapped templates. Tone intensity (confident vs.
hedged) is chosen from the candidate's rank *band* within the selected
top-K, not just raw score, so a rank-3 row reads confidently and a rank-95
row reads as a marginal pick. Concerns are only ever mentioned when a real
flag fired — never added for variety.

## Data quality observations (found during validation, not assumed upfront)

- **Keyword-stuffing tell confirmed in real data:** `CAND_0000001`'s own
  summary says *"I'm building competence on the ML side"* (i.e., not there
  yet) while her skills array claims "advanced" NLP, Speech Recognition,
  and Fine-tuning LLMs — and the platform's own `skill_assessment_scores`
  for those same skills sit at 38–54/100, confirming the self-rating is
  inflated. This is exactly the signal `skill_credibility_check` (in
  `ranker/features.py`) is built to catch, and it's the mechanism, not the
  TF-IDF semantic score, that catches it — her semantic similarity is
  still reasonably high since the vocabulary genuinely overlaps with the
  JD.
- **`career_history.description` appears to be drawn from a template pool
  independent of `career_history.title`.** Multiple candidates in the
  50-row sample have *verbatim-identical* description text attached to
  different titles and companies (e.g. the exact phrase "Business analyst
  at a consulting firm, working primarily with retail and CPG clients..."
  appears under a "Customer Support" title at one company and an
  "Accountant" title at another). This means the TF-IDF semantic score can
  pick up vocabulary that has nothing to do with what the title or
  industry suggests, purely from template-pool collision — checked
  directly: none of the off-target "Operations Manager / Marketing
  Manager / Business Analyst" decoy profiles in the sample broke into the
  top third of the ranking despite this, because `WEIGHT_TITLE` (0.30) and
  the near-zero skill-trust score for these profiles dominate the
  combination — but it's worth knowing the semantic component alone is
  noisier on this dataset than it would be on real-world resumes, where
  title and description are authored together.
- **A meaningful fraction of the pool (most of the 50-row sample) appears
  to follow a single templated "off-target professional" archetype** —
  near-identical summary text across many candidates ("Lately I've been
  curious about how AI tools could augment my work — I've experimented
  with ChatGPT..."), different surface titles, same underlying shallow-AI
  framing. Consistent with the JD's own framing that genuine matches are
  sparse in a large pool; only one candidate in the 50-row sample
  (`CAND_0000031`) reads as an unambiguous strong fit.

## Known limitations (proxies, not direct observations)

The schema doesn't give direct signal for every disqualifier the JD
names. Three rules are explicitly best-effort proxies, flagged here rather
than quietly assumed solid:

- **"Hasn't written production code in 18 months"** — approximated by
  current-title keywords (Architect/Director/VP/Head of/Engineering
  Manager). A genuinely hands-on Staff/Principal engineer with one of
  these titles would be mis-flagged; there's no field for actual code
  authorship recency.
- **"Research-only, no production deployment"** — approximated by
  `career_history.industry` tags containing research/academia. A
  candidate at a corporate research lab tagged with a generic industry
  string would not be caught.
- **"Closed-source proprietary work, 5+ years, no external validation"**
  — approximated by `github_activity_score == -1` (no GitHub linked).
  This conflates "no GitHub" with "no external validation" — a candidate
  who publishes papers or speaks at conferences instead would be
  incorrectly flagged, but the schema has no field for that.
- **"Framework-enthusiast with no scars"** (someone whose presence on
  GitHub/blogs is all tutorials, never production debugging) — **not
  implemented.** Nothing in this schema (no blog/GitHub-content field, just
  a numeric activity score) supports detecting this distinction, and a
  fabricated heuristic here would be worse than admitting the gap.

## Performance at 100K scale

Feature extraction is a single pass per candidate over small, bounded
lists (≤10 skills typical, ≤10 career_history entries, ≤5 education
entries) — no nested loops over the full pool. The dominant cost at 100K
rows is the TF-IDF fit+transform, which scikit-learn implements in
optimized sparse-matrix C code; fitting and transforming ~100K
short-to-medium documents typically completes in low tens of seconds on a
single CPU core, well inside the 5-minute budget even stacked with
feature extraction and the final sort. Measured on the 50-candidate demo:
0.01s feature extraction, 0.03s TF-IDF, <0.01s combine+rank — see
`rank.py`'s own timing output, which is printed on every run specifically
so this can be re-verified directly against the real 100K pool rather than
taken on faith.

## Audit against the source documents (second pass)

After the first working version was built and tested, every guideline
document was re-read fresh against the actual code — not from memory —
specifically looking for JD/schema language that wasn't yet reflected
anywhere in `ranker/`. Four real gaps were found and fixed:

1. **"6-8 years total experience, of which 4-5 are in applied ML/AI roles
   at product companies (not pure services)"** (JD, "how to read between
   the lines") was only half-implemented — `experience_fit_score` checked
   total years, not composition. Now blends a total-years-band score with
   a separate relevant-tenure-band score (months spent in a core/adjacent
   title at a non-consulting company, ideal band 48-60 months), 65/35.

2. **Per-skill `endorsements`** (schema field, parallel to
   `skill_assessment_scores`) wasn't used anywhere. Most skills in most
   profiles have no platform assessment at all, so the original
   trust-discount could only catch overclaiming when an assessment
   happened to exist. Added a softer fallback discount
   (`ENDORSEMENT_ZERO_DISTRUST_MULTIPLIER`) for advanced/expert claims
   with zero endorsements *and* no assessment — checked against the
   sample data and confirmed it fires rarely (1/50) rather than
   over-triggering, and correctly does *not* fire on skills that do have
   real endorsement counts even without a platform assessment.

3. **`avg_response_time_hours`** (redrob_signals_doc.md lists it right
   next to `recruiter_response_rate` as a paired signal) wasn't used —
   only the response *rate* was, not response *speed*. A candidate can
   score well on one while scoring poorly on the other. Added as a
   parallel sub-multiplier in `availability.py`.

4. **Consulting-firm detection relied entirely on a name list**, which
   will always be incomplete against a 100K-candidate pool (this is
   exactly how "Mindtree" got missed initially — see the data-quality
   notes above). The schema's own `industry` field tagging a company "IT
   Services" is a direct structural match for the JD's named category
   ("TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc."), so
   `is_consulting_only_career` now matches on name-list OR industry tag.

**A correction made during this audit, for the record:** the original
hypothesis was that fix #2 would explain why `CAND_0000032` (the .NET
Developer claiming "advanced" Embeddings) scored higher than expected.
Re-checking her actual data after implementing the fix showed her
Embeddings skill has 30 endorsements — genuine corroboration — so the new
check correctly does *not* discount her. Her score is already
appropriately suppressed by the separate self-described-hobbyist rule
(her own summary explicitly hedges her AI exposure as side-project level)
and her weak title tier. The endorsement-fallback fix is real and
verified to fire correctly elsewhere (confirmed on `CAND_0000043`,
claiming "advanced" Reinforcement Learning with zero endorsements and no
assessment) — it just wasn't the mechanism behind that particular
candidate's score, and the record should reflect that rather than
overstate the fix's effect on the example that originally motivated it.

### Schema fields deliberately not used

For transparency (and because "why didn't you use X" is a fair interview
question): the following schema fields were considered and intentionally
excluded, not overlooked —

- **`education` (institution, degree, tier, field_of_study)** — the JD
  never mentions pedigree, degree, or institution anywhere, and explicitly
  signals a culture that prioritizes shipped work over credentials ("if
  you've spent your career at Google or Meta... this isn't it"). Adding a
  pedigree signal the JD doesn't ask for would be scope creep with real
  bias risk.
- **`certifications`, `languages`** — no JD signal calls for either.
- **`company_size`** (current or in career_history) — considered as a
  proxy for "shipped at meaningful scale," but company headcount doesn't
  reliably indicate the scale of any *individual's* system (a candidate
  at a 10,000-person company could have worked on a tiny internal tool;
  someone at an 11-person startup could have built something massive).
  Too weak/noisy a proxy to add responsibly.
- **`expected_salary_range_inr_lpa`** — the JD's "On location, comp, and
  logistics" section header mentions comp but the body text never states
  a target band to compare against, so there's nothing to score against.
- **`connection_count`, `profile_completeness_score`,
  `profile_views_received_30d`, `search_appearance_30d`,
  `saved_by_recruiters_30d`** — these measure how much *others* already
  notice a candidate (market desirability), not whether *this* candidate
  is reachable/available, which is the specific framing the JD and
  signals doc both use ("for hiring purposes, not actually available").
  Folding in popularity signals would reward already-popular candidates
  for a reason the JD doesn't ask for.
- **`applications_submitted_30d`, `offer_acceptance_rate`** — plausibly
  relevant in either direction (more applications could mean actively
  job-hunting, or could mean spray-applying; a low offer-acceptance rate
  could mean hard-to-close, or could mean appropriately selective) without
  a clear signal from the JD on which reading to apply. Left out rather
  than guessing.



```
rank.py                       # CLI entrypoint — the single reproduce command
ranker/
  config.py                   # every weight/threshold/taxonomy, with rationale comments
  io_utils.py                 # streaming JSONL(.gz) load, CSV write
  features.py                 # title/skill/experience/location scoring, JD disqualifier rules
  semantic.py                 # TF-IDF semantic similarity vs JD core text
  honeypot.py                 # arithmetic honeypot risk detection
  availability.py             # behavioral-signal multiplier
  scoring.py                  # orchestration: combine, gate, sort, rank
  reasoning.py                # grounded, varied, rank-consistent reasoning text
tests/
  test_demo.py                # smoke test against sample_candidates.json
submission_metadata.yaml      # filled per submission_metadata_template.yaml
```

## Demo limitation

`sample_candidates.json` (the only candidate data shipped with this
bundle) has 50 rows, not the real 100K-row pool — `candidates.jsonl.gz`
itself wasn't part of the files used to build this. The exact same
`rank.py` command works unchanged against the real pool; the demo here
runs with `--top_k 50` (the full sample size) instead of `--top_k 100`
purely because there are only 50 rows to rank, and is provided as
`demo_submission.csv` for inspection, not as the actual competition
submission.
