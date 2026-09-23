# Changes since `pipeline-frozen-dev`

Every behaviour change made in `backend/` and `scripts/` after the pipeline
was frozen. The frozen components are unchanged: `understand.py`,
`retrieval.py`, `matcher.py`, `generate.py`, the index, and every constraint
record. All changes below live in four files:

- `backend/app/main.py` (the HTTP API)
- `backend/app/postprocess.py` (what happens to results between the frozen
  pipeline and the API response)
- `backend/requirements.txt`
- `scripts/evaluate_generate.py`

This log was compiled from the code and the work sessions, not from git.
Where the tag sits relative to the first version of `main.py` was not
checked. If the tag already contains that version, section 1 is not a
post-freeze change.

**Evidence types used below.**

- **Corpus scan:** a count over all 2,045 constraint records or their 7,939
  residual conditions.
- **Dev row:** one of the 20 dev-split gold rows.
- **Hand-made query:** a query written to exercise a case; it is not in the
  gold set.

No change was tuned on recall, MRR or extraction metrics. Nothing has been
run on the test split.

## Summary

| # | Change | Can it change a status? | Main evidence |
|---|---|---|---|
| 1 | HTTP API | No | 3 dev queries; 12 hand-made scheme-name cases |
| 2 | Certainty sort | No (order only) | 1 dev row (gold_053) |
| 3 | Caveats list | No | Hand inspection of dev rows gold_007, gold_053 |
| 4 | Requirement markers | Only through #5 | Corpus scan |
| 5 | Stricter-number downgrade (rules b/c/d) | **Yes: eligible → needs_checking only** | Corpus scan; 1 hand-made query. **Never fired on dev** |
| 6 | Quoting and count rule | No (text only) | 1 dev row (gold_007) |
| 7 | Downgrade reason templates | No (text only) | 1 hand-made query |
| 8 | Restatement rule fix | Only eligible → needs_checking, through #5 | 1 dev row (gold_007); corpus audit |
| 9 | Evaluation runs `collect()`/`finalize()` | No (measurement) | — |
| 10 | `requirements.txt` | No | — |

Across every step, nothing moves a status upward, and nothing produces
`not_eligible` that the matcher did not already produce.

## 1. HTTP API (`main.py`)

- **What:**
  - `POST /match` runs understand → hybrid retrieval (top 10) → match →
    explain.
  - It returns `profile_confidence`, `clarifying_question`, `notice` and
    `results`.
  - `GET /health` answers without loading the model. The index and bge-m3
    load once at startup.
  - A clarification is appended to the query and the whole pipeline runs
    again.
  - A clarifying question is asked only when no clarification was given. It
    is understand's fallback question, or the matcher's chosen field worded
    by `phrase_question`.
  - `notice` fires when the person names something ending in Yojana, Scheme,
    Card, Nidhi or Bima that no corpus scheme name matches (fuzzy token
    match).
  - Each result carries `documents` (the scheme's `documents_text` split into
    items), `apply_url` (the myScheme page), `last_updated` and `source_url`.
  - Results outside explain()'s five get generate.py's code template.
  - If no LLM provider answers, the API still returns results with template
    reasons.
  - CORS allows localhost and a Lovable placeholder domain.
- **Why:** the frontend's response shape.
- **Evidence:**
  - Three dev queries (English, Hindi, clarify) through a running server.
  - The notice matcher was checked on 12 hand-made names: the four invented
    gold names and eight real or generic ones.
  - The document splitter was checked on about 10 sampled records.
- **Statuses:** passed through unchanged.

## 2. Certainty sort (`postprocess.sort_key`)

- **What:** results are grouped eligible → needs_checking → not_eligible.
  Within a group:
  - first version: schemes where a condition about the person passed (state
    and board registration don't count), then fewest unverified conditions,
    then most personal passes, then retrieval rank;
  - current version: the personal-pass flag, then retrieval rank. The owner
    removed the two middle keys.
- **Why:** on gold_053, National Family Benefit Scheme (a death benefit) sat
  above the disability pension the row expects.
- **Evidence:** one dev row (gold_053). With the current key, nfbs is again
  above igndps on that row, because retrieval rank decides between them.
- **Statuses:** never; order only. The sort runs after explain() has chosen
  its five, so it doesn't change which schemes are explained.

## 3. Caveats list

- **What:** each result carries `caveats`: every unverified condition of the
  scheme that is requirement-shaped (#4) and not a restatement (#8), verbatim,
  in record order. It is filled for eligible and needs_checking results and
  empty for not_eligible ones.
- **Why:** an eligible result with unchecked conditions read as a clean
  match. gold_053's nfbs showed eligible with no hint that it requires the
  breadwinner to have died.
- **How it evolved:** hand inspection of the five eligible results on gold_007
  and gold_053 (farmer-insurance, rythu-bima, pmjjby, igndps, nfbs) drove it:
  1. a quoted first residual;
  2. a requirement filter;
  3. a filter for residuals restating passed checks;
  4. a number guard;
  5. finally the full list.
- **Statuses:** the list itself never changes a status.

## 4. Requirement markers

- **What:** a residual counts as a requirement if it contains must, should,
  shall, required, only, cannot, excluded, at least, minimum, maximum, not
  more than, not less than, up to, ineligible, not eligible, will not or
  shall not.
  - Bare "eligible" was dropped: it matched grants such as "Dwarfs are also
    eligible for this scheme."
  - An "… is/are also eligible" clause is removed before the test, so a
    sentence that grants eligibility to another group isn't a requirement,
    while one that also says "must" still is.
- **Evidence (corpus scan):**
  - Requirement-shaped residuals went from 3,903 to 4,980 of 7,939 (+1,077).
  - 35 inclusion-only sentences stay excluded.
  - An early version of the inclusion rule wrongly dropped one real
    requirement (safaew-bcew). The rule was changed to the clause removal
    above.
  - 2,959 residuals have no marker and are never caveats. That rule was
    left unchanged.
- **Statuses:** only indirectly. More requirement-shaped residuals means more
  candidates for #5.

## 5. Stricter-number downgrade (rules b, c, d)

- **What:** a residual is a "stricter number" when it is about the same
  quantity as a passed check and states a number that check's constraint
  doesn't contain. The quantities are age, income, land, or disability
  percentage against the 40% PwD benchmark. For an eligible scheme:
  - **(b)** the person gave a confident value that satisfies the bound parsed
    from the residual: it is treated as a restatement. No caveat, no change.
  - **(c)** the value is missing or low-confidence, or the bound can't be
    parsed: the status goes eligible → needs_checking.
  - **(d)** the value is known and violates the bound: eligible →
    needs_checking. It never becomes not_eligible, because regex parses are
    not confident enough to exclude anyone.
- **Why:** the matcher passes the PwD category at a fixed 40%. igndps's own
  text requires more than 80%, so a person with 45% would have been shown
  eligible.
- **Evidence:**
  - A corpus scan found residuals with a number stricter than the typed
    constraint in 101 schemes, about 5% of records. By kind: age 77,
    disability 21, land 12, income 11 residuals. This was a regex-based
    upper bound.
  - The rule was then exercised on **one hand-made query**: "I am 60 years
    old, I work as a farmer in Telangana, …". There, rythu-bima's "18-59
    years" rule moved it from eligible to needs_checking.
- **Known gap:** **the rule never fired on dev.** Status transitions over all
  200 dev results were X → X in every run, so dev gives no evidence for or
  against it. The dev row with a stricter residual (gold_053, igndps, "more
  than 80%") falls under rule (b), because the person stated 82%.
- **Known weaknesses (described, not changed):**
  - Of the 140 stricter-number candidates in a corpus simulation, 68 have
    bounds the parser can't read. Those always take rule (c) when the scheme
    is eligible. An example is "The age of the male applicant should be 60
    years or above."
  - A number next to category words is read as a disability percentage even
    without disability wording. 6 cases, such as marks percentages or
    student counts.
  - Note numbering and years count as numbers.
- **Statuses:** **yes**, eligible → needs_checking only.

## 6. Quoting and count rule

- **What:**
  - No caveat is quoted in the reason. A result with caveats gets a count
    appended: "N condition(s) of this scheme still to confirm." This applies
    to eligible and needs_checking results, in English, Hindi and Telugu.
  - The only quoted condition is the one that caused a #5 downgrade (see #7).
  - A quoted condition must be an exact substring of the scheme's residual
    conditions or `eligibility_text`, or it is dropped (the grounding check).
  - Before this, the first caveat was quoted, followed by "And N more
    condition(s) to check."
- **Why:** quoting the first caveat surfaced whichever condition came first
  in the record, not the most important. On gold_007, rythu-bima's reason
  quoted "The applicant can apply for a single policy only."
- **Evidence:** one dev row (gold_007) and the owner's decision.
- **Statuses:** never.

## 7. Downgrade reason templates

- **What:** a result moved to needs_checking by #5 gets a reason written in
  code, not generate.py's template:
  - With a known value: "You said your age is 60; this scheme's own text
    says: <residual>."
  - With an unknown value: "This scheme requires: <residual> That couldn't
    be checked from what you told us."
  - Then the checks that genuinely passed, excluding the field that caused
    the downgrade, and a count of the other caveats.
  - The quoted residual has leading punctuation, colons and quote marks
    stripped, is grounding-checked, and is never truncated.
  - Hindi and Telugu templates exist; the residual stays verbatim in English.
- **Why:** on the hand-made age-60 query, generate.py's template said "age
  (60) … checked out" for a scheme downgraded on age. It also carried a
  stray “: from the chunk text.
- **Evidence:** one hand-made query.
- **Statuses:** never; text only.

## 8. Restatement rule fix

- **What:** a residual is treated as already checked only if:
  - **(a)** it is, word for word, the clause a passed check was read from
    (compared after lowercasing and collapsing punctuation and whitespace;
    multi-clause spans are split on " | "); or
  - **(b)** rule (b) of #5 settles it.
  Three rules were removed:
  - a residual counted as checked when it merely mentioned an occupation
    word of a passed check;
  - one that mentioned a subject word (state, age, category, land, income,
    BPL, marital status, residence) whose numbers the check covered;
  - span containment in either direction.
- **Why:** on gold_007, rythu-bima's "The farmer should have a bank account
  in their name…" and "The farmer should pay the required premium…" were
  hidden, because they contain "farmer" and the occupation check passed.
- **Evidence:** one dev row (gold_007), then a corpus audit simulating every
  typed field as passed.
  - Before, suppressed as restatements: 1,397. That is 791 exact span, 143
    span containment only, 136 occupation word, 320 subject word without
    numbers, and 7 subject word with equal numbers.
  - After: 816.
  - Caveats shown in the simulation went from 3,362 to 4,024.
  - On dev, eligible results with no caveats went from 4 to 2.
- **Cost:** some genuine near-duplicates are now shown as caveats, because
  they aren't the exact source clause:
  - "The applicant should be a resident of India." on igndps;
  - "The applicant must be residing in India." on pomis;
  - "The farmers must be from Telangana state." on rythu-bima for a Telangana
    farmer.
- **Statuses:** only eligible → needs_checking, through #5. There were 12
  more age/income/land/disability candidates in the simulation, which
  containment matches used to shadow. None changed on dev or on the five
  verification queries.

## 9. Evaluation uses the API's post-processing

- **What:** `postprocess.collect()` builds `finalize()`'s input: explain()'s
  text, or generate.py's template outside the five it covers.
  - Both `main.py` and `scripts/evaluate_generate.py` call `collect()` then
    `finalize()`, so evaluation scores exactly what `/match` returns: final
    statuses, reasons, caveats and order.
  - The script gained `--limit` and `--out` for smoke tests.
- **Evidence:** a 2-row smoke test. It executed; no metrics were read.
- **Statuses:** never; measurement only.

## 10. `backend/requirements.txt`

- **What:** pinned runtime dependencies, matching the installed versions
  (verified).
- **Why:** deployment.
- **Statuses:** never.

## Known gaps

- **#5 is untested on dev.** Its only positive evidence is one hand-made
  query.
- **The frozen understand-v4.1 drops an age stated without a first-person
  word** ("I am a farmer in Telangana, 60 years old"). So the most natural
  age-60 query never reaches rule (d): the scheme is needs_checking from the
  matcher because age is unknown.
- **rythu-bima's source text states both 18-59 (three times) and 18-60
  (twice).** The typed constraint took 18-60. The downgrade in #5 relies on
  the other one.
- **Dev evidence overall is 20 rows,** and several decisions above rest on a
  single row or on a query written for the purpose.
