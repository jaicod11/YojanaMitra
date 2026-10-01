# Changes since `pipeline-frozen-dev`

Every behaviour change made in `backend/` and `scripts/` after the pipeline
was frozen.

These frozen components are unchanged: `retrieval.py`, `matcher.py`,
`generate.py`, the index, and every constraint record.

`understand.py` has one post-freeze commit (`bb0e4c4a`, 2026-09-24; see #12):
- **What it is:** logging and key bookkeeping only.
- **Why:** provider failures that used to be silent now reach the server log,
  and a key hit by per-minute rate limits recovers without a restart.
- **What it doesn't touch:** no output-affecting line (prompt, model,
  temperature, parsing or validation). Output was verified byte-identical on
  two cached queries.

All changes below live in these files:

- `backend/app/main.py` (the HTTP API)
- `backend/app/postprocess.py` (what happens to results between the frozen
  pipeline and the API response)
- `backend/requirements.txt`
- `scripts/evaluate_generate.py`
- `backend/app/understand.py` (#12 only: logging and key bookkeeping)
- `scripts/label_categories.py` (#12 only: the provider pool that
  `understand.py` shares with the batch labelling and extraction scripts)

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
| 11 | Frontend wiring: CORS from `ALLOWED_ORIGINS`, verbatim flags, dependent-note flag | No | The real UI driven against the running API; corpus scan |
| 12 | Provider failure logging and per-minute key cooldown | No line that decides a status changed. Indirectly, only through which provider answers | Stubbed-provider simulations; 1 real query; output byte-identical on 2 hand-made cached queries |

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

## 11. Frontend wiring (`main.py`)

- **What** (commit `13488d89`, 2026-09-23, tag `frontend-wired`):
  - **CORS:** the allowed origins come from `ALLOWED_ORIGINS`, set in the
    environment or in the repo-root `.env`. `main.py` loads that file by a
    path relative to its own location, not the working directory.
    - Format: comma-separated exact origins; trailing slashes are removed.
    - The default is `http://localhost:8080`, the frontend's dev server.
    - An origin containing `*` stops startup with an error.
    - Methods: POST and OPTIONS. Header: `Content-Type` only. No
      credentials.

    This replaces #1's setting of localhost plus a Lovable placeholder
    domain.
  - **`caveats_verbatim` and `unverified_verbatim`:** one boolean per entry
    of `caveats` and of `unverified_conditions`. Each says whether that entry
    appears word for word in the scheme's `eligibility_text`, using the same
    check as #6's grounding (`postprocess.is_grounded`).
  - **`requires_dependent_note`:** when a scheme requires a dependent, the
    matcher adds a fixed sentence to `unverified_conditions`: "the scheme
    requires a dependent (see the scheme text)". It isn't scheme text, so the
    API takes it out of `unverified_conditions` and reports this boolean
    instead. The frontend shows its own fixed line. `main.py` keeps a copy of
    the sentence and asserts at startup that `matcher.py` still contains it.
- **Why:**
  - The browser calls the API directly, so CORS has to name the frontend's
    origin.
  - Extraction asks for verbatim residuals but doesn't enforce it. The UI
    shouldn't present a paraphrase, or the matcher's own sentence, as the
    scheme's words.
- **Evidence:**
  - **UI runs:** the real UI, driven with Playwright against the running
    API, in five scenarios: English, Hindi, the clarify flow, an invented
    scheme name, and the server-down error screen.
  - **Preflight checks (2026-09-23):**
    - The CORS preflight was checked in the browser and with curl. An origin
      of exactly `http://localhost:8080` is allowed.
    - `http://127.0.0.1:8080`, `http://localhost:8081` and the LAN address
      get `400 Disallowed CORS origin`.
    - A non-allowed request header gets `400 Disallowed CORS headers`.
  - **Corpus scan (2026-10-02):** 118 of the 7,939 residuals (1.5%), in 64
    schemes, are not word for word in their scheme's `eligibility_text`.
    Those are the entries the flags mark `false`.
- **Statuses:** never. The fields sit alongside the existing ones, and CORS
  only decides which browser origins may call the API.

## 12. Provider failure logging and per-minute key cooldown

- **What** (commit `bb0e4c4a`, 2026-09-24). This is logging and key
  bookkeeping only. It touches three files: `scripts/label_categories.py`
  (the provider pool that `understand.py` and the batch scripts share),
  `backend/app/understand.py` (frozen) and `main.py`.
  - **Logging:**
    - Every failed provider attempt logs one line: provider, key name, error
      kind (`retryable`, `fatal`, `rate_minute` or `daily_quota`), HTTP
      status, attempt number and a truncated message.
    - Every answer logs the provider, the key and the time taken.
    - The existing messages are kept word for word.
    - The server writes these lines in uvicorn's format. The batch scripts
      still print them above their progress bar: failures and retries, but
      not answers.
  - **Swallowed errors:** `main.py` now logs the `understand()` and
    `phrase_question()` errors it swallows. The swallowing itself is
    unchanged.
  - **In `understand.py`:**
    - `_pools()` passes `minute_cooldown` to both pools. Provider order is
      unchanged: Gemini, then Groq.
    - A line is logged when a provider is skipped because no key is usable,
      and when a fatal error disables it.
    - A new read-only `provider_status()`.
  - **Per-minute key cooldown (API only):**
    - Three consecutive per-minute 429s on a key, within one call, bench
      that key for 60 s. Before, they retired it until restart.
    - A daily cap still retires a key for the life of the process.
    - The batch scripts keep permanent retirement, so a long run paces itself
      as before.
  - **`/health`:** for each provider, the number of keys in total,
    available, cooling down and retired; the seconds left on each cooldown;
    and the last failure (time, key, kind, HTTP status). No provider is
    called.
- **Why:**
  - **Silent failures.** A server log showed Groq 429s and no Gemini lines.
    Gemini's 5xx, timeout, connection and fatal errors fell through to Groq
    without any message, and successes were never logged.
  - **The cache showed what happened:** Groq answered every call from 14:59
    UTC on 2026-09-23. One Gemini answer at 15:03 proved Gemini was failing
    silently, not retired.
  - **Permanent retirement.** Three per-minute 429s retired a key until
    uvicorn restarted, though such limits clear within a minute.
  - **Why 60 s:** both providers define their per-minute limits over a 60 s
    window. In the log of the 1,904-scheme extraction run, 86 of 87
    per-minute 429s cleared after one 20 s wait, and the other after
    20 s + 40 s.
- **Evidence:**
  - **Stubbed providers:** fake keys, no API call.
    - A Gemini 503 on every attempt now logs three Gemini failure lines
      before Groq answers.
    - Three per-minute 429s put a key in a 60 s cooldown and the next key
      answers. After 61 s of real time, the first key is used again.
    - The batch-script path still retires the key, with the old messages.
  - **One real, uncached query:** three `[gemini] GEMINI_API_KEY answered`
    lines.
  - **Output unchanged:** two hand-made queries, both cached and not gold
    rows: an English Andhra Pradesh farmer and a Hindi Uttar Pradesh widow.
    Their full `/match` responses were **byte-identical** before and after
    the change. The server's caches pointed at scratch copies, so
    `data/cache` was not written.
- **Statuses:**
  - **No line touched decides a status or affects output:** no prompt, model
    name, temperature, parsing or validation line.
  - **Indirect effect:** which provider answers an uncached call can differ
    after a burst of per-minute 429s. A cooled-down Gemini key now returns
    after 60 s, where before Groq answered until restart.
- **Known weaknesses (described, not changed):**
  - A single fatal error still disables a provider until restart, for
    example one 400 from Gemini on an unusual prompt. It is now logged at
    ERROR.
  - `/health` is public and shows key variable names, though never the
    values.

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
