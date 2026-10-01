# YojanaMitra: project handoff

**As of 2026-10-02.** This document lets a new assistant session continue the
project without the earlier conversations. Every fact below was read from
files on disk on 2026-10-02, without running git. Anything that could not be
confirmed is marked **Unverified**.

- **Repository state:** `main` is at `bb0e4c4a`, "log provider failures and
  answers, and let rate-limited keys recover" (2026-09-24 07:32 IST). No file
  outside `node_modules/`, `.git/` and caches has been modified since
  2026-09-24 07:23, so the working tree probably matches that commit.
  **Unverified:** that the tree is clean, since that needs `git status`.
- **Remote:** `origin` is `https://github.com/jaicod11/YojanaMitra.git`. The
  locally recorded `origin/main` is also `bb0e4c4a`. **Unverified:** whether
  the tags were pushed.
- **No earlier handoff document exists in the repo.** Two figures from an
  earlier handoff (passed along in conversation) were stale; the corrections
  are in §6 and in `docs/CLASSIFIER.md`.

Contents:
1. Working conventions
2. What the project is
3. Current state
4. What is frozen and why
5. Results so far
6. Known limitations
7. What remains
8. The rule-first plan
9. How to run it locally
10. Things to protect

---

## 1. Working conventions

These are how the owner works with assistants on this project. Follow them
from the first message.

**The advisor writes prompts, not code.** The assistant acts as an advisor.
Every code or feature request is answered with a paste-ready prompt for
Claude Code, not with code. A good prompt includes:
- the task in one line;
- HARD CONSTRAINTS: frozen files, no git, which data folders are read-only;
- numbered steps, with a STOP after diagnosis when the cause is unclear;
- a VERIFY section;
- a REPORT BACK section.

**Claude Code never runs git commands, read-only ones included** (no
`git status`, `git diff` or `git log`). Every prompt ends with:

> Don't commit; list changed files and suggest commit messages.

Commit messages carry **no `Co-Authored-By` and no Claude attribution
trailer**. The repo's `CLAUDE.md` says the same: never commit, push or
rebase; leave all changes uncommitted; list the changed files and suggest a
commit message. The owner reviews and commits everything.

**Be direct.** Push back when something is wrong, including when the advisor
itself was wrong earlier. Say plainly what the evidence shows and what it
doesn't.

**Verify claims with shell commands; don't accept self-reports.** Have Claude
Code count files, checksum outputs, grep the code, or diff responses, and
paste the output. Examples from this project where checking caught a wrong
claim:
- The earlier handoff said there were five classes with fewer than 25
  schemes; there are four. It said 462 Groq labels; there are 327.
- The tag meant to be `postprocess-frozen` is actually named
  `postprocess-frozen,` with a trailing comma (§4).
- `docs/CHANGES_SINCE_FREEZE.md` claimed the frozen modules were unchanged
  after `understand.py` had been edited, for logging and key bookkeeping
  only. It was corrected on 2026-10-02 (§4).

**Guard evaluation integrity.**
- **Dev tuning:** no tuning on the dev set beyond the agreed budget.
  **Unverified:** the budget itself is not written down in the repo; ask the
  owner.
- **No single-row fixes:** never fix a bug using only the one row that
  revealed it. Find corpus-level evidence first (a scan over all 2,062
  constraint records or all 7,939 residuals), and say when a decision still
  rests on one row.
- **Test set:** the test split is run once, at the end, with `--final`
  (§7). After that, changing the system and re-scoring test turns it into a
  second dev set.

**Operating habits that came from incidents:**
- **Diagnose before rerunning.** State the cause of a failed or stopped run,
  with evidence, before rerunning anything. Reruns spend free-tier LLM quota.
- **Run long jobs detached.** LLM runs longer than about 10 minutes are
  started with `subprocess.Popen(..., start_new_session=True)` plus
  `caffeinate -is -w <pid>`, logging to a file next to the data. Otherwise
  give the owner the command to run with `nohup caffeinate -is`. A
  background task tied to a Claude Code session dies when the session ends,
  and this Mac idle-sleeps after one minute.
- **Stop test servers.** A server started for a test must be stopped
  afterwards unless the owner asked to keep it. On 2026-09-23 a leftover Vite
  server on 8080 caused a session of CORS 400 errors (§9).
- **Keep data read-only by default.** `data/interim`, `data/index`,
  `data/cache` and `data/gold` are read-only unless a task says otherwise. A
  live `/match` query appends to `data/cache`. When that must not happen,
  verification has used a harness that points the caches at scratch copies.
- **Never print API key values.**

## 2. What the project is

**YojanaMitra** ("scheme friend") helps a person in India find government
welfare schemes they may qualify for. The person describes their situation
in plain language. The gold set covers English, Hindi and Telugu; the API
also accepts Tamil and Bengali codes (`ta`, `bn`), which have no gold rows.
They get back up to 10 schemes from the myScheme portal, each with:
- a status: **eligible**, **needs_checking** or **not_eligible**;
- a reason in their language;
- the scheme's own conditions still to confirm;
- the documents needed;
- a link to the official myScheme page.

**Pipeline** (`backend/app/`):
1. **understand** (LLM): extracts a structured profile and an English search
   query from the description.
2. **retrieval:** hybrid BM25 plus bge-m3 dense search over chunked scheme
   text, fused with RRF; top 10.
3. **matcher:** deterministic. It compares the profile with typed
   eligibility constraints that were extracted offline from each scheme's
   eligibility text.
4. **generate** (LLM): writes reasons with citations for the top 5 results
   that are not not_eligible.
5. **postprocess:** caveats, sort order, and downgrade-only status
   corrections.
6. **FastAPI** `POST /match`, then the React frontend.

**The two design principles.** Every component must respect them.
1. **The LLM never decides eligibility.** Statuses come from typed
   constraints and the deterministic matcher. The LLMs extract the profile
   and write explanations. Validation rejects any reason that claims an
   eligibility the matcher didn't give; on dev this was measured at 0
   hallucinated claims. An LLM relevance flag can only ever move a scheme
   from eligible to needs_checking.
2. **Rule out only on hard, confident evidence; anything uncertain defaults
   to needs_checking.** not_eligible comes only from a failed matcher check
   against a confident profile value. Post-processing only ever moves
   eligible to needs_checking. It never produces not_eligible and never
   upgrades a status. Regex parses are not trusted enough to exclude anyone.

**Course context.** This is a course project. The supervised text classifier
(`docs/CLASSIFIER.md`) was built as the course's required ML component.
**Unverified (not recorded in the repo):** the course name and code, the
team members, and the deadline. The owner should fill these in here.

**Deadline status:** see §7. As of 2026-10-02, deployment and the
rule-first rebuild remain. The owner reports that the test run is complete
and the report written. Neither the test results nor the report is in this
checkout (§5, §7).

## 3. Current state

| Component | State as of 2026-10-02 | Frozen? | Where |
|---|---|---|---|
| **Corpus** | 2,153 myScheme page PDFs (public HF dataset `shrijayan/gov_myscheme`, downloaded 2026-09-05; portal data as of 28 March 2024 per the PDF footers). 87 duplicate downloads skipped, 2,066 scheme records parsed, 0 parse failures. | Not tagged; treat as fixed (everything depends on it) | `data/external/` (gitignored, not in the backup); `data/interim/schemes/` (gitignored) |
| **Category labels** | 2,066 silver labels from an LLM: gemini-3.1-flash-lite 1,662, openai/gpt-oss-120b via Groq 327, gemini-3.6-flash 77. A relabelling pass over Social Welfare changed 246 labels (the pre-pass file is `predictions.json.bak-pre-swe-merge-2026-09-21`). Hand-checked samples: round 1 (30 schemes) and round 2 (100, blind). | Not tagged; treat as fixed (the classifier's targets) | `data/interim/labels/` |
| **Constraints** | 2,062 constraint records: 2,045 extracted (1,808 by Gemini, 237 by Groq) and 17 extraction-failed stand-ins. Four schemes have no record (`cmacs`, `dlbe`, `mrin`, `wbrupashree`); **Unverified:** why. 7,939 residual conditions (eligibility text not captured by a typed field). The boolean-fields migration (`citizen_of_india`, `bpl_household`, `requires_bocw_registration`, `not_availing_other_scheme`) finished 2026-09-20, with 58 records hand-corrected. 396 records carry the `multi_branch_eligibility` flag. | **Yes** | `data/interim/constraints/`; review decisions in `data/interim/constraints_review_boolean_fields.json`; pre-migration copy in `data/interim/constraints_pre_boolean_fields/` |
| **Gold set** | 104 rows (92 en, 6 hi, 6 te): 54 positive, 30 exclusion, 15 clarify, 5 no_match. Split dev 20 / test 82 / none 2. The two `skip_scoring` rows are gold_032 and gold_069. Frozen 2026-09-21. | **Yes** | `data/gold/` (in git); see `data/gold/README.md` |
| **Index** | 15,825 chunks. BM25Okapi, plus bge-m3 (1,024-d, float16, normalised) in FAISS `IndexFlatIP`. Built 2026-09-22. `data/processed/chunks.jsonl` must match `chunks_sha256` in the manifest, or the API refuses to start. | **Yes** (`pipeline-frozen-dev`) | `data/index/` (gitignored); `chunks.jsonl` is rebuilt by `scripts/build_chunks.py` |
| **Retrieval** | Hybrid RRF (k=60), 50 candidates per list, top 10. | **Yes** (`pipeline-frozen-dev`) | `backend/app/retrieval.py` |
| **understand** | Prompt `understand-v4.1`. Gemini gemini-3.1-flash-lite first, Groq openai/gpt-oss-120b as fallback, temperature 0. Cached in `data/cache/understand.jsonl`. | **Yes** (`pipeline-frozen-dev`). Post-freeze edit: logging and key cooldown only, 2026-09-24, logged as change #12 | `backend/app/understand.py` |
| **matcher** | v2 with 7 rules: applying_for, priority, multi_branch, bocw_occupation, person_level_pass, residual_touch, bocw_gate. Occupation threshold 0.85, residual-touch threshold 0.48, minimum clarify score 0.5. | **Yes** (`pipeline-frozen-dev`) | `backend/app/matcher.py` |
| **generate** | `generate-v1`. Explains the top 5 results that are not not_eligible, in the person's language, in one LLM call. Citations validated, one retry, then a code template. Cached in `data/cache/generate.jsonl`. | **Yes**, by the owner's instruction from the API step (2026-09-23). It is not inside `pipeline-frozen-dev`, since it was built after that tag; it is inside `postprocess-frozen,` | `backend/app/generate.py` |
| **postprocess** | `collect()`/`finalize()`: caveats, certainty sort, stricter-number downgrade (eligible to needs_checking only), quoting and count rule, downgrade reason templates in en/hi/te. | **Yes** (`postprocess-frozen,`) | `backend/app/postprocess.py`; history in `docs/CHANGES_SINCE_FREEZE.md` |
| **FastAPI** | `POST /match` and `GET /health`. `/health` now also reports each provider's key state. CORS allows the exact origins in `ALLOWED_ORIGINS` (default `http://localhost:8080`; POST and OPTIONS; header `Content-Type` only; no wildcards). If no LLM answers, `/match` still returns results with template reasons. | No; changed after every freeze | `backend/app/main.py`; pinned dependencies in `backend/requirements.txt` |
| **Frontend** | Lovable-exported TanStack Start app (React, TypeScript, Tailwind), wired to the real `POST /match`. Details under the table. | Tag `frontend-wired` marks the wiring; `strictPort` came after it | `frontend/` (main view `src/routes/index.tsx`, API client `src/lib/api.ts`) |
| **Classifier** | Done. Predicts a scheme's myScheme category (15 classes). Headline macro-F1 0.742 ± 0.026 (§5). Standalone: nothing in the pipeline uses it. | **Yes** (`classifier-done`) | `scripts/classifier/`, `data/eval/classifier/`, `docs/CLASSIFIER.md` |
| **HF data backup** | Private HF dataset `devil2411/yojanamitra-data-backup` holding `data/interim`, `data/index` and `data/cache`. Restore script with `--check` and `--force`; it rebuilds `chunks.jsonl` and checks its SHA. The backup was made on 2026-09-23, before the restore script was committed at 18:15 IST (inferred; the upload time isn't recorded locally). **`data/cache` was last written 2026-09-23 20:48, so the backup is probably missing the latest cache entries.** | n/a | `scripts/restore_data_backup.py` |
| **Provider logging and key cooldown** | Done (commit `bb0e4c4a`, 2026-09-24). Details under the table. | Not tagged; post-freeze; logged as change #12 | `scripts/label_categories.py`, `backend/app/understand.py`, `backend/app/main.py` |
| **Test run** | **The owner reports it is complete; the results are not on this machine** (§5). | n/a | §5, §7 |
| **Deployment** | **Not started.** | n/a | §7 |
| **Report and deck** | **The owner reports the report is written.** No report or deck files are in the repo. | n/a | §7, §10 |
| **Rule-first rebuild** | **Not started.** No `understand_rules.py` and no feasibility doc exist. | n/a | §8 |

**Frontend details:**
- The dev server runs on port 8080 with `strictPort` (§9).
- State lives in React only: nothing is stored in localStorage, sessionStorage or cookies.
- The clarifying question appears above the results, at most 2 rounds per search.
- A "taking longer" message shows after 15 s; requests time out at 90 s.
- Each result has a "Conditions to check" list and an "Also noted for this scheme" list.
- Each result has one "View & apply on myScheme (official site)" link.
- The date label reads "Scheme details as of 28 March 2024 (myScheme)".

**Provider logging and key cooldown details:**
- Every failed attempt is logged with the provider, key name, error kind, HTTP status and message. Every answer is logged with the key and the time taken.
- In the API, a key that returns 3 per-minute 429s in a row cools down for 60 s instead of being retired until restart.
- A daily cap still retires a key for the life of the process.
- The batch scripts keep permanent retirement.
- Provider order is unchanged.

## 4. What is frozen and why

**Why freeze.** Dev metrics only describe the system that will face the test
set if that system stops changing. Components were frozen after their final
dev runs. Changing a frozen component would mean re-running dev, and would
risk tuning on dev beyond the budget.

**Tags.** These were read from `.git/refs/tags/` as plain files; git was not
run.

| Tag (exact name) | Points to | Created | Marks |
|---|---|---|---|
| `pipeline-frozen-dev` | Object `043c6055`. It is packed, so **its target can't be read without git (Unverified)**. The tag file was written 2026-09-22 19:33, one minute after commit `b2085f87` "freeze understand-v4.1 and matcher v2 before test", so it very likely marks that commit, possibly as an annotated tag. | 2026-09-22 | understand-v4.1, matcher v2, retrieval, the index and the constraint records, after the final dev run `data/eval/e2e_dev_20260922-191512.json` |
| `postprocess-frozen,` | Commit `8ab120e6` "count only exact source clauses as restatements" | 2026-09-23 13:02 | `postprocess.py`, and with it `generate.py` and the API as of that commit |
| `frontend-wired` | Commit `13488d89` "connect the frontend to POST /match" | 2026-09-23 17:43 | The frontend calling the real API, with CORS |
| `classifier-done` | Commit `d87d4cef` "measure the near-duplicate leak and widen the best model's C grid" | 2026-09-23 19:37 | The classifier, including the group-aware CV and wider C grid |

- **The postprocess tag name ends in a comma.** It was almost certainly a
  typo for `postprocess-frozen`, and a tag named `postprocess-frozen` does
  not exist. If the owner wants it fixed, it's the owner's own git command
  (not Claude Code's):
  `git tag postprocess-frozen 8ab120e6 && git tag -d 'postprocess-frozen,'`,
  plus pushing or deleting the remote tag if tags were pushed.
- **Two commits are newer than every tag:** `aed76504` (Vite `strictPort`,
  2026-09-23) and `bb0e4c4a` (provider logging and key cooldown,
  2026-09-24).

**Rule: every post-freeze change to `backend/` or `scripts/` must appear in
`docs/CHANGES_SINCE_FREEZE.md`.** Each entry says what changed, why, the
evidence (a corpus scan, a dev row, or a hand-made query), and whether it can
change a status.

**The log was brought up to date on 2026-10-02.** Before that it ended at
2026-09-23 12:51. Two changes were added:
- **#11, frontend wiring in `main.py`** (commit `13488d89`): CORS via
  `ALLOWED_ORIGINS`, the `caveats_verbatim` / `unverified_verbatim` flags, and
  `requires_dependent_note`.
- **#12, provider logging and per-minute key cooldown** (commit `bb0e4c4a`).
  This touched the frozen `understand.py`.

The log's opening line now says `understand.py` has that one post-freeze
commit:
- It is logging and key bookkeeping only; no output-affecting line was
  touched.
- Output was verified byte-identical on two cached queries. They are
  hand-made, not gold rows.
- Which provider answers can still differ after a burst of per-minute 429s.

## 5. Results so far

**All numbers below are on the 20-row dev split.** Dev is small: a
positive-row rate moves by 0.10 per row, and an exclusion-row rate by about
0.17. Read every number below with that in mind.

**Test-set results: not found on this machine.** On 2026-10-02 the owner
reported that the test run is complete and the report written. Searching
this checkout and this machine found no test results:
- **No result files.** `data/eval/` has no `test_*`, `e2e_test_*` or
  `generate_test_*` file, the names the three scripts write with
  `--split test --final`. A search of the home directory and `/tmp`, by those
  names and for any JSON containing `"split": "test"` modified since
  2026-09-24, found nothing.
- **No trace in the cache.** None of the 82 test descriptions is in
  `data/cache/understand.jsonl`, while all 20 dev descriptions are. The cache
  was last written 2026-09-23.
  - The end-to-end and generation runs call `understand()` on every row, so
    they did not run here against this data.
  - A retrieval-only run on raw descriptions doesn't call `understand()`, so
    the cache can't rule that one out. But it would still have left a
    `test_*.json`.
- **No later commits.** There is no commit after 2026-09-24 in this
  checkout.

The run may have been done in another checkout or on another machine.
**Copy its result files into `data/eval/` and add the test numbers here.**
Include all three retrieval modes, so the comparison with the BM25 keyword
baseline is recorded. **Do not re-run the test split to regenerate them;**
it is scored once.

### Retrieval

Measured by `scripts/evaluate.py` on 10 positive and 6 exclusion dev rows.
- **R@k:** recall@k of the expected scheme.
- **MRR:** MRR@10.
- **FIR:** the share of exclusion rows whose excluded scheme appears in the
  top 10. Retrieval isn't meant to exclude; the matcher does that.

| Query sent to retrieval | Mode | R@5 | R@10 | MRR | FIR |
|---|---|---:|---:|---:|---:|
| Raw description | BM25 | 0.10 | 0.30 | 0.059 | 0.17 |
| Raw description | Dense | 0.40 | 0.60 | 0.267 | 0.67 |
| Raw description | Hybrid | 0.40 | 0.50 | 0.216 | 0.67 |
| understand-v1 rewrite | Hybrid | 0.30 | 0.60 | 0.287 | 1.00 |
| understand-v2 rewrite | Hybrid | 0.40 | 0.60 | 0.228 | 0.67 |

The frozen pipeline uses the understand-v4.1 rewrite with hybrid search. In
the final end-to-end dev run, the expected scheme was in the top 10 for
**6 of 10** positive rows (recall@10 0.60). **MRR was not computed for
v4.1;** no retrieval-only run used it.

### Matcher (end to end)

From `data/eval/e2e_dev_20260922-191512.json`, the final run before the
freeze, with the frozen configuration. Each target scheme is also matched
directly, so its status is known even when retrieval missed it.

- **Positive rows (10):** the expected scheme was eligible 2, needs_checking
  8, not_eligible 0.
  - **False-exclusion rate: 0.0.**
  - End-to-end rate (in the top 10 and not not_eligible): **0.60**.
- **Exclusion rows (6):** the excluded scheme was correctly marked
  not_eligible on 2 rows (0.33), **wrongly eligible on 0 (0.0)**, and left at
  needs_checking on 4 (0.67).
- **All 200 candidates** (20 rows × 10): eligible 8, needs_checking 125,
  not_eligible 67.
- **Profile extraction:**
  - state 11/11, age 5/6 (gold_065's Telugu age is dropped), gender 2/2,
    land 2/2;
  - occupation 9/12, using approximate matching;
  - false fills: occupation 1 of 8 rows with no occupation, income 1 of 20.
- **Trajectory over the matcher rules:**
  - with understand-v3 and no matcher rules, 0.33 of exclusions were wrongly
    eligible;
  - with v4 and 6 rules, 0.0;
  - eligible candidates went from 20 (v3, no rules) to 10 (v4, 6 rules) to
    8 (v4.1, 7 rules). The last step changed both understand and the
    matcher, so it isn't attributable to the 7th rule alone.
- **Clarify rows** are reported qualitatively: the field chosen, next to the
  gold notes. There is no metric.

### Generation

**These numbers date from 22 Sep and predate post-processing, so they do
not describe what `/match` returns today.** The run (2026-09-22 20:28) came
before two later changes:
- the post-processing layer (2026-09-23): downgrades, caveats, sort;
- routing `evaluate_generate.py` through `collect()`/`finalize()`.

Since then only a 2-row smoke test has run, and its metrics weren't read.

From `data/eval/generate_dev_20260922-202840.json` (generate-v1, top 5
explained):
- **Volume:** 24 LLM calls over 20 rows (one row needed no call); 84 schemes
  explained.
- **Hallucinated eligibility: 0** (also 0 against the matcher's status).
  Claims caught by validation: 0.
- **Citation validity:** 187/188 (99.5%) on the first attempt, 230/231 over
  all attempts. The 243 final citations have 0 invalid.
- **Fallback template:** 1 of 84 schemes (gold_003, `ysrkn`, "no valid answer
  after the retry"). 5 rows needed a retry.
- **Relevance flags:** 38, which downgraded 0 eligible schemes.
- **no_match rows** naming the invented scheme: 0.

### Classifier

From `docs/CLASSIFIER.md`; 2,066 schemes, 15 classes, stratified 5-fold CV,
seed 42.

| Model / treatment | Macro-F1 | Weighted-F1 | Accuracy |
|---|---|---|---|
| Majority class | 0.024 ± 0.000 | 0.075 | 0.213 |
| Best TF-IDF model: LinearSVC (balanced) | 0.688 ± 0.021 | 0.813 | 0.817 |
| **bge-m3 + LogisticRegression (balanced): headline** | **0.742 ± 0.026** | **0.831** | **0.830** |
| Same model, group-aware CV (34 near-duplicate groups kept within one fold) | 0.742 ± 0.023 | 0.831 | 0.830 |
| Same model, near-duplicates dropped (2,015 schemes) | 0.748 ± 0.033 | 0.833 | 0.831 |
| Same model, wider C grid up to 1,000 (C chosen 30, 10, 100, 10, 30) | 0.748 ± 0.022 | 0.835 | 0.835 |

**Why 0.742 is the headline:**
- Group-aware CV gives the same value (0.7421 vs 0.7419), so near-duplicates
  don't inflate it.
- The wider C grid adds only 0.006, within the fold spread.
- Keeping the original grid leaves the best model on the same terms as the
  other rows, which weren't re-tuned.
- The per-class table, confusion matrices, error sample and label-ceiling
  comparison all come from that run.

Read 0.742 as slightly conservative. Public Safety, Law & Justice has only 7
schemes and F1 0.18; the doc treats that as a finding about corpus
imbalance, not a defect to fix.

### Label ceiling

**Round 2:** 100 schemes, hand-labelled blind, sampled only from flash-lite
labels. "Reweighted" corrects for how the sample was stratified.

| Comparison | Raw | Reweighted |
|---|---:|---:|
| Silver label vs hand label | 0.89 | 0.889 |
| Classifier vs hand label | 0.82 | 0.840 |
| Classifier vs silver label | 0.87 | 0.903 |

- In all 8 cases where both classifier and silver label were wrong, the
  classifier made the same mistake as the silver label.
- **Round 1** (30 schemes): classifier 0.80, silver 0.90.
- **What it means:** the ceiling is set by label quality. Agreement with the
  silver labels measures consistency with the labelling model, not accuracy
  on the true taxonomy.
- **Uncertainty:** with n=100, these figures are uncertain by about ±7
  points.
- **Not hand-checked:** none of the 327 Groq-labelled schemes or the 77
  gemini-3.6-flash ones were in round 2.

## 6. Known limitations

Everything from the existing docs, plus the items added for this handoff.

**Data and labels**
- **The corpus is a snapshot.** Portal data is as of 28 March 2024. Schemes
  may have changed, closed or been added since.
- **Gaps in the constraints.** 17 schemes have extraction-failed stand-ins,
  and 4 schemes have no constraint record (`cmacs`, `dlbe`, `mrin`,
  `wbrupashree`; **Unverified:** why).
- **Multi-branch error mode (unfixed).** When a scheme lists several eligible
  groups and only one carries a condition, extraction can apply the
  condition to all of them. For example, mjpjay's BOCW registration belongs
  to one of categories A/B/C. mjpjay was fixed by hand. 396 records carry the
  `multi_branch_eligibility` flag; the 25 migrated ones with a BOCW or BPL
  `true` were reviewed and found clean.
- **Self-contradicting source records.** Two gold rows are `skip_scoring`
  for this reason: gold_069 (`pmmvy`, second-child rules) and gold_032
  (`pmsscsn`, which says Scheduled Caste in its name and Scheduled Tribe in
  its eligibility text).
- **Telegraphic input is untested.** The gold set has no telegraphic, non-first-person
  descriptions. All 92 English rows contain a first-person word; that was
  checked, while the 12 Hindi and Telugu rows were not. Input like "farmer,
  AP, 2 acres" is therefore untested, and understand-v4.1's first-person
  rules make it a real risk (next section).
- **Dev is small and thin in Hindi and Telugu.** It has 20 rows. Its only
  Telugu row is gold_065, swapped in by rule, and its only Hindi positive is
  gold_053.
- **Silver labels match hand labels on about 89% of the round-2 sample**
  (§5).

**understand and matcher (limits accepted at the freeze)**
- **gold_065 (Telugu) drops the age.** The model's English evidence phrase is
  a fragment without "I", or not a verbatim copy of the translation. The
  owner declined a fix rather than run dev again.
- **understand-v4.1 drops an age stated without a first-person word**, as in
  "I am a farmer in Telangana, 60 years old".
- **Residual-touch threshold T=0.48.** Similarity doesn't separate genuine
  from unrelated pairs: unrelated pairs reach 0.650. T was chosen with
  gold_084's 0.493 in view, so gold_084 doesn't validate it, and gold_014 is
  missed at 0.442. See `data/eval/residual_touch_calibration.json`.
- **cmpvy:** the priority rule probably misfires on it. This is unfixed;
  mj-fapm's provenance span was trimmed instead.
- **rythu-bima's own text states the age range both as 18–59 (three times)
  and 18–60 (twice).** The typed constraint took 18–60.

**Post-processing** (`docs/CHANGES_SINCE_FREEZE.md`)
- **The requirement-marker filter hides about 2,959 of the 7,939 residuals.**
  Those have no marker word (must, should, shall, required, only, …), so they
  are never shown as caveats, real requirements worded without a marker
  included.
- **The stricter-number downgrade (rules b/c/d) never fired on dev.** Its only
  positive evidence is one hand-made query.
- **The stricter-number parser can't read the bound in 68 of 140
  candidates.** Those always take rule (c) and downgrade an eligible scheme.
  Example: "The age of the male applicant should be 60 years or above."
- **6 category residuals are misread as disability percentages**, such as
  marks percentages or student counts next to category words. Note numbering
  and years also count as numbers.
- **Some genuine near-duplicates now show as caveats.** The exact-clause
  restatement rule only hides the exact source clause. Examples: "The
  applicant should be a resident of India." on igndps; "The farmers must be
  from Telangana state." on rythu-bima for a Telangana farmer.
- **Several decisions rest on a single dev row** (gold_007, gold_053) or on
  hand-made queries. They are flagged as weaknesses in the change log, not
  precedents.

**Generation**
- **No current measurement.** Generation's dev numbers predate
  post-processing (§5).
- **One Hindi caveat looks truncated: "Must be below the poverty"** (leprosy
  pension `lpsup`, seen on the Hindi Uttar Pradesh widow query). This is not
  a UI cut. The source eligibility text reads "Must be below the poverty and
  should be under Rs. 56,460/- per annum…", without the word "line", and
  extraction split that sentence at "and".

**API, frontend and operations**
- **`/health` is public and unauthenticated.** It now exposes provider names,
  models, key variable names (`GEMINI_API_KEY_2` and so on, never values),
  key counts, and the kind of the last failure.
- **The "Also noted for this scheme" list is computed in the frontend, not
  the API.** It is `unverified_conditions` minus `caveats`, so other API
  clients don't get it.
- **One fatal provider error disables that provider until restart.** For
  example, a single 400 from Gemini for an unusual prompt does this. It is
  now logged at ERROR.
- **Free-tier provider caps, observed and recorded as constants in
  `scripts/label_categories.py`:** about 20 Gemini requests per key per day,
  and about 200,000 Groq tokens per key per day. There are 3 keys of each.
- **Waits can exceed the frontend's timeout.** Per-minute 429 waits are 20 s
  plus 40 s per key, so one `/match` can outlast the frontend's 90 s
  timeout.
- **One tag name has a stray comma** (§4).
- **The HF backup is older than the latest `data/cache` writes** (§3).

**Classifier** (`docs/CLASSIFIER.md`)
- Its targets are silver labels (above).
- Public Safety has 7 schemes.
- Three other model rows also chose the top of the original C grid in every
  fold and weren't re-tuned.
- bge-m3's pretraining data may include myScheme pages.
- Near-duplicate groups use a single cosine threshold (0.95).

## 7. What remains

### Deployment (not started)
- **Frontend.** `npm run build` (`vite build`) uses nitro with **Cloudflare
  as its default target**, according to the Lovable config header in
  `frontend/vite.config.ts`.
  - **Build-time setting:** `VITE_API_BASE_URL` must point at the deployed
    API.
  - **API setting:** the deployed frontend origin must be added to
    `ALLOWED_ORIGINS`, as an exact origin with no wildcard.
- **Backend memory.** It holds bge-m3 in memory, about **2.2 GB** (the
  owner's figure; **Unverified:** not re-measured here), plus the FAISS index
  and the BM25 pickle. That **rules out Render's free tier**.
- **Candidate host: a Hugging Face Docker Space.** **Unverified:** check the
  current free-tier CPU and RAM limits, and sleep and cold-start behaviour.
  The API keys go in as Space secrets. The index was built on Apple `mps`;
  query embedding will run on CPU.
- **Getting the data into the container.** The API needs `data/interim`,
  `data/index`, `data/cache` and the rebuilt `data/processed/chunks.jsonl`.
  The restore script took **11 min 14 s**, including 14 HTTP 429 rate-limit
  waits, because the backup is 6,236 small files. A container start is only
  viable once the backup also carries **a single archive**, such as a
  tarball of those folders. Refresh the backup at the same time (§3).

### The one-shot test run (82 rows: 43 positive, 23 exclusion, 12 clarify, 4 no_match)
- **Status:** the owner reports this run is complete, but its results aren't
  on this machine (§5). If it has been run, it must not be run again. The
  steps below record how it is run.
- **Commands:**
  - retrieval: `python scripts/evaluate.py --split test --final`
  - end to end: `python scripts/evaluate_e2e.py --split test --final`
  - generation: `python scripts/evaluate_generate.py --split test --final`

  All three refuse the test split without `--final` and drop `skip_scoring`
  rows.
- **Before running:**
  - make sure `docs/CHANGES_SINCE_FREEZE.md` is current (it was brought up
    to date on 2026-10-02);
  - decide whether the system tested is `HEAD` (including the post-freeze
    logging commit) and record that;
  - decide the rule-first question in §8.
- **Quota.** Test descriptions are not in the LLM cache, so every row makes
  fresh calls (roughly 2–4 per row). The Gemini caps (about 60 requests a day
  across 3 keys) mean some calls will fall back to Groq. The cache records
  which provider answered each call, and the server log now does too.
  Report the provider mix with the results.
- **Run it detached** (§1). Whatever comes out is what gets reported; don't
  re-run.

### Report and deck
The owner reports the report is written. Neither it nor a deck is in the
repo, so they couldn't be checked against this document. §10 lists the
material they must not lose.

### Rule-first extraction rebuild
See §8.

### Housekeeping
- Copy the test results into `data/eval/` and record them in §5.
- Fix the tag name if the owner wants it.
- Refresh the HF backup and add an archive.

## 8. The rule-first plan

**Agreed approach:** build `backend/app/understand_rules.py` **alongside**
the frozen `understand.py`, selected by a config flag, **never in place of
it**. That keeps the frozen pipeline and its dev metrics valid, and lets both
paths be scored against the same gold set with the same scripts.

What that implies for whoever writes the prompt:
- **Default off.** With the flag off, nothing changes. `understand.py`,
  `matcher.py`, `generate.py`, `postprocess.py`, retrieval and the index
  stay untouched.
- **Same output shape.** `understand_rules.py` returns the same structure as
  `understand()`: `search_query_en`, `profile`, `confidence`, `evidence`,
  `clarifying_question`, `status`. Then the matcher and everything after it
  run unchanged.
- **Own identity.** It gets its own version string, and its own cache file
  or cache-key prefix, so its results never mix with understand-v4.1's in
  `data/cache/understand.jsonl`.
- **Scored by flag.** The evaluation scripts gain a way to select the path.
  The frozen path's existing dev results stay the reference.
- **Logged.** The work goes in the change log.

**Open question for the owner:** the test set can be scored once.
- If the test run is complete (§5), the rule-first path can only be reported
  on dev, unless it was part of that run.
- If the run hasn't happened, the rule-first path must be finished and
  frozen before it, and both paths scored in that single run.

**Unverified:** the detailed rule design and its motivation are not written
down in the repo; there is no `docs/RULES_FEASIBILITY.md`. Get them from the
owner rather than reconstructing them.

## 9. How to run it locally

**Prerequisites.**
- **Python:** 3.13.5 is the one in use, via pyenv. Install the dependencies
  with `pip install -r backend/requirements.txt`; they are pinned (FastAPI
  0.141.1, uvicorn 0.52.1, torch 2.14.0, sentence-transformers 6.1.0,
  faiss-cpu 1.15.1, google-genai 2.22.0, groq 1.7.0).
- **Node and npm:** run `npm install` in `frontend/`.
- **Repo-root `.env`:** holds `GEMINI_API_KEY`, `_2`, `_3` and
  `GROQ_API_KEY`, `_2`, `_3` (see `.env.example`), and optionally
  `ALLOWED_ORIGINS`. It is gitignored.
- **Data:** `data/interim`, `data/index`, `data/cache` and
  `data/processed/chunks.jsonl`. If they are missing, run
  `python scripts/restore_data_backup.py`, which needs a Hugging Face login
  with access to the private repo. It also rebuilds `chunks.jsonl`.
- **`frontend/.env`** must contain `VITE_API_BASE_URL=http://localhost:8000`.
  It is gitignored, so a fresh checkout must create it. Vite reads it at
  start-up; restart the dev server after changing it.

**The two commands**, each in its own terminal:

```
cd ~/Downloads/YojanaMitra/backend && uvicorn app.main:app --port 8000
cd ~/Downloads/YojanaMitra/frontend && npm run dev
```

Then open **http://localhost:8080**.
- **Starting from the repo root:** the backend command must use
  `--app-dir backend` (`uvicorn app.main:app --app-dir backend --port 8000`).
  Without it, it fails with `No module named 'app'`. The module resolution
  was checked, but that form wasn't started.
- **Checking the backend:** `curl http://localhost:8000/health` returns
  `{"status": "ok", "providers": [...]}`.

**Why port 8080 and `strictPort` matter.** The API accepts browser requests
only from `ALLOWED_ORIGINS`, which defaults to exactly
`http://localhost:8080`.
- **What happened on 2026-09-23:** a leftover dev server held 8080, and Vite
  silently moved a new one to 8081. The page loaded, but every `/match`
  preflight got `400 Disallowed CORS origin` and the UI said it couldn't
  reach the server.
- **Now:** `strictPort: true` in `frontend/vite.config.ts` makes a busy port
  fail with "Port 8080 is already in use". Find and stop the old process
  with `lsof -nP -iTCP:8080 -sTCP:LISTEN`.
- **Use `localhost`:** `127.0.0.1:8080` and the "Network" URL are different
  origins, and they are rejected unless added to `ALLOWED_ORIGINS`.

**What to expect.** These timings were measured on this laptop.

| Situation | Time |
|---|---|
| First `/match` after start-up, even for a cached description (one-time warm-up) | 18–25 s (measured 18.6 s and 25.2 s) |
| A new, uncached description | about 15–20 s (measured 16.1 s: three Gemini calls of 3–4 s each) |
| A cached description | about 2 s, and about 0.1 s for an immediate repeat |

- **Provider lines** appear in the backend console, for example
  `[gemini] GEMINI_API_KEY answered in 3.7s (attempt 1)`, plus a WARNING line
  for every failed attempt.
- **Key state:** `/health` shows keys available, cooling down and retired,
  and the last failure.

## 10. Things to protect

- **The freeze.**
  - **Don't change** `understand.py`, `matcher.py`, `retrieval.py`,
    `generate.py`, `postprocess.py`, the index or the constraint records.
  - **Allowed post-freeze changes:** observability or bookkeeping that is
    verified not to change output (byte-identical responses, as was done for
    the logging change). Each one goes in the change log.
  - **New behaviour** goes alongside the frozen code behind a flag (§8).
- **Dev/test discipline.**
  - Tuning looks at dev only, within the agreed budget.
  - Never fix a bug on the single row that revealed it.
  - The test split runs once with `--final`.
  - `skip_scoring` rows are never scored.
  - Once test has been scored, nothing is tuned and re-scored.
- **The HF backup is the only off-laptop copy** of `data/interim` (scheme
  records, silver labels, hand-verification sheets, constraint records
  including the hand corrections and review files), `data/index` and
  `data/cache`. All three are gitignored. By contrast:
  - The raw PDFs (`data/external`) aren't backed up, but can be downloaded
    again from the public `shrijayan/gov_myscheme` dataset.
  - `chunks.jsonl` is rebuildable.
  - `data/gold` and `data/eval` are in git.

  Keep the backup current; it is already behind the cache.
- **API keys.** Never print, log or commit them. `.env` is gitignored.
- **Methodological material the report must not lose:**
  - the two design principles (§2), and the downgrade-only post-processing
    that enforces them;
  - **the gold set:**
    - typed rows;
    - per-pair verification against verbatim scheme clauses
      (`supported` / `supported_unstated`);
    - a dated corrections table;
    - invented scheme names checked absent from the corpus;
    - `skip_scoring` for self-contradicting source records;
  - the dev/test split: stratified, seeded, with the language swap
    documented before any evaluation read it, and scripts that refuse test
    without `--final`;
  - `evaluate_e2e.py --compare`, which credits every changed verdict to the
    step that changed it;
  - the residual-touch calibration, with its honest caveat (gold_084 was in
    view);
  - the change log's structure: evidence type per change, a "can it change a
    status?" column, and known gaps stated plainly (rule #5 never fired on
    dev);
  - the generation measurements: hallucinated eligibility, citation validity
    per attempt, fallback rate;
  - **the classifier's evaluation design:**
    - stratified 5-fold and nested CV;
    - explicit leakage statements;
    - the confound check;
    - group-aware CV for near-duplicates;
    - the label ceiling against blind hand labels;
    - an error analysis with a judgement on each case;
  - **incidents and their fixes:**
    - silent provider failures, now logged;
    - permanent key retirement on per-minute 429s, now a cooldown;
    - the port-drift CORS failure, now `strictPort`.
