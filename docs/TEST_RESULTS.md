# YojanaMitra: test-set results

**Run date:** 2026-10-29, 19:15–20:03 IST (the four result files' timestamps).
**Pipeline commit:** `6b1ac224` "Added README" (committed 2026-10-29 00:46 IST).
**Scope:** the single scored run of the test split, **82 rows** (43 positive,
23 exclusion, 12 clarify, 4 no_match). The 2 `skip_scoring` rows (gold_032,
gold_069) are excluded: the gold set gives them split `none`, and the run logs
print "(2 skip_scoring rows excluded)".

Every number below comes from the four files in this table unless it says
otherwise. Numbers marked † were computed from the per-row records in the same
file rather than read from its metrics block. Dev figures appear only for
comparison, and each names its dev file.

| File | What it scores | Written (IST) |
|---|---|---|
| `data/eval/test_20261029-191521.json` | Retrieval, raw descriptions | 19:15:21 |
| `data/eval/test_20261029-192759.json` | Retrieval, understand-v4.1 rewrites; profile extraction | 19:27:59 |
| `data/eval/e2e_test_20261029-193348.json` | End to end: understand, hybrid retrieval, matcher | 19:33:48 |
| `data/eval/generate_test_20261029-200355.json` | Generation, through `collect()`/`finalize()` | 20:03:55 |

### What was run

- **Configuration recorded in the files:**
  - understand-v4.1;
  - matcher with 7 rules (applying_for, priority, multi_branch,
    bocw_occupation, person_level_pass, residual_touch, bocw_gate), occupation
    threshold 0.85, residual-touch threshold 0.48, minimum clarify score 0.5;
  - generate-v1, explaining the top 5;
  - hybrid retrieval: RRF k=60, 50 candidates per list, top 10.
- **Index and gold set:** 15,825 chunks, `chunks_sha256` `04b4c023…`, built
  2026-09-22; `gold_sha256` `9ebc22f0…`. All four files agree on both hashes,
  and both match the files on disk.
- **The commit is not recorded in any result file.**
  - It was read from `.git/refs/heads/main` and `.git/logs/HEAD`, without
    running git.
  - No file under `backend/` or `scripts/` has changed since 2026-09-24 07:22,
    so the code is that of `bb0e4c4a`. That includes the post-freeze logging
    and key-cooldown change (#12 in `CHANGES_SINCE_FREEZE.md`).
  - The three later commits are documentation, going by their messages. They
    were not diffed.
- **Providers:** all 82 understand results came from Gemini (the `provider`
  field in the rewritten-retrieval file). All 107 generation calls went to
  gemini-3.1-flash-lite. No step used Groq.
- **The understand calls predate the scored run.**
  - The understand outputs scored here were produced on 2026-10-29 between
    00:49:11 and 00:58:28 IST (`created` in `data/cache/understand.jsonl`).
    That is about 18 hours before the scored files.
  - Every scored run read them from the cache: 82 of 82 rows are "cached" in
    both `test_retrieval_rewritten.log` and `test_e2e.log`.
  - gold_001 appears in the cache twice, 15 s apart.
  - Nothing on disk records which process made those calls, and `data/eval/`
    holds no earlier test result file. **The owner should confirm that nothing
    was scored at that time.**
  - Generation calls were fresh in the scored run (0 cached).

**Units.**
- Retrieval metrics are per row: 43 positive rows, 23 exclusion rows.
- End-to-end metrics are per (row, scheme) pair: 52 positive pairs and 24
  exclusion pairs, because some rows list more than one scheme.
- At these sizes, one positive pair moves a rate by 0.019, one exclusion pair
  by 0.042, and one positive row by 0.023.

## 1. Retrieval

**For English rows, "raw" and "rewritten" are the same query.**
- † For all 72 English rows, the understand-v4.1 search query is
  character-identical to the description.
- The two query sources differ only on 9 of the 10 Hindi and Telugu rows.
  gold_071 (Telugu) got an invalid understand result, so the rewritten run
  also fell back to its raw description.
- So the English figures below are identical by construction, and the
  raw-versus-rewritten comparison is really a translation test on 9 rows.

Positive rows, n = 43. Recall is computed per row (the fraction of that row's
expected schemes found) and averaged over rows.

| Query source | Mode | R@5 | R@10 | MRR@10 | Rows with any expected scheme in top 10 † |
|---|---|---:|---:|---:|---:|
| Raw | BM25 | 0.2093 | 0.3953 | 0.2094 | 17 |
| Raw | Dense | 0.3721 | 0.4419 | 0.3413 | 20 |
| Raw | Hybrid | 0.3605 | 0.3721 | 0.2783 | 17 |
| Rewritten | BM25 | 0.2326 | 0.4186 | 0.2172 | 18 |
| Rewritten | Dense | **0.3953** | **0.4651** | **0.3491** | **21** |
| Rewritten | Hybrid (deployed) | 0.3488 | 0.3953 | 0.2667 | 18 |

By language. One Hindi row moves a Hindi rate by up to 0.33, and one Telugu
row moves a Telugu rate by up to 0.25.

| Lang (positive rows) | Mode | Raw R@5 | Raw R@10 | Raw MRR@10 | Rewritten R@5 | Rewritten R@10 | Rewritten MRR@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| en (36) | BM25 | 0.25 | 0.4722 | 0.2501 | 0.25 | 0.4722 | 0.2501 |
| en (36) | Dense | 0.4028 | 0.4861 | 0.366 | 0.4028 | 0.4861 | 0.366 |
| en (36) | Hybrid | 0.3889 | 0.4028 | 0.2954 | 0.3889 | 0.4028 | 0.2954 |
| hi (3) | BM25 | 0.0 | 0.0 | 0.0 | 0.3333 | 0.3333 | 0.1111 |
| hi (3) | Dense | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| hi (3) | Hybrid | 0.5 | 0.5 | 0.4444 | 0.3333 | 0.5 | 0.2222 |
| te (4) | BM25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| te (4) | Dense | 0.0 | 0.0 | 0.0 | 0.25 | 0.25 | 0.0833 |
| te (4) | Hybrid | 0.0 | 0.0 | 0.0 | 0.0 | 0.25 | 0.0417 |

**Dense won.** It is best on every positive metric, for both query sources.
- **Hybrid's R@10 is below BM25's alone:** 0.3953 vs 0.4186 rewritten, and
  0.3721 vs 0.3953 raw. Both still find at least one expected scheme on the
  same number of rows (18 rewritten, 17 raw).
- **Per row, on rewritten queries †:**
  - dense's top 10 held more of a row's expected schemes than hybrid's on 3
    rows, and never fewer;
  - dense ranked the first expected scheme higher on 10 rows, hybrid on 1.

**That does not match the deployed mode.** The frozen, deployed mode is
hybrid, and the end-to-end and generation files both record
`retrieval_mode: "hybrid"`.

Median retrieval latency per query (rewritten run): BM25 0.034 s, dense
0.056 s, hybrid 0.08 s.

## 2. Exclusion

The false-inclusion rate is the share of the 23 exclusion rows with any
excluded scheme in the top 10.
- Retrieval isn't meant to exclude anyone, so this is a diagnostic.
- All 23 exclusion rows are English, so raw and rewritten are identical.
- "Via an exclusions chunk" means the excluded scheme was matched through its
  exclusions section: the clause that rules the person out.

| Mode | False-inclusion rate | Rows (of 23) † | Of those, via an exclusions chunk |
|---|---:|---:|---:|
| BM25 | 0.3913 | 9 | 5 |
| Dense | 0.4348 | 10 | 2 |
| Hybrid (deployed) | 0.3913 | 9 | 3 |

What became of the 9 hybrid inclusions in the end-to-end run †:

| Row | Excluded scheme | Rank | Section matched | Matcher status |
|---|---|---:|---|---|
| gold_016 | ysrrb | 1 | eligibility | needs_checking |
| gold_040 | ignwps | 6 | exclusions | not_eligible |
| gold_051 | disabled-pension | 1 | overview | not_eligible |
| gold_070 | pmmvy | 8 | exclusions | needs_checking |
| gold_073 | ssy | 1 | eligibility | needs_checking |
| gold_080 | pmay-u | 6 | overview | needs_checking |
| gold_091 | pmuy | 3 | overview | needs_checking |
| gold_097 | apy | 1 | eligibility | not_eligible |
| gold_098 | ysrrb | 1 | exclusions | needs_checking |

None was shown as eligible: 3 were ruled out, and 6 were left at
needs_checking.

## 3. Profile extraction

These figures come from the rewritten-retrieval file; the end-to-end file
reports identical ones.
- **Accuracy** is over rows whose gold value is non-null.
- **False fill** is over rows whose gold value is null.
- **Occupation** uses approximate (word-overlap) matching.
- **The dev column** comes from `e2e_dev_20260922-191512.json`, the final
  understand-v4.1 run. Dev false fill was 2 / 107 = 0.0187.

| Field | Gold non-null | Correct | Accuracy | Gold null | Filled | False-fill rate | Dev (correct / non-null) |
|---|---:|---:|---:|---:|---:|---:|---|
| occupation | 50 | 42 | 0.84 | 32 | 1 | 0.0312 | 9 / 12 |
| state | 48 | 47 | 0.9792 | 34 | 1 | 0.0294 | 11 / 11 |
| age | 18 | 15 | 0.8333 | 64 | 0 | 0.0 | 5 / 6 |
| gender | 13 | 8 | 0.6154 | 69 | 0 | 0.0 | 2 / 2 |
| annual_income_inr | 24 | 24 | 1.0 | 58 | 0 | 0.0 | 0 / 0 |
| land_acres | 9 | 9 | 1.0 | 73 | 0 | 0.0 | 2 / 2 |
| caste_category | 15 | 15 | 1.0 | 67 | 0 | 0.0 | 0 / 0 |
| **Overall** | **177** | **160** | **0.904** | **397** | **2** | **0.005** | **29 / 33 = 0.8788** |

By language:

| Lang | Correct / gold non-null | Accuracy | False-fill rate |
|---|---:|---:|---:|
| en | 134 / 147 | 0.9116 | 0.0056 |
| hi | 14 / 15 | 0.9333 | 0.0 |
| te | 12 / 15 | 0.8 | 0.0 |

**Compared with dev: higher overall (0.904 vs 0.879), but level on the fields
both splits measured.**
- Dev had no gold income or caste values. On test, those two fields are 39 of
  39 correct.
- On the other five fields, test is 121/138 = 0.8768 †, against dev's 29/33 =
  0.8788.
- False fill is lower on test: 0.005 vs 0.0187.

The misses †:
- **age (3):**
  - gold_008, "I am a farmer in Telangana, aged 55." This is the documented
    limit: an age stated without a first-person word is dropped.
  - gold_052: the son's age, 15.
  - gold_071.
- **gender (5):** all five gold values are "female", and the model left gender
  empty each time:
  - gold_011 and gold_100 (widows);
  - gold_066 (pregnant);
  - gold_091 (the description doesn't state gender);
  - gold_071.
- **state (1):** gold_071.
- **false fills (2):**
  - gold_004, state "Maharashtra": the person lives abroad, and the family's
    farmland is in Maharashtra;
  - gold_090, occupation "employee", from "in my job".
- **All 3 Telugu misses are gold_071.** Its understand result was invalid, so
  its profile is empty.

## 4. End to end

These results come from `e2e_test_20261029-193348.json`.
- **The statuses are the matcher's,** before post-processing.
- **Each expected or excluded scheme is also matched directly,** so its status
  is known even when retrieval missed it.
- **This run used the same understand output as the rewritten retrieval
  run** (cached). † Its ranks for all 76 target pairs match that run's hybrid
  ranks.

### Positive pairs (52, over 43 rows)

The row and column totals come from the metrics block; the cells are
computed †.

| Matcher status | In top 10 | Not in top 10 (rank None) | Total |
|---|---:|---:|---:|
| eligible | 4 | 4 | 8 |
| needs_checking | 15 | 29 | 44 |
| not_eligible | 0 | 0 | 0 |
| **Total** | **19** | **33** | **52** |

- **False-exclusion rate: 0.0** (0 of 52).
- **Retrieved at all:** 19 of 52 expected schemes had a rank; 33 had rank
  None. Ranks of the 19: 1 (×9), 2 (×2), 3 (×2), 4 (×2), 5, 6 (×2), 10.
- **End-to-end rate** (in the top 10 and not not_eligible): **0.3654** (19 of
  52).
- **By row †:** all expected schemes retrieved on 16 rows, some on 2, none on
  25.

**These are two different failures:**
- **The matcher got it wrong:** never, on these pairs. 0 of 52 expected
  schemes were ruled out, and 0 of 24 excluded schemes were marked eligible
  (see the exclusion table).
- **Retrieval never surfaced it:** 33 of 52 (0.6346 †).
  - **Every positive pair lost end to end was lost in retrieval.**
  - For 4 of those 33, the matcher would have said eligible.

By language †:

| Lang | Positive rows | Pairs | In top 10 | eligible | needs_checking | not_eligible |
|---|---:|---:|---:|---:|---:|---:|
| en | 36 | 42 | 16 | 6 | 36 | 0 |
| hi | 3 | 4 | 2 | 1 | 3 | 0 |
| te | 4 | 6 | 1 | 1 | 5 | 0 |

**After post-processing** (from the generation file, which has the same
understand output, retrieval and matcher results †):
- the 19 retrieved expected schemes ended eligible 3, needs_checking 16,
  not_eligible 0;
- post-processing downgraded one of them from eligible to needs_checking.

### Exclusion pairs (24, over 23 rows)

| Matcher status | In top 10 | Not in top 10 | Total | Rate |
|---|---:|---:|---:|---:|
| not_eligible (correct) | 3 | 1 | 4 | 0.1667 |
| needs_checking (not ruled out) | 6 | 14 | 20 | 0.8333 |
| eligible (wrongly included) | 0 | 0 | 0 | 0.0 |
| **Total** | **9** | **15** | **24** | |

On exclusion pairs, the failure is the matcher's caution, not a wrong verdict.
- 20 of 24 excluded schemes stay at needs_checking, because the profile gave
  no hard evidence against them.
- Only the 6 that were also retrieved would reach the person, labelled needs
  checking.

### All candidates

- **820 candidates** (82 rows × 10): the matcher marked 21 eligible, 439
  needs_checking and 360 not_eligible.
- **The 4 no_match rows' 40 candidates †:** 20 needs_checking, 20
  not_eligible, 0 eligible.

### Clarify rows (12)

The e2e file has no clarify metric, so this is reported qualitatively, as on
dev.
- **The question is chosen in code** (`select_clarifying_field`) and phrased
  from a template.
- **Understand's own question** is recorded as `fallback_question` (present on
  6 of the 12), but it isn't the question asked.
- **11 of 12 rows got a question:** state on 9, annual income on 1, occupation
  (refine) on 1.
- **gold_059 got none:** no field was selected.

| Row | Lang | Field asked | What the gold notes say is needed | Asked for a field the notes name? (my reading, not a metric) |
|---|---|---|---|---|
| gold_009 | en | state | state and landholding | yes |
| gold_013 | en | annual income | exact land area (given in bigha) | no |
| gold_033 | en | state | daughter's age/class, state, income, caste | yes |
| gold_034 | en | state | caste, income, state | yes |
| gold_042 | en | state | age and BPL/income | no |
| gold_047 | en | state | exact age and BPL/income | no |
| gold_054 | en | state | disability %, age, state, income/BPL | yes |
| gold_059 | en | none | where the cart operates (inside the town's limits or not) | no |
| gold_060 | en | state | loan amount, business type, location | yes (location) |
| gold_076 | en | state | daughter's exact age first; then state, school type, income | yes, but not the field the notes lead with |
| gold_087 | en | state | exact trade, state, welfare-board registration | yes |
| gold_088 | hi | occupation (refine) | welfare-board registration | no |

By this reading, 7 of the 12 rows asked for a field the notes name, and 5 did
not. State dominates because a missing state blocks most of the top 10.

## 5. Generation

These results come from `generate_test_20261029-200355.json`. Every result
passes through `collect()`/`finalize()`, as `/match` does. **The dev figures
predate post-processing, so the comparison is not like for like.**

Where the 820 returned reasons came from †:

| Source | Reasons | Sent to the LLM? |
|---|---:|---|
| LLM | 307 | yes |
| Fallback template (no valid LLM answer after the retry) | 10 | yes, but the answer was rejected |
| Code, for not_eligible results | 360 | no, by design |
| Template, for results outside the five explained | 143 | no, by design |

The model path (LLM plus fallback) covered 307 + 10 = 317 schemes.

| Metric | Test | Dev (`generate_dev_20260922-202840.json`) |
|---|---|---|
| Schemes explained (model path) | 317 | 84 |
| LLM calls | 107: 51 rows with 1 call, 28 rows with 2 †; 3 rows had nothing to explain | 24 |
| Rows retried | 28 | 5 |
| **Hallucinated eligibility** (vs final status, all 820 reasons) | **0** | 0 |
| Hallucinated eligibility vs matcher status | 0 | 0 |
| Claims caught and retried by validation | 1 | 0 |
| **Citation validity, first attempt** | **668 / 677 = 0.9867** | 187 / 188 = 0.9947 |
| Citation validity, all attempts | 891 / 909 = 0.9802 | 230 / 231 = 0.9957 |
| Final citations | 1,153, 0 invalid | 243, 0 invalid |
| **Fallback template instead of the LLM** | **10 / 317 = 0.0315**, in 8 rows | 1 / 84 = 0.0119 |
| Relevance flags | 156 | 38 |
| Eligible schemes downgraded | 2 (gold_011 fseg, gold_085 hefas-pbaocwwb †) | 0 |
| no_match rows naming the invented scheme | 0 | 0 |

- **The caught claim:** on gold_036 (Telugu), a needs_checking scheme's reason
  said "అర్హత ఉంది" ("is eligible"). Validation rejected it and the row was
  retried.
- **The 10 fallbacks** all read "no valid answer after the retry":
  - ysrrb ×4 (gold_005, gold_011, gold_098, gold_100);
  - pmay-g ×3 (gold_010, gold_015, gold_098);
  - svsgcfrss (gold_088);
  - nbcfdc-gls and bby (gold_090).
- **The quote errors fall on the same schemes.** Validation recorded 8 bad
  quotes for ysrrb, 8 for pmay-g and 2 for svsgcfrss.

**Against dev:**
- Hallucinated eligibility stayed at 0, now over 317 explained schemes instead
  of 84.
- First-attempt citation validity fell, from 0.9947 to 0.9867.
- The fallback rate rose, from 0.0119 to 0.0315.
- Neither run had an invalid citation in its final output.

## 6. Honest reading

**What the numbers support: the two design principles held on unseen data.**
No expected scheme was ruled out (0 of 52), and no excluded scheme was called
eligible (0 of 24). No explanation claimed an eligibility the final status
didn't give (0 of 820 reasons), and no final citation was invalid (0 of
1,153). Extraction held up too. On the five fields dev measured, test was
level with dev (0.8768 vs 0.8788), and the two fields dev couldn't measure,
income and caste, were 24/24 and 15/15. What the numbers don't show is a
decisive system. 44 of 52 expected schemes came back needs_checking rather
than eligible, and only 4 of 24 excluded schemes were ruled out. That safety
record is partly the cost of the caution, so these results measure the caution
as much as the accuracy.

**Retrieval is where the system loses, and hybrid was not the right deployed
choice on these numbers.** 33 of 52 expected schemes never reached the top
10, and that accounts for every end-to-end loss on positive pairs. Dense beat
hybrid on every positive metric for both query sources, and hybrid's R@10 was
below BM25's alone. The R@10 gap rests on only 3 rows (a sign test on 3 to 0
gives p = 0.25 †), but no row went the other way. The ranking gap is clearer:
dense put the first expected scheme higher on 10 rows and hybrid on 1
(p ≈ 0.012 †). Dev pointed the same way: on raw queries, R@10 was 0.60 for
dense against 0.50 for hybrid (`dev_20260922-130736.json`). I found no recorded
rationale for choosing hybrid in the repo. Hybrid's only edge here is one
fewer exclusion row with the excluded scheme in the top 10 (9 vs 10). Two
limits apply. These are retrieval-only metrics, so a dense pipeline's
end-to-end rate was never measured. And switching now and re-scoring would
turn the test split into a second dev set.

**Telugu: the test set supports no claim beyond "it failed on most of the few
rows tested".** The same goes for Hindi. There are 4 Telugu and 3 Hindi
positive rows, and no exclusion rows in either language. With the raw
description, retrieval found nothing for any Telugu row in any mode. With the
rewrite, it found one Telugu scheme (gold_043, ignwps), and end to end 1 of
the 6 Telugu expected schemes was retrieved. Translation doesn't explain all
of it: gold_017 and gold_036 were rewritten into clean English, and retrieval
still missed. Understand failed outright on one row, gold_071, which returned
invalid with an empty profile. That row accounts for all three Telugu
extraction misses; the other Telugu fields were 12 of 12. For Hindi, the
rewrite helped BM25 but moved gold_018's first expected scheme from rank 1 to
rank 6 in hybrid. One row moves a Telugu rate by 0.25 and a Hindi rate by 0.33.

**The false-inclusion rate sounds worse than it is, and precision here is only
a lower bound.** Hybrid's false-inclusion rate of 0.3913 (9 of 23 rows) is a
retrieval diagnostic. What the person sees is the matcher's verdict on those
9: 3 ruled out, 6 shown as needs_checking, none as eligible. In 3 of the 9,
the clause that rules the person out is what pulled the scheme in. The cost is
clutter, and a needs-checking label on schemes the person can't get, but not a
wrong yes. As for precision, these files contain no precision metric, and the
expected-scheme lists are not exhaustive. A retrieved or eligible scheme
missing from the gold list may be one the person truly qualifies for, so any
precision figure computed from this gold set is a lower bound. For example, of
the 21 candidates the matcher called eligible, 4 are gold-expected and none is
gold-excluded †. The other 17 are unlabelled, not known to be wrong. By the
same token, recall is measured only against the schemes the gold rows list.

## 7. Not in the files

- **The commit:** no result file records it (see the header).
- **The skip_scoring count:** not in the JSON; it appears only in the logs and
  the gold set.
- **Precision:** no file has a precision metric.
- **A clarify-row metric:** there is none; clarify behaviour is qualitative
  only.
- **Per-language end-to-end and generation metrics:** the per-language
  end-to-end table above was computed from rows. Generation has no
  per-language breakdown.
- **End-to-end MRR:** the e2e file has ranks but no MRR.
- **Hindi and Telugu exclusion behaviour:** the test split has no exclusion
  rows in either language.
- **Accuracy for the ten newer profile fields** (education class and stage,
  residence, marital status, disability, minority, BPL, BOCW registration,
  prior schemes, applying_for): the e2e file gives fill counts only. `family`
  isn't scored.
- **The provider mix, as a metric:** it was taken from per-row `provider`
  fields and the caches.
- **Latency outside retrieval:** there is no end-to-end or generation timing.
