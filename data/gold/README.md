# Gold set

`gold_set.jsonl` is the evaluation set for scheme matching: 100 user
descriptions, each paired with the profile a system should extract from it
and the schemes it should or should not recommend.

**Frozen 2026-09-21.** The `FROZEN` marker in this directory means the entries
are fixed. Do not edit them to fit system output. If an entry is wrong, fix it
deliberately, say why in its `notes`, and update the counts and date here.

## Format

One JSON object per line:

| field | meaning |
|---|---|
| `id` | `gold_001` … `gold_100` |
| `description` | what the user says |
| `language` | `en` (90), `hi` (5), `te` (5) |
| `expected_profile` | the profile to extract: `occupation`, `state`, `age`, `gender`, `annual_income_inr`, `land_acres`, `caste_category`, `family` (`null` when not stated) |
| `expected_schemes` | slugs the system should recommend (empty unless `positive`) |
| `excluded_schemes` | `exclusion` rows only: slugs the system must **not** recommend |
| `test_type` | `positive`, `exclusion`, `clarify` or `no_match` (see below) |
| `notes` | why the expectation holds, usually citing the scheme's own rule |
| `skip_scoring` | present and `true` only on rows that must not be scored |

Slugs are filenames in `data/interim/schemes/` without `.json`. That
directory is gitignored, so checking slugs needs the local corpus.

## Test types

| test_type | count | what the system should do |
|---|---:|---|
| `positive` | 54 | recommend every slug in `expected_schemes` (46 rows expect 1 scheme, 7 expect 2, 1 expects 3) |
| `exclusion` | 29 | not recommend the slug(s) in `excluded_schemes`: the user looks close to eligible, but a specific rule in the scheme text rules them out |
| `clarify` | 14 | ask for the missing detail (state, income, age, class, registration status…) instead of guessing |
| `no_match` | 3 | recommend nothing: no scheme in the corpus fits, or the named scheme doesn't exist |

In total, the rows reference 38 distinct slugs.

## `skip_scoring`

Scoring code **must skip rows with `skip_scoring: true`** and report how many
it skipped.

One row carries it: **gold_069**, an exclusion test on `pmmvy` (Pradhan Mantri
Matru Vandana Yojana) for a second pregnancy with a single girl. The source
record contradicts itself. Its `eligibility_text` says the scheme covers the
first live birth only, with twins/triplets in the second pregnancy as the
exception. Its `description` and `benefits_text` give ₹6,000 for any second
child who is a girl, which is likely PMMVY 2.0 rules mixed with older text. A
system that reads those sections would be marked wrong for a defensible
answer, so the row stays in the file but is left out of metrics.

## How the set was typed (2026-09-21)

The descriptions, profiles, expected schemes and notes came before this step
and were not changed by it. The step added `test_type` to all 100 rows and
`excluded_schemes` to the exclusion rows:

1. `positive` was set on every row with non-empty `expected_schemes`.
2. `clarify`, `exclusion` and `no_match` were assigned by hand to the rows with
   empty `expected_schemes`.
3. For each exclusion row, the scheme described in `notes` was found in
   `data/interim/schemes/`, and the specific rule the note cites was checked
   in that scheme's `eligibility_text`. For example, gold_029's 60%
   attendance rule is in `csspremsobcsi`, not the PM-YASASVI pre-matric
   scheme, which requires 75%. gold_094 lists both `pm-kisan` and
   `namo-shetkari-mahasanman-nidhi-yojana` because both apply the same
   family-level exclusion.
4. Corrections:
   - gold_004's slug was misspelled (`…mahasamman…`) and was corrected to
     `namo-shetkari-mahasanman-nidhi-yojana`.
   - gold_086 (an unregistered Haryana construction worker) is `clarify`, not
     `exclusion`. Its note refers to "these schemes" without naming one, and
     one Haryana welfare board scheme (`aduw-hbocwwb`) is specifically for
     unregistered workers.
5. Validation: every slug in `expected_schemes` and `excluded_schemes` exists
   in `data/interim/schemes/`, every row has a `test_type`, `positive` rows
   have non-empty `expected_schemes`, and `exclusion` rows have non-empty
   `excluded_schemes`.

`pm-yasasvipmsobcebcdnts` and `pmyasasvipmsobcebcdnts` look like one slug
with a typo, but they are two different schemes: the PM-YASASVI **Post**-Matric
and **Pre**-Matric scholarships. gold_025 (college) uses the first, and
gold_037 (class 9) uses the second.
