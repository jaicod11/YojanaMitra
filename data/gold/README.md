# Gold set

`gold_set.jsonl` is the evaluation set for scheme matching: 104 user
descriptions, each paired with the profile a system should extract from it
and the schemes it should or should not recommend.

**Frozen 2026-09-21.** The `FROZEN` marker in this directory means the entries
are fixed. Do not edit them to fit system output. If an entry is wrong, fix it
deliberately, record it under [Corrections](#corrections), and update the
counts here.

## Provenance

Descriptions were drafted with AI assistance. On 2026-09-21 every
expected_schemes and excluded_schemes value was checked against the scheme's
eligibility text in data/interim/schemes/ with AI assistance; disagreements
were adjudicated by hand and are listed under Corrections. One misspelled slug
(gold_004) was corrected.

## Format

One JSON object per line:

| field | meaning |
|---|---|
| `id` | `gold_001` … `gold_104` |
| `description` | what the user says |
| `language` | `en` (92), `hi` (6), `te` (6) |
| `expected_profile` | the profile to extract: `occupation`, `state`, `age`, `gender`, `annual_income_inr`, `land_acres`, `caste_category`, `family` (`null` when not stated) |
| `expected_schemes` | slugs the system should recommend (empty unless `positive`) |
| `excluded_schemes` | `exclusion` rows only: slugs the system must **not** recommend |
| `test_type` | `positive`, `exclusion`, `clarify` or `no_match` (see below) |
| `verification` | `positive` rows only: for each expected slug, whether the description meets the scheme's conditions (see below) |
| `notes` | why the expectation holds, usually citing the scheme's own rule |
| `skip_scoring` | present and `true` only on rows that must not be scored |
| `split` | `dev`, `test`, or `none` for the `skip_scoring` rows (see below) |

Slugs are filenames in `data/interim/schemes/` without `.json`. That
directory is gitignored, so checking slugs needs the local corpus.

## Test types

| test_type | count | what the system should do |
|---|---:|---|
| `positive` | 54 | recommend every slug in `expected_schemes` (46 rows expect 1 scheme, 7 expect 2, 1 expects 3), flagging any conditions `verification` marks as unstated |
| `exclusion` | 30 | not recommend the slug(s) in `excluded_schemes`: the user looks close to eligible, but a specific rule in the scheme text rules them out |
| `clarify` | 15 | ask for the missing detail (state, income, age, class, registration status…) instead of guessing |
| `no_match` | 5 | say the named scheme can't be found, rather than describe it or map it to another scheme. All five name an invented scheme. |

In total, the rows reference 40 distinct slugs.

## Dev/test split

**Tuning may only look at `dev`. `test` is run once, at the end.** Anything
chosen by looking at results, such as prompts, retrieval settings,
thresholds or fusion weights, must be chosen on `dev` alone. Once `test` has
been scored, changing the system and scoring `test` again makes it a second
dev set.

| split | rows | positive | exclusion | clarify | no_match |
|---|---:|---:|---:|---:|---:|
| `dev` | 20 | 10 | 6 | 3 | 1 |
| `test` | 82 | 43 | 23 | 12 | 4 |
| `none` | 2 | 1 | 1 | 0 | 0 |

The 102 scoreable rows are stratified by `test_type`. The 20 dev rows are
allocated across types in proportion (largest remainder), then drawn with a
fixed seed within each type. `scripts/split_gold.py` reproduces the split
exactly. The two `skip_scoring` rows get `none`.

The stratified draw put no Telugu row in dev. So the script then makes one
seeded swap for each language that has no *positive* row in dev, meaning no
query that should retrieve a real scheme. A positive test row in that
language trades places with an English positive dev row, which leaves the
per-type counts unchanged. This swaps **gold_065** (Telugu, `positive`, a
goldsmith expecting `pmv`) into dev and **gold_005** (English, `positive`)
into test. Dev is 18 English rows, 1 Telugu row and 1 Hindi row, and its
positive rows are 8 English, 1 Hindi (gold_053) and 1 Telugu.

An earlier version of the rule swapped on any `test_type` and picked
gold_102 (Telugu, `no_match`) for dev in place of gold_089. That row asks
about an invented scheme, so it can't show whether Telugu queries retrieve
real schemes. The rule was changed to positive rows only on 2026-09-22,
before any evaluation had read the split, and gold_089 and gold_102 returned
to their stratified-draw splits (dev and test).

## `verification`

Each positive row has one entry per expected slug:

```json
"verification": {"ysrrb": {"status": "supported", "clause": "The benefit of the scheme shall be provided to all the landholder farmer families…", "reason": "AP farmer owning 2 acres."}}
```

- `clause` is quoted verbatim from the scheme's `eligibility_text`. `...`
  joins excerpts that aren't adjacent. Text in `[brackets]` comes from another
  field: PM Vishwakarma's list of 18 trades is in its `faq_text`.
- `reason` says how the description meets, or leaves open, that clause.
- `status` is one of:
  - `supported` (49 pairs): the description states every must-have condition
    about the person (income, caste, age, land, BPL status, enrolment…) and
    indicates no exclusion.
  - `supported_unstated` (14 pairs): nothing contradicts the scheme, but at
    least one person-level must-have condition isn't stated. The system should
    recommend the scheme and flag those conditions as needing checking, not
    claim eligibility.
  - `contradicted`: the description violates a stated condition. The two
    pairs found were resolved (gold_011, gold_059 under Corrections), so none
    remain.

Four things don't count against `supported`, because a user can't be expected
to state them:
- exclusion clauses the description doesn't mention
- application paperwork (documents, bank account)
- citizenship and residence in India
- administrative coverage, such as whether a crop is notified in the district
  or a town is covered by the scheme

## `skip_scoring`

Scoring code **must skip rows with `skip_scoring: true`** and report how many
it skipped. Two rows carry it. In both, the scheme's own record contradicts
itself, so a system could be marked wrong for a defensible answer. Each row
stays in the file but is left out of metrics.

- **gold_069**, an exclusion test on `pmmvy` (Pradhan Mantri Matru Vandana
  Yojana) for a second pregnancy with a single girl. Its `eligibility_text`
  says the scheme covers the first live birth only, with twins/triplets in the
  second pregnancy as the exception. Its `description` and `benefits_text`
  give ₹6,000 for any second child who is a girl, likely PMMVY 2.0 rules mixed
  with older text.
- **gold_032**, a positive test on `pmsscsn`. The scheme's name and category
  tag say Scheduled Caste, but its eligibility text, documents list and FAQ
  say Scheduled Tribe of Nagaland. The raw page
  (`data/external/gov_myscheme/text_data/pmsscsn.pdf`) has the same
  inconsistency, so it is not a parser error.

## How the set was built (2026-09-21)

**Typing.** Each row was given a `test_type`, and exclusion rows were given
`excluded_schemes`:

1. `positive` was set on every row with non-empty `expected_schemes`.
2. `clarify`, `exclusion` and `no_match` were assigned by hand to the rows with
   empty `expected_schemes`.
3. For each exclusion row, the scheme described in `notes` was found in
   `data/interim/schemes/`. For example, gold_029's 60% attendance rule is in
   `csspremsobcsi`, not the PM-YASASVI pre-matric scheme, which requires 75%.
   gold_094 lists both `pm-kisan` and `namo-shetkari-mahasanman-nidhi-yojana`
   because both apply the same family-level exclusion.
4. gold_086 (an unregistered Haryana construction worker) is `clarify`, not
   `exclusion`. Its note refers to "these schemes" without naming one, and
   one Haryana welfare board scheme (`aduw-hbocwwb`) is specifically for
   unregistered workers.

**Verification.**
- Every `expected_schemes` value (63 row–scheme pairs) was read against the
  scheme's `eligibility_text` and classified as above.
- For every `excluded_schemes` value (30 rows), the excluding clause was
  confirmed to be in the excluded scheme's `eligibility_text`.
- Disagreements were adjudicated by hand and are listed under Corrections.

**Invented names.** None of the five invented names appears in any scheme
name or text field in `data/interim/schemes/`. For gold_101–104, no state or
central scheme in the corpus targets that occupation in that state either, so
there is no real scheme a system could fairly offer instead. gold_089 names
no state, so only its name was checked.

**Validation.** The file passes these checks:
- every slug exists in `data/interim/schemes/`
- every row has a `test_type`
- `positive` rows have non-empty `expected_schemes` and a `verification` entry
  for each of them, whose clause appears verbatim in the scheme's text
- `exclusion` rows have non-empty `excluded_schemes`
- no other test type has `expected_schemes`

`pm-yasasvipmsobcebcdnts` and `pmyasasvipmsobcebcdnts` look like one slug
with a typo, but they are two different schemes: the PM-YASASVI **Post**-Matric
and **Pre**-Matric scholarships. gold_025 (college) uses the first, and
gold_037 (class 9) uses the second.

## Corrections

All label changes were made on 2026-09-21, before any system output was
scored against the set.

| row | change | why |
|---|---|---|
| gold_004 | `excluded_schemes`: `namo-shetkari-mahasamman-nidhi-yojana` → `namo-shetkari-mahasanman-nidhi-yojana` | The old slug was misspelled and has no file in the corpus. |
| gold_069 | `skip_scoring` added, `notes` extended | The source record contradicts itself (see above). |
| gold_081 | `clarify` → `exclusion`, `excluded_schemes: ["pmay-u"]` | The user owns a house and previously received a housing benefit. PMAY-U excludes owners of a pucca house and past recipients of Government of India housing schemes. The description doesn't say "pucca" or "Government of India", so this relies on the plain reading. `pmay-g` was not added: it excludes households *living in* pucca houses and has no "previously availed" clause. |
| gold_099 | `no_match` → `clarify` | PM Vishwakarma covers Coir Weaver but not handloom weaving, so "weaver" needs a follow-up question. |
| gold_011 | `expected_schemes`: `["pm-kisan", "ysrrb"]` → `["ysrrb"]` | PM-KISAN requires land "in their names", and her land is still being transferred. The spouse rule the note cites is in `ysrrb`'s text only. |
| gold_059 | `positive` → `clarify`, `expected_schemes`: `["pm-svanidhi"]` → `[]` | PM SVANidhi covers peri-urban vendors who hold a Letter of Recommendation and vend *inside* the town's limits. This vendor sells in a village outside the town, so the right move is to ask where the cart operates. |
| gold_090 | `no_match` → `positive`, `expected_schemes`: `[]` → `["kvps", "nscs"]` | Kisan Vikas Patra and National Savings Certificates are open to "Any individual who is a resident of India", with no income test. PMJJBY, PMSBY and POMIS would also fit once age is known. |
| gold_032 | `skip_scoring` added, `notes` extended | The source record contradicts itself (see above). |
| gold_101–104 | added, `no_match` | `no_match` had only one row, too few to report a hallucination rate. Each asks about an invented scheme; gold_102 is in Telugu and gold_103 in Hindi. |

The `notes` of gold_011, gold_059 and gold_090 keep their original reasoning
and end with a line beginning "Corrected 2026-09-21:" that explains the
current label. Those lines were added on 2026-09-22 and changed no labels.
