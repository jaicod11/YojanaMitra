/**
 * Types the UI renders. They are produced from the API's response by
 * toMatchResponse() in ./api.ts, which is the only place the two shapes meet.
 */

/** The languages the UI offers, and the code the API expects for each. Every
 * request takes its language code from this table. */
export const LANGUAGE_CODES = {
  English: "en",
  Hindi: "hi",
  Telugu: "te",
} as const;

export type Language = keyof typeof LANGUAGE_CODES;
export type LanguageCode = (typeof LANGUAGE_CODES)[Language];
export const LANGUAGES = Object.keys(LANGUAGE_CODES) as Language[];

export type Confidence = "high" | "medium" | "low";

export type ProfileConfidence = {
  field: string;
  value: string | number | boolean | string[] | null;
  confidence: Confidence | null;
};

export type Status = "eligible" | "needs_checking" | "not_eligible";

/** One condition shown on a card. verbatim is false when the text is not found
 * word for word in the scheme's eligibility text. */
export type Condition = {
  text: string;
  verbatim: boolean;
};

export type SchemeMatch = {
  slug: string;
  scheme_name: string;
  level: "central" | "state" | null;
  status: Status;
  reason: string;
  matched_clause: string | null;
  /** Every condition the matcher could not check, unfiltered. Kept for
   * completeness; cards render caveat_items and extra_items instead. */
  unverified_conditions: string[];
  unverified_verbatim: boolean[];
  caveats: string[];
  caveats_verbatim: boolean[];
  requires_dependent_note: boolean;
  /** caveats paired with their verbatim flags ("Conditions to check"). */
  caveat_items: Condition[];
  /** unverified_conditions not in caveats, in order, deduplicated, with their
   * verbatim flags ("Also noted for this scheme"). */
  extra_items: Condition[];
  documents: string[];
  apply_url: string | null;
  last_updated: string | null;
  source_url: string | null;
};

export type MatchResponse = {
  profile_confidence: ProfileConfidence[];
  clarifying_question: string | null;
  notice: string | null;
  results: SchemeMatch[];
};
