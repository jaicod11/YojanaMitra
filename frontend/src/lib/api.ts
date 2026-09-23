import { MOCK_RESPONSE } from "@/lib/mock-matches";
import {
  LANGUAGE_CODES,
  type Condition,
  type Confidence,
  type Language,
  type MatchResponse,
  type ProfileConfidence,
  type SchemeMatch,
  type Status,
} from "@/lib/scheme-matches";

/** Longest a /match request may take before it is abandoned. The first
 * request for a new description makes several model calls on the server. */
export const REQUEST_TIMEOUT_MS = 90_000;

// ---------------------------------------------------------------------------
// The API's response as it arrives (backend/app/main.py, MatchResponse)
// ---------------------------------------------------------------------------

export type ApiProfileField = {
  field: string;
  value: unknown;
  confidence: string | null;
};

export type ApiSchemeMatch = {
  slug: string;
  scheme_name: string | null;
  level: string | null;
  status: string;
  reason: string;
  matched_clause: string | null;
  unverified_conditions: string[];
  unverified_verbatim: boolean[];
  caveats: string[];
  caveats_verbatim: boolean[];
  requires_dependent_note: boolean;
  documents: string[];
  apply_url: string | null;
  last_updated: string | null;
  source_url: string | null;
};

export type ApiMatchResponse = {
  profile_confidence: ApiProfileField[];
  clarifying_question: string | null;
  notice: string | null;
  results: ApiSchemeMatch[];
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export type ApiErrorKind = "network" | "timeout" | "http" | "response" | "config";

/** A failed /match call. kind says what failed, so the UI can say so. The
 * message never contains what the person typed. */
export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;

  constructor(kind: ApiErrorKind, message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/**
 * POST /match. clarification is the person's answer(s) to a clarifying
 * question; the query stays the original description, and the server combines
 * the two. With VITE_USE_MOCK=true the built-in mock is returned instead.
 */
export async function fetchMatches(query: string, language: Language, clarification?: string): Promise<MatchResponse> {
  if (import.meta.env.VITE_USE_MOCK === "true") {
    await new Promise((resolve) => setTimeout(resolve, 1100));
    return toMatchResponse(MOCK_RESPONSE);
  }

  const base = (import.meta.env.VITE_API_BASE_URL ?? "").trim().replace(/\/+$/, "");
  if (!base) {
    throw new ApiError("config", "This app is not connected to a server: VITE_API_BASE_URL is not set.");
  }

  const body: { query: string; language: string; clarification?: string } = {
    query,
    language: LANGUAGE_CODES[language],
  };
  if (clarification?.trim()) {
    body.clarification = clarification.trim();
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response: Response;
  let text: string;
  try {
    response = await fetch(`${base}/match`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    text = await response.text();
  } catch {
    if (controller.signal.aborted) {
      throw new ApiError(
        "timeout",
        `The server did not answer within ${REQUEST_TIMEOUT_MS / 1000} seconds. It may be busy; please try again.`,
      );
    }
    // fetch() rejects without a response: the server is down or unreachable,
    // the connection dropped, or the browser blocked the request (CORS).
    throw new ApiError("network", "We couldn't reach the YojanaMitra server. It may be offline, or your connection may have dropped.");
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    throw new ApiError("http", errorDetail(text) ?? "Something went wrong. Please try again.", response.status);
  }
  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch {
    throw new ApiError("response", "The server sent a response we couldn't read.", response.status);
  }
  return toMatchResponse(payload);
}

/** FastAPI's error body: {"detail": "..."} or, for invalid input,
 * {"detail": [{"msg": "...", ...}, ...]}. */
function errorDetail(text: string): string | null {
  try {
    const parsed: unknown = JSON.parse(text);
    if (!isRecord(parsed)) return null;
    const detail = parsed["detail"];
    if (typeof detail === "string" && detail.trim()) return detail.trim();
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => (isRecord(item) && typeof item["msg"] === "string" ? item["msg"] : null))
        .filter((msg): msg is string => Boolean(msg));
      return messages.length ? messages.join("; ") : null;
    }
  } catch {
    // not JSON (a plain-text 500, for example)
  }
  return null;
}

// ---------------------------------------------------------------------------
// Adapter: the one place the API's shape is turned into the UI's
// ---------------------------------------------------------------------------

const STATUSES: readonly Status[] = ["eligible", "needs_checking", "not_eligible"];
const CONFIDENCES: readonly Confidence[] = ["high", "medium", "low"];

/**
 * Checks the response's shape and converts it for the UI. It never changes a
 * status, reason or condition; it only fills in defaults and pairs lists:
 *
 * - notice, clarifying_question: string, or null when absent or empty
 * - profile_confidence value: string / number / boolean / string[] kept as
 *   they are; anything else becomes text; null stays null
 * - profile_confidence confidence: high / medium / low, anything else null
 * - status: must be eligible / needs_checking / not_eligible; any other value
 *   rejects the response rather than being reinterpreted
 * - level: central / state, anything else null (no badge)
 * - scheme_name: falls back to the slug when null
 * - caveat_items: caveats paired with caveats_verbatim; a missing flag counts
 *   as not verbatim, so the "verify" note is shown rather than hidden
 * - extra_items: unverified_conditions not in caveats, order kept,
 *   duplicates dropped, each with its own unverified_verbatim flag
 * - list fields default to [], requires_dependent_note to false
 */
export function toMatchResponse(payload: unknown): MatchResponse {
  if (!isRecord(payload) || !Array.isArray(payload["results"])) {
    throw new ApiError("response", "The server sent a response we couldn't read.");
  }
  const profile = Array.isArray(payload["profile_confidence"]) ? payload["profile_confidence"] : [];
  return {
    profile_confidence: profile.filter(isRecord).map(toProfileField),
    clarifying_question: nonEmptyString(payload["clarifying_question"]),
    notice: nonEmptyString(payload["notice"]),
    results: payload["results"].map(toSchemeMatch),
  };
}

function toProfileField(item: Record<string, unknown>): ProfileConfidence {
  const value = item["value"];
  const confidence = item["confidence"];
  return {
    field: String(item["field"] ?? ""),
    value:
      value === null || value === undefined
        ? null
        : typeof value === "string" || typeof value === "number" || typeof value === "boolean"
          ? value
          : Array.isArray(value)
            ? value.map(String)
            : JSON.stringify(value),
    confidence: CONFIDENCES.includes(confidence as Confidence) ? (confidence as Confidence) : null,
  };
}

function toSchemeMatch(raw: unknown): SchemeMatch {
  if (!isRecord(raw) || typeof raw["slug"] !== "string") {
    throw new ApiError("response", "The server sent a response we couldn't read.");
  }
  const status = raw["status"];
  if (!STATUSES.includes(status as Status)) {
    throw new ApiError("response", `The server sent a result with an unknown status (${String(status)}).`);
  }
  const caveats = stringList(raw["caveats"]);
  const caveatsVerbatim = booleanList(raw["caveats_verbatim"]);
  const unverified = stringList(raw["unverified_conditions"]);
  const unverifiedVerbatim = booleanList(raw["unverified_verbatim"]);
  const level = raw["level"];

  const caveatSet = new Set(caveats);
  const seen = new Set<string>();
  const extraItems: Condition[] = [];
  unverified.forEach((text, i) => {
    if (caveatSet.has(text) || seen.has(text)) return;
    seen.add(text);
    extraItems.push({ text, verbatim: unverifiedVerbatim[i] === true });
  });

  return {
    slug: raw["slug"],
    scheme_name: nonEmptyString(raw["scheme_name"]) ?? raw["slug"],
    level: level === "central" || level === "state" ? level : null,
    status: status as Status,
    reason: typeof raw["reason"] === "string" ? raw["reason"] : "",
    matched_clause: nonEmptyString(raw["matched_clause"]),
    unverified_conditions: unverified,
    unverified_verbatim: unverifiedVerbatim,
    caveats,
    caveats_verbatim: caveatsVerbatim,
    requires_dependent_note: raw["requires_dependent_note"] === true,
    caveat_items: caveats.map((text, i) => ({ text, verbatim: caveatsVerbatim[i] === true })),
    extra_items: extraItems,
    documents: stringList(raw["documents"]),
    apply_url: nonEmptyString(raw["apply_url"]),
    last_updated: nonEmptyString(raw["last_updated"]),
    source_url: nonEmptyString(raw["source_url"]),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function booleanList(value: unknown): boolean[] {
  return Array.isArray(value) ? value.map((item) => item === true) : [];
}
