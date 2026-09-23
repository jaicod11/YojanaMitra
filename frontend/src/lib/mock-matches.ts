import type { ApiMatchResponse } from "@/lib/api";

/**
 * Offline stand-in for POST /match, served when VITE_USE_MOCK=true. It has the
 * API's raw shape, so it goes through the same adapter as a real response.
 */
export const MOCK_RESPONSE: ApiMatchResponse = {
  profile_confidence: [
    { field: "occupation", value: "farmer", confidence: "high" },
    { field: "state", value: "Andhra Pradesh", confidence: "high" },
    { field: "land_acres", value: 2, confidence: "medium" },
    { field: "bpl_household", value: true, confidence: "low" },
  ],
  clarifying_question: "Could you please provide the total annual income for your household?",
  notice: null,
  results: [
    {
      slug: "pm-kisan",
      scheme_name: "Pradhan Mantri Kisan Samman Nidhi",
      level: "central",
      status: "eligible",
      reason:
        "You are a farmer with land in your name, which this scheme requires. 1 condition of this scheme still to confirm.",
      matched_clause:
        "All landholding farmers' families, which have cultivable land holding in their names are eligible to get benefit under the scheme.",
      unverified_conditions: [
        "All Persons who paid Income Tax in last assessment year",
        "Farmer families holding land must have their names in the land records.",
      ],
      unverified_verbatim: [true, false],
      caveats: ["Farmer families holding land must have their names in the land records."],
      caveats_verbatim: [false],
      requires_dependent_note: false,
      documents: ["Aadhaar Card", "Landholding papers", "Savings Bank Account"],
      apply_url: "https://www.myscheme.gov.in/schemes/pm-kisan",
      last_updated: "2024-03-28",
      source_url: "https://www.myscheme.gov.in/schemes/pm-kisan",
    },
    {
      slug: "mock-girl-child-scholarship",
      scheme_name: "Girl Child Scholarship (mock)",
      level: "state",
      status: "needs_checking",
      reason:
        "Your daughter's class matches this scheme. Your family income must still be confirmed. 2 conditions of this scheme still to confirm.",
      matched_clause: "Girl child enrolled in Class 9–12 in a recognised institution",
      unverified_conditions: [
        "The annual family income should not exceed ₹2,50,000.",
        "The student must have scored at least 60% in the previous class.",
      ],
      unverified_verbatim: [true, true],
      caveats: [
        "The annual family income should not exceed ₹2,50,000.",
        "The student must have scored at least 60% in the previous class.",
      ],
      caveats_verbatim: [true, true],
      requires_dependent_note: true,
      documents: [],
      apply_url: "https://www.myscheme.gov.in/",
      last_updated: "2024-03-28",
      source_url: "https://www.myscheme.gov.in/",
    },
    {
      slug: "pm-sym",
      scheme_name: "Pradhan Mantri Shram Yogi Maan-Dhan",
      level: "central",
      status: "not_eligible",
      reason: "This scheme's age limit is 18-40; you said 45.",
      matched_clause: "The applicant should be between 18 and 40 years of age.",
      unverified_conditions: [],
      unverified_verbatim: [],
      caveats: [],
      caveats_verbatim: [],
      requires_dependent_note: false,
      documents: ["Aadhaar card", "Savings bank account"],
      apply_url: "https://www.myscheme.gov.in/schemes/pm-sym",
      last_updated: "2024-03-28",
      source_url: "https://www.myscheme.gov.in/schemes/pm-sym",
    },
  ],
};
