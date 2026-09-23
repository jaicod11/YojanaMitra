import { useMutation } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import {
  ArrowLeft,
  ArrowUpRight,
  Check,
  Clock,
  FileText,
  Info,
  Languages,
  Leaf,
  LoaderCircle,
  Mic,
  Search,
  ShieldCheck,
  TriangleAlert,
  Users,
  WifiOff,
} from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, fetchMatches } from "@/lib/api";
import {
  LANGUAGES,
  type Condition,
  type Language,
  type MatchResponse,
  type ProfileConfidence,
  type SchemeMatch,
} from "@/lib/scheme-matches";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "YojanaMitra — Find government schemes for you" },
      { name: "description", content: "Describe your situation in simple words and discover government welfare schemes you may qualify for." },
      { property: "og:title", content: "YojanaMitra — Find government schemes for you" },
      { property: "og:description", content: "A multilingual assistant matching Indian citizens with relevant welfare schemes." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

/** Clarifying questions answered per search before we stop asking. */
const MAX_CLARIFY_ROUNDS = 2;
/** How long a request runs before the loading message says it can take a while. */
const SLOW_AFTER_MS = 15_000;

type MatchVariables = { query: string; language: Language; clarification?: string };

function Index() {
  const [query, setQuery] = useState("");
  const [language, setLanguage] = useState<Language>("English");
  // The description the person searched with. It stays the query for every
  // clarification round; their answers travel separately as `clarification`.
  const [originalQuery, setOriginalQuery] = useState("");
  const [answers, setAnswers] = useState<string[]>([]);
  const [answer, setAnswer] = useState("");
  const [questionSkipped, setQuestionSkipped] = useState(false);
  const [response, setResponse] = useState<MatchResponse | null>(null);

  const mutation = useMutation({
    mutationFn: ({ query, language, clarification }: MatchVariables) => fetchMatches(query, language, clarification),
    onSuccess: (data) => {
      setResponse(data);
      setQuestionSkipped(false);
    },
  });
  const slow = useSlowFlag(mutation.isPending, mutation.submittedAt);
  const error = mutation.error ? toApiError(mutation.error) : null;

  function search(event: FormEvent) {
    event.preventDefault();
    const description = query.trim();
    if (!description) return;
    setOriginalQuery(description);
    setAnswers([]);
    setAnswer("");
    setQuestionSkipped(false);
    setResponse(null);
    mutation.mutate({ query: description, language });
  }

  function submitAnswer(event: FormEvent) {
    event.preventDefault();
    const text = answer.trim();
    if (!text) return;
    // Every answer so far goes with the original description, so a second
    // round does not lose the first answer.
    const next = [...answers, text];
    setAnswers(next);
    setAnswer("");
    mutation.mutate({ query: originalQuery, language, clarification: next.join(" ") });
  }

  function retry() {
    if (mutation.variables) mutation.mutate(mutation.variables);
  }

  function startOver() {
    mutation.reset();
    setQuery("");
    setOriginalQuery("");
    setAnswers([]);
    setAnswer("");
    setQuestionSkipped(false);
    setResponse(null);
  }

  // A question is shown above the results it came with, at most
  // MAX_CLARIFY_ROUNDS times per search; after that the results stand as they are.
  const question =
    response?.clarifying_question && answers.length < MAX_CLARIFY_ROUNDS && !questionSkipped && !mutation.isPending
      ? response.clarifying_question
      : null;

  return (
    <main className="min-h-screen bg-background">
      <Header language={language} setLanguage={setLanguage} />
      {!response && !mutation.isPending && !error && <InputScreen query={query} setQuery={setQuery} onSubmit={search} />}
      {!response && mutation.isPending && <LoadingScreen slow={slow} />}
      {!response && !mutation.isPending && error && <ErrorScreen error={error} onRetry={retry} onStartOver={startOver} />}
      {response && (
        <ResultsScreen
          response={response}
          question={question}
          answer={answer}
          setAnswer={setAnswer}
          onAnswer={submitAnswer}
          onSkip={() => setQuestionSkipped(true)}
          updating={mutation.isPending}
          slow={slow}
          error={mutation.isPending ? null : error}
          onRetry={retry}
          onStartOver={startOver}
        />
      )}
    </main>
  );
}

/** True once the current request has been running for SLOW_AFTER_MS. */
function useSlowFlag(pending: boolean, startedAt: number): boolean {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    setSlow(false);
    if (!pending) return undefined;
    const timer = window.setTimeout(() => setSlow(true), SLOW_AFTER_MS);
    return () => window.clearTimeout(timer);
  }, [pending, startedAt]);
  return slow;
}

function toApiError(error: Error): ApiError {
  return error instanceof ApiError ? error : new ApiError("response", "Something went wrong. Please try again.");
}

function Header({ language, setLanguage }: { language: Language; setLanguage: (value: Language) => void }) {
  return (
    <header className="border-b border-border bg-background/95">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-5 py-5 sm:px-8">
        <div className="flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-md bg-primary text-primary-foreground" aria-hidden="true">
            <Leaf className="size-5" />
          </div>
          <div>
            <p className="text-xl font-bold text-foreground">Yojana<span className="text-primary">Mitra</span></p>
            <p className="hidden text-xs text-muted-foreground sm:block">Government schemes, made simpler</p>
          </div>
        </div>
        <div className="w-36 sm:w-44">
          <Select value={language} onValueChange={(value) => setLanguage(value as Language)}>
            <SelectTrigger aria-label="Choose language" className="h-11 border-border bg-surface text-base shadow-none">
              <Languages className="mr-2 size-4 text-primary" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {LANGUAGES.map((item) => <SelectItem key={item} value={item} className="text-base">{item}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
      </div>
    </header>
  );
}

function InputScreen({ query, setQuery, onSubmit }: { query: string; setQuery: (value: string) => void; onSubmit: (event: FormEvent) => void }) {
  return (
    <section className="mx-auto flex max-w-4xl flex-col px-5 pb-20 pt-16 sm:px-8 sm:pt-24">
      <div className="max-w-3xl">
        <p className="mb-4 flex items-center gap-2 text-sm font-semibold text-primary"><ShieldCheck className="size-4" /> Trusted guidance, in your language</p>
        <h1 className="text-4xl font-bold leading-tight text-foreground sm:text-6xl">Find the right government support for you.</h1>
        <p className="mt-6 max-w-2xl text-lg leading-8 text-muted-foreground">Tell us about your work, family, income, or needs in your own words. We’ll help identify schemes that may fit.</p>
      </div>

      <form onSubmit={onSubmit} className="mt-12 rounded-lg border border-border bg-card p-5 shadow-2xl shadow-background/30 sm:p-8">
        <label htmlFor="situation" className="mb-3 block text-lg font-semibold text-card-foreground">Describe your situation in a few sentences</label>
        <div className="relative">
          <Textarea
            id="situation"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="For example: I am a farmer in Andhra Pradesh with 2 acres of land. My daughter is studying in Class 11."
            className="min-h-44 resize-none border-border bg-surface px-5 py-4 pr-14 text-lg leading-8 shadow-none placeholder:text-muted-foreground/80 focus-visible:ring-2"
          />
          <Button type="button" variant="ghost" size="icon" aria-label="Voice input is coming soon" title="Voice input — coming soon" className="absolute bottom-4 right-4 size-10 text-primary hover:bg-accent">
            <Mic className="size-5" />
          </Button>
        </div>
        <div className="mt-5 flex flex-col-reverse items-start justify-between gap-4 sm:flex-row sm:items-center">
          <p className="text-sm leading-6 text-muted-foreground">Do not include Aadhaar numbers or bank details.</p>
          <Button type="submit" variant="hero" size="xl" disabled={!query.trim()} className="w-full sm:w-auto">
            Find schemes <Search className="size-5" />
          </Button>
        </div>
      </form>
    </section>
  );
}

function LoadingScreen({ slow }: { slow: boolean }) {
  return (
    <section className="mx-auto flex min-h-[68vh] max-w-xl flex-col items-center justify-center px-5 text-center" aria-live="polite">
      <div className="flex size-16 items-center justify-center rounded-full border border-primary/30 bg-surface">
        <LoaderCircle className="size-8 animate-spin text-primary" />
      </div>
      <h1 className="mt-7 text-2xl font-semibold text-foreground">Understanding your situation...</h1>
      <p className="mt-3 text-base text-muted-foreground">We’re checking your details against relevant schemes.</p>
      {slow && <p className="mt-3 text-base font-medium text-foreground">Still checking — this can take up to a minute.</p>}
    </section>
  );
}

const errorDetails = {
  network: { title: "We can’t reach the server", icon: WifiOff },
  timeout: { title: "This is taking too long", icon: Clock },
  http: { title: "The server ran into a problem", icon: TriangleAlert },
  response: { title: "Something went wrong", icon: TriangleAlert },
  config: { title: "The app isn’t connected to a server", icon: TriangleAlert },
};

function ErrorScreen({ error, onRetry, onStartOver }: { error: ApiError; onRetry: () => void; onStartOver: () => void }) {
  const details = errorDetails[error.kind];
  const Icon = details.icon;
  return (
    <section className="mx-auto flex min-h-[68vh] max-w-xl flex-col items-center justify-center px-5 text-center" role="alert">
      <div className="flex size-16 items-center justify-center rounded-full border border-warning/30 bg-surface">
        <Icon className="size-8 text-warning" />
      </div>
      <h1 className="mt-7 text-2xl font-semibold text-foreground">{details.title}</h1>
      <p className="mt-3 text-base leading-7 text-muted-foreground">{error.message}</p>
      {error.status !== null && <p className="mt-2 text-xs text-muted-foreground">Error code {error.status}</p>}
      <div className="mt-8 flex flex-wrap justify-center gap-3">
        <Button variant="hero" size="xl" onClick={onRetry}>Try again</Button>
        <Button variant="ghost" size="xl" onClick={onStartOver}>Start over</Button>
      </div>
    </section>
  );
}

const statusOrder: Record<SchemeMatch["status"], number> = {
  eligible: 0,
  needs_checking: 1,
  not_eligible: 2,
};

type ResultsScreenProps = {
  response: MatchResponse;
  question: string | null;
  answer: string;
  setAnswer: (value: string) => void;
  onAnswer: (event: FormEvent) => void;
  onSkip: () => void;
  updating: boolean;
  slow: boolean;
  error: ApiError | null;
  onRetry: () => void;
  onStartOver: () => void;
};

function ResultsScreen({ response, question, answer, setAnswer, onAnswer, onSkip, updating, slow, error, onRetry, onStartOver }: ResultsScreenProps) {
  const sortedResults = [...response.results].sort(
    (a, b) => statusOrder[a.status] - statusOrder[b.status]
  );

  const counts = {
    eligible: sortedResults.filter((scheme) => scheme.status === "eligible").length,
    needs_checking: sortedResults.filter((scheme) => scheme.status === "needs_checking").length,
    not_eligible: sortedResults.filter((scheme) => scheme.status === "not_eligible").length,
  };
  const countSummary = [
    counts.eligible > 0 && `${counts.eligible} eligible`,
    counts.needs_checking > 0 && `${counts.needs_checking} needs checking`,
    counts.not_eligible > 0 && `${counts.not_eligible} not eligible`,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <section className="mx-auto max-w-6xl px-5 pb-24 pt-10 sm:px-8 sm:pt-14">
      <Button variant="ghost" onClick={onStartOver} className="mb-8 px-0 text-muted-foreground hover:bg-transparent hover:text-foreground"><ArrowLeft /> Start over</Button>
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="text-sm font-semibold text-primary">Your matches</p>
          <h1 className="mt-2 text-3xl font-bold text-foreground sm:text-4xl">Schemes worth exploring</h1>
        </div>
        <p className="text-base text-muted-foreground">
          {response.results.length} potential matches found
          {countSummary && <span className="block text-sm sm:text-right">{countSummary}</span>}
        </p>
      </div>

      {updating && (
        <div className="mt-6 flex items-center gap-3 rounded-lg border border-primary/30 bg-surface px-5 py-4" aria-live="polite">
          <LoaderCircle className="size-5 shrink-0 animate-spin text-primary" />
          <p className="text-base text-foreground">
            Updating your results with your answer...
            {slow && <span className="block text-sm text-muted-foreground">Still checking — this can take up to a minute.</span>}
          </p>
        </div>
      )}

      {error && <ErrorBanner error={error} onRetry={onRetry} />}

      {response.notice && (
        <div className="mt-6 flex gap-3 rounded-lg border border-warning/40 bg-warning/10 px-5 py-4" role="status">
          <Info className="mt-0.5 size-5 shrink-0 text-warning" />
          <p className="text-base font-medium leading-7 text-foreground">{response.notice}</p>
        </div>
      )}

      {question && <ClarifyPanel question={question} answer={answer} setAnswer={setAnswer} onAnswer={onAnswer} onSkip={onSkip} />}

      <ProfileSummary profile={response.profile_confidence} />

      {sortedResults.length === 0 ? (
        <div className="mt-8 rounded-lg border border-border bg-card px-6 py-10 text-center">
          <p className="text-lg font-semibold text-card-foreground">No schemes found for this description</p>
          <p className="mt-2 text-base text-muted-foreground">Try describing your work, state, age, income or family in a little more detail.</p>
        </div>
      ) : (
        <div className="mt-8 grid gap-5">
          {sortedResults.map((scheme) => <SchemeCard key={scheme.slug} scheme={scheme} />)}
        </div>
      )}
      <p className="mt-8 text-sm leading-6 text-muted-foreground">Eligibility is based on the information you shared. Please confirm final requirements on the official scheme website before applying.</p>
    </section>
  );
}

function ErrorBanner({ error, onRetry }: { error: ApiError; onRetry: () => void }) {
  const details = errorDetails[error.kind];
  const Icon = details.icon;
  return (
    <div className="mt-6 flex flex-col gap-3 rounded-lg border border-warning/40 bg-surface px-5 py-4 sm:flex-row sm:items-center sm:justify-between" role="alert">
      <div className="flex gap-3">
        <Icon className="mt-0.5 size-5 shrink-0 text-warning" />
        <p className="text-base text-foreground">
          <span className="font-semibold">{details.title}.</span> {error.message}
          <span className="block text-sm text-muted-foreground">The results below are from before your answer.</span>
        </p>
      </div>
      <Button variant="hero" size="lg" onClick={onRetry} className="shrink-0">Try again</Button>
    </div>
  );
}

function ClarifyPanel({ question, answer, setAnswer, onAnswer, onSkip }: {
  question: string;
  answer: string;
  setAnswer: (value: string) => void;
  onAnswer: (event: FormEvent) => void;
  onSkip: () => void;
}) {
  return (
    <form onSubmit={onAnswer} className="mt-6 rounded-lg border border-primary/40 bg-card p-5 sm:p-6">
      <p className="text-sm font-semibold text-primary">One quick question</p>
      <label htmlFor="clarify-answer" className="mt-2 block text-xl font-semibold leading-8 text-card-foreground">{question}</label>
      <Textarea
        id="clarify-answer"
        value={answer}
        onChange={(event) => setAnswer(event.target.value)}
        className="mt-4 min-h-24 bg-surface p-4 text-base"
      />
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button type="submit" variant="hero" size="lg" disabled={!answer.trim()}>Update results</Button>
        <Button type="button" variant="ghost" size="lg" onClick={onSkip}>Skip this question</Button>
        <p className="text-sm text-muted-foreground">Your answer is added to what you told us, and the results below are checked again.</p>
      </div>
    </form>
  );
}

function ProfileSummary({ profile }: { profile: ProfileConfidence[] }) {
  return (
    <div className="mt-8 border-y border-border bg-surface px-5 py-5 sm:px-6">
      <p className="text-sm font-semibold text-gold-soft">Here’s what we understood</p>
      {profile.length === 0 ? (
        <p className="mt-3 text-base text-muted-foreground">We couldn’t pick out any specific details from your description yet.</p>
      ) : (
        <div className="mt-3 flex flex-wrap gap-x-8 gap-y-3">
          {profile.map((item) => (
            <div key={item.field} className="flex items-center gap-2 text-base">
              <span className="capitalize text-muted-foreground">{item.field.replaceAll("_", " ")}</span>
              <span className="font-semibold text-foreground">{formatProfileValue(item.value)}</span>
              {item.confidence && <span className="text-xs capitalize text-muted-foreground">{item.confidence} confidence</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function formatProfileValue(value: ProfileConfidence["value"]): string {
  if (value === null) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}

/** "28 March 2024" from the API's "2024-03-28" (read as a calendar date, so
 * the day does not shift with the viewer's time zone). */
function formatSchemeDate(value: string | null): string | null {
  if (!value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
}

const statusDetails = {
  eligible: { label: "Eligible", className: "border-success/30 bg-success/15 text-success", icon: Check },
  needs_checking: { label: "Needs checking", className: "border-warning/30 bg-warning/15 text-warning", icon: Search },
  not_eligible: { label: "Not eligible", className: "border-neutral-status/40 bg-neutral-status/15 text-neutral-status-foreground", icon: ShieldCheck },
};

function ConditionList({ items }: { items: Condition[] }) {
  return (
    <ul className="space-y-2 text-sm leading-6 text-muted-foreground">
      {items.map((item, index) => (
        <li key={`${index}-${item.text}`} className="flex gap-2">
          <span aria-hidden="true">•</span>
          <span>
            {item.text}
            {!item.verbatim && (
              <span className="ml-1 text-xs text-muted-foreground/70">(our reading of the scheme text — verify on the official page)</span>
            )}
          </span>
        </li>
      ))}
    </ul>
  );
}

function SchemeCard({ scheme }: { scheme: SchemeMatch }) {
  const status = statusDetails[scheme.status];
  const StatusIcon = status.icon;
  const date = formatSchemeDate(scheme.last_updated);
  const hasDocuments = scheme.documents.length > 0;

  return (
    <article className="rounded-lg border border-border bg-card p-5 sm:p-7">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            {scheme.level && <Badge variant="outline" className="border-primary/30 bg-primary/10 uppercase text-primary">{scheme.level}</Badge>}
            <Badge variant="outline" className={status.className}><StatusIcon className="mr-1.5 size-3.5" />{status.label}</Badge>
          </div>
          <h2 className="mt-4 text-2xl font-bold text-card-foreground">{scheme.scheme_name}</h2>
          <p className="mt-3 max-w-3xl text-base leading-7 text-muted-foreground">{scheme.reason}</p>
        </div>
        {scheme.apply_url && (
          <Button asChild variant="hero" size="lg" className="w-full shrink-0 sm:w-auto">
            <a href={scheme.apply_url} target="_blank" rel="noreferrer">View &amp; apply on myScheme (official site) <ArrowUpRight /></a>
          </Button>
        )}
      </div>

      {scheme.requires_dependent_note && (
        <p className="mt-5 flex items-center gap-2 text-sm text-foreground">
          <Users className="size-4 shrink-0 text-warning" />
          This scheme also has conditions about dependents — check the official page.
        </p>
      )}

      <div className={`mt-6 grid gap-5 border-t border-border pt-5 ${hasDocuments ? "md:grid-cols-[1.3fr_1fr] md:gap-8" : ""}`}>
        <Accordion type="multiple">
          {scheme.caveat_items.length > 0 && (
            <AccordionItem value="caveats" className="border-none">
              <AccordionTrigger className="py-2 text-base text-foreground hover:text-primary hover:no-underline">
                Conditions to check ({scheme.caveat_items.length})
              </AccordionTrigger>
              <AccordionContent className="pt-2"><ConditionList items={scheme.caveat_items} /></AccordionContent>
            </AccordionItem>
          )}
          {scheme.extra_items.length > 0 && (
            <AccordionItem value="also-noted" className="border-none">
              <AccordionTrigger className="py-2 text-base text-foreground hover:text-primary hover:no-underline">
                Also noted for this scheme ({scheme.extra_items.length})
              </AccordionTrigger>
              <AccordionContent className="pt-2"><ConditionList items={scheme.extra_items} /></AccordionContent>
            </AccordionItem>
          )}
          {scheme.matched_clause && (
            <AccordionItem value="rule" className="border-none">
              <AccordionTrigger className="py-2 text-base text-foreground hover:text-primary hover:no-underline">See the exact eligibility rule</AccordionTrigger>
              <AccordionContent className="pt-2 text-base leading-7 text-muted-foreground">“{scheme.matched_clause}”</AccordionContent>
            </AccordionItem>
          )}
        </Accordion>
        {hasDocuments && (
          <div>
            <p className="flex items-center gap-2 text-sm font-semibold text-gold-soft"><FileText className="size-4" /> Documents you may need</p>
            <ul className="mt-3 grid gap-2 text-sm text-muted-foreground sm:grid-cols-2 md:grid-cols-1">
              {scheme.documents.map((document, index) => (
                <li key={`${index}-${document}`} className="flex items-center gap-2"><span className="size-1.5 rounded-full bg-primary" />{document}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="mt-6 border-t border-border pt-4">
        <p className="text-xs text-muted-foreground">
          {date ? `Scheme details as of ${date} (myScheme)` : "Scheme details date not available (myScheme)"}
        </p>
      </div>
    </article>
  );
}
