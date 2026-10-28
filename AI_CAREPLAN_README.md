# Care Plan AI Assistant — How It Works

A reference doc on the AI-generated care plan feature: what data goes in, what
comes back, and the exact prompt/guardrails the model is given.

> **Two versions.** Most of this document describes **v1** (one-shot draft, no
> version header). **v2** (`X-CarePlan-Version: 2`, `src/careplan_ai/agents/care_plan/`)
> is the agent that follows "Care Plan AI integration ppt final.pdf": one-by-one
> Add / View / Dismiss suggestions, every statement traceable to a quote, and
> code-level "nothing missed, nothing invented" checks. v2 reads the assessment,
> previous care plans, review notes **and** carer / nursing / external
> professional / relative notes, and uses a per-care-home **style library
> (RAG, OPENCARE-51)** for tone only — see section 8.

## 1. Who does what

| Layer | Role |
|---|---|
| **Core Application** (main app + database) | Reads the resident's record, packages it up, sends it to the AI service. Never talks to the model directly. Displays the draft; only saves it if a staff member accepts. |
| **AI Drafting Service** (this microservice) | Turns the package into a prompt, applies the guardrails below, manages size limits/retries, calls the model, returns a clean structured answer. Runs as its own Docker container — pluggable into anything over HTTP. |
| **AI Model** (GPT-5, Azure OpenAI, private deployment) | Returns exactly one thing: `action`, `outcome`, an optional `care_plan_title`, and a `rationale`. Nothing else — the output shape is enforced, not just requested. |

Nothing is auto-saved. The AI service has no write access to the database at all.

## 2. Why it's a separate service, not part of the main app

- Runs identically anywhere as one Docker container (laptop, server, any cloud) — no environment drift.
- Plugs into anything that can make an HTTP call — not tied to this one application.
- If the AI is slow, being updated, or briefly down, the rest of the system is unaffected.
- Swapping the model or provider later means changing one small service, not the whole system.
- Scales independently — busy AI usage never forces scaling the whole app.

## 3. The data assembled per request (from the Core Application's database)

All four are sent **in full** — none of this is capped, sampled, or summarised before sending:

1. **Current assessment** — the Q&A of the assessment being actioned right now.
2. **Earlier attempts** — this resident's own past completions of the *same* assessment (to see how answers changed over time).
3. **Care plan history** — every previous care-plan entry for this resident on this topic (title/action/outcome).
4. **Review notes** — every review note logged for this resident on this topic.

## 4. How history is "remembered"

There is no separate AI memory store. The model holds nothing between requests
(it is stateless). Every single draft request re-reads the resident's record
live from the database and sends it again in full — so the model is always
working from the current state of the record, never a cached or stale copy.

## 5. The exact system prompt (verbatim)

This is the complete, unmodified instruction given to the model before every
request — nothing paraphrased:

```
You are a clinical care-planning assistant for a residential and community care provider. Given a resident's completed assessment, their care-plan history, recent review notes, and (when available) their own earlier completions of this same assessment, draft ONE concrete care plan entry: an action, an expected outcome, and (when appropriate) a short title.

Rules:
- Base the draft strictly on the information provided below. Do not invent
  diagnoses, risks, or facts that are not present in the given context.
- Prefer specific, actionable language a carer can follow over vague advice.
- Write the way an experienced, caring member of staff would write in a resident's
  notes: warm and person-centred. If the resident's name happens to appear in the
  given context (e.g. a review note mentions them by name), use it instead of
  "the resident" - otherwise write "the resident"/"they", don't invent a name.
  Avoid stiff clinical or corporate phrasing ("the service user presents with...",
  "it is recommended that...") - prefer plain, direct sentences.
- Keep the action and outcome each to a few sentences.
- Review notes record how previous care plans have gone - use them to judge
  whether a previous action is still relevant, needs updating, or is no longer
  needed (e.g. a note saying a need has resolved means don't repeat that action).
- This resident's earlier completions of the same assessment (when provided) let
  you compare answers over time - if something has changed (e.g. a risk that
  used to be absent is now present, or vice versa), reflect that change in the
  draft rather than treating the latest answers in isolation.
- The draft is reviewed by a qualified carer before use - if the provided
  context is too sparse to draft something specific, say so plainly in the
  action field rather than fabricating detail.
```

### The output is structurally locked, not just instructed

On top of the text above, the model is bound to this exact schema
(`extra="forbid"` — a field the model invents is rejected, not silently accepted):

```
action           : string, required, 1–2000 characters
outcome          : string, optional, up to 2000 characters
care_plan_title  : string, optional, up to 500 characters
rationale        : string, optional, up to 1000 characters
                   ("Brief justification tying the draft back to the given assessment/history.")
```

## 6. The exact per-request template (how the resident's data is written into the prompt)

Whatever data exists (see section 3) is poured into this exact template
before being sent as the "real request" message, after the system prompt above:

```
## Assessment Type
{assessment type, or "Not specified"}

## Assessment Responses (latest completion)
{each answered question as "- Q: ...\n  A: ..." — or "None provided"}

## This Resident's Earlier Completions Of This Same Assessment
{each past completion's date + its own Q&A block — or "None provided"}

## This Resident's Previous Care Plans
{each entry as "- [Title] Action: ... | Outcome: ..." — or "None provided"}

## Recent Review Notes
{each note, with date if known — or "None provided"}

## Title Hint
{the title hint, or "None provided - propose a short, descriptive title if appropriate"}
```

## 7. The 3 worked examples shown to the model (verbatim, exact)

Before the real request, the model is shown 3 example request→response pairs
(taken directly from the real code — not hand-written duplicates, so they can't
drift out of sync with the real format). These calibrate tone and conciseness
beyond what the rules alone achieve.

**Example 1 — a specific, well-supported case:**

```
## Assessment Type
Oral Health Assessment

## Assessment Responses (latest completion)
- Q: Does the resident have dentures?
  A: Yes, upper dentures
- Q: Oral cleanliness
  A: Some particles/tartar/plaque in a few areas

## This Resident's Earlier Completions Of This Same Assessment
None provided

## This Resident's Previous Care Plans
None provided

## Recent Review Notes
- Emily has been finding it harder to manage her dentures independently since her hand grip weakened

## Title Hint
None provided - propose a short, descriptive title if appropriate
```
→
```json
{
  "action": "Support Emily with her denture care each day: offer to help her take her upper dentures out and clean them with a soft brush and denture cleanser, since her grip has made this harder to manage alone. After meals, encourage and assist Emily to rinse her mouth and check around her gums for any trapped food, given the plaque noted at her last check.",
  "outcome": "Emily's dentures are cleaned and cared for daily even as her grip changes, and her mouth stays comfortable with reduced plaque build-up.",
  "care_plan_title": "Denture and Oral Care",
  "rationale": "Assessment notes dentures and some plaque; the review note explains why hands-on support, not just prompting, is now needed, and gives the resident's name."
}
```

**Example 2 — deliberately empty (teaches "say so, don't guess"):**

```
## Assessment Type
Not specified

## Assessment Responses (latest completion)
None provided

## This Resident's Earlier Completions Of This Same Assessment
None provided

## This Resident's Previous Care Plans
None provided

## Recent Review Notes
None provided

## Title Hint
None provided - propose a short, descriptive title if appropriate
```
→
```json
{
  "action": "There isn't enough information yet to draft a specific action - no completed assessment, care-plan history, or review notes are available for this resident. Please complete a relevant assessment first, or add details here about the specific need this care plan should address.",
  "outcome": "Once a specific need is identified, an outcome can be set that reflects it.",
  "rationale": "No assessment responses, previous care plans, or review notes were provided, so there's nothing concrete to base a specific plan on."
}
```

**Example 3 — teaches "notice change over time":**

```
## Assessment Type
Falls Risk Assessment

## Assessment Responses (latest completion)
- Q: Has the resident fallen in the last 3 months?
  A: Yes, once
- Q: Uses a walking aid?
  A: Yes, a frame

## This Resident's Earlier Completions Of This Same Assessment
- Completed 2026-02-01:
    - Q: Has the resident fallen in the last 3 months?
      A: No
    - Q: Uses a walking aid?
      A: No

## This Resident's Previous Care Plans
None provided

## Recent Review Notes
None provided

## Title Hint
None provided - propose a short, descriptive title if appropriate
```
→
```json
{
  "action": "The resident has started using a walking frame and had a fall in the last 3 months, which is a change since their last falls assessment when they weren't using any aid and hadn't fallen. Support them to use their frame for all walking, keep their usual routes clear of clutter and trailing cables, and stay close during transfers in case they need steadying. Check in on their confidence with the frame and let the senior on duty know if they seem unsteady or reluctant to use it.",
  "outcome": "The resident moves around safely with their frame, their confidence with it improves, and they have no further falls.",
  "care_plan_title": "Falls Prevention with Walking Frame",
  "rationale": "The latest assessment shows a new fall and new frame use compared to the previous assessment (no falls, no aid) - this is a meaningful change in risk that the action addresses directly."
}
```

## 7a. The prompting techniques used, and why

Four distinct techniques, stacked together — each doing a different job:

1. **Role prompting** — the system prompt (section 5) is a fixed persona + rule set given before every single request. This sets tone and behaviour globally, not per-request.
2. **Few-shot prompting** — the 3 worked examples above are shown before the real request. Rules alone don't fully calibrate style and conciseness; showing the model real input→output pairs does that better.
3. **Full-context grounding (not RAG)** — the resident's entire relevant history is sent directly, in full, every time. Nothing is retrieved from a vector database or document store — there's no retrieval step at all. This works specifically because one resident's history is small enough to send whole (see section 9's scale numbers).
4. **Structured output / schema-constrained generation** — the model's answer is bound to a strict schema at the API level (section 5's schema), not just asked for in the instructions. This is a hard constraint layered on top of the prompt, not a prompting technique on its own.

**What this deliberately is not:**
- **Not zero-shot** — real worked examples are given, not just instructions.
- **Not full chain-of-thought** — the model gives a brief one-line `rationale`, not a step-by-step reasoning trace.
- **Not RAG for facts** — the resident's own records are never retrieved by similarity search (v2's style library, section 8, is used for tone only).

## 8. RAG is for style only — never for facts

**The resident's facts are not retrieved.** A resident's relevant history is
small enough to hand over in full, and the backend already knows exactly which
records apply (this resident, this pathway), so a database query beats a
similarity search — and can't silently miss a relevant note.

**v2's style library (OPENCARE-51) is the only retrieval.** Each care home has
its own library (one Pinecone namespace per home) of its anonymised past care
plans, tagged by pathway, date, author role, status and source. When drafting
for home X and pathway Y, a fixed graph step (not a tool the model chooses)
fetches the top 5 active, non-excluded examples from home X tagged Y. They are
given to the model as STYLE EXAMPLES — tone, structure, length, vocabulary — and:

- are never evidence: they have no evidence id, can't be cited, and never
  appear behind "View";
- are checked in code: a suggestion that borrows a detail (e.g. "hoist") that
  only an example contains is dropped, and its sources are re-suggested;
- are anonymised before indexing (names, rooms, dates, CHI/NHS numbers, phone,
  email, postcode → placeholders); the retrieval query is anonymised too;
- are audited: each session and saved trace records the example ids used.

No examples for a home/pathway (or the library is off or down) → the neutral
house style; drafting never depends on the library. Endpoints for the backend:
`POST /library/homes/{homeId}/care-plans` (index or re-tag, batch of ≤100 for
backfill), `DELETE …/{carePlanId}`, `POST …/{carePlanId}/exclusion`.

**Data residency:** the Pinecone free tier runs in AWS us-east-1, not the UK.
Only anonymised text is stored, but the DPIA must cover it before real care
plans are indexed.

## 9. Scale, measured from a real request

| | |
|---|---|
| Care plan entries sent | 5 |
| Review notes sent | 3 |
| Total reading used | ~1,714 tokens |
| Available capacity (GPT-5) | 124,500 tokens |
| Used | **~1.4%** |

124,500 tokens is roughly a long novel's worth of reading capacity — that real
resident's entire history used about as much of it as a single paragraph. It
would take **~2,000 combined entries** for one resident on one topic before the
safety-net trimming (oldest-first: review notes → past completions → care
plans) would ever engage. No real resident is close to that.

## 10. Reliability & operations

- **Retries**: up to 3 attempts with exponential backoff, except a
  "deployment not found" error (misconfiguration) — that fails immediately
  rather than retrying a guaranteed failure.
- **Circuit breaker**: if the AI service keeps failing, the Core Application
  stops calling it for ~30s and fails fast with a clear message, instead of
  every request hanging out its timeout.
- **Dedicated capacity**: AI requests run on their own bounded thread pool on
  the Core Application side — a slow AI call never blocks other staff using
  the rest of the system.

## 11. Who can access it, and how

There's no login page and no per-user account on the AI service itself — it
isn't meant to be reached directly by a person. It trusts exactly one caller:
the Core Application, identified by a shared secret.

- **`X-Api-Key` header** — a fixed secret only the Core Application knows.
  Missing or wrong → immediate `401 authentication_failed`, before any other
  code runs. This is a *service-to-service* credential, not a staff login.
- **Individual staff never call it directly.** A staff member authenticates
  into the main app as normal; the AI service only ever hears from the Core
  Application on their behalf, over the app's own internal network call.
- **`X-Caller-Fingerprint` header** — identified-but-anonymous. The Core
  Application computes a one-way scrambled value from the real staff member's
  ID (a salted hash it alone can produce) and sends *that* instead of the real
  ID. The AI service can tell "this is the same person as an earlier request"
  (needed for fair per-person rate limiting below) but can never reverse it
  back to who that person actually is. Real identity is tracked properly,
  once, in the Core Application's own audit log — not duplicated here.

## 12. Rate limiting — protecting the service from being overwhelmed

- **Default: 10 requests per 60 seconds, per real end user** (both numbers
  configurable, no code change needed).
- **Keyed by the caller fingerprint, not the shared API key.** This matters:
  every real request comes from the same Core Application using the same API
  key, so keying on the key alone would bucket the *entire organisation's*
  traffic under one shared limit — a handful of busy staff in the same minute
  could lock everyone else out. Keying per fingerprinted person instead means
  one busy user can never starve anyone else.
- Falls back to the API key (then to raw network address) only if a caller
  somehow sends no fingerprint at all — a safety default, not the normal path.
- Exceeding the limit returns a proper `429` with a `Retry-After` header
  telling the caller exactly how long to wait — never a silent failure or an
  indefinite hang.

## 13. Schema validation — so a bad request (or a bad AI answer) can't break anything

Validation happens at **every boundary**, not just once:

- **Inbound**: every request is checked against a strict schema before any
  other code runs. Wrong types, missing required fields, malformed JSON — all
  rejected immediately with a clear `422 invalid_request`, never reaching the
  model call at all.
- **Outbound (the model's own answer)**: the model's response is bound to an
  equally strict schema — `action`/`outcome`/`care_plan_title`/`rationale`
  only, with real length limits, and `extra="forbid"` means a field the model
  invents on its own is *rejected*, not silently passed through.
- **Three separate contracts, deliberately**: what the Core Application sends,
  what's asked of the model, and what's promised back to the Core Application
  are three independent schemas. Changing one (e.g. tuning what's asked of the
  model) can't accidentally change what the caller receives.
- **One consistent error shape for every failure** — auth failure, bad
  request, rate limit, AI failure, or a genuine bug — always
  `{error_code, message, request_id}`. A raw stack trace is never returned to
  the caller, even when something unexpected breaks.

## 14. Logs & traceability

- **Structured JSON, one line per event**, written straight to the container's
  own standard output — no separate log file to manage; picked up
  automatically by whatever hosts the container.
- **One ID ties everything together.** Every request gets a correlation ID —
  either passed in by the Core Application (so its own logs, this service's
  logs, and the resident's audit trail all share the same ID) or freshly
  generated if none was sent. It's echoed back in the response and included in
  every error, so any failure can be traced end to end from a single ID.
- **No personal details, ever** — log lines contain only reference IDs and
  counts (e.g. "3 review notes sent"), never a resident's or staff member's
  real identity.
- Verbosity is configurable (`INFO` by default) with no code change.
