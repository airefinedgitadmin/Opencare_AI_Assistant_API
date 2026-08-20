# Care Plan AI API

A FastAPI microservice that drafts a care plan entry (`action` / `outcome` /
`carePlanTitle`) from a resident's completed assessment plus their care-plan
history. It is called by [`care-plan-main-backend`](../care-plan-main-backend)
(the Java service that owns care plan persistence) - this service only
generates a *draft*; the Java backend (and a human reviewer) decides whether
and how it gets saved via its existing `POST /care-plan` endpoint.

## How it fits together

```
care-plan-main-frontend  --(user fills in "Add Care Plan")-->  care-plan-main-backend
                                                                       |
                                              (assembles assessment + history context)
                                                                       v
                                                        care-plan-ai-api  POST /v1/care-plans/generate
                                                                       |
                                                      (LLM drafts action/outcome/title)
                                                                       v
                                              care-plan-main-backend  POST /care-plan  (after review)
```

The Java backend is responsible for retrieving the resident's current
assessment answers and relevant historical care plans from SQL Server and
sending them as context in the request body - this service does not talk to
any database directly. It calls Azure OpenAI to produce a structured draft,
validates the shape of what comes back, and returns JSON whose field names
mirror the Java `AssessmentCarePlanOutDTO` / `AssessmentCarePlanResponseDTO`
so the caller can forward it with no remapping.

## Project structure

```
src/careplan_ai/
├── main.py                  # thin uvicorn entrypoint
├── api/
│   ├── app.py                # create_app() factory + lifespan
│   ├── dependencies.py       # shared FastAPI Depends() (API-key auth)
│   ├── routers/
│   │   ├── care_plan.py      # POST /v1/care-plans/generate
│   │   └── health.py         # /healthz, /readyz
│   └── schemas/
│       ├── base.py           # CamelModel - camelCase wire format
│       ├── request.py        # inbound request models
│       └── response.py       # outbound response model (matches Java DTO)
├── core/
│   ├── config.py              # Settings (pydantic-settings, CAREPLAN_AI_* env)
│   ├── model_config.py        # MODEL_REGISTRY - known models' context windows
│   ├── rate_limit.py          # per-caller-fingerprint rate limiting
│   ├── correlation.py         # correlation-ID middleware
│   ├── exceptions.py          # single exception hierarchy
│   ├── exception_handlers.py  # global handlers -> HTTP + ErrorResponse
│   └── logging.py             # structlog JSON-to-stdout config
├── domain/
│   └── care_plan.py           # CarePlanDraft - strict LLM structured-output schema
└── llm/
    ├── client.py               # AzureChatOpenAI construction (cached)
    ├── prompts/
    │   ├── system_prompt.py     # the system-turn instructions given to the model
    │   ├── formatters.py        # renders a request into the prompt's plain-text sections
    │   └── few_shot_examples.py # worked examples shown before the real request
    ├── context_window.py        # trims oldest history to fit the model's context window
    └── generator.py             # retryable structured-output call

tests/
├── unit/          # mirrors src/careplan_ai/<pkg>/ - no network, no app
└── integration/   # real FastAPI app via TestClient; only the LLM call is faked
```

## Prerequisites

- Python 3.11+
- An Azure OpenAI resource with a chat-completions deployment (e.g. GPT-4o)

## Local setup

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env       # then fill in real values
```

## Environment variables

All variables are prefixed `CAREPLAN_AI_` and can be set in the environment
or in a `.env` file (see `.env.example`).

| Variable | Required | Default | Description |
|---|---|---|---|
| `CAREPLAN_AI_ENVIRONMENT` | no | `local` | `local` \| `development` \| `staging` \| `production` |
| `CAREPLAN_AI_LOG_LEVEL` | no | `INFO` | Python logging level |
| `CAREPLAN_AI_API_KEY` | **yes** | - | Shared secret the caller sends as `X-Api-Key` |
| `CAREPLAN_AI_AZURE_OPENAI_ENDPOINT` | **yes** | - | e.g. `https://your-resource.openai.azure.com` |
| `CAREPLAN_AI_AZURE_OPENAI_API_KEY` | **yes** | - | Azure OpenAI resource key |
| `CAREPLAN_AI_AZURE_OPENAI_DEPLOYMENT` | **yes** | - | Chat-completions deployment name |
| `CAREPLAN_AI_AZURE_OPENAI_API_VERSION` | no | `2024-08-01-preview` | Azure OpenAI API version |
| `CAREPLAN_AI_LLM_TEMPERATURE` | no | `0.2` | Sampling temperature |
| `CAREPLAN_AI_LLM_MAX_RETRIES` | no | `3` | Retries on transient LLM failures |
| `CAREPLAN_AI_LLM_REQUEST_TIMEOUT_SECONDS` | no | `30` | Per-request timeout to Azure OpenAI |

## Running locally

```bash
uvicorn careplan_ai.main:app --reload --port 8000
```

## API reference

### `POST /v1/care-plans/generate`

Headers: `X-Api-Key: <CAREPLAN_AI_API_KEY>`

Request:

```json
{
  "assessmentTemplateUserId": 42,
  "assessmentTemplateId": 7,
  "assessmentType": "Falls Risk",
  "assessmentResponses": [
    { "question": "Has the resident fallen recently?", "answer": "Yes, twice in the last month" }
  ],
  "previousCarePlans": [
    { "carePlanTitle": "Falls Prevention", "action": "Weekly mobility check", "outcome": "No falls since last review" }
  ],
  "reviewNotes": [
    { "note": "Resident continues to need close supervision on stairs", "createdDate": "2026-07-01" }
  ],
  "previousAssessmentCompletions": [
    {
      "completedDate": "2026-02-01",
      "responses": [
        { "question": "Has the resident fallen recently?", "answer": "No" }
      ]
    }
  ],
  "carePlanTitle": null
}
```

All list fields are optional and default to empty - the Java backend sends
every one of this resident's own previous care plans, review notes, and
earlier completions of this same assessment (no fixed count cap; this
service's own context-window sizing is the only size control - see
`llm/context_window.py`). There is deliberately no "similar residents" or
free-text "additional context" field: this service never receives another
resident's data, and there's no source in the Java backend for arbitrary
extra context today.

Response (`200 OK`):

```json
{
  "assessmentTemplateUserId": 42,
  "assessmentTemplateId": 7,
  "action": "Support resident with a supervised mobility exercise plan twice daily.",
  "outcome": "Reduced fall frequency and improved confidence with walking aids.",
  "carePlanTitle": "Falls Prevention",
  "rationale": "Based on two recent falls reported in the assessment."
}
```

`assessmentTemplateUserId`, `assessmentTemplateId`, `action`, `outcome`, and
`carePlanTitle` map directly onto the Java backend's care-plan DTOs.
`rationale` is additive - it is not part of that schema and can be dropped or
shown to a reviewer at the caller's discretion.

Error responses share one shape:

```json
{ "error_code": "authentication_failed", "message": "Missing or invalid API key", "request_id": "..." }
```

| Status | `error_code` | Cause |
|---|---|---|
| 401 | `authentication_failed` | Missing/incorrect `X-Api-Key` |
| 422 | `invalid_request` | Request body malformed, or no context to generate from |
| 429 | `rate_limit_exceeded` | Caller exceeded `CAREPLAN_AI_RATE_LIMIT_REQUESTS` per window; response includes a `Retry-After` header |
| 502 | `llm_generation_failed` | Azure OpenAI call failed or returned an unparseable draft |
| 500 | `internal_error` | Unexpected error |

## Rate limiting

`POST /v1/care-plans/generate` is rate-limited per caller, keyed by the
`X-Api-Key` value sent (not by IP - every real caller is a backend service
sharing one identity). Defaults to 20 requests/60s; tune via
`CAREPLAN_AI_RATE_LIMIT_REQUESTS` / `CAREPLAN_AI_RATE_LIMIT_WINDOW_SECONDS`.
This protects both this service and the Azure OpenAI deployment's own
provisioned-throughput quota from being overwhelmed by one caller. State is
in-memory, which is enough for a single-instance deployment; if this ever
runs as more than one replica, point slowapi at a shared backend (e.g. Redis)
instead.

## Docker

Build the image:

```bash
docker build -t care-plan-ai-api:local .
```

Run it (reading config from `.env`):

```bash
docker run --rm -p 8000:8000 --env-file .env care-plan-ai-api:local
```

Or via Compose:

```bash
docker compose up --build
```

The image is a non-root, multi-stage build (`python:3.12-slim`) with a
built-in `HEALTHCHECK` against `/healthz`.

## Testing

```bash
pytest                       # unit + integration
pytest -m "not integration"  # unit only, no FastAPI app spin-up
pytest --cov=careplan_ai     # with coverage
```

Integration tests exercise the real FastAPI app end-to-end; the only thing
faked is the LLM call itself (`generate_care_plan_draft`), so auth, request
validation, and response shaping are all tested for real.

## Code quality

```bash
ruff check .
mypy src
```
