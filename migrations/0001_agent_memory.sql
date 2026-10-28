-- Agent memory for the care plan agent (X-CarePlan-Version: 2).
-- Own schema + row level security: on Supabase this keeps the tables out of
-- the auto-generated public API; the service connects as the owner.

create schema if not exists careplan_ai;

-- Drafting sessions: the evidence pack, suggestions and staff decisions.
-- Rows expire after CAREPLAN_AI_AGENT_SESSION_TTL_HOURS and are purged by the app.
create table if not exists careplan_ai.agent_sessions (
    session_id uuid primary key,
    assessment_template_user_id bigint not null,
    status text not null,
    state jsonb not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    expires_at timestamptz not null
);
create index if not exists agent_sessions_expires_idx
    on careplan_ai.agent_sessions (expires_at);
alter table careplan_ai.agent_sessions enable row level security;

-- Per-statement trace of each saved care plan - powers "View" (deck slide 20).
create table if not exists careplan_ai.care_plan_traces (
    care_plan_id bigint primary key,
    session_id uuid not null,
    assessment_template_user_id bigint not null,
    trace jsonb not null,
    created_at timestamptz not null default now()
);
alter table careplan_ai.care_plan_traces enable row level security;
