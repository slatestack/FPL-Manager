-- Phase 4: every optimiser run, with its inputs, solver statistics, plan and chip analysis.
CREATE TABLE optimization_runs (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    team_id                  INT NOT NULL,
    base_gameweek            INT NOT NULL,
    prediction_run_timestamp TIMESTAMPTZ NOT NULL REFERENCES prediction_runs(run_timestamp),
    inputs                   JSONB NOT NULL,     -- team state (squad, bank, selling prices, FTs, chips), raw picks
    params                   JSONB NOT NULL,
    solver                   JSONB NOT NULL,     -- status, objective, timings, model size per solve
    plan                     JSONB NOT NULL,     -- per-gameweek transfers / XI / bench / captaincy
    chip_evaluation          JSONB NOT NULL,
    chip_recommended         VARCHAR(16),
    objective                NUMERIC(8,3) NOT NULL,
    baseline_objective       NUMERIC(8,3) NOT NULL, -- same horizon with no transfers or chips
    execution_log_id         UUID REFERENCES execution_logs(id)
);
CREATE INDEX optimization_runs_team_idx ON optimization_runs (team_id, created_at DESC);
