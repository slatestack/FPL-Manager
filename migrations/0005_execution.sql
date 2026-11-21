-- Phases 5-6: execution bookkeeping, deadline actions, manager history for the audit view.

ALTER TABLE system_config
    ADD COLUMN auth_last_check_at TIMESTAMPTZ,
    ADD COLUMN auth_last_check_ok BOOLEAN,
    ADD COLUMN auth_last_error    TEXT,
    ADD COLUMN last_auth_alert_at TIMESTAMPTZ;

-- Each deadline action (dry_run at T-24h, final_lock at T-30m, commit at T-15m) runs once per gameweek.
CREATE TABLE scheduled_actions (
    team_id     INT NOT NULL,
    gameweek    INT NOT NULL,
    action      VARCHAR(16) NOT NULL CHECK (action IN ('dry_run', 'final_lock', 'commit')),
    status      VARCHAR(16) NOT NULL CHECK (status IN ('RUNNING', 'SUCCESS', 'FAILED', 'SKIPPED', 'MISSED')),
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    detail      JSONB,
    PRIMARY KEY (team_id, gameweek, action)
);

-- Per-gameweek results of the managed entry (public entry history), for projected vs actual.
CREATE TABLE manager_history (
    entry_id             INT NOT NULL,
    gameweek             INT NOT NULL,
    points               INT NOT NULL,
    total_points         INT,
    overall_rank         INT,
    bank                 INT,
    value                INT,
    event_transfers      INT,
    event_transfers_cost INT,
    points_on_bench      INT,
    updated_at           TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (entry_id, gameweek)
);

CREATE INDEX execution_logs_team_gw_idx ON execution_logs (team_id, gameweek, created_at DESC);

-- Execution audit detail: trigger, guardrail report, request payloads and FPL responses.
ALTER TABLE execution_logs
    ADD COLUMN trigger VARCHAR(16),
    ADD COLUMN details JSONB,
    ADD COLUMN finished_at TIMESTAMPTZ;
