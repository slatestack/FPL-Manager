-- Phase 3: historical archive (training data), xP run audit and richer prediction output.

-- ---------------------------------------------------------------- past seasons (public FPL archive)
CREATE TABLE archive_players (
    season     VARCHAR(7) NOT NULL,            -- e.g. '2025-26'
    element    INT NOT NULL,                   -- FPL id within that season
    code       INT NOT NULL,                   -- stable cross-season player code (= players.code)
    web_name   VARCHAR(64) NOT NULL,
    position   VARCHAR(3) NOT NULL CHECK (position IN ('GKP', 'DEF', 'MID', 'FWD')),
    team_code  INT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (season, element)
);

CREATE TABLE archive_fixtures (
    season       VARCHAR(7) NOT NULL,
    fixture_id   INT NOT NULL,
    gameweek_id  INT,
    team_h_code  INT NOT NULL,
    team_a_code  INT NOT NULL,
    kickoff_time TIMESTAMPTZ,
    finished     BOOLEAN NOT NULL,
    team_h_score INT,
    team_a_score INT,
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (season, fixture_id)
);

CREATE TABLE archive_player_matches (
    season                  VARCHAR(7) NOT NULL,
    element                 INT NOT NULL,
    fixture_id              INT NOT NULL,
    gameweek_id             INT NOT NULL,
    was_home                BOOLEAN NOT NULL,
    kickoff_time            TIMESTAMPTZ,
    minutes                 INT NOT NULL,
    starts                  INT,
    total_points            INT NOT NULL,
    goals_scored            INT NOT NULL,
    assists                 INT NOT NULL,
    clean_sheets            INT NOT NULL,
    goals_conceded          INT NOT NULL,
    own_goals               INT NOT NULL,
    penalties_saved         INT NOT NULL,
    penalties_missed        INT NOT NULL,
    yellow_cards            INT NOT NULL,
    red_cards               INT NOT NULL,
    saves                   INT NOT NULL,
    bonus                   INT NOT NULL,
    bps                     INT NOT NULL,
    defensive_contribution  INT,                   -- NULL before 2025-26 (rule did not exist)
    expected_goals          NUMERIC(6,2),
    expected_assists        NUMERIC(6,2),
    expected_goals_conceded NUMERIC(6,2),
    value                   INT,
    fpl_xp                  NUMERIC(6,2),          -- FPL's own pre-match xP where the archive has it
    updated_at              TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (season, element, fixture_id)
);

-- ---------------------------------------------------------------- xP runs
CREATE TABLE prediction_runs (
    run_timestamp  TIMESTAMPTZ PRIMARY KEY,        -- joins player_predictions.run_timestamp
    model_version  VARCHAR(32) NOT NULL,
    base_gameweek  INT NOT NULL,                   -- first gameweek predicted (GW+1)
    horizon        INT NOT NULL,
    params         JSONB NOT NULL,                 -- every tunable constant used
    inputs         JSONB NOT NULL,                 -- hashes / timestamps of the data snapshot used
    diagnostics    JSONB,                          -- training sizes, fitted league constants, timings
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE player_predictions
    ADD COLUMN n_fixtures        INT,
    ADD COLUMN p_start           NUMERIC(4,3),
    ADD COLUMN expected_minutes  NUMERIC(5,1),
    ADD COLUMN expected_goals    NUMERIC(5,3),
    ADD COLUMN expected_assists  NUMERIC(5,3),
    ADD COLUMN p_clean_sheet     NUMERIC(4,3),
    ADD COLUMN xp_appearance     NUMERIC(5,2),
    ADD COLUMN xp_deductions     NUMERIC(5,2);
CREATE INDEX player_predictions_run_idx ON player_predictions (run_timestamp, target_gw);

ALTER TABLE player_predictions
    ADD CONSTRAINT player_predictions_run_fk FOREIGN KEY (run_timestamp) REFERENCES prediction_runs(run_timestamp);

-- Team goal expectations used by each run (model vs odds blend), for auditing fixture swings.
CREATE TABLE fixture_predictions (
    run_timestamp     TIMESTAMPTZ NOT NULL REFERENCES prediction_runs(run_timestamp),
    fixture_id        INT NOT NULL REFERENCES fixtures(fixture_id),
    lambda_home       NUMERIC(5,3) NOT NULL,
    lambda_away       NUMERIC(5,3) NOT NULL,
    lambda_home_model NUMERIC(5,3) NOT NULL,
    lambda_away_model NUMERIC(5,3) NOT NULL,
    odds_weight       NUMERIC(4,3) NOT NULL,       -- 0 when no odds snapshot was available
    PRIMARY KEY (run_timestamp, fixture_id)
);

-- Latest run's predictions, convenient for Phase 4 and dashboards.
CREATE VIEW latest_player_predictions AS
SELECT pp.*
FROM player_predictions pp
WHERE pp.run_timestamp = (SELECT MAX(run_timestamp) FROM prediction_runs);

CREATE TABLE model_backtests (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    season      VARCHAR(7) NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    params      JSONB NOT NULL,
    metrics     JSONB NOT NULL
);
